# import some stuff and define helper functions
import os, argparse, time
import numpy as np
import sys

from env.env import JSP_Env
from agent.DQN.agent import DQN_Agent
import uuid

import logging

# Configure the logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def eval_dqn(model, instance_path, args, arrival_times: list | None = None):
    '''Solve the JSSP instance in file instance_path using given model
    
      Args:
        model - path of model to use
        instance_path - path of instance to solve
        args - args from argparser
        partial_plan - steps to execute at start instead of using the agent

      Return:
        makespan
        start_times - start time of operations executed in given order
        plan - sequence of actions
    '''
    # load env
    env = JSP_Env(args)
    env.load_instance(instance_path, arrival_times)
    state = env.get_graph_data(args.device)

    # load agent
    agent = DQN_Agent(args, out_dim=len(env.rules))
    agent.load(model)     

    # run the model
    i = 0
    while True:
        # choose action from partial plan if given
        action = agent.select_action(state, random=False, test_only=True)
        state, _, done, _, _ = env.step(action)
        i += 1
        if done:
            break

    return env.get_makespan()

def parse_instance_taillard(filename):
    '''Parses instance written in Taillard specification: http://jobshop.jjvh.nl/explanation.php
    
      Args:
        filename - file containing the instance in Taillard specification

      Returns:
        number of jobs,
        number of machines,
        the processor times for each operation,
        the order for visiting the machines
    '''

    with open(filename, 'r') as f:
        # parse number of jobs J and machines M
        J, M = map(int, f.readline().split())

        # Initialize two empty numpy arrays with dimensions J x M
        processor_times = np.empty((J, M), dtype=int)
        orders_of_machines = np.empty((J, M), dtype=int)
    
        # Read the next J lines containing processor times
        for i in range(J):
            processor_times[i] = list(map(int, f.readline().split()))
    
        # Read the next J lines containing orders of machines
        for i in range(J):
            orders_of_machines[i] = list(map(int, f.readline().split()))

        return J, M, processor_times, orders_of_machines

def taillard_to_standard(taillard_instance, seed: int | None = None):
    # parse taillard instance
    J, M, processor_times, orders_of_machines = parse_instance_taillard(taillard_instance)
    if seed is not None:
        np.random.seed(seed)
        indices = np.arange(J)
        np.random.shuffle(indices)
        processor_times = processor_times[indices]
        orders_of_machines = orders_of_machines[indices]

    # save as standard instance
    standard_instance = "/tmp/standard_" + str(uuid.uuid4()) + '_' + taillard_instance.split("/")[-1]
    with open(standard_instance, 'w') as f:
        # save number of jobs and machines
        f.write(f"{J}\t{M}\n")

        for job in range(J):
            for machine in range(M):
                f.write(f'{orders_of_machines[job][machine] - 1}\t{processor_times[job][machine]} ')

            f.write('\n')

    return standard_instance

def get_all_instances_in_taillard_specification():
    '''Lists all instances in Taillard specification'''
    matching_files = []
    root_dir = "../../../benchmarks/jssp/"
    target_string = "Taillard_specification"

    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            if target_string in filepath:
                matching_files.append(filepath)

    return matching_files

def get_dynamic_jssp(instance: str, load_factor: float = 1.0, seed: int | None = None):
    '''Turns static JSSP instance to dynamic

      Args:
        filename of static JSSP instance
        load factor of the job shop
        seed for the random generator

      Returns:
        list of jobs known at the beginning
        dictionary of arriving jobs as  as {time_of_arrival: (operations, machines)} 
    '''
    J, _, processor_times, _ = parse_instance_taillard(instance)

    indices = np.arange(J)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    # separate jobs into known jobs and arriving jobs
    arriving_jobs_indeces = indices[:J//2]

    # calculate beta = 1/lambda
    average_time_between_arrivals = processor_times.mean() / load_factor

    arrival_times = np.zeros(J, dtype=int)
    t = 1
    for index in arriving_jobs_indeces:
        t += int(np.random.exponential(scale=average_time_between_arrivals)) + 1
        arrival_times[index] = t

    return list(arrival_times)


if __name__ == '__main__':
    MODELS_PATH = 'agent/DQN/weight'
    BENCHMARKS_PATH = "../../../benchmarks/jssp/ta_instances/Taillard_specification/"
    DATA_FILE = f'experiment_dynamic_jssp_ieee_icce_rl_jsp.csv'

    # need command line arguments for the model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cpu')
    # arguments for DQN
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=100000, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=.01, type=float)
    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    parser.add_argument('--double', action='store_true')
    parser.add_argument(
        '--max_process_time',
        type=int,
        default=100,
        help='Maximum Process Time of an Operation')
    args = parser.parse_args(args=[])

    # prepare csv file header
    with open(DATA_FILE, 'w') as f:
        f.write("seed,model,instance,load_factor,makespan,runtime\n")

    # load models and instances and seeds
    models = os.listdir(MODELS_PATH)
    instances = sorted(get_all_instances_in_taillard_specification())
    load_factors = [1, 2, 4]
    seeds = list(range(10))

    # run experiments
    count = 0
    total = len(seeds) * len(models) * len(instances) * len(load_factors)
    logging.info(f"{seeds=}")
    logging.info(f"{models=}")
    logging.info(f"{load_factors=}")
    for model in models:
        for load_factor in load_factors:
            for instance in instances:
                for seed in seeds:
                    # oad model and prepare instance
                    model_path = os.path.join(MODELS_PATH, model)

                    # prepare dynamic arrival times
                    arrival_times = get_dynamic_jssp(instance, load_factor=load_factor, seed=seed)
                    
                    # run experiment
                    standard_instance = taillard_to_standard(instance, seed=seed)
                    start = time.time()
                    makespan = eval_dqn(model_path, standard_instance, args, arrival_times=arrival_times)
                    runtime = time.time() - start

                    instance_name = instance.split('/')[-1]
                    with open(DATA_FILE, 'a') as f:
                        f.write(f"{seed},{model},{instance_name},{load_factor},{makespan},{runtime}\n")

                    count += 1
                    logging.info(f"{count=} {total=} {seed=} {model=} {instance_name=} {load_factor=} {makespan=} {runtime=}")
