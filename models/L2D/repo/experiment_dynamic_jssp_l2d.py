from datetime import datetime

from mb_agg import *
from agent_utils import *
from Params import configs
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO

import torch
import os
import numpy as np
import itertools
import time
import sys
import uuid
import pandas as pd

import logging

# Configure the logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

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

def solve_instance(instance: str, model: str, plan=None, device='cpu', machine_start_times=None, t:int=0):
    '''Solves instance using given model and prints out the makespan
    
      Args:
          instance - file with the instance in Taillard specification
          model - network model to use as an agent

      Returns:
          makespan of the solution,
          dispatch times of operations,
          actions executed by the agent
    '''
    # parse the instance
    jobs, machines, times, orders = parse_instance_taillard(instance)

    # load agents environment
    env = SJSSP(n_j=jobs, n_m=machines)
    adj, fea, candidate, mask = env.reset((times, orders))
    ep_reward = - env.max_endTime
    if machine_start_times is not None:
        env.mchsStartTimes = machine_start_times

    # load the agent
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=jobs,
              n_m=machines,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    ppo.policy.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)

    # run the experiment
    for i in itertools.count():
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)

        action = None
        # choose action from partial plan if given
        if plan is not None and i < len(plan):
            env.set_current_time(plan[i][1])
            action = np.int64(plan[i][0])
            # assert action in candidate, f"Action {action} not in candidate {candidate}" 

        # if no action was chosen from partial plan, use agent to choose
        if action is None:
            env.set_current_time(t)
            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                action = greedy_select_action(pi, candidate)


        adj, fea, reward, done, candidate, mask = env.step(action)
        ep_reward += reward

        if done:
            break

    start_times = env.LBs - times
    return env.posRewards - ep_reward, start_times, env.partial_sol_sequeence


def get_dynamics_jssp(instance, load_factor: float = 1.0, seed: int | None = None):
    '''Turns static JSSP instance to dynamic

      Args:
        filename of static JSSP instance
        load_factor
        seed

      Returns:
        list of jobs known at the beginning
        dictionary of arriving jobs as  as {time_of_arrival: (operations, machines)} 
    '''
    J, _, processor_times, orders_of_machines = parse_instance_taillard(instance)

    indices = np.arange(J)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    # separate jobs into known jobs and arriving jobs
    jobs_known_at_the_beginning = [(processor_times[i], orders_of_machines[i]) for i in indices[J//2:]]
    arriving_jobs_indeces = indices[:J//2]

    # calculate beta = 1/lambda
    average_time_between_arrivals = processor_times.mean() / load_factor
    
    t = 0
    arriving_jobs = {}
    for index in arriving_jobs_indeces:
        t += int(np.random.exponential(scale=average_time_between_arrivals)) + 1
        arriving_jobs[t] = (processor_times[index], orders_of_machines[index])
        
    return jobs_known_at_the_beginning, arriving_jobs

def save_static_jssp_taillard(jobs):
    '''Saves list of jobs as static JSSP instance in taillards specification
        
      Args:
        list of jobs to save

      Returns:
        filename where JSSP instance was saved to
    '''
    J, M = len(jobs), len(jobs[0][0])
    formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    file_uuid = uuid.uuid4()
    with open(f"/tmp/{J}_{M}_{formatted_datetime}_{str(file_uuid)}.txt", 'w') as f:
        f.write(f"{J} {M}\n")
        for job in jobs:
            times, _ = job
            f.write(" ".join(map(str, times)) + '\n')
        for job in jobs:
            _, orders = job
            f.write(" ".join(map(str, orders)) + '\n')  

    return f"/tmp/{J}_{M}_{formatted_datetime}_{str(file_uuid)}.txt"

def solve_dynamic_jssp(instance, model, load_factor, seed):
    '''Turns static JSSP in Taillard specification instance to dynamic and solves it

      Args: 
        instance to solve

      Returns: 
        makespan
    '''
    # turn static JSSP instance to dynamic
    known_jobs, arriving_jobs = get_dynamics_jssp(instance, load_factor, seed)
    
    # solve static JSSP with jobs known initially
    makespan, start_times, actions = solve_instance(save_static_jssp_taillard(known_jobs), model)
    t = 0
    plan = []
    while True:
        t += 1
        
        # no jobs left
        if not arriving_jobs:
            break
    
        # no job arrived
        if not t in arriving_jobs:
            continue
    
        # new job arrived, remove not yet executed operations from the plan
        J, M = len(known_jobs), len(known_jobs[0][0])
        for i in range(len(actions)):
            row = actions[i] // M
            col = actions[i] % M

            if not start_times[row][col] < t:
                continue
                
            if (actions[i], start_times[row][col]) in plan:
                continue
    
            plan.append((actions[i], start_times[row][col]))

        # add new job to the plan, with times shifted to current time t
        new_job = arriving_jobs.pop(t)
        known_jobs.append(new_job)
        
        # create new schedule WHILE REUSING THE ALREADY EXECUTED PLAN
        makespan, start_times, actions = solve_instance(save_static_jssp_taillard(known_jobs), model, plan=plan, t=t)

    return makespan, plan, start_times


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


if __name__ == '__main__':
    # MODELS = [os.path.join("SavedNetwork", model) for model in os.listdir("SavedNetwork")]

    LOAD_FACTORS = [1, 2, 4]
    INSTANCES = sorted(get_all_instances_in_taillard_specification())

    if not len(sys.argv) > 1:
        raise ValueError('Model not given!')
    
    model = sys.argv[1]
    DATA_FILE = f'experiment_dynamic_jssp_l2d_seed_{model.replace("/", "_")}.csv'
    PREV_DATA_FILE = f'experiment_dynamic_jssp_l2d_seed_{model.replace("/", "_")}_prev.csv'
    with open(DATA_FILE, 'w') as f:
        f.write("seed,model,instance,load_factor,makespan,runtime\n")

    df = pd.read_csv(PREV_DATA_FILE)
    print(df)
    count = 0
    total = 2 * len(INSTANCES) * len(LOAD_FACTORS)
    for load_factor in LOAD_FACTORS:
        for instance in INSTANCES:
            for seed in [8, 9]:
                instance_name = instance.split('/')[-1]
                filtered_df = df[(df['seed'] == seed) & (df['model'] == model) & (df['instance'] == instance_name) & (df['load_factor'] == load_factor)]
                if not filtered_df.empty:
                    makespan = filtered_df['makespan'].iloc[0]
                    runtime = filtered_df['runtime'].iloc[0]
                else:
                    # run the experiment
                    start = time.time()
                    makespan, plan, start_times = solve_dynamic_jssp(instance, model, load_factor, seed)
                    runtime = time.time() - start

                # save the data
                with open(DATA_FILE, 'a') as f:
                    f.write(f"{seed},{model},{instance_name},{load_factor},{makespan},{runtime}\n")

                # log the progress
                count += 1
                logging.info(f"{count=} {total=} {seed=} {model=} {instance_name=} {load_factor=} {makespan=} {runtime=} ")
