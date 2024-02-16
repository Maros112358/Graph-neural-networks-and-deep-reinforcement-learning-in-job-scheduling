from jssp.models.agent import Agent

# import Wheatley jssp solver
from jssp.solve import solve_instance
from jssp.utils.loaders import load_problem

# load trained agent
import os
import time
import uuid
import logging
import numpy as np
import pandas as pd

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
        processor_times = []
        orders_of_machines = []
    
        # Read the next J lines containing processor times
        for i in range(J):
            processor_times.append(list(map(int, f.readline().split())))
    
        # Read the next J lines containing orders of machines
        for i in range(J):
            orders_of_machines.append(list(map(int, f.readline().split())))

        return J, M, processor_times, orders_of_machines
    
def shuffle_taillard(file_path, seed=None):
    """Shuffle the rows of a file, keeping the first row intact, with an optional random seed.
    
    Args:
      file_path - path to the file to be shuffled.
      seed - seed for the random number generator

    Returns:
      file name with shuffled data rows
    """
    # Set the seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Keep the first line intact and shuffle the rest
    header = lines[0]
    data = lines[1:]
    half = int(len(data) / 2)
    jobs = np.array(data[:half])
    machines = np.array(data[half:])
    indices = list(range(half))
    np.random.shuffle(indices)
    jobs = list(jobs[indices])
    machines = list(machines[indices])
    data = jobs + machines

    # Write the shuffled data back to the file
    new_file_path = "/tmp/shuffled_" + str(uuid.uuid4()) + '_' + file_path.split("/")[-1]
    with open(new_file_path, 'w') as file:
        file.write(header)
        file.writelines(data)

    return new_file_path

def solve_instance_taillard(instance, agent):
    '''Solves JSSP instance in Taillard specification
    
      Args:
        instance: instance to solve
        agent: agent to use for solving the problem

      Returns:
        solution
    '''
    n_j, n_m, affectations, durations = load_problem(
        instance,
        taillard_offset=True,
        deterministic=True
    )

    assert agent.env_specification.max_n_jobs >= n_j
    assert agent.env_specification.max_n_machines >= n_m
        
    start = time.time()
    solution = solve_instance(
        agent, affectations, durations, True
    )
    runtime = time.time() - start

    return solution, runtime

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
    SEEDS = list(range(10))
    MODELS = [os.path.join("saved_models", model, 'agent.pkl') for model in os.listdir("saved_models") if 'SAVE' not in model] 
    INSTANCES = sorted(get_all_instances_in_taillard_specification())
    DATA_FILE = 'experiment_static_jssp_wheatley.csv'
    PREV_DATA_FILE = 'experiment_static_jssp_wheatley_save.csv'
    df = pd.read_csv(PREV_DATA_FILE)

    # prepare data file
    with open(DATA_FILE, 'w') as f:
        f.write("seed,model,instance,makespan,runtime\n")

    # run the experiment
    count = 0
    total = len(SEEDS) * len(MODELS) * len(INSTANCES)
    for model in MODELS:
        for instance in INSTANCES:
            for seed in SEEDS:
                instance_name = instance.split('/')[-1]

                filtered_df = df[(df['seed'] == seed) & (df['model'] == model) & (df['instance'] == instance_name)]
                if not filtered_df.empty:
                    makespan = filtered_df['makespan'].iloc[0]
                    runtime = filtered_df['runtime'].iloc[0]
                else:
                    agent = Agent.load(model)
                    J, M, _, _ = parse_instance_taillard(instance)
                    assert agent.env_specification.max_n_jobs >= J
                    assert agent.env_specification.max_n_machines >= M
                
                    shuffled_instance = shuffle_taillard(instance, seed=seed)
                    solution, runtime = solve_instance_taillard(shuffled_instance, agent)
                    makespan = solution.get_makespan()
                    
                with open(DATA_FILE, 'a') as f:
                    f.write(f"{seed},{model},{instance_name},{makespan},{runtime}\n")

                count += 1
                logging.info(f"{count=} {total=} {seed=} {model=} {instance_name=} {makespan=} {runtime=} ")
