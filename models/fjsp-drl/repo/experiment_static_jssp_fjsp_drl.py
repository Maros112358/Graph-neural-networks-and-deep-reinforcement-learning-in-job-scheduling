import torch, json, PPO_model, copy, os, gym, time
import numpy as np
from env.load_data import nums_detec
import uuid
import pandas as pd

import logging

# Configure the logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def shuffle_file_rows(file_path, seed=None):
    """Shuffle the rows of a file, keeping the first row intact, with an optional random seed.
    
    Args:
      file_path - path to the file to be shuffled.
      see - seed for the random number generator

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
    np.random.shuffle(data)

    # Write the shuffled data back to the file
    new_file_path = "/tmp/shuffled_" + str(uuid.uuid4()) + '_' + file_path.split("/")[-1]
    with open(new_file_path, 'w') as file:
        file.write(header)
        file.writelines(data)

    return new_file_path


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

def jssp_taillard_to_fjsp(filename: str, seed: int = None) -> str:
    '''Transforms JSSP instance in Taillard's specification to FJSP instance
       and stores it in a temporary file
    
      Args:
        filename - name of the file with JSSP instance in Taillard's specification
        
      Returns:
        string - filename of the equivalent FJSP instance 
    '''
    # parse JSSP Taillard instance
    J, M, processor_times, orders_of_machines = parse_instance_taillard(filename)
    if seed is not None:
        np.random.seed(seed)
        indices = np.arange(J)
        np.random.shuffle(indices)
        processor_times = processor_times[indices]
        orders_of_machines = orders_of_machines[indices]
    
    # convert JSSP to FJSP
    with open("/tmp/fjsp_" + filename.split("/")[-1], 'w') as f:
        # write number of jobs, number of machines, and jobs/machines (which is always 1 for JSSP)
        f.write(str(J) + "   " + str(M) + "   1\n")
        
        # each line is a job
        for i in range(J):
            # each line starts with the number of operations in a job
            number_of_operations = len(processor_times[i])
            f.write(str(number_of_operations) + "  ")
            
            # print the operation as a tuple (number of available machines, current machine, processing time)
            for j in range(number_of_operations):
                f.write(" 1   " + str(orders_of_machines[i][j]) + "   " + str(processor_times[i][j]) + "  ")
                
            f.write('\n')
            
    return "/tmp/fjsp_" + filename.split("/")[-1]
    
def get_model(model_path: str):
    '''Loads model and returns model object and memories object'''
    # setup torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        
    # load configuration of the experiment
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
        
    model_paras = load_dict['model_paras']
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras['device'] = device
    env_paras = load_dict['env_paras']
    env_paras['device'] = device
    env_paras['batch_size'] = 1

    # create model and its parameters from the checkpoint
    memories = PPO_model.Memory()
    model_CKPT = torch.load(model_path, map_location=torch.device('cpu'))
    model = PPO_model.PPO(model_paras, load_dict['train_paras'])
    model.policy.load_state_dict(model_CKPT)
    model.policy_old.load_state_dict(model_CKPT)
    return model, memories, env_paras

def solve_instance(instance_file, model, memories, env_paras, flag_sample=False):
    '''Solves FJSP instance using given model and memories
    
      Args:
          instance_file - name of the file with the FJSP instance
          model - model used to solve the instance
          memories - model's memories
    
      Returns:
          makespan and time it took to solve the instance
    '''
    with open(instance_file, 'r') as file_object:
        # load the parameters of the instance
        line = file_object.readlines()
        ins_num_jobs, ins_num_mas, _ = nums_detec(line)
        env_paras["num_jobs"] = ins_num_jobs
        env_paras["num_mas"] = ins_num_mas
        
        # create env and get states and completion signal
        env = gym.make('fjsp-v0', case=[instance_file], env_paras=env_paras, data_source='file')
        state = env.state
        dones = env.done_batch
        done = False  # Unfinished at the beginning
        
        # perform scheduling
        start = time.time()
        while not done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
            state, rewards, dones = env.step(actions)  # environment transit
            done = dones.all()
        run_time = time.time() - start  # The time taken to solve this environment (instance)

        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            raise Exception("Scheduling error")

        return copy.deepcopy(env.makespan_batch), run_time
    
def get_all_jssp_instances_in_taillard_specification():
    '''Lists all instances in Taillard specification'''
    matching_files = []
    root_dir = "../../../benchmarks/jssp"
    target_string = "Taillard_specification"

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            if target_string in filepath:
                matching_files.append(filepath)

    return matching_files
    

if __name__ == '__main__':
    DATA_FILE = 'experiment_static_jssp_fjsp_drl.csv'
    PREV_DATA_FILE = 'experiment_static_jssp_fjsp_drl_prev.csv'
    EXCEPTIONS = [
        '../../../benchmarks/jssp/orb_instances/Taillard_specification/orb07.txt' # this instance gets stuck while solving, I haven't figured out why
    ]
    with open(DATA_FILE, 'w') as f:
        f.write("seed,model,instance,makespan,runtime\n")

    models = [os.path.join('results', file_path) for file_path in os.listdir('results')] + ['model/save_10_5.pt']
    seeds = list(range(10))
    instances = sorted(get_all_jssp_instances_in_taillard_specification())

    df = pd.read_csv(PREV_DATA_FILE)

    count = 0
    total = len(seeds) * len(models) * len(instances)
    for model_path in models:
        # load model
        model, memories, env_paras = get_model(model_path)
        for instance in instances:
            if instance in EXCEPTIONS:
                logging.info("Skipping instance '{}'".format(instance))
                continue
            
            for seed in seeds:
                instance_name = instance.split('/')[-1]
                filtered_df = df[(df['seed'] == seed) & (df['model'] == model_path) & (df['instance'] == instance_name)]
                if not filtered_df.empty:
                    makespan = filtered_df['makespan'].iloc[0]
                    run_time = filtered_df['runtime'].iloc[0]
                else:
                    # load instance
                    fjsp_instance = jssp_taillard_to_fjsp(instance, seed=seed)

                    # solve instance
                    makespan, run_time = solve_instance(fjsp_instance, model, memories, env_paras)
                    makespan = makespan.item()

                # save results to data file
                with open(DATA_FILE, 'a') as f:
                    f.write(f"{seed},{model_path},{instance_name},{makespan},{run_time}\n")

                # log message
                count += 1
                logging.info("{} {} {} {} {} {} {}".format(
                    'count=' + str(count),
                    'total=' + str(total),
                    'seed=' + str(seed),
                    'model=' + str(model_path),
                    'instance_name=' + str(instance_name),
                    'makespan=' + str(makespan),
                    'runtime=' + str(run_time)
                ))
