import os
import numpy as np
import uuid

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

def get_all_fjsp_instances():
    '''Lists all FJSP instances'''
    matching_files = []
    root_dir = "../../../../benchmarks/fjsp"
    target_string = ".fjs"

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            if target_string in filepath:
                matching_files.append(filepath)

    return matching_files

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

def jssp_taillard_to_fjsp(filename: str, seed: int|None = None) -> str:
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

import time
from Params import configs
from validation_realWorld import test
from torch.utils.data import DataLoader
from PPOwithValue import PPO
from DataRead import getdata
import torch
import os
import copy
import itertools
from FJSP_Env import FJSP
from mb_agg import g_pool_cal
from mb_agg import *
from copy import deepcopy

def validate(vali_set,batch_size, policy_job,policy_mch,num_operation,number_of_task,Data, plan: list | None = None):
    policy_job = copy.deepcopy(policy_job)
    policy_mch = copy.deepcopy(policy_mch)
    policy_job.eval()
    policy_mch.eval()
    def eval_model_bat(bat, plan: list | None = None):
        actions = []
        start_times = []
        with torch.no_grad():
            data = bat.numpy()

            env = FJSP(n_j=Data['n'], n_m=configs.n_m,EachJob_num_operation=num_operation)
            device = torch.device(configs.device)
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [batch_size, number_of_task, number_of_task]),
                                     n_nodes=number_of_task,
                                     device=device)

            adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time = env.reset(data)
            first_task = []
            pretask = []

            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool=None
            for j in itertools.count():
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), number_of_task)

                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
                env_mask = torch.from_numpy(np.copy(mask)).to(device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
                
                if plan is not None and j < len(plan):
                    # choose action from partial plan if possible
                    action, a_idx, log_a, action_node, _, mask_mch_action, hx = plan[j]
                else:
                    # if no action was chosen from partial plan, use agent to choose
                    action, a_idx, log_a, action_node, _, mask_mch_action, hx = policy_job(x=env_fea,
                                                                                                   graph_pool=g_pool_step,
                                                                                                   padded_nei=None,
                                                                                                   adj=env_adj,
                                                                                                   candidate=env_candidate
                                                                                                   , mask=env_mask
    
                                                                                                   , mask_mch=env_mask_mch
                                                                                                   , dur=env_dur
                                                                                                   , a_index=0
                                                                                                   , old_action=0
                                                                                           , mch_pool=pool
                                                                                                   ,old_policy=True,
                                                                                                    T=1
                                                                                                   ,greedy=True
                                                                                                   )
                actions.append((action, a_idx, log_a, action_node, _, mask_mch_action, hx))
                pi_mch,_,pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time)
                _, mch_a = pi_mch.squeeze(-1).max(1)

                if j == 0:
                    first_task = action.type(torch.long).to(device)
                pretask = action.type(torch.long).to(device)

                # make an action
                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a)

                action = action.cpu().numpy()
                row = np.where(action[0] <= env.last_col[0])[0][0]
                col = action[0] - env.first_col[0][row]
                start_times.append(env.temp1[0][row][col] - env.dur_a)

                if env.done():
                    break

            cost = env.mchsEndTimes.max(-1).max(-1)
            # assert cost[0] == cost[1], f/"First cost and second cost are not equal {cost[0], cost[1]}"
        return cost[0], actions, start_times

    for bat in vali_set:
        assert torch.equal(bat[0], bat[1]), "First and second matrix in batch are not equal"
        return eval_model_bat(bat, plan)

def test(filepath, datafile, plan: list | None = None):

    Data = getdata(datafile)
    n_j = Data['n']
    n_m = Data['m']

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                n_j=n_j,
                n_m=n_m,
                num_layers=configs.num_layers,
                neighbor_pooling_type=configs.neighbor_pooling_type,
                input_dim=configs.input_dim,
                hidden_dim=configs.hidden_dim,
                num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                num_mlp_layers_actor=configs.num_mlp_layers_actor,
                hidden_dim_actor=configs.hidden_dim_actor,
                num_mlp_layers_critic=configs.num_mlp_layers_critic,
                hidden_dim_critic=configs.hidden_dim_critic)

    job_path = './{}.pth'.format('policy_job')
    mch_path = './{}.pth'.format('policy_mch')

    job_path = os.path.join(filepath,job_path)
    mch_path = os.path.join(filepath, mch_path)

    ppo.policy_job.load_state_dict(torch.load(job_path, map_location=torch.device('cpu')))
    ppo.policy_mch.load_state_dict(torch.load(mch_path, map_location=torch.device('cpu')))

    batch_size = 2
    num_operations = []
    num_operation = []
    for i in Data['J']:
        num_operation.append(Data['OJ'][i][-1])
    num_operation_max = np.array(num_operation).max()

    time_window = np.zeros(shape=(Data['n'], num_operation_max, Data['m']))

    data_set = []
    for i in range(Data['n']):
        for j in Data['OJ'][i+1]:
            mchForJob = Data['operations_machines'][(i + 1, j)]
            for k in mchForJob:
                time_window[i][j-1][k - 1] = Data['operations_times'][(i + 1, j, k)]


    for i in range(batch_size):
        num_operations.append(num_operation)
        data_set.append(time_window)
    data_set = np.array(data_set)

    num_operation = np.array(num_operations)
    number_of_tasks = num_operation.sum(axis=1)[0]
    number_of_tasks = int(number_of_tasks)

    valid_loader = DataLoader(data_set, batch_size=batch_size)
    makespan, actions, start_times = validate(valid_loader,batch_size, ppo.policy_job, ppo.policy_mch,num_operation,number_of_tasks,Data, plan)

    return makespan, actions, start_times

def solve_fjsp_instance(model, instance, plan: list | None = None):
    '''Solves FJSP instance using given model

    Args:
      model - model to use for solving the instance
      instance - instance to be solved
      plan - preset actions to make by agent

    Returns:
      makespan of the instance,
      actions made by agent,
      time it took to solve the instance
    '''
    # get number of machines
    with open(instance, 'r') as f:
        M = int(f.readline().strip().split()[1])

    # override the value of number of machines in the configs according to the instance 
    # this is not handled in the original code and causes errors
    setattr(configs, 'n_m', M)

    # solve the instance
    start = time.time()
    makespan, actions, start_times = test(model, instance, plan)
    end = time.time()

    return makespan, actions, start_times, end - start



if __name__ == '__main__':
    DATA_FILE = 'experiment_static_fjsp_end_to_end_drl_for_fjsp.csv'
    with open(DATA_FILE, 'w') as f:
        f.write("seed,model,instance,makespan,runtime\n")

    models = ['saved_network/FJSP_J15M15/best_value0']
    seeds = list(range(10))
    instances = sorted(get_all_fjsp_instances())

    count = 0
    total = len(seeds) * len(models) * len(instances)
    exceptions = [
        '../../../../benchmarks/fjsp/1_Brandimarte/BrandimarteMk7.fjs',
        '../../../../benchmarks/fjsp/1_Brandimarte/BrandimarteMk8.fjs',
        '../../../../benchmarks/fjsp/1_Brandimarte/BrandimarteMk9.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2a_Hurink_sdata/HurinkSdata63.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2b_Hurink_edata/HurinkEdata63.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2c_Hurink_rdata/HurinkRdata63.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2d_Hurink_vdata/HurinkVdata63.fjs'
    ]

    for model in models:
        for seed in seeds:
            for instance in instances:
                if instance in exceptions:
                    # logging.info(f"Skipping '{instance}'")
                    continue

                # run experiment
                makespan, _, _, runtime = solve_fjsp_instance(model, shuffle_file_rows(instance, seed=seed))
                
                # save result to data file
                instance_name = instance.split('/')[-1]
                with open(DATA_FILE, 'a') as f:
                    f.write(f"{seed},{model},{instance_name},{makespan},{runtime}\n")
                
                count += 1
                logging.info(f"{count=} {total=} {seed=} {model=} {instance_name=} {makespan=} {runtime=} ")
