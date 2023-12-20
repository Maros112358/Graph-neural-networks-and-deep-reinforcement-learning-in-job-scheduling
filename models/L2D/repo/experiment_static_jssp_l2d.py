from mb_agg import *
from agent_utils import *
from Params import configs
from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO

import os
import time
import torch
import numpy as np
import itertools

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
    
def solve_instance(instance: str, model: str, plan=None, device='cpu', machine_start_times=None, t:int=0, seed: int | None=None):
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
    if seed is not None:
      np.random.seed(seed)
      indices = np.arange(jobs)
      np.random.shuffle(indices)
      times = times[indices]
      orders = orders[indices]
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
    g_pool_step = g_pool_step.to(device)

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
            assert action in candidate, f"Action {action} not in candidate {candidate}"

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
    MODELS = [os.path.join("SavedNetwork", model) for model in os.listdir("SavedNetwork")]
    INSTANCES = get_all_instances_in_taillard_specification()
    DATA_FILE = 'experiment_static_jssp_l2d.csv'
    BENCHMARKS_PATH = "../../../benchmarks/jssp/ta_instances/Taillard_specification/"

    with open(DATA_FILE, 'w') as f:
        f.write("seed,model,instance,makespan,runtime\n")

    count = 0
    total = len(SEEDS) * len(MODELS) * len(INSTANCES)
    for seed in SEEDS:
        for model in MODELS:
            for instance in INSTANCES:
                start = time.time()
                makespan, _, _ = solve_instance(instance, model, seed=seed)
                runtime = time.time() - start
                instance = instance.split('/')[-1]
                with open(DATA_FILE, 'a') as f:
                    f.write(f"{seed},{model},{instance},{makespan},{runtime}\n")

                count += 1
                logging.info(f"{count=} {total=} {seed=} {model=} {instance=} {makespan=} {runtime=} ")
