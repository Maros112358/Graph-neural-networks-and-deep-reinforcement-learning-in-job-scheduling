from mb_agg import *
from Params import configs
from copy import deepcopy
from FJSP_Env import FJSP
from mb_agg import g_pool_cal
import copy
from DataRead import getdata
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import os
from uniform_instance import uni_instance_gen,FJSPDataset
from Params import configs

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

def jssp_taillard_to_fjsp(filename: str) -> str:
    '''Transforms JSSP instance in Taillard's specification to FJSP instance
       and stores it in a temporary file
    
      Args:
        filename - name of the file with JSSP instance in Taillard's specification
        
      Returns:
        string - filename of the equivalent FJSP instance 
    '''
    # parse JSSP Taillard instance
    J, M, processor_times, orders_of_machines = parse_instance_taillard(filename)
    
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

def validate(vali_set,batch_size, policy_job,policy_mch,num_operation,number_of_task,Data):
    # print(f"{vali_set,batch_size, policy_job,policy_mch,num_operation,number_of_task,Data=}")
    policy_job = copy.deepcopy(policy_job)
    policy_mch = copy.deepcopy(policy_mch)
    policy_job.eval()
    policy_mch.eval()
    def eval_model_bat(bat):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()

            # print(Data['n'], Data['m'])
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
            j = 0

            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            pool=None
            while True:
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), number_of_task)

                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(candidate)).long().to(device)
                env_mask = torch.from_numpy(np.copy(mask)).to(device)
                env_mch_time = torch.from_numpy(np.copy(mch_time)).float().to(device)
                # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
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

                pi_mch,_,pool = policy_mch(action_node, hx, mask_mch_action, env_mch_time)

                _, mch_a = pi_mch.squeeze(-1).max(1)

                if j == 0:
                    first_task = action.type(torch.long).to(device)

                pretask = action.type(torch.long).to(device)

                adj, fea, reward, done, candidate, mask,job,_,mch_time,job_time = env.step(action.cpu().numpy(), mch_a)
                #rewards += reward

                j += 1
                # print(j)
                if env.done():
                    # print('done')
                    break

            cost = env.mchsEndTimes.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)
    
    totall_cost = torch.cat([eval_model_bat(bat) for bat in vali_set], 0)

    return totall_cost

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

def test(filepath, datafile):
    from torch.utils.data import DataLoader
    from PPOwithValue import PPO
    import torch
    import os

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

    SEEDs = [200]
    mean_makespan = []
    for SEED in SEEDs:
        valid_loader = DataLoader(data_set, batch_size=batch_size)
        vali_result = validate(valid_loader,batch_size, ppo.policy_job, ppo.policy_mch,num_operation,number_of_tasks,Data).mean()
        mean_makespan.append(vali_result)

    return np.array(mean_makespan).mean()

if __name__ == '__main__':

    import numpy as np
    import time
    import argparse
    from Params import configs

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=6, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=6, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=6, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=6, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=-1, help='LB of duration')
    parser.add_argument('--high', type=int, default=1, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args(args=[])


    filename = './FJSSPinstances/M15'
    filename = get_imlist(filename)
    print(filename)
    filepath = 'saved_network'
    filepath = os.path.join(filepath, 'FJSP_J%sM%s' % (15,15))
    filepaths = get_imlist(filepath)
    print(filepaths)
    #---------------------------------------------------------------------------------------------
    '''data_file = './FJSSPinstances/0_BehnkeGeiger/Behnke13.fjs'
    result = []
    for filepath in filepaths:
        a = test(filepath, data_file)
        result.append(a)
    min = np.array(result).min()
    print('min', min)'''

    #---------------------------------------------------------------------------------------------
    J,M, _, _ = parse_instance_taillard("../../../../benchmarks/orb_instances/Taillard_specification/orb07.txt")
    setattr(configs, 'n_m', M)    
    a = test("saved_network/FJSP_J15M15/best_value0", jssp_taillard_to_fjsp("../../../../benchmarks/orb_instances/Taillard_specification/orb07.txt"))
