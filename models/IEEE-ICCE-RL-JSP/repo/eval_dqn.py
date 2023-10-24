import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Process
from collections import defaultdict, Counter
from agent.DQN.agent import DQN_Agent
from env.env import JSP_Env

seed = 1000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def eval_dqn(weight_path, instance_path, args):
    start = time.time()
    env = JSP_Env(args)
    agent = DQN_Agent(args, out_dim=len(env.rules))
    agent.load(weight_path)
    avai_ops = env.load_instance(instance_path)
    job_num = env.jsp_instance.initial_job_num
    machine_num = env.jsp_instance.machine_num
    state = env.get_graph_data(args.device)
    tic = datetime.now()
    while True:
        action = agent.select_action(state, random=False, test_only=True)
        state, reward, done, info = env.step(action)
        if done:
            break

    makespan = env.get_makespan()
    tardiness = env.get_tardiness()
    setup_count = env.jsp_instance.setup_count
    toc = datetime.now()
    # print(
    #     f"{instance_path}\t"
    #     f"{job_num}\t"
    #     f"{machine_num}\t"
    #     f"{weight_path}\t"
    #     f"{makespan}\t"
    #     f"{tardiness}\t"
    #     f"{setup_count}\t"
    #     f"{env.rules_count}\t"
    #     f"{round((toc - tic).total_seconds(), 2)}\t")
    return makespan, time.time() - start

def eval_ta(weight_path, model_name):
    print("Evaluating model with name", model_name)
    total_gap = 0
    total_case_num = 0
    total_rules_count = Counter()
    ta_dir = "./JSPLIB/TA"
    size_list = os.listdir(ta_dir)
    size_list = [
        '15x15',
        '20x15',
        '20x20',
        '30x15',
        '30x20',
        '50x15',
        '50x20',
        '100x20',
    ]
    rows = []
    tic = datetime.now()
    for size in size_list:
        size_gap = 0
        case_num = 0
        lines = open(os.path.join(ta_dir, size)).readlines()
        for line in lines:
            case_num += 1
            line = line.rstrip('\n').split(',')
            instance, J, M, op_ms = line[0],int(line[1]), int(line[2]), int(line[3])
            env = JSP_Env(args)
            agent = DQN_Agent(args, out_dim=len(env.rules))
            agent.load(weight_path)
            avai_ops = env.load_instance("./JSPLIB/instances/" + instance)
            state = env.get_graph_data(args.device)
            start_time = time.time()
            while True:
                action = agent.select_action(
                    state, random=False, test_only=True)
                state, reward, done, info = env.step(action)
                if done:
                    end_time = time.time()
                    makespan = env.get_makespan()
                    instance_gap = (makespan - op_ms) / op_ms
                    size_gap += instance_gap
                    print(f"{model_name, instance, J, M, makespan, instance_gap, start_time - end_time=}")
                    row = {
                        'Model': model_name,
                        'Instance': instance,
                        'J': J,
                        'M': M,
                        'Makespan': makespan,
                        'Gap': instance_gap,
                        'Time': start_time - end_time
                    }
                    rows.append(row)
                    break
            total_rules_count += Counter(env.rules_count)
        total_gap += size_gap
        total_case_num += case_num
        print(f"size: {size}\tgap: {round(size_gap / case_num, 3)}")
    toc = datetime.now()
    print(f"total gap: {round(total_gap / total_case_num, 3)}")
    print(f"total rules count: {total_rules_count}")
    print(f"{round((toc - tic).total_seconds(), 2)}")
    return rows

if __name__ == "__main__":
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
    args = parser.parse_args()

    # pandas

    rows = []
    for x in os.listdir("agent/DQN/weight"):
        if x.startswith("DQN_ep"):
            rows += eval_ta(os.path.join("agent/DQN/weight", x), x)

    

    DF = pd.DataFrame(rows)
    DF.to_csv('eval_dqn.csv', index=False)
    # weight_path = "agent/DQN/weight/DQN_ep1400"
    # print(weight_path)
    # eval_ta(weight_path)

