from Params import configs
import numpy as np
import copy
from FJSP_Env import FJSP
from torch.utils.data import DataLoader
from DataRead import getdata
import os, uuid
import time
import pandas as pd

import logging

# Configure the logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def permissibleLeftShift(a,mch_a, durMat, mchMat, mchsStartTimes, opIDsOnMchs,mchEndTime,row,col,first_col,last_col):#
    #a=action, durMat=self.dur, mchMat=mchaine, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs

    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a,mch_a, mchMat, durMat, mchsStartTimes, opIDsOnMchs,row,col,first_col,last_col)

    dur_a = durMat[row][col][mch_a]

    startTimesForMchOfa = mchsStartTimes[mch_a]#机器mch_a的start数组
    endtineformch0fa=mchEndTime[mch_a]
    #print('starttimesformchofa',startTimesForMchOfa)
    opsIDsForMchOfa = opIDsOnMchs[mch_a]#机器mch_a处理task的数组
    flag = False
    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]

    #machine中以调度的task的开始时间大于job中action的上一个task的完工时间

    if len(possiblePos) == 0:

        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a)

    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a,mch_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa,first_col,last_col)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')

    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)

    startTimesForMchOfa[index] = startTime_a

    opsIDsForMchOfa[index] = a
    endtineformch0fa[index]=startTime_a+dur_a

    return startTime_a


def calLegalPos(dur_a,mch_a,jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa,first_col,last_col):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]#possiblepos有可能是一个有可能是多个task，找到machine中tasks的starttimefomach
    durOfPossiblePos=[]
    for possiblePo in possiblePos:
        row = np.where(opsIDsForMchOfa[possiblePo] <= last_col)[0][0]
        col = opsIDsForMchOfa[possiblePo] - first_col[row]
        durOfPossiblePos.append(durMat[row,col][mch_a])

    durOfPossiblePos=np.array(durOfPossiblePos)#tasks的加工时间
    if possiblePos[0] != 0:
        row1 = np.where(opsIDsForMchOfa[possiblePos[0] - 1] <= last_col)[0][0]
        col1 = opsIDsForMchOfa[possiblePos[0]-1] - first_col[row1]


        startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + durMat[row1,col1][mch_a])
    else:
        startTimeEarlst = max(jobRdyTime_a,0)

    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care

    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos

    idxLegalPos = np.where(dur_a <= possibleGaps)[0]

    legalPos = np.take(possiblePos, idxLegalPos)

    return idxLegalPos, legalPos, endTimesForPossiblePos

def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    endtineformch0fa[:]=np.insert(endtineformch0fa, earlstPos, startTime_a+dur_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]

    return startTime_a


def calJobAndMchRdyTimeOfa(a, mch_a,mchMat, durMat, mchsStartTimes, opIDsOnMchs,row,col,first_col,last_col):
    #numpy.take（a，indices，axis = None，out = None，mode ='raise' ）取矩阵中所有元素的第a个元素
    # cal jobRdyTime_a
    if col != 0:
        jobPredecessor = a - 1

    else:
        jobPredecessor = None

    #jobPredecessor = a - 1 if col != 0 else None#if a % mchMat.shape[1] = 0即该job调度完成或为第一个调度的task

    #job中action前一个task
    if jobPredecessor is not None:
        mchJobPredecessor = mchMat[row][col-1]  # 处理该task的机器

        durJobPredecessor = durMat[row,col-1,mchJobPredecessor]#加工时间

        jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()#opIDsOnMchs->对应mchJobPredecessor----shape（machine,n_job）
        #找到数组opIDsOnMchs[mchJobPredecessor]中等于jobPredecessor的索引值####opIDsOnMchs->shape(machine,job)
    else:
        jobRdyTime_a = 0
    #cal mchRdyTime_a
    mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None

    #machine中action前一个task

    if mchPredecessor is not None:

        row_1 = np.where(mchPredecessor <= last_col)[0][0]
        col_1 = mchPredecessor - first_col[row_1]
        durMchPredecessor = durMat[row_1,col_1,mch_a]

        mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()

         #np.where()返回一个索引数组，这里返回在该machine中以调度task的索引。最后返回machine中action上一个task的结束时间
    else:

        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a

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


def runDRs(rule, datafile, batch_size):
    Data = getdata(datafile)
    n_j = Data['n']
    n_m = Data['m']

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

    data_loader = DataLoader(copy.deepcopy(data_set), batch_size=batch_size, shuffle=False)
    result = []
    for _, data_set in enumerate(data_loader):
        data_set = data_set.numpy()
        batch_size = data_set.shape[0]
        env = FJSP(n_j=n_j, n_m=n_m, EachJob_num_operation=num_operation)
        _, _, omega, mask, mch_mask, _, _, _ = env.reset(copy.deepcopy(data_set), rule)
        rewards = []

        d = 0
        while True:
            action = []
            mch_a = []
            # print(mask)
            for i in range(batch_size):
                # print(f"{omega[i][np.where(mask[i] == 0)]=}")

                a = np.random.choice(omega[i][np.where(mask[i] == 0)])
                row = np.where(a <= env.last_col[i])[0][0]

                col = a - env.first_col[i][row]
                m = np.random.choice(np.where(mch_mask[i][row][col] == 0)[0])

                action.append(a)
                mch_a.append(m)
                # print(i, 'a', a, row, col, m)
            d += 1
            _, _, reward, _, omega, mask, _, mch_mask, _, _ = env.step(action, mch_a)

            rewards.append(reward)
            if env.done():
                break

        result.append(env.mchsEndTimes.max(-1).max(-1))

    return result

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

if __name__ == "__main__":
    DATA_FILE = 'experiment_static_fjsp_pdrs.csv'
    PREV_DATA_FILE = 'experiment_static_fjsp_pdrs_save.csv'
    PDRS = ['FIFO_SPT', 'FIFO_EET', 'MOPNR_SPT', 'MOPNR_EET', 'LWKR_SPT', 'LWKR_EET', 'MWKR_SPT', 'MWKR_EET']
    INSTANCES = sorted(get_all_fjsp_instances())
    BATCH_SIZE = 10

    exceptions = [
        '../../../../benchmarks/fjsp/1_Brandimarte/BrandimarteMk7.fjs',
        '../../../../benchmarks/fjsp/1_Brandimarte/BrandimarteMk8.fjs',
        '../../../../benchmarks/fjsp/1_Brandimarte/BrandimarteMk9.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2a_Hurink_sdata/HurinkSdata63.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2b_Hurink_edata/HurinkEdata63.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2c_Hurink_rdata/HurinkRdata63.fjs',
        '../../../../benchmarks/fjsp/2_Hurink/2d_Hurink_vdata/HurinkVdata63.fjs'
    ]
    
    with open(DATA_FILE, 'w') as f:
        f.write("run,pdr,instance,makespan,runtime\n")

    df = pd.read_csv(PREV_DATA_FILE)
    count = 0
    total = BATCH_SIZE * len(INSTANCES) * len(PDRS)
    for instance in INSTANCES:
        print(instance)
        if instance in exceptions:
            # skip exceptions
            continue

        for pdr in PDRS:
            instance_name = instance.split('/')[-1]
            filtered_df = df[(df['pdr'] == pdr) & (df['instance'] == instance_name)]
            if filtered_df.shape[0] == BATCH_SIZE:
                for run in range(BATCH_SIZE):
                    row = filtered_df[(filtered_df['pdr'] == pdr) & (filtered_df['instance'] == instance_name) & (filtered_df['run'] == run)]
                    makespan = filtered_df['makespan'].iloc[0]
                    avg_runtime = filtered_df['runtime'].iloc[0]
                    with open(DATA_FILE, 'a') as f:
                        f.write(f"{run},{pdr},{instance_name},{makespan},{avg_runtime}\n")

                    count += 1
                    logging.info(f"{count=} {total=} {run=} {pdr=} {instance_name=} {makespan=} {avg_runtime=} ")
            else:
                start = time.time()
                makespans = runDRs(pdr, instance, batch_size=BATCH_SIZE)
                avg_runtime = (time.time() - start) / BATCH_SIZE

                for run in range(BATCH_SIZE):
                    makespan = int(makespans[0][run])
                    with open(DATA_FILE, 'a') as f:
                        f.write(f"{run},{pdr},{instance_name},{makespan},{avg_runtime}\n")

                    count += 1
                    logging.info(f"{count=} {total=} {run=} {pdr=} {instance_name=} {makespan=} {avg_runtime=} ")