import bisect
import random
from env.utils.mach_job_op import *
from env.utils.generator import *
from env.utils.graph import Graph
from tools.logger import Logger


class JSP_Instance:
    def __init__(self, args):
        self.args = args
        self.process_time_range = [1, args.max_process_time]
        self.no_op_avai_ops = []
        self.num_job_type = 3
        self.setup_count = 0

    # basic functions
    def generate_case(self):
        # initial jobs, not dynamic
        self.insert_jobs(job_num=self.initial_job_num)

    def insert_jobs(self, job_num):
        for i in range(job_num):
            job_id = len(self.jobs)
            self.register_time(self.arrival_time)
            op_config = gen_operations(
                self.op_num,
                self.machine_num,
                self.process_time_range)
            self.jobs.append(Job(job_id=job_id,
                                 arrival_time=self.arrival_time,
                                 op_config=op_config,
                                 job_type=random.randint(0, self.num_job_type)))
            self.graph.add_job(self.jobs[-1])

    def reset(self):
        self.initial_job_num = random.randint(3, self.args.data_size)
        self.machine_num = random.randint(3, self.initial_job_num)
        self.op_num = self.machine_num
        self.jobs = []
        self.machines = [Machine(machine_id)
                         for machine_id in range(self.machine_num)]
        self.arrival_time = 0
        self.current_time = 0
        self.time_stamp = []
        self.graph = Graph(self.machine_num)
        self.logger = Logger()
        self.generate_case()

    def load_instance(self, filename, arrival_times: list | None = None):
        self.jobs = []
        self.arrival_time = 0
        self.current_time = 0
        self.time_stamp = []
        if arrival_times is not None:
            self.time_stamp = arrival_times

        f = open(filename)
        line = f.readline()
        while line[0] == '#':
            line = f.readline()
        line = line.split()
        self.initial_job_num, self.machine_num = int(line[0]), int(line[1])
        self.machines = [Machine(machine_id)
                         for machine_id in range(self.machine_num)]
        self.graph = Graph(self.machine_num)
        self.logger = Logger()

        for i in range(self.initial_job_num):
            arrival_time = 0
            if arrival_times is not None:
                arrival_time = arrival_times[i]
            op_config = []
            line = f.readline().split()
            for j in range(self.machine_num):
                machine_id, process_time = int(
                    line[j * 2]), int(line[j * 2 + 1])
                op_config.append({"id": j,
                                  "machine_id": machine_id,
                                  "process_time": process_time})
            self.jobs.append(Job(job_id=i,
                                 arrival_time=arrival_time,
                                 job_type=i % self.num_job_type,
                                 op_config=op_config))
            self.graph.add_job(self.jobs[-1])

    def done(self):
        for job in self.jobs:
            if job.done() == False:
                return False
        return True

    def current_avai_ops(self):
        avai_ops = self.available_ops()
        return avai_ops

    def get_graph_data(self, device):
        self.graph.update_feature(
            self.jobs,
            self.machines,
            self.current_time,
            self.args.max_process_time)
        data = self.graph.get_data().to(device)
        return data

    def assign(self, avai_ops, idx, rule_name=None):
        job_id, op_id = avai_ops[idx]['job_id'], avai_ops[idx]['op_id']
        job = self.jobs[job_id]
        assert op_id == job.current_op_id, \
            f"op_id: {op_id}\tself.jobs[{job_id}].current_op_id: {self.jobs[job_id].current_op_id}"
        op = job.current_op()
        op.rule_name = rule_name
        machine = self.machines[op.machine_id]
        setup_time = machine.get_setup_time(op, active=False)
        if setup_time != 0:
            self.setup_count += 1
        op_info = {
            "job_id": job_id,
            "op_id": op.op_id,
            "current_time": max(self.current_time, op.avai_time),
            "process_time": op.process_time,
            "job_type": op.job_type,
        }
        start_time, avai_time = machine.process_op(op_info, setup_time)
        job.current_op().update(start_time)
        # print(f"\tO{op_info['job_id']},{op_info['op_id']}"
        #       f"\tself.current_time: {self.current_time}"
        #       f"\top.avai_time: {op.avai_time}")
        # add op to logger
        self.logger.add_op(op)
        if job.next_op() != -1:
            job.update_current_op(avai_time=avai_time)
        self.register_time(avai_time)

    # about time control
    def register_time(self, time):
        # maintain a list in sorted order
        bisect.insort(self.time_stamp, time)

    def update_time(self):
        self.current_time = self.time_stamp.pop(0)

    def available_ops(self):
        if self.done() == True:
            return None
        avai_ops = []
        for m in self.machines:
            for job in self.jobs:
                if job.done() == False and job.current_op().avai_time <= self.current_time:
                    if m.avai_time() <= self.current_time and m.machine_id == job.current_op().machine_id:
                        avai_ops.append({'m_id': m.machine_id, 'job_id': job.job_id, 'op_id': job.current_op_id, 'node_id': job.current_op().node_id})
            
        if len(avai_ops) == 1:
            self.assign(avai_ops, 0)
        else:
            if len(avai_ops) > 1:
                return avai_ops
            
        self.update_time()
        return self.available_ops()

    # def available_ops(self):
    #     if self.done():
    #         return None
    #     avai_ops = []
    #     no_op_avai_ops = []
    #     for m in self.machines:
    #         if m.avai_time() > self.current_time:
    #             continue

    #         min_waiting_time = 1e6
    #         for job in self.jobs:
    #             if job.done() or m.machine_id != job.current_op().machine_id:
    #                 continue

    #             if job.current_op().avai_time <= self.current_time:
    #                 avai_ops.append({'m_id': m.machine_id,
    #                                  'job_id': job.job_id,
    #                                  'op_id': job.current_op_id,
    #                                  'node_id': job.current_op().node_id})
    #                 min_waiting_time = min(
    #                     min_waiting_time, job.current_op().process_time)

    #         if min_waiting_time == 1e6:
    #             continue

    #         for job in self.jobs:
    #             if job.done() or m.machine_id != job.current_op().machine_id:
    #                 continue

    #             if job.current_op().avai_time > self.current_time and job.current_op(
    #             ).avai_time < self.current_time + min_waiting_time:
    #                 no_op_avai_ops.append({'m_id': m.machine_id,
    #                                        'job_id': job.job_id,
    #                                        'op_id': job.current_op_id,
    #                                        'node_id': job.current_op().node_id})

    #     if len(avai_ops) == 1 and len(no_op_avai_ops) == 0:
    #         self.assign(avai_ops, 0)
    #     else:
    #         if len(avai_ops) >= 1:
    #             return avai_ops + no_op_avai_ops

    #     self.update_time()
    #     return self.available_ops()
    
    
