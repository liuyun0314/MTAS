import random

import numpy as np
import simpy
import copy
import actions
from actions import *
import concurrent.futures
from Environments.Task import Task
from Environments.Job import Job
from Environments.Operation import Operation
from Environments.Machine import Machine
from Environments.Reward import *

rules = ['SPTM', 'NINQ', 'WINQ', 'LWT', 'SPT', 'LPT', 'MWKR', 'EDD', 'MOPNR']

class JobShop:
    def __init__(self, env, args, seed):
        # job shop
        self.env = env
        self.args = args
        self.decision_points = [0]
        self.dynamic_contor = 0
        self.random_state = random.Random(seed)
        self.num_actoin = args.n_action
        self.num_tasks = args.num_tasks
        self.num_warmup_jobs = args.num_warmup_jobs
        self.num_new_job = args.num_new_jobs
        self.num_jobs = self.num_new_job + self.num_tasks * self.num_warmup_jobs
        self.num_machines = args.num_machines
        self.num_ops_range = args.num_ops_range
        self.num_cand_machines_range = args.num_cand_machines_range
        self.weights = args.weights  # the number of operations of a job is randomly generated from the range
        self.processing_time_range = args.processing_time_range  # the processing time of rach operation is assigned by a range
        self.due_time_multiplier = args.due_time_multiplier
        self.arrival_interval = self.newJob_arrival_interval(args)
        self.repair_time_range = args.repair_time_range
        self.next_new_job_AT = self.arrival_interval[0]
        if self.arrival_interval[0] not in self.decision_points:
            self.decision_points.append(self.arrival_interval[0])
        # self.starTime = np.random.randint(1, 2000)    # the time point when a machine fails
        self.starTime = self.random_state.randint(1, 2000)    # the time point when a machine fails
        if self.starTime not in self.decision_points:
            self.decision_points.append(self.starTime)
        # repair_time = np.random.randint(self.repair_time_range[0], self.repair_time_range[1])  # 修复时间ref{Multi-objective reinforcement learning framework for dynamic flexible job shop scheduling problem with uncertain events}
        repair_time = self.random_state.randint(self.repair_time_range[0], self.repair_time_range[1])  # 修复时间ref{Multi-objective reinforcement learning framework for dynamic flexible job shop scheduling problem with uncertain events}
        self.finishTime = self.starTime + repair_time  # the time point when a machine is repaired
        if self.finishTime not in self.decision_points:
            self.decision_points.append(self.finishTime)
        self.decision_points = sorted(self.decision_points)
        self.action_space = [['SPTM', 'NINQ', 'WINQ', 'LWT'], ['SPTM', 'NINQ', 'WINQ', 'LWT'], ['SPTM', 'NINQ', 'WINQ', 'LWT'], ['SPT', 'LPT', 'MWKR', 'EDD', 'MOPNR']]

        self.index_job = 0  # the index of the jobs that arrives in the job shop system
        self.in_system_job_num = 0
        self.in_system_job_dic = {}  # {'arrival_time': self.in_system_job_num}
        self.tasks_list = []
        self.jobs = []
        self.machines = []
        self.failure_machine = None
        self.done = False
        self.num_finished = 0
        self.span = 0
        self.Task_Initializationr()
        self.Machine_Initializationr()
        self.completed_jobs = {}
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []
        self.machine_queue = {}
        for m in self.machines:
            self.machine_queue[m.name] = []

    def newJob_arrival_interval(self, args):
        avg = np.average(args.processing_time_range) - 0.5
        beta = avg / args.E_utliz
        arrival_interval = np.random.exponential(beta, args.num_new_jobs).round() + 1
        return arrival_interval

    def reset(self, seed, arrival_interval):
        '''
        # reset the job shop environment
        :return:
        '''
        self.done = False
        self.env = simpy.Environment()
        # self.random_state = random.Random(seed)
        self.decision_points = [0]
        self.index_job = 0  # the index of the jobs that arrives in the job shop system
        self.dynamic_contor = 0
        self.in_system_job_num = 0
        self.in_system_job_dic = {}  # {'arrival_time': self.in_system_job_num}
        self.tasks_list = []
        self.jobs = []
        self.machines = []
        self.span = 0
        # self.starTime = -1
        # self.finishTime = -1
        self.failure_machine = None
        self.num_finished = 0
        self.arrival_interval = arrival_interval
        self.next_new_job_AT = self.arrival_interval[0]
        self.num_new_job = self.args.num_new_jobs
        self.num_jobs = self.num_new_job + self.num_tasks * self.num_warmup_jobs
        if self.arrival_interval[0] not in self.decision_points:
            self.decision_points.append(self.arrival_interval[0])
        if self.starTime not in self.decision_points:
            self.decision_points.append(self.starTime)
        # repair_time = np.random.randint(self.repair_time_range[0], self.repair_time_range[1])   # 修复时间ref{Multi-objective reinforcement learning framework for dynamic flexible job shop scheduling problem with uncertain events}
        # self.finishTime = self.starTime + repair_time
        if self.finishTime not in self.decision_points:
            self.decision_points.append(self.finishTime)
        self.decision_points = sorted(self.decision_points)
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []
        self.Task_Initializationr()
        self.Machine_Initializationr()
        self.completed_jobs = {}
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []
        self.machine_queue = {}
        for m in self.machines:
            self.machine_queue[m.name] = []
        obs, ava = self.obs_encode()
        state = obs.copy()
        return obs, state, ava

    def Jobs_Initializationr(self, idTask, arrival_time):
        if idTask == 0 or idTask == 2:
            job_weight = self.random_state.choices([1, 2, 4], weights=self.weights)[0]
        else:
            job_weight = self.random_state.choices([1, 5, 10], weights=self.weights)[0]
        job = self.Job_Generator(idTask, job_weight, arrival_time, self.num_ops_range[1])
        return job

    def Job_Generator(self, task_id, weight, arrival_time, max_operation_num):
        # generate the operation list
        operations_list = []
        operations_num = self.random_state.randint(1, max_operation_num)
        for i in range(operations_num):
            operation = self.Operation_Generator(i, self.tasks_list[task_id].job_counter, task_id, arrival_time)
            operations_list.append(operation)
        job = Job(task_id, self.tasks_list[task_id].job_counter, weight, arrival_time, operations_list)
        self.tasks_list[task_id].job_counter += 1
        return job

    def Operation_Generator(self, id_operation, id_job, taskID, arrival_time):

        candidate_machines = {}
        # generate the information of candidate machine
        candidate_machine_num = self.random_state.randint(1, self.num_machines)
        sample_index = self.random_state.sample(range(self.num_machines), candidate_machine_num)
        # process_time = self.random_state.randint(self.processing_time_range[0], self.processing_time_range[1], candidate_machine_num)
        process_time = self.random_state.sample(range(self.processing_time_range[0], self.processing_time_range[1]),
                                                candidate_machine_num)
        count = 0
        for m in sample_index:
            machine_name = 'M' + str(m)
            candidate_machines[machine_name] = process_time[count]
            count += 1
        operation = Operation(id_operation, id_job, taskID, candidate_machines, arrival_time)
        return operation

    def Machine_Initializationr(self):
        current_time = 0
        for m in range(self.num_machines):
            machine = Machine(m, current_time)
            self.machines.append(machine)

    def Task_Initializationr(self):
        for i in range(self.num_tasks):
            task = self.Task_Generator(i)
            self.tasks_list.append(task)
            for i in range(self.num_warmup_jobs):
                job = self.Jobs_Initializationr(task.idTask,0)
                # task.job_counter += 1
                self.jobs.append(job)
                task.jobsList.append(job)

    def Task_Generator(self, id):
        objectives = ['WTmean', 'WFmean', 'WTmax']
        objective = self.random_state.choices(objectives)
        task = Task(id, objectives[id])
        return task

    def Exponential_arrival(E_ave, total_jobs_num):

        A = np.random.exponential(E_ave, total_jobs_num).round()
        A = [A[i] + 1 for i in range(len(A))]
        A = sorted(A)
        A = np.cumsum(A)
        return A

    def new_job_arrival(self):
        if self.index_job < self.num_new_job:
            if self.env.now == self.next_new_job_AT:
                task_id = self.random_state.randint(0, self.num_tasks - 1)
                job = self.Jobs_Initializationr(task_id, self.env.now)
                self.jobs.append(job)
                self.index_job += 1
                self.record_jobs_arrival()
                self.tasks_list[task_id].jobsList.append(job)

                self.dynamic_contor += 1
                if self.index_job < self.num_new_job:
                    self.next_new_job_AT += self.arrival_interval[self.index_job]
                    if self.next_new_job_AT not in self.decision_points:
                        self.decision_points.append(self.next_new_job_AT)
                    if self.next_new_job_AT == self.env.now:  # 可能出现arrival_interval[self.index_job]=0的情况
                        self.new_job_arrival()
                if self.env.now not in self.decision_points:
                    self.decision_points.append(self.env.now)
                self.decision_points = sorted(self.decision_points)

    def record_jobs_arrival(self):
        self.in_system_job_num += 1
        self.in_system_job_dic[self.env.now] = self.in_system_job_num

    def machine_failure(self):
        if self.env.now == self.starTime:
            # 随机选择一个机器，然后把该机器的状态设置为故障
            machine = self.random_state.choice(self.machines)
            machine.available = False
            if len(machine.assignedOpera):
                processingOpera = machine.assignedOpera[-1]
                if processingOpera.endTime > self.env.now and processingOpera.startTime <= self.env.now:  # 在当前时刻，该机器正在处理一个操作
                    self.putBactOpera(processingOpera)  # 把正在处理的操作放回到工作列表中
                    del machine.assignedOpera[-1]
                    if len(machine.assignedOpera) == 0:  # 如果这种情况发生，说明到目前位置machine只处理过一个操作就坏了
                        machine.currentTime = 0
                    else:    # 否则，把下一个操作的开始时间作为当前时间
                        machine.currentTime = machine.assignedOpera[-1].endTime
            self.failure_machine = copy.deepcopy(machine)
            self.machines.remove(machine)
            self.clean_machine_queue(machine.name)   # 将损坏机器的队列清空
            # repairTime = random.randint(1, 99)
            # self.finishTime = self.starTime + repairTime
            # if self.finishTime not in self.decision_points:
            #     self.decision_points.append(self.finishTime)
            #     self.decision_points = sorted(self.decision_points)
    def putBactOpera(self, opera):
        opera.assignedMachine = ""
        opera.assigned = False
        opera.completed = False
        opera.startTime = 0
        opera.duration = 0
        opera.endTime = 0
        opera.endTime = 0
        J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
        if opera.idOpertion == 0:
            J.RT = 0
        if opera.idOpertion == len(J.operation_list) - 1:
            J.completed = False
            self.num_finished -= 1
            J.endTime = 0
            del self.completed_jobs[J.idTask][-1]

    def clean_machine_queue(self, machine_name):
        queue = self.machine_queue[machine_name]
        if len(queue) != 0:
            for opera in queue:
                opera.assigned = False
                opera.assignedMachine = ""
            # self.machine_queue[machine_name] = []
        del self.machine_queue[machine_name]

    def machine_repair(self):
        if self.env.now == self.finishTime:
            self.failure_machine.available = True
            self.failure_machine.currentTime = self.env.now
            self.machines.append(self.failure_machine)
            self.machine_queue[self.failure_machine.name] = []
            if self.env.now not in self.decision_points:
                self.decision_points.append(self.env.now)
                self.decision_points = sorted(self.decision_points)

    def step_heuristicRule(self, jobs, machines, action):
        last_endTime = C_last(self.tasks_list)
        eWTmean = estimated_WTmean(self.tasks_list[0].jobsList, self.machines)
        eWFmean = estimated_WFmean(self.tasks_list[1].jobsList, self.machines)
        eWTmax = estimated_WTmax(self.tasks_list[2].jobsList, self.machines)
        pre_obj = [eWTmean, eWFmean, eWTmax]
        pre_C_last = last_PT(self.jobs)

        assigned_opera_mac, self.machine_queue = eval(action)(jobs, machines, self.env.now, self.machine_queue)
        all_values_not_none = all(value is None for value in assigned_opera_mac.values())
        PTs = []
        if not all_values_not_none:
            for mac, opera in assigned_opera_mac.items():
                if opera is not None:
                    machine = None
                    for m in machines:
                        if m.name == mac:
                            machine = m
                            break
                    opera.assignedMachine = mac
                    opera.assigned = True
                    opera.startTime = self.env.now
                    opera.duration = opera.cMachines[mac]
                    # opera.endTime = opera.getEndTime()
                    opera.endTime = opera.startTime + opera.duration
                    opera.completed = True  # 这句话不可以放在yield sim.env.timeout(opera.duration)之前。因为，这样可能导致实际系统的时间还没有到达该Opera完成的时间，但是在接下来的重调度时，由于该Opera的completed被标为了True,所以后驱Opera会被选择处理，但事实上，该Opera还没有执行完成
                    machine.currentTime = opera.endTime
                    machine.assignedOpera.append(opera)
                    # machine.state = 'busy'
                    J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                    # J = jobs[opera.jobID]
                    # J.span = J.getSpan()

                    # 如果这个操作是该工作的第一个操作，那么就把该工作的开始时间标记为该操作的开始时间
                    if opera.idOpertion == 0:
                        J.RT = opera.startTime
                    # 如果这个操作是该工作的最后一个操作，那么就把该工作的结束时间标记为该操作的结束时间，把该工作的completed标记为True，且加入到已经完成的工作列表中
                    if opera.idOpertion == len(J.operation_list) - 1:
                        J.completed = True
                        self.num_finished += 1
                        J.endTime = J.getEndTime()
                        J.span = J.endTime - J.RT
                        self.completed_jobs[J.idTask].append(J)

                    # 机器处理完一个操作的时间点为一个决策点，所以把机器处理完一个操作的时间点加入到决策点列表中
                    if machine.currentTime not in self.decision_points:
                        self.decision_points.append(machine.currentTime)
                        self.decision_points = sorted(self.decision_points)

                    # 如果所有的工作都已经到达车间，且所有的工作都已经完成，那么就把done标记为True
                    if self.num_finished == self.num_jobs:
                        self.done = True
                        # 值得注意的是，在所有新工作均到达车间之前，属于task i的全部工作被执行完成时，不可以将其completed标为True，因为后续可能会有属于task i的新工作到达
                        for i in range(self.num_tasks):
                            self.tasks_list[i].completed = True  # 所有机器都已经被完成了，所以把所有的task都标记为完成
                            if len(self.completed_jobs[i]) > 0:
                                # task_i完成列表中最后一个job的完工时间即为该task的完工时间
                                last_J = self.completed_jobs[i][-1]
                                self.tasks_list[i].endTime = last_J.endTime
                            else:
                                self.tasks_list[i].endTime = 0
                    # machine.state = 'idle'
                    PTs.append(opera.endTime)
        r = cal_Reward2(self.jobs, self.tasks_list, self.machines, pre_obj, self.done, pre_C_last, last_endTime)
        return r

    def step_heuristic(self, a_m, a_s):
        last_PT = C_last(self.tasks_list)
        # last_evaluation = laset_evaluations(self.tasks_list, self.machines)
        last_span = estimated_span(self.jobs, self.machines)
        last_evaluation = laset_evaluations(self.tasks_list, self.machines, self.completed_jobs)
        # last_evaluation = laset_evaluations(self.tasks_list, self.machines, self.jobs)
        self.machine_queue = eval(a_m)(self.machines, self.jobs, self.env.now, self.machine_queue)
        for mac in self.machines:
            if mac.currentTime > self.env.now:  # the machine is busy
                continue
            opera = eval(a_s)(self.tasks_list, mac, self.machine_queue)   # decide an operation to be processed
            if opera is not None:
                if opera in mac.assignedOpera:
                    input("条件满足，按Enter键继续")
                opera.assigned = True
                opera.completed = True
                opera.assignedMachine = mac.name
                opera.startTime = self.env.now
                opera.duration = opera.cMachines[mac.name]
                opera.endTime = self.env.now + opera.duration
                mac.currentTime = opera.endTime
                mac.assignedOpera.append(opera)
                self.machine_queue[mac.name].remove(opera)
                J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                # 如果这个操作是该工作的第一个操作，那么就把该工作的开始时间标记为该操作的开始时间
                if opera.idOpertion == 0:
                    J.RT = opera.startTime
                # 如果这个操作是该工作的最后一个操作，那么就把该工作的结束时间标记为该操作的结束时间，把该工作的completed标记为True，且加入到已经完成的工作列表中
                if opera.idOpertion == len(J.operation_list) - 1:
                    J.completed = True
                    self.num_finished += 1
                    J.endTime = J.getEndTime()
                    self.completed_jobs[J.idTask].append(J)

                # 机器处理完一个操作的时间点为一个决策点，所以把机器处理完一个操作的时间点加入到决策点列表中
                if mac.currentTime not in self.decision_points:
                    self.decision_points.append(mac.currentTime)
                    self.decision_points = sorted(self.decision_points)

                # 如果所有的工作都已经到达车间，且所有的工作都已经完成，那么就把done标记为True
                if self.num_finished == self.num_jobs:
                    self.done = True
                    # 值得注意的是，在所有新工作均到达车间之前，属于task i的全部工作被执行完成时，不可以将其completed标为True，因为后续可能会有属于task i的新工作到达
                    for i in range(self.num_tasks):
                        self.tasks_list[i].completed = True  # 所有机器都已经被完成了，所以把所有的task都标记为完成
                        if len(self.completed_jobs[i]) > 0:
                            # task_i完成列表中最后一个job的完工时间即为该task的完工时间
                            last_J = self.completed_jobs[i][-1]
                            self.tasks_list[i].endTime = last_J.endTime
                        else:
                            self.tasks_list[i].endTime = 0
        rewards = Rewarder(last_evaluation, self.tasks_list, self.machines, self.completed_jobs)
        r = interval_time(self.tasks_list, last_PT)
        return r

    def step_DR(self, opera_lists, idle_machines):
        last_PT = C_last(self.tasks_list)
        for machine, opera in opera_lists.items():
            for m in idle_machines:
                if m.name == machine:
                    mac = m
                    break
            if opera is not None:
                opera.assigned = True
                opera.completed = True
                opera.assignedMachine = mac.name
                opera.startTime = self.env.now
                opera.duration = opera.cMachines[mac.name]
                opera.endTime = self.env.now + opera.duration
                mac.currentTime = opera.endTime
                mac.assignedOpera.append(opera)
                J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                # 如果这个操作是该工作的第一个操作，那么就把该工作的开始时间标记为该操作的开始时间
                if opera.idOpertion == 0:
                    J.RT = opera.startTime
                # 如果这个操作是该工作的最后一个操作，那么就把该工作的结束时间标记为该操作的结束时间，把该工作的completed标记为True，且加入到已经完成的工作列表中
                if opera.idOpertion == len(J.operation_list) - 1:
                    J.completed = True
                    self.num_finished += 1
                    J.endTime = J.getEndTime()
                    self.completed_jobs[J.idTask].append(J)

                # 机器处理完一个操作的时间点为一个决策点，所以把机器处理完一个操作的时间点加入到决策点列表中
                if mac.currentTime not in self.decision_points:
                    self.decision_points.append(mac.currentTime)
                    self.decision_points = sorted(self.decision_points)

                # 如果所有的工作都已经到达车间，且所有的工作都已经完成，那么就把done标记为True
                if self.num_finished == self.num_jobs:
                    self.done = True
                    # 值得注意的是，在所有新工作均到达车间之前，属于task i的全部工作被执行完成时，不可以将其completed标为True，因为后续可能会有属于task i的新工作到达
                    for i in range(self.num_tasks):
                        self.tasks_list[i].completed = True  # 所有机器都已经被完成了，所以把所有的task都标记为完成
                        if len(self.completed_jobs[i]) > 0:
                            # task_i完成列表中最后一个job的完工时间即为该task的完工时间
                            last_J = self.completed_jobs[i][-1]
                            self.tasks_list[i].endTime = last_J.endTime
                        else:
                            self.tasks_list[i].endTime = 0
        r = interval_time(self.tasks_list, last_PT)
        return r

    def step1(self, a_m, a_s):
        last_PT = C_last(self.tasks_list)
        # last_evaluation = laset_evaluations(self.tasks_list, self.machines)
        last_span = estimated_span(self.jobs, self.machines)
        last_evaluation = laset_evaluations(self.tasks_list, self.machines, self.completed_jobs)
        # last_evaluation = laset_evaluations(self.tasks_list, self.machines, self.jobs)
        # for i in range(self.num_tasks):
        #     self.machine_queue = eval(a_m)(self.machines, self.tasks_list[i].jobsList, self.env.now, self.machine_queue)
        self.machine_queue = eval(a_m)(self.machines, self.jobs, self.env.now, self.machine_queue)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # 使用列表推导式创建一个需要执行的任务列表
        #     futures = [executor.submit(eval(a_m), self.machines, self.tasks_list[i].jobsList, self.env.now, self.machine_queue) for i in range(self.num_tasks)]
        #     # # 调用parallel_execution方法来并行执行操作,等待所有任务完成并获取结果
        #     for future in concurrent.futures.as_completed(futures):
        #         self.machine_queue = future.result()

        for mac in self.machines:
            if mac.currentTime > self.env.now:  # the machine is busy
                continue
            opera = eval(a_s)(self.tasks_list, mac, self.machine_queue)   # decide an operation to be processed
            if opera is not None:
                if opera in mac.assignedOpera:
                    input("条件满足，按Enter键继续")
                opera.assigned = True
                opera.completed = True
                opera.assignedMachine = mac.name
                opera.startTime = self.env.now
                opera.duration = opera.cMachines[mac.name]
                opera.endTime = self.env.now + opera.duration
                mac.currentTime = opera.endTime
                mac.assignedOpera.append(opera)
                self.machine_queue[mac.name].remove(opera)
                J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                # 如果这个操作是该工作的第一个操作，那么就把该工作的开始时间标记为该操作的开始时间
                if opera.idOpertion == 0:
                    J.RT = opera.startTime
                # 如果这个操作是该工作的最后一个操作，那么就把该工作的结束时间标记为该操作的结束时间，把该工作的completed标记为True，且加入到已经完成的工作列表中
                if opera.idOpertion == len(J.operation_list) - 1:
                    J.completed = True
                    self.num_finished += 1
                    J.endTime = J.getEndTime()
                    self.completed_jobs[J.idTask].append(J)

                # 机器处理完一个操作的时间点为一个决策点，所以把机器处理完一个操作的时间点加入到决策点列表中
                if mac.currentTime not in self.decision_points:
                    self.decision_points.append(mac.currentTime)
                    self.decision_points = sorted(self.decision_points)

                # 如果所有的工作都已经到达车间，且所有的工作都已经完成，那么就把done标记为True
                if self.num_finished == self.num_jobs:
                    self.done = True
                    # 值得注意的是，在所有新工作均到达车间之前，属于task i的全部工作被执行完成时，不可以将其completed标为True，因为后续可能会有属于task i的新工作到达
                    for i in range(self.num_tasks):
                        self.tasks_list[i].completed = True  # 所有机器都已经被完成了，所以把所有的task都标记为完成
                        if len(self.completed_jobs[i]) > 0:
                            # task_i完成列表中最后一个job的完工时间即为该task的完工时间
                            last_J = self.completed_jobs[i][-1]
                            self.tasks_list[i].endTime = last_J.endTime
                        else:
                            self.tasks_list[i].endTime = 0
        rewards = Rewarder(last_evaluation, self.tasks_list, self.machines, self.completed_jobs)
        r = interval_time(self.tasks_list, last_PT)
        return r

    def get_obs(self):
        '''Return all agent observations in a list'''
        # the information of machines
        all_U_m = []
        CT = 0
        BTs = []
        for m in self.machines:
            if m.currentTime > CT:
                CT = m.currentTime
            assigned_opera = m.assignedOpera
            CT_m = m.currentTime  # Time when machine m completes the current last scheduled operation
            busy_time = 0
            for op in assigned_opera:
                busy_time += op.duration
            U_m = busy_time / CT_m
            all_U_m.append(U_m)
            BTs.append(busy_time)
        U_avg = np.mean(all_U_m)  # 1. average utilization rate of all machines
        U_std = np.std(all_U_m)  # 2. standard deviation of all machine utilization rate
        W = [x / CT for x in BTs]
        W_avg = np.mean(W)  # 3. Average normalized machine workload
        W_std = np.std(W)  # 4. Standard deviation of normalized machine workload
        mac_state = [U_avg, U_std, W_avg, W_std]

        state = []
        for i in range(self.num_tasks):
            obs_i = self.get_agent_obs(i)
            state.extend(obs_i)
        state.extend(mac_state)
        return state

    def get_agent_obs(self, agent_id):
        '''Returns observation for agent_id'''

        CRJs = []  # the completion rate of jobs
        TR = []  #
        jobs = self.tasks_list[agent_id].jobsList
        for j in jobs:
            OP_j = 0  # current operation number that has been completed of job J_i
            ETL_i = 0  # estimated completion time of the remaining operations of job J_i
            C_j = 0  # The completion time of the last scheduled operation of job J_i until decision point
            for index, o in enumerate(j.operation_list):
                if o.completed:
                    OP_j += 1
                    C_j = o.endTime
                else:
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    ETL_i += mean_PT
            CRJ = OP_j / len(j.operation_list)
            CRJs.append(CRJ)
            TR_j = (C_j + ETL_i - j.DT) / (C_j + ETL_i)  # the job processing delay rate
            TR.append(TR_j)
        CRJ_avg = np.mean(CRJs)  # 5. mean completion rate of jobs
        CRJ_std = np.std(CRJs)  # 6. standard deviation of completion rate of jobs
        TR_avg = np.mean(TR)  # 7. mean processing delay rate of jobs
        TR_std = np.std(TR)  # 8. standard deviation of processing delay rate of jobs
        task_routing_state = [CRJ_avg, CRJ_std, TR_avg, TR_std]

        return task_routing_state

    ##### for multi-agent Transformer#####

    def step(self, a_m):
        last_endTime = C_last(self.tasks_list)
        # pre_C1 = C_lastOpera(self.tasks_list[0].jobsList)
        # pre_C = C_lastOpera(self.tasks_list[1].jobsList)
        # pre_C3 = C_lastOpera(self.tasks_list[2].jobsList)
        # preSpan = C_lastOpera(self.jobs)
        pre_idlTime = idleTime(self.machines, self.env.now)
        eWTmean = estimated_WTmean(self.tasks_list[0].jobsList, self.machines)
        eWFmean = estimated_WFmean(self.tasks_list[1].jobsList, self.machines)
        eWTmax = estimated_WTmax(self.tasks_list[2].jobsList, self.machines)
        pre_obj = [eWTmean, eWFmean, eWTmax]
        pre_C_last = last_PT(self.jobs)
        # pre_slack = slack_time(self.jobs, self.machines)
        last_evaluation = laset_evaluations(self.tasks_list, self.machines, self.completed_jobs)
        # last_evaluation = laset_evaluations(self.tasks_list, self.machines, self.jobs)
        # rules = ['SPTM', 'NINQ', 'WINQ', 'LWT', 'SPT', 'LPT', 'MWKR', 'EDD', 'MOPNR']
        actions = [rules[i] for i in a_m]    # am的前三个元素只能[0, 1, 2, 3]，第四个元素只能[4, 5, 6, 7, 8]
        for i in range(self.num_tasks):
            self.machine_queue = eval(actions[i])(self.machines, self.tasks_list[i].jobsList, self.env.now, self.machine_queue)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # 使用列表推导式创建一个需要执行的任务列表
        #     futures = [executor.submit(eval(actions[i]), self.machines, self.tasks_list[i].jobsList, self.env.now, self.machine_queue) for i in range(self.num_tasks)]
        #     # # 调用parallel_execution方法来并行执行操作,等待所有任务完成并获取结果
        #     for future in concurrent.futures.as_completed(futures):
        #         self.machine_queue = future.result()

        for mac in self.machines:
            if mac.currentTime > self.env.now:  # the machine is busy
                continue
            opera = eval(actions[-1])(self.tasks_list, mac, self.machine_queue)   # decide an operation to be processed
            if opera is not None:
                if opera in mac.assignedOpera:
                    input("条件满足，按Enter键继续")
                opera.assigned = True
                opera.completed = True
                opera.assignedMachine = mac.name
                opera.startTime = self.env.now
                opera.duration = opera.cMachines[mac.name]
                opera.endTime = self.env.now + opera.duration
                mac.currentTime = opera.endTime
                mac.busyTime += opera.duration
                mac.assignedOpera.append(opera)
                self.machine_queue[mac.name].remove(opera)
                J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                # 如果这个操作是该工作的第一个操作，那么就把该工作的开始时间标记为该操作的开始时间
                if opera.idOpertion == 0:
                    J.RT = opera.startTime
                # 如果这个操作是该工作的最后一个操作，那么就把该工作的结束时间标记为该操作的结束时间，把该工作的completed标记为True，且加入到已经完成的工作列表中
                if opera.idOpertion == len(J.operation_list) - 1:
                    J.completed = True
                    self.num_finished += 1
                    J.endTime = J.getEndTime()
                    self.completed_jobs[J.idTask].append(J)

                # 机器处理完一个操作的时间点为一个决策点，所以把机器处理完一个操作的时间点加入到决策点列表中
                if mac.currentTime not in self.decision_points:
                    self.decision_points.append(mac.currentTime)
                    self.decision_points = sorted(self.decision_points)

                # 如果所有的工作都已经到达车间，且所有的工作都已经完成，那么就把done标记为True
                if self.num_finished == self.num_jobs:
                    self.done = True
                    # 值得注意的是，在所有新工作均到达车间之前，属于task i的全部工作被执行完成时，不可以将其completed标为True，因为后续可能会有属于task i的新工作到达
                    for i in range(self.num_tasks):
                        self.tasks_list[i].completed = True  # 所有机器都已经被完成了，所以把所有的task都标记为完成
                        if len(self.completed_jobs[i]) > 0:
                            # task_i完成列表中最后一个job的完工时间即为该task的完工时间
                            last_J = self.completed_jobs[i][-1]
                            self.tasks_list[i].endTime = last_J.endTime
                        else:
                            self.tasks_list[i].endTime = 0
        obs, ava = self.obs_encode()
        state = obs.copy()
        # rewards = Rewarder(last_evaluation, self.tasks_list, self.machines)
        rewards = Rewarder(last_evaluation, self.tasks_list, self.machines, self.completed_jobs)
        # r = get_r(last_evaluation, self.completed_jobs)
        # r = get_reward(self.jobs, self.machines, last_span)
        # r = interval_time(self.tasks_list, last_PT)
        # r = reward_slackTime(pre_slack, self.jobs, self.machines)
        # r = getRewards2(pre_C1, pre_C2, pre_C3, preSpan, self.tasks_list, self.jobs)
        # r = final_reward4(self.tasks_list, last_PT, self.machines)
        # r = final_reward8(self.tasks_list, last_PT, [eWTmean, eWFmean, eWTmax], self.done, self.machines, weight,
        #                   self.num_finished / self.num_jobs)
        # r, pre_idleTime = final_reward9(self.tasks_list,[eWTmean, eWFmean, eWTmax], self.machines, pre_idleTime, now)
        # r = final_reward11(self.tasks_list, self.machines)
        # r = test_reward(self.tasks_list, self.machines, now, pre_idleTime[1], self.done, optimal_span, last_PT)
        # r = hope_reward(self.tasks_list, self.machines, now, pre_idleTime[1], self.done, optimal_span, last_PT)
        # r = last_reward(self.tasks_list, self.machines, now, pre_idleTime[1], self.done, optimal_span, last_PT)
        # r = last_reward2(self.tasks_list, self.machines, now, pre_idleTime[1], self.done, optimal_span, last_PT)
        # r = test_reward(self.tasks_list, self.machines, now, pre_idleTime[1], self.done, optimal_span, last_PT)
        # r = cal_Reward(self.jobs, self.tasks_list, self.machines, pre_obj, self.done, pre_C_last)
        # r = cal_Reward2(self.jobs, self.tasks_list, self.machines, pre_obj, self.done, pre_C_last)  # 这个尝试可行
        # r = cal_Reward2(self.tasks_list, pre_obj, self.done, last_endTime)
        r = get_reward_func(self.tasks_list, pre_obj, last_endTime, self.done)
        return obs, state, r, rewards, self.done, ava
    def obs_encode(self):
        '''Return all agent observations in a list'''
        obs = []
        ava = []

        # task agent
        for i in range(self.num_tasks + 1):
            avail = self.get_avail_action(i)
            if i == 3:   # the agent is for sequencing
                mac_state = self.get_sequence_features()
                state = {
                    "state": mac_state,
                    # "avail_action": avail
                }
            else:    # the agents are for routing for each task
                obs_tasks = self.get_routing_features(i)
                state = {
                    "state": obs_tasks,
                    # "avail_action": avail
                }

            obs_cat = np.hstack(
                # [np.array(state[k], dtype=np.float32).flatten() for k in sorted(state)]
                [np.array(state[k], dtype=np.float32).flatten() for k in state]
            )
            # if np.all(obs_cat == 0):
            #     obs_cat = [0.001 for _ in obs_cat]
            obs.append(obs_cat)
            ava.append(avail)
        return obs, ava

    # def get_sequence_features1(self):
    #     '''Returns observation for agent_id'''
    #     # the information of machines
    #     all_U_m = []
    #     CT = 0
    #     BTs = []
    #     wt = []
    #     m_que = []
    #     workload = []
    #     m_durations = []
    #     for m in self.machines:
    #         # wt.append(self.env.now - m.currentTime)
    #         pt = 0
    #         for op in self.machine_queue[m.name]:
    #             pt += op.cMachines[m.name]
    #         m_durations.append(pt)
    #         m_que.append(len(self.machine_queue[m.name]))
    #         if m.currentTime > CT:
    #             CT = m.currentTime
    #         assigned_opera = m.assignedOpera
    #         CT_m = m.currentTime  # Time when machine m completes the current last scheduled operation
    #         busy_time = 0
    #         for op in assigned_opera:
    #             busy_time += op.duration
    #         workload.append(busy_time)
    #         if busy_time == 0:
    #             U_m = 0
    #         else:
    #             U_m = busy_time / CT_m
    #         all_U_m.append(U_m)
    #         wt.append(self.env.now - busy_time)
    #         # BTs.append(busy_time)
    #
    #     U_avg = np.mean(all_U_m)  # 1. average utilization rate of all machines
    #     U_std = np.std(all_U_m)  # 2. standard deviation of all machine utilization rate
    #     max_workload = max(workload)
    #     if max_workload == 0:
    #         W_avg, W_std = 0, 0
    #     else:
    #         workload = [x / max_workload for x in workload]
    #         W_avg = np.mean(workload)  # 3. Average normalized machine workload
    #         W_std = np.std(workload)  # 4. Standard deviation of normalized machine workload
    #
    #     m_durations = [x / np.sum(m_durations) if np.sum(m_durations) > 0 else 0 for x in m_durations]
    #     m_que_avg = np.mean(
    #         m_durations)  # 5. Average processing time required for all machines to process all operations in the queue
    #
    #     wt_avg = np.std(
    #         m_durations)  # 6. Std of processing time required for all machines to process all operations in the queue
    #
    #     mac_state = np.array([U_avg, U_std, W_avg, W_std, wt_avg, m_que_avg])
    #
    #     # 使用 Clipping 和 Normalization 处理特征
    #     mac_state = np.clip(mac_state, 0, 2)  # 限制在合理的范围内
    #     mac_state = self.normalize_features(mac_state)  # 归一化
    #
    #     return mac_state
    #
    # def get_sequence_features2(self):
    #     '''Returns observation for agent_id'''
    #     # the information of machines
    #     all_U_m = []
    #     CT = 0
    #     BTs = []
    #     wt = []
    #     m_que = []
    #     workload = []
    #     m_durations = []
    #     for m in self.machines:
    #         # wt.append(self.env.now - m.currentTime)
    #         pt = 0
    #         for op in self.machine_queue[m.name]:
    #             pt += op.cMachines[m.name]
    #         m_durations.append(pt)
    #         m_que.append(len(self.machine_queue[m.name]))
    #         if m.currentTime > CT:
    #             CT = m.currentTime
    #         assigned_opera = m.assignedOpera
    #         CT_m = m.currentTime  # Time when machine m completes the current last scheduled operation
    #         busy_time = 0
    #         for op in assigned_opera:
    #             busy_time += op.duration
    #         workload.append(busy_time)
    #         if busy_time == 0:
    #             U_m = 0
    #         else:
    #             U_m = busy_time / CT_m
    #         all_U_m.append(U_m)
    #         wt.append(self.env.now - busy_time)
    #         # BTs.append(busy_time)
    #
    #     # avg_workload = np.mean(workload)
    #     U_avg = np.mean(all_U_m)  # 1. average utilization rate of all machines
    #     U_std = np.std(all_U_m)  # 2. standard deviation of all machine utilization rate
    #     # W = [x / CT if x > 0 else 0 for x in BTs]
    #     max_workload = max(workload)
    #     if max_workload == 0:
    #         W_avg, W_std = 0, 0
    #     else:
    #         workload = [x / max_workload for x in workload]
    #         W_avg = np.mean(workload)  # 3. Average normalized machine workload
    #         W_std = np.std(workload)  # 4. Standard deviation of normalized machine workload
    #
    #     # 5. Average processing time required for all machines to process all operations in the queue
    #     m_durations = [x / np.sum(m_durations) if np.sum(m_durations) > 0 else 0 for x in m_durations]
    #     m_que_avg = np.mean(m_durations)
    #
    #     # 6. Std of processing time required for all machines to process all operations in the queue
    #     wt_avg = np.std(m_durations)
    #
    #     mac_state = np.array([U_avg, U_std, W_avg, W_std, wt_avg, m_que_avg])
    #     # mac_state = (mac_state - np.mean(mac_state)) / np.std(mac_state)  # normalization
    #     # mac_state += 1
    #
    #     # return self.get_normalization_features(mac_state)
    #     return mac_state

    def get_sequence_features(self):
        '''Returns observation for agent_id'''
        epsilon = 1e-6  # 防止除零

        all_U_m = []
        CT = 0
        workload = []
        m_durations = []

        for m in self.machines:
            pt = sum(op.cMachines[m.name] for op in self.machine_queue[m.name])
            m_durations.append(pt)

            if m.currentTime > CT:
                CT = m.currentTime

            assigned_opera = m.assignedOpera
            CT_m = m.currentTime  # 机器完成当前任务的时间
            busy_time = sum(op.duration for op in assigned_opera)

            workload.append(busy_time)
            U_m = busy_time / (CT_m + epsilon)  # 避免0/0
            all_U_m.append(U_m)

        # 1. 计算机器利用率
        U_avg = np.mean(all_U_m)
        U_std = np.std(all_U_m)

        # 2. 归一化工作负载
        max_workload = max(workload, default=0)
        if max_workload == 0:
            W_avg, W_std = epsilon, epsilon
        else:
            workload = [x / (max_workload + epsilon) for x in workload]
            W_avg = np.mean(workload)
            W_std = np.std(workload)

        # 3. 计算所有机器的队列处理时间
        if np.sum(m_durations) == 0:
            m_que_avg = epsilon
        else:
            m_durations = [x / (np.sum(m_durations) + epsilon) for x in m_durations]
            m_que_avg = np.mean(m_durations)

        # 4. 计算队列处理时间的标准差
        wt_avg = np.std(m_durations) if len(m_durations) > 1 else epsilon

        # 5. 进行 `log1p()` 变换以控制尺度
        U_avg = np.log1p(U_avg)
        U_std = np.log1p(U_std)
        W_avg = np.log1p(W_avg)
        W_std = np.log1p(W_std)
        m_que_avg = np.sqrt(m_que_avg)  # 平滑变换

        mac_state = np.array([U_avg, U_std, W_avg, W_std, wt_avg, m_que_avg])

        return mac_state

    def get_routing_features(self, agent_id):
        '''Returns observation for agent_id'''

        epsilon = 1e-6  # 避免除零错误
        CRJs = []  # 完成率
        TR = []  # 任务延迟率
        jobs = self.tasks_list[agent_id].jobsList
        CKs = [m.currentTime for m in self.machines]
        T_cure = (np.mean(CKs) + epsilon) / self.num_machines  # 避免 0 值

        for j in jobs:
            OP_j = 0  # 已完成的操作数
            ETL_i = 0  # 预计剩余时间
            C_j = 0  # 任务 j 的最近完成时间
            for index, o in enumerate(j.operation_list):
                if o.completed:
                    OP_j += 1
                    C_j = o.endTime
                else:
                    cMachines = o.cMachines
                    mean_PT = sum(cMachines.values()) / (len(cMachines) + epsilon)  # 避免除 0
                    ETL_i += mean_PT
            CRJ = OP_j / (len(j.operation_list) + epsilon)  # 避免 0/0
            CRJs.append(CRJ)
            TR_j = (max(T_cure, C_j) + ETL_i - j.DT) / (C_j + ETL_i + epsilon)  # 归一化
            TR.append(TR_j)

        # 计算特征，避免全 0
        CRJ_avg = max(epsilon, np.mean(CRJs))
        CRJ_std = max(epsilon, np.std(CRJs))
        TR_avg = max(epsilon, np.mean(TR))
        TR_std = max(epsilon, np.std(TR))

        CTs = [m.currentTime for m in self.machines]
        T_cure = (np.mean(CTs) + epsilon) / self.num_machines
        min_CT = min(CTs)

        N_tard, N_left, N_Aleaft = 0, 0, 0
        for J in jobs:
            T_left = 0
            j = 0
            op_J = 0
            C_last = 9999
            for o in J.operation_list:
                if o.completed:
                    op_J += 1
                    C_last = o.endTime
                else:
                    N_left += 1
                    cMachines = o.cMachines
                    mean_PT = sum(cMachines.values()) / (len(cMachines) + epsilon)
                    T_left += mean_PT
                if T_left + max(T_cure, C_last) > J.DT:
                    j += 1
            N_tard += j
            if max(C_last, min_CT) > J.DT:
                N_Aleaft += len(J.operation_list) - op_J

        # 归一化 Tard_e 和 Tard_a 避免过大
        Tard_e = min(1, N_tard / (N_left + epsilon))
        Tard_a = min(1, N_Aleaft / (N_left + epsilon))

        # 处理数值差异：使用 log(1+x) 变换
        task_routing_state = np.array([
            np.log1p(CRJ_avg),
            np.log1p(CRJ_std),
            np.log1p(abs(TR_avg)),
            np.log1p(abs(TR_std)),
            np.log1p(Tard_e),
            np.log1p(Tard_a)
        ])
        return task_routing_state

    # def get_routing_features1(self, agent_id):
    #     '''Returns observation for agent_id'''
    #
    #     CRJs = []  # 作业完成率
    #     TR = []  # 任务处理延迟率
    #     jobs = self.tasks_list[agent_id].jobsList
    #     CKs = [m.currentTime for m in self.machines]
    #
    #     T_cure = np.mean(CKs) / max(1, self.num_machines)  # 避免除零
    #     for j in jobs:
    #         OP_j = 0  # 已完成工序数
    #         ETL_i = 0  # 剩余工序的估计完成时间
    #         C_j = 0  # 该作业最后调度的工序完成时间
    #
    #         for o in j.operation_list:
    #             if o.completed:
    #                 OP_j += 1
    #                 C_j = o.endTime
    #             else:
    #                 mean_PT = np.mean(list(o.cMachines.values())) if o.cMachines else 0
    #                 ETL_i += mean_PT
    #
    #         CRJ = OP_j / max(1, len(j.operation_list))  # 避免除零
    #         CRJs.append(CRJ)
    #
    #         TR_j = (max(T_cure, C_j) + ETL_i - j.DT) / max(1e-6, (C_j + ETL_i))  # 避免除零
    #         TR.append(max(TR_j, 0))  # 确保 TR 不小于 0
    #
    #     CRJ_avg = np.mean(CRJs) if CRJs else 0
    #     CRJ_std = np.std(CRJs) if CRJs else 0
    #     TR_avg = np.mean(TR) if TR else 0
    #     TR_std = np.std(TR) if TR else 0
    #
    #     CTs = [m.currentTime for m in self.machines]
    #     T_cure = np.mean(CTs) / max(1, self.num_machines)
    #     min_CT = min(CTs) if CTs else 0
    #
    #     N_tard, N_left, N_Aleaft = 0, 0, 0
    #     for J in jobs:
    #         T_left = 0
    #         j = 0
    #         op_J = 0  # 已完成工序数
    #         C_last = 9999  # 该作业最后调度的工序完成时间
    #
    #         for o in J.operation_list:
    #             if o.completed:
    #                 op_J += 1
    #                 C_last = o.endTime
    #             else:
    #                 N_left += 1
    #                 mean_PT = np.mean(list(o.cMachines.values())) if o.cMachines else 0
    #                 T_left += mean_PT
    #
    #             if T_left + max(T_cure, C_last) > J.DT:
    #                 j += 1
    #
    #         N_tard += j
    #         if max(C_last, min_CT) > J.DT:
    #             N_Aleaft += len(J.operation_list) - op_J
    #
    #     Tard_e = N_tard / max(1, N_left)  # 避免除零
    #     Tard_a = N_Aleaft / max(1, N_left)  # 避免除零
    #
    #     task_routing_state = np.array([CRJ_avg, CRJ_std, TR_avg, TR_std, Tard_e, Tard_a])
    #     task_routing_state = self.normalize_features(task_routing_state)  # 归一化
    #
    #     return task_routing_state
    #
    # def get_routing_features2(self, agent_id):
    #     '''Returns observation for agent_id'''
    #
    #     # tasks_routing_states = []
    #     # the information of jobs
    #     CRJs = []  # the completion rate of jobs
    #     TR = []   #
    #     jobs = self.tasks_list[agent_id].jobsList
    #     CKs = []
    #     for m in self.machines:
    #         CKs.append(m.currentTime)
    #     T_cure = np.mean(CKs) / self.num_machines
    #     for j in jobs:
    #         OP_j = 0  # current operation number that has been completed of job J_i
    #         ETL_i = 0  # estimated completion time of the remaining operations of job J_i
    #         C_j = 0 # The completion time of the last scheduled operation of job J_i until decision point
    #         for index, o in enumerate(j.operation_list):
    #             if o.completed:
    #                 OP_j += 1
    #                 C_j = o.endTime
    #             else:
    #                 cMachines = o.cMachines
    #                 total_sum = sum(cMachines.values())
    #                 mean_PT = total_sum / float(len(cMachines))
    #                 ETL_i += mean_PT
    #         CRJ = OP_j / len(j.operation_list)
    #         CRJs.append(CRJ)
    #         TR_j = (max(T_cure, C_j) + ETL_i - j.DT) / (C_j + ETL_i)   # the job processing delay rate
    #         TR.append(TR_j)
    #     CRJ_avg = np.mean(CRJs)  # 1. mean completion rate of jobs
    #     CRJ_std = np.std(CRJs)   # 2. standard deviation of completion rate of jobs
    #     TR_avg = np.mean(TR)  # 3. mean processing delay rate of jobs
    #     TR_std = np.std(TR)  # 4. standard deviation of processing delay rate of jobs
    #
    #     CTs = []
    #     for m in self.machines:
    #         CTs.append(m.currentTime)
    #     T_cure = np.mean(CTs) / self.num_machines
    #     min_CT = min(CTs)
    #
    #     N_tard, N_left = 0, 0
    #     N_Aleaft = 0
    #     for J in jobs:
    #         T_left = 0
    #         j = 0
    #         op_J = 0  # the number of operations that have been completed
    #         C_last = 9999  # the completion time of the last scheduled operation of job J_i until decision point
    #         for o in J.operation_list:
    #             if o.completed:
    #                 op_J += 1
    #                 C_last = o.endTime
    #             else:
    #                 N_left += 1  # The number of operations that have not been completed
    #                 cMachines = o.cMachines
    #                 total_sum = sum(cMachines.values())
    #                 mean_PT = total_sum / float(len(cMachines))
    #                 T_left += mean_PT
    #             if T_left + max(T_cure, C_last) > J.DT:
    #                 j += 1
    #         N_tard += j
    #         if max(C_last, min_CT) > J.DT:
    #             N_Aleaft += len(J.operation_list) - op_J
    #     try:
    #         Tard_e = N_tard / N_left   # 5. Estimated tardiness rate Tard_e
    #     except:
    #         Tard_e = 9999
    #
    #     try:
    #         Tard_a = N_Aleaft / N_left   # 6. Actual tardiness rate Tard_a
    #     except:
    #         Tard_a = 9999
    #
    #     task_routing_state = np.array([CRJ_avg, CRJ_std, TR_avg, TR_std, Tard_e, Tard_a])
    #     # task_routing_state += 1
    #     # task_routing_state = (task_routing_state - np.mean(task_routing_state)) / np.std(task_routing_state)  # normalization
    #     # return self.get_normalization_features(task_routing_state)
    #     return task_routing_state

    # def normalize_features(self, features):
    #     """对特征进行归一化，防止极端值影响训练"""
    #     max_threshold = np.percentile(features, 95)  # 计算 95% 分位数
    #     features = np.clip(features, 0, max_threshold)  # 限制最大值
    #     mean_val = np.mean(features)
    #     std_val = np.std(features)
    #     return (features - mean_val) / max(1e-6, std_val)  # 标准化，避免除零
    #
    # def get_normalization_features(self, mac_state):
    #     '''Returns observation for agent_id'''
    #
    #     # 归一化处理
    #     if len(mac_state) > 0:
    #         max_data = np.max(mac_state)
    #         min_data = np.min(mac_state)
    #         mac_state = (mac_state - min_data) / (max_data-min_data + 1e-6)
    #         mac_state = np.nan_to_num(mac_state, nan=1e-6)
    #
    #         # mean = np.mean(mac_state)
    #         # std = np.std(mac_state)
    #         # # 防止除零和溢出
    #         # std = std if std > 1e-8 else 1e-8
    #         # mac_state = (mac_state - mean) / std
    #         # mac_state = np.nan_to_num(mac_state, nan=1e-6)
    #     else:
    #         mac_state = np.ones_like(mac_state)
    #
    #     return mac_state

    def get_avail_action(self, agent_id):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            SPTM,
            NINQ,
            WINQ,
            LWT,
            SPT,
            LPT,
            MWKR,
            EDD,
            MOPNR
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        if agent_id == 0 or agent_id == 1 or agent_id == 2:
            (
                avail[SPT],
                avail[LPT],
                avail[MWKR],
                avail[EDD],
                avail[MOPNR]
             ) = (0, 0, 0, 0, 0)
        else:
            (
                avail[SPTM],
                avail[NINQ],
                avail[WINQ],
                avail[LWT]
            ) = (0, 0, 0, 0)

        return np.array(avail)

    def Features(self, agent_id):

        jobs = self.tasks_list[agent_id].jobsList
        CTs = []
        for m in self.machines:
            CTs.append(m.currentTime)
        T_cure = np.mean(CTs)/self.num_machines
        min_CT = min(CTs)

        # 1 Estimated tardiness rate Tard_e
        N_tard, N_left = 0, 0
        N_Aleaft = 0
        for J in jobs:
            T_left = 0
            j = 0
            op_J = 0   # the number of operations that have been completed
            C_last = 9999   # the completion time of the last scheduled operation of job J_i until decision point
            for o in J.operation_list:
                if o.completed:
                    op_J += 1
                    C_last = o.endTime
                else:
                    N_left += 1    # The number of operations that have not been completed
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    T_left += mean_PT
                if T_left + max(T_cure, C_last) > J.DT:
                    j += 1
            N_tard += j
            if max(C_last, min_CT) > J.DT:
                N_Aleaft += len(J.operation_list) - op_J
        try:
            Tard_e = N_tard / N_left
        except:
            Tard_e = 9999

        try:
            Tard_a = N_Aleaft / N_left
        except:
            Tard_a = 9999

        return Tard_e, Tard_a



##################################################################### for AMDQN  #################################################################################
    def Features_AMDQN(self, jobs):
        '''Returns observation for agent_id'''

        # tasks_routing_states = []
        # the information of jobs
        CRJs = []  # the completion rate of jobs
        TR = []   #
        CKs = []
        for m in self.machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs) / self.num_machines
        for j in jobs:
            OP_j = 0  # current operation number that has been completed of job J_i
            ETL_i = 0  # estimated completion time of the remaining operations of job J_i
            C_j = 0 # The completion time of the last scheduled operation of job J_i until decision point
            for index, o in enumerate(j.operation_list):
                if o.completed:
                    OP_j += 1
                    C_j = o.endTime
                else:
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    ETL_i += mean_PT
            CRJ = OP_j / len(j.operation_list)
            CRJs.append(CRJ)
            TR_j = (max(T_cure, C_j) + ETL_i - j.DT) / (C_j + ETL_i)   # the job processing delay rate
            TR.append(TR_j)
        CRJ_avg = np.mean(CRJs)  # 1. mean completion rate of jobs
        CRJ_std = np.std(CRJs)   # 2. standard deviation of completion rate of jobs
        TR_avg = np.mean(TR)  # 3. mean processing delay rate of jobs
        TR_std = np.std(TR)  # 4. standard deviation of processing delay rate of jobs

        CTs = []
        for m in self.machines:
            CTs.append(m.currentTime)
        T_cure = np.mean(CTs) / self.num_machines
        min_CT = min(CTs)

        N_tard, N_left = 0, 0
        N_Aleaft = 0
        for J in jobs:
            T_left = 0
            j = 0
            op_J = 0  # the number of operations that have been completed
            C_last = 9999  # the completion time of the last scheduled operation of job J_i until decision point
            for o in J.operation_list:
                if o.completed:
                    op_J += 1
                    C_last = o.endTime
                else:
                    N_left += 1  # The number of operations that have not been completed
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    T_left += mean_PT
                if T_left + max(T_cure, C_last) > J.DT:
                    j += 1
            N_tard += j
            if max(C_last, min_CT) > J.DT:
                N_Aleaft += len(J.operation_list) - op_J
        try:
            Tard_e = N_tard / N_left   # 5. Estimated tardiness rate Tard_e
        except:
            Tard_e = 9999

        try:
            Tard_a = N_Aleaft / N_left   # 6. Actual tardiness rate Tard_a
        except:
            Tard_a = 9999
        return CRJ_avg, CRJ_std, TR_avg, TR_std, Tard_e, Tard_a

    def step_AMDQN(self, a_m, a_s):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用列表推导式创建一个需要执行的任务列表
            futures = [executor.submit(eval(a_m), self.machines, self.tasks_list[i].jobsList, self.env.now, self.machine_queue)
                       for i in range(self.num_tasks)]
            # # 调用parallel_execution方法来并行执行操作,等待所有任务完成并获取结果
            for future in concurrent.futures.as_completed(futures):
                self.machine_queue = future.result()
        for mac in self.machines:
            if mac.currentTime > self.env.now:  # the machine is busy
                continue
            opera = eval(a_s)(self.tasks_list, mac, self. machine_queue)  # decide an operation to be processed
            if opera is not None:
                if opera in mac.assignedOpera:
                    input("条件满足，按Enter键继续")
                opera.assigned = True
                opera.completed = True
                opera.assignedMachine = mac.name
                opera.startTime = self.env.now
                opera.duration = opera.cMachines[mac.name]
                opera.endTime = self.env.now + opera.duration
                mac.currentTime = opera.endTime
                mac.assignedOpera.append(opera)
                self.machine_queue[mac.name].remove(opera)
                J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                # 如果这个操作是该工作的第一个操作，那么就把该工作的开始时间标记为该操作的开始时间
                if opera.idOpertion == 0:
                    J.RT = opera.startTime
                # 如果这个操作是该工作的最后一个操作，那么就把该工作的结束时间标记为该操作的结束时间，把该工作的completed标记为True，且加入到已经完成的工作列表中
                if opera.idOpertion == len(J.operation_list) - 1:
                    J.completed = True
                    self.num_finished += 1
                    J.endTime = J.getEndTime()
                    self.completed_jobs[J.idTask].append(J)

                # 机器处理完一个操作的时间点为一个决策点，所以把机器处理完一个操作的时间点加入到决策点列表中
                if mac.currentTime not in self.decision_points:
                    self.decision_points.append(mac.currentTime)
                    self.decision_points = sorted(self.decision_points)

                # 如果所有的工作都已经到达车间，且所有的工作都已经完成，那么就把done标记为True
                if self.num_finished == self.num_jobs:
                    self.done = True
                    # 值得注意的是，在所有新工作均到达车间之前，属于task i的全部工作被执行完成时，不可以将其completed标为True，因为后续可能会有属于task i的新工作到达
                    for i in range(self.num_tasks):
                        self.tasks_list[i].completed = True  # 所有机器都已经被完成了，所以把所有的task都标记为完成
                        if len(self.completed_jobs[i]) > 0:
                            # task_i完成列表中最后一个job的完工时间即为该task的完工时间
                            last_J = self.completed_jobs[i][-1]
                            self.tasks_list[i].endTime = last_J.endTime
                        else:
                            self.tasks_list[i].endTime = 0

    def Features_PPO(self):

        jobs = self.jobs
        CTQ = 0      # the number of completed jobs
        for m, queue in self.machine_queue.items():
            CTQ += len(queue)
        Job_system = len(jobs) - CTQ

        remain_jobs = 0
        for i in range(self.num_tasks):
            remain_jobs += len(self.completed_jobs[i])

        TTDs = []
        queue_job_num = 0
        ROPs = np.zeros(len(jobs))
        RPTs = np.zeros(len(jobs))
        NPTs = np.zeros(len(jobs))
        STs = np.zeros(len(jobs))
        WTQs = np.zeros(len(jobs))
        NTQs = np.zeros(len(jobs))
        for count, job in enumerate(jobs):
            DDT = job.DDT
            TTD = DDT - self.env.now
            TTDs.append(TTD)
            NPT = None
            flag = False
            for index, opera in enumerate(job.operation_list):
                if opera.assigned and not opera.completed:
                    flag = True
                    WTQs[count] += opera.cMachines[opera.assignedMachine]
                    NTQs[count] += 1

                if not opera.completed:
                    ROPs[count] += 1
                    PT = sum(opera.cMachines.values()) / len(opera.cMachines)
                    RPTs[count] += PT

                if index == 0:
                    if not opera.completed:
                        NPT = sum(opera.cMachines.values()) / len(opera.cMachines)
                    else:
                        if index != len(job.operation_list) - 1:
                            if not job.operation_list[index + 1].completed:
                                tmp_opera = job.operation_list[index + 1]
                                NPT = sum(tmp_opera.cMachines.values()) / len(tmp_opera.cMachines)
                else:
                    if index != len(job.operation_list) - 1:
                        if not opera.completed:
                            if job.operation_list[index - 1].completed:
                                NPT = sum(opera.cMachines.values()) / len(opera.cMachines)
                        else:
                            if not job.operation_list[index + 1].completed:
                                tmp_opera = job.operation_list[index + 1]
                                NPT = sum(tmp_opera.cMachines.values()) / len(tmp_opera.cMachines)
                    else:
                        if not opera.completed:
                            if job.operation_list[index - 1].completed:
                                NPT = sum(opera.cMachines.values()) / len(opera.cMachines)
                        else:
                            NPT = 0
                NPTs[count] = NPT
            STs[count] = TTD-RPTs[count]
            if flag:
                queue_job_num += 1

        Job_queue = np.sum(NTQs)

        WTQs_avg = np.mean(WTQs)
        WTQs_std = np.std(WTQs)
        WTQs_max = np.max(WTQs)
        WTQs_min = np.min(WTQs)

        NTQs_avg = np.mean(NTQs)
        NTQs_std = np.std(NTQs)
        NTQs_max = np.max(NTQs)
        NTQs_min = np.min(NTQs)

        TTD_avg = np.mean(TTDs)
        TTD_std = np.std(TTDs)
        TTD_max = np.max(TTDs)
        TTD_min = np.min(TTDs)

        ROP_avg = np.mean(ROPs)
        ROP_std = np.std(ROPs)
        ROP_max = np.max(ROPs)
        ROP_min = np.min(ROPs)

        RPT_avg = np.mean(RPTs)
        RPT_std = np.std(RPTs)
        RPT_max = np.max(RPTs)
        RPT_min = np.min(RPTs)

        NPT_avg = np.mean(NPTs)
        NPT_std = np.std(NPTs)
        NPT_max = np.max(NPTs)
        NPT_min = np.min(NPTs)

        ST_avg = np.mean(STs)
        ST_std = np.std(STs)
        ST_max = np.max(STs)
        ST_min = np.min(STs)

        return Job_system, Job_queue, WTQs_avg, WTQs_std, WTQs_max, WTQs_min, NTQs_avg, NTQs_std, NTQs_max, NTQs_min, \
            TTD_avg, TTD_std, TTD_max, TTD_min, ROP_avg, ROP_std, ROP_max, ROP_min, RPT_avg, RPT_std, RPT_max, RPT_min, \
            NPT_avg, NPT_std, NPT_max, NPT_min, ST_avg, ST_std, ST_max, ST_min

    def Features_DMDDQN(self, jobs):
        '''Returns observation for agent_id'''

        # tasks_routing_states = []
        # the information of jobs
        CRJs = []  # the completion rate of jobs
        TR = []   #
        CKs = []
        for m in self.machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs) / self.num_machines
        for j in jobs:
            OP_j = 0  # current operation number that has been completed of job J_i
            ETL_i = 0  # estimated completion time of the remaining operations of job J_i
            C_j = 0 # The completion time of the last scheduled operation of job J_i until decision point
            for index, o in enumerate(j.operation_list):
                if o.completed:
                    OP_j += 1
                    C_j = o.endTime
                else:
                    cMachines = o.cMachines
                    total_sum = sum(cMachines.values())
                    mean_PT = total_sum / float(len(cMachines))
                    ETL_i += mean_PT
            CRJ = OP_j / len(j.operation_list)
            CRJs.append(CRJ)
            TR_j = (max(T_cure, C_j) + ETL_i - j.DT) / (C_j + ETL_i)   # the job processing delay rate
            TR.append(TR_j)
        CRJ_avg = np.mean(CRJs)  # 1. mean completion rate of jobs
        CRJ_std = np.std(CRJs)   # 2. standard deviation of completion rate of jobs
        TR_avg = np.mean(TR)  # 3. mean processing delay rate of jobs
        TR_std = np.std(TR)  # 4. standard deviation of processing delay rate of jobs

        if self.env.now == 0:
            UR_avg = 0
            UR_std = 0
        else:
            URs = []
            for m in self.machines:
                pt = 0
                for op in m.assignedOpera:
                    pt += op.cMachines[m.name]
                URs.append(pt / self.env.now)
            UR_avg = np.mean(URs)
            UR_std = np.std(URs)

        return UR_avg, UR_std, CRJ_avg, CRJ_std, TR_avg, TR_std


    def get_states_HMAPPO(self, jobs_list, machines_list, previous_feature):
        # total number of machines for the task
        num_machines = len(machines_list)
        UK = []
        # total number of jobs for the task in current system
        n_jobs = len(jobs_list)
        # Average utilization rate of machines
        workloads = []
        for m in machines_list:
            CT = m.currentTime
            sum_pt = 0
            for j in jobs_list:
                for o in j.operation_list:
                    if o.assignedMachine == m.name:
                        sum_pt += o.cMachines[m.name]
            workloads.append(sum_pt)
            if CT == 0:
                u = 0
            else:
                u = sum_pt / CT
            UK.append(u)
        U_avg = sum(UK) / num_machines

        # the normalized workload of machines
        normal_workloads = []
        max_w = max(workloads)
        for item in workloads:
            if max_w == 0:
                n_w = 0
            else:
                n_w = item / max_w
            normal_workloads.append(n_w)

        # Average normalized machine workload
        W_avg = sum(normal_workloads) / num_machines

        # Standard deviation of normalized machine workload
        sum_w = 0
        for item in workloads:
            sum_w += pow(item - W_avg, 2)
        W_std = sum_w / num_machines

        # Standard deviation of machine utilization rate
        sum_U = 0
        for u in UK:
            sum_U += pow(u - U_avg, 2)
        U_std = math.sqrt(sum_U / num_machines)

        OP = np.zeros(len(jobs_list))
        CRJ = np.zeros(len(jobs_list))
        for job in jobs_list:
            counter = 0
            for opera in job.operation_list:
                if opera.completed:
                    OP[counter] += 1
                else:
                    break
            CRJ[counter] = OP[counter] / len(job.operation_list)
            counter += 1
        # Completion rate of operations
        if n_jobs:
            CRO = 0
            CRJ_avg = 0
            CRJ_std = 0
        else:
            CRO = sum(OP) / n_jobs
            # Average job completion rate
            CRJ_avg = sum(CRJ) / n_jobs
            # Standard deviation of job completion rate
            sum_CRJ = 0
            for i in range(n_jobs):
                sum_CRJ += pow(CRJ[i] - CRJ_avg, 2)
            CRJ_std = math.sqrt(sum_CRJ / n_jobs)

        # Estimated Tardiness Rate Tard_e and Actual Tardiness Rate Tard_a
        CKs = []
        ETO = []
        for m in self.machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs)
        min_CK = min(CKs)
        num_unfinished_jobs = 0
        sum_unOpera = []
        ATJ = []
        UC = []
        ATJ_num_inOpera = []
        UC_num_inOpera = []
        for j in jobs_list:
            C_last = j.RT + j.span  # job j中最后一个被schedule的操作的完工时间
            if not j.completed:
                UC.append(j)
                num_unfinished_jobs += 1
                all_t_avg = []
                sum_t_avg = 0
                uncompleted_num = 0
                for o in j.operation_list:
                    if not o.completed:
                        uncompleted_num += 1
                        cMachines = o.cMachines
                        t_avg = np.mean(list(cMachines.values()))
                        sum_t_avg += t_avg
                        if max(T_cure, C_last) + sum_t_avg > j.DT:
                            ETO.append(o)
                sum_unOpera.append(uncompleted_num)
            UC_num_inOpera.append(num_unfinished_jobs)
            if num_unfinished_jobs > 0 and max(C_last, min_CK) > j.DT:
                ATJ.append(j)
                ATJ_num_inOpera.append(num_unfinished_jobs)
        if len(sum_unOpera) == 0:
            Tard_e = 0
        else:
            Tard_e = len(ETO) / sum(sum_unOpera)  # Estimated Tardiness Rate Tard_e
        if len(UC_num_inOpera) == 0 or len(ATJ_num_inOpera) == 0:
            Tard_a = 0
        else:
            Tard_a = sum(ATJ_num_inOpera) / sum(UC_num_inOpera)  # Actual Tardiness Rate Tard_a
        current_feature = [num_machines, U_avg, U_std, CRO, CRJ_avg, CRJ_std, W_avg, W_std, Tard_e, Tard_a]
        D = [current_feature[i] - previous_feature[i] for i in range(len(current_feature))]
        current_feature.extend(D)
        return current_feature












