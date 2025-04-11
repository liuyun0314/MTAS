from collections import Counter
import numpy as np

# \ref{Dynamic scheduling for flexible job shop using a deep reinforcement learning approach}
# \ref{Deep reinforcement learning for dynamic flexible job shop scheduling problem considering variable processing times}
# obtain the ready operations
def ready_operations(jobs, currentTime):
    # 这里应该补充这个operation是否已经添加到队列中
    ready_operations = {}
    # jobs = jobsSet.jobsList
    for index, job in enumerate(jobs):
        opera_list = job.operation_list
        if len(opera_list) != 0:
            if not opera_list[0].completed and not opera_list[0].assigned:
                ready_operations[index] = opera_list[0]
            else:
                pre_opera = opera_list[0]
                for opera in opera_list[1:]:
                    if not opera.completed and not opera.assigned:
                        if pre_opera.completed and pre_opera.endTime <= currentTime:
                            ready_operations[index] = opera
                            break
                    else:
                        pre_opera = opera
    return ready_operations

## action for machine assignment rule (routing)

def SPTM(macs, jobs, current_time, machine_queue):
    '''
    Select the machine with the smallest processing time of the operation
    :param macs:
    :param jobs:
    :param current_time:
    :param machine_queue:
    :return:
    '''
    ready_opera = ready_operations(jobs, current_time)
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                selected_mac = min(cMachines, key=cMachines.get)
                if selected_mac not in machine_queue.keys():   # 说明这个机器坏了，次小的
                    filtered_dict = {k: v for k, v in cMachines.items() if v != cMachines[selected_mac]}
                    if len(filtered_dict):
                        selected_mac = min(filtered_dict, key=filtered_dict.get)
                    else:
                        selected_mac = None
                if selected_mac is not None:
                    machine_queue[selected_mac].append(opera)
                    opera.assigned = True
                    opera.assignedMachine = selected_mac
    return machine_queue

def NINQ(macs, jobs, current_time, machine_queue):
    '''
    Select the machine with the smallest number of jobs in the queue
    :param macs:
    :param jobs:
    :param current_time:
    :param machine_queue:
    :return:
    '''
    ready_opera = ready_operations(jobs, current_time)
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                min_num = np.Inf
                selected_mac = None
                for mac in cMachines.keys():
                    if mac in machine_queue.keys():
                        opera_list = machine_queue[mac]
                        if len(opera_list) == 0:
                            selected_mac = mac
                            # machine_queue[mac].append(opera)
                            break
                        else:
                            job_counter = Counter([op.jobID for op in opera_list])
                            num_jobs = len(job_counter)
                            if num_jobs < min_num:
                                min_num = num_jobs
                                selected_mac = mac
                        # jobs_count.append(num_jobs)
                # selected_index = jobs_count.index(min(jobs_count))
                # selected_mac = macs[selected_index].name
                if selected_mac is not None:
                    opera.assigned = True
                    machine_queue[selected_mac].append(opera)
                    opera.assignedMachine = selected_mac
    return machine_queue

def WINQ(macs, jobs, current_time, machine_queue):
    '''
    Select the machine with the smallest workload, i.e., the sum of operation processing time of jobs in the queue
    :param macs:
    :param jobs:
    :param current_time:
    :param machine_queue:
    :return:
    '''
    ready_opera = ready_operations(jobs, current_time)
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                # all_sum_PT = list(cMachines.values())[0]
                # selected_mac = list(cMachines.keys())[0]
                all_sum_PT = np.Inf
                selected_mac = None
                for name in cMachines.keys():
                    if name in machine_queue.keys():
                        opera_list = machine_queue[name]
                        if len(opera_list) != 0:
                            sum_PT = 0
                            for op in opera_list:
                                sum_PT += op.cMachines[name]
                            if sum_PT < all_sum_PT:
                                selected_mac = name
                                all_sum_PT = sum_PT
                            # all_sum_PT.append(sum_PT)
                        else:
                            selected_mac = name
                            break
                # min_index = all_sum_PT.index(min(all_sum_PT))
                if selected_mac is not None:
                    opera.assigned = True
                    machine_queue[selected_mac].append(opera)
                    opera.assignedMachine = selected_mac
    return machine_queue

def LWT(macs, jobs, current_time, machine_queue):
    '''
    Select the machine with the less total processing time
    :param macs:
    :param jobs:
    :param current_time:
    :param machine_queue:
    :return:
    '''
    ready_opera = ready_operations(jobs, current_time)
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                all_total_PT = []
                max_PT = np.Inf
                selected_mac = None
                for name in cMachines.keys():
                    if name in machine_queue.keys():
                        opera_list = machine_queue[name]
                        total_PT = sum(obj.cMachines[name] for obj in opera_list)
                        # if a machine is processing an operation, then the total processing time should minus the processing time of the operation that has finished
                        machine = [obj for obj in macs if obj.name == name]
                        if len(machine) != 0:
                            machine = machine[0]
                        else:
                            # This situation indicates that a machine is broken down
                            continue
                        if machine.currentTime > current_time: # the machine is busy at the step
                            diff_PT = machine.currentTime - current_time # It will diff_PT to complete the operation
                            total_PT += diff_PT
                        if total_PT < max_PT:
                            selected_mac = name
                            max_PT = total_PT
                    # all_total_PT.append(total_PT)
                # min_index = all_total_PT.index(min(all_total_PT))
                if selected_mac is not None:
                    # If selected is still None at this point, there is only one candidate machine for this operation, and that machine is currently broken.
                    opera.assigned = True
                    machine_queue[selected_mac].append(opera)
                    opera.assignedMachine = selected_mac
    return machine_queue

def noAction(macs, jobs, current_time, machine_queue):
    '''
    No operation is assigned
    :param macs:
    :param jobs:
    :param current_time:
    :param machine_queue:
    :return:
    '''
    return machine_queue

# actions for operation sequencing rules (sequencing)

def SPT(task_list, mac, machine_queue):
    '''
    Select the operation with the shortest processing time
    :param task_list:
    :param mac:
    :param current_time:
    :param machine_queue:
    :return:
    '''

    if len(machine_queue[mac.name]) == 0:
        return None
    else:
        queue = machine_queue[mac.name]
        selected_opera = queue[0]
        for op in queue:
            if op.cMachines[mac.name] < selected_opera.cMachines[mac.name]:
                selected_opera = op
        # selected_opera = min(machine_queue, key=lambda x: x.cMachineNames[mac.name])
        return selected_opera

def LPT(task_list, mac, machine_queue):
    '''
    Select the operation with the longest processing time
    :param task_list:
    :param mac:
    :param current_time:
    :param machine_queue:
    :return:
    '''
    if len(machine_queue[mac.name]) == 0:
        return None
    else:
        queue = machine_queue[mac.name]
        selected_opera = queue[0]
        for op in queue:
            if op.cMachines[mac.name] > selected_opera.cMachines[mac.name]:
                selected_opera = op
        # selected_opera = max(machine_queue, key=lambda x: x.cMachineNames[mac.name])
        return selected_opera

def MWKR(task_list, mac, machine_queue):
    '''
    Select the job with the shortest average remaining processing time
    :param task_list:
    :param mac:
    :param current_time:
    :param machine_queue:
    :return:
    '''

    # which jobs in the queue are ready to be processed
    # ready_jobs = set(op.jobID for op in machine_queue[mac.name])
    # ready_jobs = sorted(ready_jobs)
    if len(machine_queue[mac.name]) == 0:
        return None
    else:
        # the average remaining processing time of the jobs in the queue
        all_avg_PT = []
        for o in machine_queue[mac.name]:
            task_id = o.taskID
            job = task_list[task_id].jobsList[o.jobID]
            sum_PT = 0
            for o in job.operation_list:
                if not o.completed:
                    avg_PT = sum(o.cMachines.values()) / len(o.cMachines)
                    sum_PT += avg_PT
            all_avg_PT.append(sum_PT)
        index = np.argmin(all_avg_PT)
        selected_opera = machine_queue[mac.name][index]
        return selected_opera

def EDD(task_list, mac, machine_queue):
    '''
    Select the job with the shortest due-date
    :param task_list:
    :param mac:
    :param current_time:
    :param machine_queue:
    :return:
    '''

    if len(machine_queue[mac.name]) == 0:
        return None
    else:
        # the average remaining processing time of the jobs in the queue
        DTs = []
        for o in machine_queue[mac.name]:
            task_id = o.taskID
            job = task_list[task_id].jobsList[o.jobID]
            DTs.append(job.DT)
        index = np.argmin(DTs)
        selected_opera = machine_queue[mac.name][index]
        return selected_opera

def MOPNR(task_list, mac, machine_queue):
    '''
    Select the job with most operations remaining
    :param jobs:
    :param mac:
    :param current_time:
    :param machine_queue:
    :return:
    '''

    if len(machine_queue[mac.name]) == 0:
        return None
    else:
        all_remained_op = []
        for o in machine_queue[mac.name]:
            task_id = o.taskID
            job = task_list[task_id].jobsList[o.jobID]
            num_remained_op = 0   # the number of remainning operations belonging to the job
            for o in job.operation_list:
                if not o.completed:
                    num_remained_op += 1
            all_remained_op.append(num_remained_op)
        index = np.argmax(all_remained_op)
        selected_opera = machine_queue[mac.name][index]
        return selected_opera