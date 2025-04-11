import math
import numpy as np
import torch
from scipy.special import expit
from MAT.utils.valuenorm import ValueNorm

def span(completed_jobs):
    tasks_Cma = []
    for k, v in completed_jobs.items():
        if len(v) == 0:
            Cmax = 0

        else:
            Cmax = v[-1].endTime
        tasks_Cma.append(Cmax)
    makespan = max(tasks_Cma)
    return makespan

def get_r(last_reward, completed_jobs):
    makespan = span(completed_jobs)
    gap = makespan - last_reward[3]
    if gap == 0:   # 如果gap=0说明这一次没有任务被分配
        reward = -1
    else:
        reward = 1 / gap
    return np.float64(reward)

def Rewarder(last_reward, task_list, machines, completed_jobs):
    rewards = []
    r_WTmean = reward_WT_mean(last_reward[0], task_list[0].jobsList, machines)
    r_WFmean = reward_WF_mean(last_reward[1], task_list[1].jobsList, machines)
    r_WTmax = reward_WT_max(last_reward[2], task_list[2].jobsList, machines)
    r_UR = reward_global_U(last_reward[3], machines)
    makespan = span(completed_jobs)
    # tasks_Cma = []
    # for k, v in completed_jobs.items():
    #     if len(v) == 0:
    #         Cmax = 0
    #     else:
    #         Cmax = v[-1].endTime
    #     tasks_Cma.append(Cmax)
    # makespan = max(tasks_Cma)
    gap = makespan - last_reward[3]
    if gap == 0:   # 如果gap=0说明这一次没有任务被分配
        r = -1
    else:
        r = 1 / gap

    rates = dynamic_rates(task_list, machines, completed_jobs)
    reward_dynamic = np.dot(rates, [r_WTmean, r_WFmean, r_WTmax, r])
    # return [r_WTmean, r_WFmean, r_WTmax, r_UR]
    return reward_dynamic

def dynamic_rates(task_list, machines, completed_jobs):

    EWTmean = estimated_WT_mean(task_list[0].jobsList, machines)
    EWFmean = estimated_WF_mean(task_list[1].jobsList, machines)
    EWTmax = estimated_WT_max(task_list[2].jobsList, machines)
    tasks_Cma = []
    for k, v in completed_jobs.items():
        if len(v) == 0:
            Cmax = 0
        else:
            Cmax = v[-1].endTime
        tasks_Cma.append(Cmax)
    makespan = max(tasks_Cma)
    objs = [EWTmean, EWFmean, EWTmax, makespan]
    objs_array = np.array(objs)
    objs_normalized = (objs_array - objs_array.min()) / (objs_array.max() - objs_array.min())
    sum_obj = np.sum(objs_normalized)
    dy_rate = [x/sum_obj for x in objs_normalized]
    return dy_rate

def laset_evaluations(task_list, machines, completed_jobs):
# def laset_evaluations(task_list, machines, jobs):
    EWTmean = estimated_WT_mean(task_list[0].jobsList, machines)
    EWFmean = estimated_WF_mean(task_list[1].jobsList, machines)
    EWTmax = estimated_WT_max(task_list[2].jobsList, machines)
    UR = machine_utilizatioin_ratio(machines)
    makespan = span(completed_jobs)
    # makespan = estimated_span(jobs, machines)
    # tasks_Cma = []
    # for k, v in completed_jobs.items():
    #     if len(v) == 0:
    #         Cmax = 0
    #     else:
    #         Cmax = v[-1].endTime
    #     tasks_Cma.append(Cmax)
    # makespan = max(tasks_Cma)
    # return [EWTmean, EWFmean, EWTmax, UR]
    return [EWTmean, EWFmean, EWTmax, makespan]


#######版本1#######
def reward_WT_mean2(r, jobs, machines):
    reward = 0
    r_next = estimated_WT_mean(jobs, machines)
    if r_next < r:
        reward = 1
    else:
        # if r_next == r:
        if r_next < 1.1 * r:
            reward = 0
        else:
            reward = -1
    return reward

def reward_WT_max2(r, jobs, machines):
    reward = 0
    r_next = estimated_WT_max(jobs, machines)
    if r_next < r:
        reward = 1
    else:
        # if r_next == r:
        if r_next < 1.1 * r:
            reward = 0
        else:
            reward = -1
    return reward

def reward_WF_mean2(r, jobs, machines):
    reward = 0
    r_next = estimated_WF_mean(jobs, machines)
    if r_next < r:
        reward = 1
    else:
        # if r_next == r:
        if r_next < 1.1 * r:
            reward = 0
        else:
            reward = -1
    return reward

def reward_global_U(r, machines):
    reward = 0
    r_next = machine_utilizatioin_ratio(machines)
    if r_next > r:
        reward = 1
    else:
        if r_next >= 0.97 * r:  # ref{Real-Time Scheduling for Dynamic Partial-No-Wait Multiobjective Flexible Job Shop by Deep Reinforcement Learning}
            reward = 0
        else:
            reward = -1
    return reward

############版本2############
def reward_WT_mean(r, jobs, machines):
    reward = 0
    r_next = estimated_WT_mean(jobs, machines)
    if r_next < r:
        reward = expit(r_next - r)
        # reward = 1 / (1 + np.exp(r_next - r))
    else:
        if r_next == r:
            reward = 0
        else:
            reward = -expit(r_next - r)
            # reward = -1 / (1 + np.exp(r_next - r))
    return reward

def reward_WF_mean(r, jobs, machines):
    reward = 0
    r_next = estimated_WF_mean(jobs, machines)
    if r_next < r:
        reward = expit(r_next - r)
        # reward = 1 / (1 + np.exp(r_next - r))
    else:
        if r_next == r:
            reward = 0
        else:
            reward = -expit(r_next - r)
            # reward = -1 / (1 + np.exp(r_next - r))
    return reward

def reward_WT_max(r, jobs, machines):
    reward = 0
    r_next = estimated_WT_max(jobs, machines)
    if r_next < r:
        reward = expit(r_next - r)
        # reward = 1 / (1 + np.exp(r_next - r))
    else:
        if r_next == r:
            reward = 0
        else:
            # reward = -1 / (1 + np.exp(r_next - r))
            reward = -expit(r_next - r)
    return reward

def estimated_WT_mean(jobs, machines):
    CKs = []
    ETWTs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + j.span
            ETWT = j.weight * max(0, max(T_cure, C_last) + sum_t_avg - j.DT)
            ETWTs.append(ETWT)
        else:
            ETWT = j.weight * max(0, j.endTime - j.DT)
            ETWTs.append(ETWT)
    return np.mean(ETWTs)


def estimated_WT_max(jobs, machines):
    CKs = []
    ETWTs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + j.span
            ETWT = j.weight * max(0, max(T_cure, C_last) + sum_t_avg - j.DT)
            ETWTs.append(ETWT)
        else:
            ETWT = j.weight * max(0, j.endTime - j.DT)
            ETWTs.append(ETWT)
    return max(ETWTs)

def estimated_WF_mean(jobs, machines):
    CKs = []
    ETWFs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            ETWF = j.weight * (max(T_cure, j.RT + j.span) + sum_t_avg - j.RT)
            ETWFs.append(ETWF)
        else:
            ETWF = j.weight * (j.endTime - j.RT)
            ETWFs.append(ETWF)
    return np.mean(ETWFs)

def machine_utilizatioin_ratio2(machines, now):
    MR = []
    PTs = []
    if now != 0:
        for m in machines:
            pt = 0
            for o in m.assignedOpera:
                pt += o.duration
            if m.currentTime > now:
                pt = pt - m.assignedOpera[-1].endTime - now
            MR.append(pt / now)
        return np.mean(MR)
    else:
        return 0

def machine_utilizatioin_ratio(machines):
    Cmax = []
    PTs = []
    for m in machines:
        Cmax.append(m.currentTime)
        pt = 0
        for o in m.assignedOpera:
            pt += o.duration
        PTs.append(pt)
    makespan = max(Cmax)
    if makespan == 0:
        return 0
    else:
        MR = [i/makespan for i in PTs]
        return np.mean(MR)


def WT_mean_func(jobs):
    WTs = []
    for j in jobs:
        WT = j.weight * max(0, j.endTime - j.DT)
        WTs.append(WT)
    return np.mean(WTs)

def WT_max_func(jobs):
    WTs = []
    for j in jobs:
        WT = j.weight * max(0, j.endTime - j.DT)
        WTs.append(WT)
    return max(WTs)

def WF_mean_func(jobs):
    WFs = []
    for j in jobs:
        WF = j.weight * max(0, j.endTime - j.RT)
        WFs.append(WF)
    return np.mean(WFs)

############################### test 1 ################################

def estimated_span(jobs, machines):
    CKs = []
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)

    E_makespan = 0
    if len(jobs) == 0:
        return E_makespan
    for j in jobs:
        if not j.completed:
            all_t_avg = []
            C_last = 0
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
                else:
                    C_last = o.endTime
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + C_last
            Cmax = max(T_cure, C_last) + sum_t_avg
        else:
            Cmax = j.endTime
        if Cmax > E_makespan:
            E_makespan = Cmax
    return E_makespan

def get_reward(jobs, mahcines, last_span):
    makespan = estimated_span(jobs, mahcines)
    gap = makespan - last_span
    if gap == 0:   # 如果gap=0说明这一次没有任务被分配
        reward = -1
    else:
        reward = 1 / gap
    return np.float64(reward)

def machine_utilizatioin_ratio3(machines, now):
    URs = []
    for m in machines:
        idleTime = 0
        operations = m.assignedOpera
        if len(operations) == 0:
            for i in range(len(operations)-1):
                idleTime += operations[i+1].startTime - operations[i].endTime
            if operations[-1].endTime < now:   # 如果最后一个工序结束时间小于当前时间，说明此时机器空闲，因此应该加上当前机器的等待时间
                idleTime += now - operations[-1].endTime
            UR = (now - idleTime) / now
        else:
            UR = 0
        URs.append(UR)
    return np.mean(URs)
#
def C_last(task_list):

    all_C_last = []
    for i in range(len(task_list)):
        jobs = task_list[i].jobsList
        C_last = []
        for j in jobs:
            last_opera = None
            for opera in j.operation_list:
                if opera.completed:
                    last_opera = opera
                else:
                    break
            if last_opera is not None:
                Cj = last_opera.endTime
            else:
                Cj = 0  # 说明该任务还没有被分配
            C_last.append(Cj)
        all_C_last.append(C_last)
    return all_C_last

def interval_time(task_list, all_C_last):
    ### the best one ###
    now_C_last = C_last(task_list)
    interval_time = []
    for i in range(len(task_list)):
        # intervals = []
        for now, old, job in zip(now_C_last[i], all_C_last[i], task_list[i].jobsList):
            interval = job.weight * (now - old)
            interval_time.append(interval)
    avg_interval_time = -np.mean(interval_time)
    rew = np.exp(avg_interval_time)
    return rew

def interval_time2(task_list, all_C_last):
    now_C_last = C_last(task_list)
    interval_time = []
    for i in range(len(task_list)):
        intervals = []
        for now, old, job in zip(now_C_last[i], all_C_last[i], task_list[i].jobsList):
            interval = job.weight * (now - old)
            intervals.append(interval)
        interval_time.append(intervals)
    tmp1 = np.exp(-np.mean(interval_time[0]))
    tmp2 = np.exp(-np.mean(interval_time[1]))
    tmp3 = np.exp(-np.max(interval_time[2]))
    return [tmp1, tmp2, tmp3]

def slack_time(jobs, machines):
    CKs = []
    all_slackTime = []
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)

    for j in jobs:
        if not j.completed:
            all_t_avg = []   # the processing time of all uncompleted operations
            C_last = 0
            for o in j.operation_list:
                if not o.completed:
                    if o.assigned:
                        t = o.cMachines[o.assignedMachine]
                    else:
                        cMachines = o.cMachines
                        t = np.mean(list(cMachines.values()))
                    all_t_avg.append(t)
                else:
                    C_last = o.endTime    # the endTime of the last completed operation
            sum_t_avg = np.sum(all_t_avg)   # the expected total processing time of all uncompleted operations
            e_span = max(T_cure, C_last) + sum_t_avg
        else:
            e_span = j.endTime
        slackTime = j.weight * (j.DT - e_span)
        all_slackTime.append(slackTime)
    return all_slackTime

def reward_slackTime(pre_slackTime, jobs, machines):
    all_ST_diff = []
    now_ST = slack_time(jobs, machines)
    for pre, new in zip(pre_slackTime, now_ST):
        all_ST_diff.append(new - pre)
    avg_interval_time = -np.mean(all_ST_diff)
    rew = expit(avg_interval_time)
    return rew
############################### test 2 ################################

def estimated_WF(jobs, machines):
    CKs = []
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    WFs = []
    for j in jobs:
        if not j.completed:
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + j.span
            Cmax = max(T_cure, C_last) + sum_t_avg
        else:
            Cmax = j.endTime
        WFj = j.weight * (Cmax - j.RT)
        WFs.append(WFj)
    return WFs

def addPT(jobs, all_C_last):
    now_C_last = C_lastOpera(jobs)
    interval_time = []
    for i in range(len(jobs)):
        for now, old, job in zip(now_C_last, all_C_last, jobs):
            interval = job.weight * (now - old)
            interval_time.append(interval)
    avg_interval_time = -np.mean(interval_time)
    rew = np.exp(avg_interval_time)
    return rew

def C_lastOpera(jobs):

    C_last = []
    for j in jobs:
        last_opera = None
        for opera in j.operation_list:
            if opera.completed:
                last_opera = opera
            else:
                break
        if last_opera is not None:
            Cj = last_opera.endTime
        else:
            Cj = 0  # 说明该任务还没有被分配
        C_last.append(Cj)
    return C_last

def getRewards(pre_C1, pre_C2, eWF, preSpan, task_list, jobs, machines):
    rew1 = addPT(task_list[0].jobsList, pre_C1)
    now_eWF = estimated_WF(task_list[1].jobsList, machines)
    rew3 = addPT(task_list[2].jobsList, pre_C2)

    WF_diff = []
    for pre, now in zip(eWF, now_eWF):
        interval = now - pre
        WF_diff.append(interval)
    avg_interval_time = np.mean(WF_diff)
    rew2 = expit(avg_interval_time)
    rew4 = addPT(jobs, preSpan)

    return (rew1 + rew2 + rew3 +rew4) / 4

def getRewards2(pre_C1, pre_C2, pre_C3, preSpan, task_list, jobs):
    rew1 = addPT(task_list[0].jobsList, pre_C1)
    rew2 = addPT(task_list[1].jobsList, pre_C2)
    rew3 = addPT(task_list[2].jobsList, pre_C3)
    rew4 = addPT(jobs, preSpan)

    ## 归一化处理
    reward_avg = np.mean([rew1, rew2, rew3, rew4])
    return reward_avg

def reward_avg(task_list, machines, pre_r, pre_UR):
    r1 = reward_WT_mean(pre_r[0], task_list[0].jobsList, machines)
    r2 = reward_WF_mean(pre_r[1], task_list[1].jobsList, machines)
    r3 = reward_WT_max(pre_r[2], task_list[2].jobsList, machines)
    r4 = reward_global_U(pre_UR, machines)
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def reward_avg2(task_list, machines, pre_r, pre_UR):
    r1 = reward_WT_mean2(pre_r[0], task_list[0].jobsList, machines)
    r2 = reward_WF_mean2(pre_r[1], task_list[1].jobsList, machines)
    r3 = reward_WT_max2(pre_r[2], task_list[2].jobsList, machines)
    r4 = reward_global_U(pre_UR, machines)
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def reward_avg3(task_list, machines, pre_r):
    r1 = reward_WT_mean2(pre_r[0], task_list[0].jobsList, machines)
    r2 = reward_WF_mean2(pre_r[1], task_list[1].jobsList, machines)
    r3 = reward_WT_max2(pre_r[2], task_list[2].jobsList, machines)
    r4 = r1 + r2 + r3
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def reward_avg4(task_list, machines, pre_r):
    r1 = reward_WT_mean(pre_r[0], task_list[0].jobsList, machines)
    r2 = reward_WF_mean(pre_r[1], task_list[1].jobsList, machines)
    r3 = reward_WT_max(pre_r[2], task_list[2].jobsList, machines)
    r4 = r1 + r2 + r3
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def reward_Nor(rewards):
    epsilon = 1e-100
    min_reward = min(rewards) if rewards else epsilon
    max_reward = max(rewards) if rewards else epsilon
    if max_reward == min_reward:
        max_reward += epsilon
    normalized_rewards = [(reward - min_reward) / (max_reward - min_reward) for reward in rewards]

    return normalized_rewards



############################################ test 3 #######################################################
def estimated_slack(jobs, T_cure):
    STs = []
    EFs = []
    for job in jobs:
        E_makespan = 0
        if job.completed:
            Cmax = job.endTime
        else:
            all_t_avg = []
            last_operaPT = 0
            Cmax = 0
            for o in job.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
                else:
                    last_operaPT = o.endTime
                sum_t_avg = sum(all_t_avg)
                C_last = job.RT + last_operaPT
                Cmax = max(T_cure, C_last) + sum_t_avg
        EFs.append(job.weight * Cmax)
        slackTime = job.weight * (job.DT - Cmax)
        STs.append(slackTime)
    return STs, EFs

def final_reward(tasks_list, machines, pre_wfs):
    CKs = []
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    WST1, _ = estimated_slack(tasks_list[0].jobsList, T_cure)
    WST1_avg = np.mean(WST1)
    if WST1_avg >= 0:
        r1 = 1
    else:
        r1 = -1

    _, wfs = estimated_slack(tasks_list[1].jobsList, T_cure)
    wfs_avg = np.mean(wfs)
    if pre_wfs > wfs_avg:
        r2 = 1
    else:
        r2 = -1

    WST3, _ = estimated_slack(tasks_list[2].jobsList, T_cure)
    WST3_avg = np.max(WST3)
    if WST3_avg >= 0:
        r3 = 1
    else:
        r3 = -1

    r4 = (r1 + r2 + r3)/3
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward


def final_reward2(tasks_list, machines, pre_wfs):
    CKs = []
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    WST1, _ = estimated_slack(tasks_list[0].jobsList, T_cure)
    WST1_avg = np.mean(WST1)
    r1 = np.tanh(WST1_avg)

    _, wfs = estimated_slack(tasks_list[1].jobsList, T_cure)
    wfs_avg = np.mean(wfs)
    r2 = 1/np.exp(wfs_avg)

    WST3, _ = estimated_slack(tasks_list[2].jobsList, T_cure)
    WST3_avg = np.max(WST3)
    r3 = np.tanh(WST3_avg)

    r4 = (r1 + r2 + r3)/3
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def final_reward3(tasks_list, machines, all_C_last):
    CKs = []
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    WST1, _ = estimated_slack(tasks_list[0].jobsList, T_cure)
    WST1_avg = np.mean(WST1)
    r1 = np.tanh(WST1_avg)

    _, wfs = estimated_slack(tasks_list[1].jobsList, T_cure)
    wfs_avg = np.mean(wfs)
    r2 = expit(wfs_avg)

    WST3, _ = estimated_slack(tasks_list[2].jobsList, T_cure)
    WST3_avg = np.max(WST3)
    r3 = np.tanh(WST3_avg)

    r4 = interval_time(tasks_list, all_C_last)
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def final_reward4(tasks_list, all_C_last, machines):
    r1, r2, r3 = interval_time2(tasks_list, all_C_last)
    # TEST 1
    r4 = np.exp(np.mean(UR(machines)) - 0.95)
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def final_reward5(tasks_list, all_C_last, machines, optimal_obj, pre_espan):
    r1, r2, r3 = interval_time2(tasks_list, all_C_last)
    all_reward = r1 + r2 + r3
    r4 = all_reward/3 + interval_time(tasks_list, all_C_last)
    # now_espan = eSpan(tasks_list[1].jobsList, machines)
    # r2 = np.tanh(pre_espan - now_espan)
    # URs = UR(machines)
    # UR_avg = np.mean(URs)
    # all_dUR = 0
    # for ur in URs:
    #     all_dUR += (ur - UR_avg) ** 2
    # r4 = np.sqrt(all_dUR/len(machines))
    # r4 = np.exp(UR(machines) - 0.95)
    # r4 = interval_time(tasks_list, all_C_last)
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward, optimal_obj

def final_reward6(tasks_list, all_C_last, pre_obj, done, machines, weight, weights):
    r1, r2, r3 = interval_time2(tasks_list, all_C_last)
    r1 = weights * r1 + (1 - weights) * reward_WT_mean(pre_obj[0], tasks_list[0].jobsList, machines)
    r2 = weights * r2 + (1 - weights) * reward_WF_mean(pre_obj[1],tasks_list[1].jobsList, machines)
    r3 = weights * r3 + (1 - weights) * reward_WT_max(pre_obj[2],tasks_list[2].jobsList, machines)
    spareR1, spareR2, spareR3 = spare_reward(pre_obj[1], tasks_list, done)
    spareR = spareR1 + spareR2 + spareR3
    # r = r1 + r2 + r3
    r4 = r1 + r2 + r3 + spareR
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def final_reward7(tasks_list, all_C_last, pre_obj, done, machines, weight, weights):
    eWTmean = estimated_WT_mean(tasks_list[0].jobsList, machines)
    if eWTmean <= 0:
        r1 = 1
    else:
        r1 = -1
    r2 = reward_WF_mean(pre_obj[1], tasks_list[1].jobsList, machines)
    eWTmax = estimated_WT_max(tasks_list[2].jobsList, machines)
    if eWTmax <= 0:
        r3 = 1
    else:
        r3 = -1
    all_list = [item for subitem in [tasks_list[0].jobsList, tasks_list[1].jobsList, tasks_list[2].jobsList] for item in subitem]
    eWT = estimated_WT(all_list, machines)
    eWT_min = np.min(eWT)
    if eWT_min <= 0:
        r4 = 1
    else:
        r4 = -1
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def final_reward8(tasks_list, all_C_last, pre_obj, done, machines, weight, weights):
    eWTmean = estimated_WT_mean(tasks_list[0].jobsList, machines)
    if eWTmean <= 0:
        r1 = 1
    else:
        r1 = -1
    r2 = reward_WF_mean(pre_obj[1], tasks_list[1].jobsList, machines)
    eWTmax = estimated_WT_max(tasks_list[2].jobsList, machines)
    if eWTmax <= 0:
        r3 = 1
    else:
        r3 = -1
    r4 = r2
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def final_reward9(tasks_list, pre_obj, machines, pre_idlTime, now):
    eWTmean = estimated_WT_mean(tasks_list[0].jobsList, machines)
    if eWTmean <= 0:
        r1 = 1
    else:
        r1 = -1
    r2 = reward_WF_mean(pre_obj[1], tasks_list[1].jobsList, machines)
    eWTmax = estimated_WT_max(tasks_list[2].jobsList, machines)
    if eWTmax <= 0:
        r3 = 1
    else:
        r3 = -1
    now_idleTime = idleTime(machines, now)
    de_reward = pre_idlTime - now_idleTime
    if de_reward > 0:
        r4 = min(1, de_reward / (pre_idlTime + 1e-10))
    else:
        r4 = max(-1, de_reward / (pre_idlTime + 1e-10))
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward, now_idleTime

def final_reward10(tasks_list, pre_obj, machines, pre_idlTime, pre_C, now):
    eWTmean = estimated_WT_mean(tasks_list[0].jobsList, machines)
    if eWTmean <= 0:
        r1 = 1
    else:
        r1 = -np.tanh(eWTmean)
    r2 = addPT(tasks_list[1].jobsList, pre_C)
    # r2 = reward_WF_mean(pre_obj[1], tasks_list[1].jobsList, machines)
    eWTmax = estimated_WT_max(tasks_list[2].jobsList, machines)
    if eWTmax <= 0:
        r3 = 1
    else:
        r3 = -np.tanh(eWTmax)
    now_idleTime = idleTime(machines, now)
    de_reward = pre_idlTime - now_idleTime
    if de_reward > 0:
        r4 = min(1, de_reward / (pre_idlTime + 1e-10))
    else:
        r4 = max(-1, de_reward / (pre_idlTime + 1e-10))
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward, now_idleTime

def final_reward11(tasks_list, machines):
    all_list = [item for subitem in [tasks_list[0].jobsList, tasks_list[1].jobsList, tasks_list[2].jobsList] for item in subitem]
    eWT = estimated_WT(all_list, machines)
    eWT_min = np.min(eWT)
    if eWT_min == 0:
        r4 = 10
    else:
        r4 = -15
    reward = np.array([r4, r4, r4, r4])
    reward = reward.reshape(1, 4, 1)
    return reward
def idleTime(machines, now):
    CT = 0
    for m in machines:
        if m.currentTime > CT:
            CT = m.currentTime
    idle_times = 0
    for m in machines:
        operList = m.assignedOpera
        if len(operList) == 0:
            continue
        idle_times += operList[0].startTime  # 如果该机器第一个处理的操作的开始时间不是0，则该操作开始之前，该机器为空闲
        if len(operList) == 1:
            if operList[0].endTime < now:
                idle_times += (CT - operList[0].endTime)
        else:
            for i in range(len(operList)-1):
                idle_times += operList[i+1].startTime - operList[i].endTime
    return idle_times / len(machines)

def spare_reward(pre_obj, tasksList, done):
    spareR1 = 0
    spareR2 = 0
    spareR3 = 0
    if done:
        r1 = WT_mean_func(tasksList[0].jobsList)
        r2 = WF_mean_func(tasksList[1].jobsList)
        r3 = WT_max_func(tasksList[2].jobsList)
        if r1 <= 0:
            spareR1 = 2
        else:
            spareR1 = -2
        if r2 < pre_obj:
            spareR2 = 2
        else:
            if r2 == pre_obj:
                spareR2 = 0
            if r2 > pre_obj:
                spareR2 = -2
        if r3 <= 0:
            spareR3 = 2
        else:
            spareR3 = -2
    return spareR1, spareR2, spareR3

def UR(machines):
    CT = 0
    for m in machines:
        if m.currentTime > CT:
            CT = m.currentTime
    if CT == 0:
        return 0
    else:
        URs = []
        for m in machines:
            idle_Time = 0
            opera_list = m.assignedOpera
            for i in range(len(opera_list[:-1])):
                idle_Time += opera_list[i+1].startTime - opera_list[i].endTime
            URs.append((CT - idle_Time) / CT)
        return URs


def eSpan(jobs, machines):
    CKs = []
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    ESP = []
    for job in jobs:
        E_makespan = 0
        if job.completed:
            Cmax = job.endTime
        else:
            all_t_avg = []
            last_operaPT = 0
            Cmax = 0
            for o in job.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
                else:
                    last_operaPT = o.endTime
                sum_t_avg = sum(all_t_avg)
                C_last = job.RT + last_operaPT
                Cmax = max(T_cure, C_last) + sum_t_avg
        ESP.append(job.weight * Cmax)
    return np.mean(ESP)

def estimated_WT(jobs, machines):
    CKs = []
    ETWTs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + j.span
            ETWT = j.weight * max(0, max(T_cure, C_last) + sum_t_avg - j.DT)
            ETWTs.append(ETWT)
        else:
            ETWT = j.weight * max(0, j.endTime - j.DT)
            ETWTs.append(ETWT)
    return ETWTs

def makeSpan(tasks_List):
    makeSpan = 0
    for i in range(len(tasks_List)):
        Cmax = tasks_List[i].jobsList[-1].endTime
        if Cmax > makeSpan:
            makeSpan = Cmax
    return makeSpan

############################################ Good luck ############################################
def test_reward(tasks_List, machines, now, pre_obj, done, opti_makespan, C_last):
    epsilon = 1e-10
    alpha = 0.7
    beta = 0.2
    gamma = 0
    Mur = machineUR(machines, now)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)
    # 归一化
    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_EWTmean_avg = np.mean(normalized_EWTmean)
    normalized_EWFmean_avg = np.mean(normalized_EWFmean)
    normalized_EWTmax_avg = np.mean(normalized_EWTmax)
    if normalized_EWTmean_avg == 0:
        agent1_r = 1
    else:
        agent1_r = -normalized_EWTmean_avg
    agent2_r = - normalized_EWFmean_avg
    if normalized_EWTmax_avg == 0:
        agent3_r = 1
    else:
        agent3_r = -normalized_EWTmax_avg
    # normalized_EWTmean = (EWTmean - np.mean(EWTmean)) / (np.std(EWTmean) + epsilon)
    # normalized_EWFmean = (EWFmean - np.mean(EWFmean)) / (np.std(EWFmean) + epsilon)
    # normalized_EWTmax = (EWFmax - np.mean(EWFmax)) / (np.std(EWFmax) + epsilon)
    winR = win_reward(tasks_List, pre_obj, done, opti_makespan)
    globalR = global_reward(tasks_List, machines, C_last, now)
    r1 = (
            + alpha * agent1_r
            + beta * globalR
            + gamma * winR
    )
    r2 = (
            + alpha * agent2_r
            + beta * globalR
            + gamma * winR
    )
    r3 = (
            + alpha * agent3_r
            + beta * globalR
            + gamma * winR
    )
    r4 = globalR + gamma * winR
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def global_reward(tasks_List, machines, all_C_last, now):
    epsilon = 1e-10
    PT = increasePT(tasks_List, all_C_last)
    PR_avg = np.max(PT)
    try:
        if PR_avg == 0:  # 如果PT=0，说明在该决策点处没有任何一个新的操作被执行，带来了资源浪费，所以记忆惩罚
            globalReward = -1
        else:
            globalReward = 1 / PR_avg
    except:
        globalReward = -1
    # jobs = [job for taskList in tasks_List for job in taskList.jobsList]
    # EWTmean = estimated_WT_mean(tasks_List[0].jobsList, machines)
    # EWFmean = estimated_WF_mean(tasks_List[1].jobsList, machines)
    # EWFmax = estimated_WT_max(tasks_List[2].jobsList, machines)

    # MURs = machineUR(machines, now)
    #
    # # 归一化
    # normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    # normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    # normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)
    #
    # # normalized_EWTmean = (EWTmean - np.mean(EWTmean)) / (np.std(EWTmean) + epsilon)
    # # normalized_EWFmean = (EWFmean - np.mean(EWFmean)) / (np.std(EWFmean) + epsilon)
    # # normalized_EWFmax = (EWFmax - np.mean(EWFmax)) / (np.std(EWFmax) + epsilon)
    #
    # globalReward = (
    #         - normalized_EWTmean  # 惩罚较高的平均加权迟到时间
    #         - normalized_EWFmean  # 惩罚较高的平均加权流动时间
    #         - normalized_EWTmax  # 惩罚较高的最大加权迟到时间
    #         # + 0.1 * np.mean(MURs)  # 奖励较高的机器利用率
    # )
    return globalReward

def increasePT(task_list, all_C_last):
    PTs = []
    task_Cmax = np.array((len(all_C_last),2))
    now_C_last = C_last(task_list)
    for i in range(len(now_C_last)):
        newPT = np.max(now_C_last[i]) - np.max(all_C_last[i])
        PTs.append(newPT)
    return PTs


def inter_time(task_list, all_C_last):
    ### the best one ###
    now_C_last = C_last(task_list)
    interval_time = []
    for i in range(len(task_list)):
        # intervals = []
        for now, old, job in zip(now_C_last[i], all_C_last[i], task_list[i].jobsList):
            interval = job.weight * (now - old)
            interval_time.append(interval)
    avg_interval_time = np.mean(interval_time)
    return avg_interval_time

def machineUR(machines, now):
    URs = [0 for i in range(len(machines))]
    if now == 0:
        return URs
    for m in machines:
        if m.currentTime <= now:
            ur = m.busyTime / now
        else:
            ur = (m.busyTime - (now - m.currentTime)) / now
        URs.append(ur)
    return URs


def estimated_WTmean(jobs, machines):
    CKs = []
    ETWTs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + j.span
            ETWT = j.weight * max(0, max(T_cure, C_last) + sum_t_avg - j.DT)
            ETWTs.append(ETWT)
        else:
            ETWT = j.weight * max(0, j.endTime - j.DT)
            ETWTs.append(ETWT)
    return np.mean(ETWTs)

def estimated_WTmax(jobs, machines):
    CKs = []
    ETWTs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            C_last = j.RT + j.span
            ETWT = j.weight * max(0, max(T_cure, C_last) + sum_t_avg - j.DT)
            ETWTs.append(ETWT)
        else:
            ETWT = j.weight * max(0, j.endTime - j.DT)
            ETWTs.append(ETWT)
    return np.max(ETWTs)

def estimated_WFmean(jobs, machines):
    CKs = []
    ETWFs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            for o in j.operation_list:
                if not o.completed:
                    cMachines = o.cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            ETWF = j.weight * (max(T_cure, j.RT + j.span) + sum_t_avg - j.RT)
            ETWFs.append(ETWF)
        else:
            ETWF = j.weight * (j.endTime - j.RT)
            ETWFs.append(ETWF)
    return np.mean(ETWFs)

def win_reward(tasks_List, pre_obj, done, opti_makespan):
    reward = 0
    if done:
        WTmean = WT_mean_func(tasks_List[0].jobsList)
        if WTmean == 0:
            reward += 0.5
        else:
            reward -= 0.5
        r2 = r_WF_mean(pre_obj, tasks_List[1].jobsList)
        reward += r2
        WTmean = WT_mean_func(tasks_List[0].jobsList)
        if WTmean == 0:
            reward += 0.5
        else:
            reward -= 0.5
        # makespan = makeSpan(tasks_List)
        # if makespan < opti_makespan:
        #     reward += 0.5
        # else:
        #     if makespan > opti_makespan:
        #         reward -= 0.5
    return reward

def r_WF_mean(r, jobs):
    reward = 0
    r_next = WF_mean_func(jobs)
    if r_next < r:
        reward = 1 / (1 + np.exp(r_next - r)) #5
    else:
        if r_next == r:
            reward = 0
        else:
            reward = -1 / (1 + np.exp(r_next - r))   #-5
    # r_next = WF_mean_func(jobs)
    # if r_next < r:
    #     reward = 1 / (1 + np.exp(r_next - r))
    # else:
    #     if r_next == r:
    #         reward = 0
    #     else:
    #         reward = -1 / (1 + np.exp(r_next - r))
    return reward

######################################### Hope ########################################
def hope_reward(tasks_List, machines, now, pre_obj, done, opti_makespan, C_last):
    epsilon = 1e-10
    alpha = 0.3
    beta = 0.7
    gamma = 1
    globalR = interval_time(tasks_List, C_last)
    Mur = machineUR(machines, now)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)
    # 归一化
    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_EWTmean_avg = np.mean(normalized_EWTmean)
    normalized_EWFmean_avg = np.mean(normalized_EWFmean)
    normalized_EWTmax_avg = np.mean(normalized_EWTmax)
    if normalized_EWTmean_avg == 0:
        agent1_r = 1
    else:
        agent1_r = -normalized_EWTmean_avg
    agent2_r = - normalized_EWFmean_avg
    if normalized_EWTmax_avg == 0:
        agent3_r = 1
    else:
        agent3_r = -normalized_EWTmax_avg
    winR = win_reward(tasks_List, pre_obj, done, opti_makespan)
    # globalR = global_reward(tasks_List, machines, C_last, now)
    r1 = (
            + alpha * agent1_r
            + beta * globalR
            # + gamma * winR
    )
    r2 = (
            + alpha * agent2_r
            + beta * globalR
            # + gamma * winR
    )
    r3 = (
            + alpha * agent3_r
            + beta * globalR
            # + gamma * winR
    )
    r4 = globalR
    reward = np.array([r1, r2, r3, r4])
    reward = reward.reshape(1, 4, 1)
    return reward

def last_reward(tasks_List, machines, now, pre_obj, done, opti_makespan, C_last):
    epsilon = 1e-10
    alpha = 0.3
    beta = 0.7
    gamma = 1
    globalR = interval_time(tasks_List, C_last)
    Mur = machineUR(machines, now)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)
    # 归一化
    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    r = np.mean(normalized_EWTmean) + np.mean(normalized_EWFmean) + np.max(normalized_EWTmax)

    # reward = np.array([r, r, r, r])
    # reward = reward.reshape(1, 4, 1)
    return r

def last_reward(tasks_List, machines, now, pre_obj, done, opti_makespan, C_last):
    epsilon = 1e-10
    alpha = 0.3
    beta = 0.7
    gamma = 1
    globalR = interval_time(tasks_List, C_last)
    Mur = machineUR(machines, now)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)
    # 归一化
    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_EWTmean_avg = np.mean(normalized_EWTmean)
    normalized_EWFmean_avg = np.mean(normalized_EWFmean)
    normalized_EWTmax_avg = np.max(normalized_EWTmax)
    if normalized_EWTmean_avg == 0:
        agent1_r = 1
    else:
        agent1_r = -(1 + normalized_EWTmean_avg)
    agent2_r = reward_WF_mean2(pre_obj, tasks_List[1].jobsList, machines)
    # agent2_r = - (1 + normalized_EWFmean_avg)
    if normalized_EWTmax_avg == 0:
        agent3_r = 1
    else:
        agent3_r = - (1 + normalized_EWTmax_avg)
    # r = -np.mean(normalized_EWTmean) - np.mean(normalized_EWFmean) - np.max(normalized_EWTmax)
    r = agent1_r + agent2_r + agent3_r + globalR
    return r

def last_reward2(tasks_List, machines, now, pre_obj, done, opti_makespan, C_last):
    epsilon = 1e-10
    alpha = 0.3
    beta = 0.7
    gamma = 1
    globalR = interval_time(tasks_List, C_last)
    Mur = machineUR(machines, now)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    # 归一化
    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_EWTmean_avg = np.mean(normalized_EWTmean)
    normalized_EWFmean_avg = np.mean(normalized_EWFmean)
    normalized_EWTmax_avg = np.max(normalized_EWTmax)
    if normalized_EWTmean_avg == 0:
        agent1_r = 1
    else:
        agent1_r = -(1 + normalized_EWTmean_avg)
    agent2_r = reward_WF_mean2(pre_obj, tasks_List[1].jobsList, machines)
    # agent2_r = - (1 + normalized_EWFmean_avg)
    if normalized_EWTmax_avg == 0:
        agent3_r = 1
    else:
        agent3_r = - (1 + normalized_EWTmax_avg)
    # r = -np.mean(normalized_EWTmean) - np.mean(normalized_EWFmean) - np.max(normalized_EWTmax)
    if done:
        endReward = end_reward(EWTmean, EWFmean, EWFmax, pre_obj)
        r = agent1_r + agent2_r + agent3_r + globalR + endReward
    else:
        r = agent1_r + agent2_r + agent3_r + globalR
    return r

def end_reward(EWTmean, EWFmean, EWTmax, opti_makespan):
    WTmean = np.mean(EWTmean)
    if WTmean == 0:
        r1 = 10
    else:
        r1 = -10
    WFmean = np.mean(EWFmean)
    if WFmean < opti_makespan:
        r2 = 10
    else:
        if WFmean == opti_makespan:
            r2 = 0
        else:
            r2 = -10
    WFmax = np.max(EWTmax)
    if WFmax == 0:
        r3 = 10
    else:
        r3 = -10
    return r1+r2+r3

def total_reward(tasks_List, machines, now, pre_obj, done, opti_makespan, last_PT):

    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    r1 = interval_time(tasks_List, last_PT)
    if done:
        r2 = end_reward(EWTmean, EWFmean, EWFmax, pre_obj)
        r = r1 + r2
    else:
        r = r1
    return r

def cal_Reward(jobs, tasks_List, machines, pre_obj, done, pre_C_last):
    epsilon = 1e-10
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_preEWTmean = (pre_obj[0] - np.min(pre_obj[0])) / (np.max(pre_obj[0]) - np.min(pre_obj[0]) + epsilon)
    normalized_preEWFmean = (pre_obj[1] - np.min(pre_obj[1])) / (np.max(pre_obj[1]) - np.min(pre_obj[1]) + epsilon)
    normalized_preEWTmax = (pre_obj[2] - np.min(pre_obj[2])) / (np.max(pre_obj[2]) - np.min(pre_obj[2]) + epsilon)

    # peneltyR = penelty([np.mean(normalized_preEWTmean), np.mean(normalized_preEWFmean), np.max(normalized_preEWTmax)],
    #                    [np.mean(normalized_EWTmean), np.mean(normalized_EWFmean), np.max(normalized_EWTmax)])
    peneltyR = penelty([np.mean(pre_obj[0]), np.mean(pre_obj[1]), np.max(pre_obj[2])],
                       [np.mean(EWTmean), np.mean(EWFmean), np.max(EWFmax)])
    reward = Sit_reward(jobs, pre_C_last)
    r = peneltyR + reward
    return r

def penelty(pre_obj, now_obj):
    # 延迟惩罚项
    k = 2 # 惩罚系数
    r = 0
    for i in range(3):
        # 下述方式可以收敛
        r += np.tanh(max(now_obj[i] - pre_obj[i], 0))
    return -r

def penelty2(pre_obj, now_obj):
    # 延迟惩罚项
    k = 0.5  # 惩罚系数
    r = 0
    epsilon = 1e-5  # 防止除以0
    diffs = [max(now_obj[i] - pre_obj[i], 0) for i in range(3)]
    max_diff = max(diffs)
    min_diff = min(diffs)
    for diff in diffs:
        # 标准化差值
        normalized_diff = (diff - min_diff) / (max_diff - min_diff + epsilon)
        # 累加惩罚值
        r += (normalized_diff) ** k
    return -r

def step_penelty(step_taken):
    step_penalty_coeff = -0.01  # 步骤惩罚系数
    return step_penalty_coeff * step_taken

def Sit_reward(jobs, pre_C_last):
    ### the best one ###
    now_C_last = last_PT(jobs)
    interval = now_C_last - pre_C_last
    reward = 1 / np.exp(interval)
    return reward

def last_PT(jobs):
    C_last = []
    for j in jobs:
        last_opera = None
        for opera in j.operation_list:
            if opera.completed:
                last_opera = opera
            else:
                break
        if last_opera is not None:
            Cj = last_opera.endTime
        else:
            Cj = 0  # 说明该任务还没有被分配
        C_last.append(Cj)
    return max(C_last)

def cal_Reward2(tasks_List, pre_obj, done, last_PT):
    epsilon = 1e-10
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    if done:
        EWTmean = WT_mean_func(tasks_List[0].jobsList)
        EWFmean = WF_mean_func(tasks_List[1].jobsList)
        EWFmax = WT_max_func(tasks_List[2].jobsList)

        # normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
        # normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
        # normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)
        #
        # normalized_preEWTmean = (pre_obj[0] - np.min(pre_obj[0])) / (np.max(pre_obj[0]) - np.min(pre_obj[0]) + epsilon)
        # normalized_preEWFmean = (pre_obj[1] - np.min(pre_obj[1])) / (np.max(pre_obj[1]) - np.min(pre_obj[1]) + epsilon)
        # normalized_preEWTmax = (pre_obj[2] - np.min(pre_obj[2])) / (np.max(pre_obj[2]) - np.min(pre_obj[2]) + epsilon)

        # peneltyR = penelty([np.mean(normalized_preEWTmean), np.mean(normalized_preEWFmean), np.mean(normalized_preEWTmax)],
        #                    [np.mean(normalized_EWTmean), np.mean(normalized_EWFmean), np.mean(normalized_EWTmax)])

        peneltyR = penelty([np.mean(pre_obj[0]), np.mean(pre_obj[1]), np.max(pre_obj[2])], [EWTmean, EWFmean, EWFmax])
        r = peneltyR + reward
    else:
        r = reward
    return r

def cal_Reward3(jobs, tasks_List, machines, pre_obj, done, pre_C_last, last_PT, step_taken):
    epsilon = 1e-10
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    if done:
        EWTmean = WT_mean_func(tasks_List[0].jobsList)
        EWFmean = WF_mean_func(tasks_List[1].jobsList)
        EWFmax = WT_max_func(tasks_List[2].jobsList)
        peneltyR = penelty2([np.mean(pre_obj[0]), np.mean(pre_obj[1]), np.max(pre_obj[2])], [EWTmean, EWFmean, EWFmax])
        r = peneltyR + reward
    else:
        r = reward
    return r

def cal_Reward4(tasks_List, machines, Epre_obj, last_PT):
    epsilon = 1e-10
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    # normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    # normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    # normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)
    #
    # normalized_preEWTmean = (Epre_obj[0] - np.min(Epre_obj[0])) / (np.max(Epre_obj[0]) - np.min(Epre_obj[0]) + epsilon)
    # normalized_preEWFmean = (Epre_obj[1] - np.min(Epre_obj[1])) / (np.max(Epre_obj[1]) - np.min(Epre_obj[1]) + epsilon)
    # normalized_preEWTmax = (Epre_obj[2] - np.min(Epre_obj[2])) / (np.max(Epre_obj[2]) - np.min(Epre_obj[2]) + epsilon)

    peneltyR = penelty([np.mean(Epre_obj[0]), np.mean(Epre_obj[1]), np.max(Epre_obj[2])],[np.mean(EWTmean), np.mean(EWFmean), np.max(EWFmax)])
    r = peneltyR + reward
    return r

def cal_Reward5(tasks_List, machines, Epre_obj, last_PT):
    epsilon = 1e-10
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    # normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    # normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    # normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)
    #
    # normalized_preEWTmean = (Epre_obj[0] - np.min(Epre_obj[0])) / (np.max(Epre_obj[0]) - np.min(Epre_obj[0]) + epsilon)
    # normalized_preEWFmean = (Epre_obj[1] - np.min(Epre_obj[1])) / (np.max(Epre_obj[1]) - np.min(Epre_obj[1]) + epsilon)
    # normalized_preEWTmax = (Epre_obj[2] - np.min(Epre_obj[2])) / (np.max(Epre_obj[2]) - np.min(Epre_obj[2]) + epsilon)

    peneltyR = penelty([np.mean(Epre_obj[0]), np.mean(Epre_obj[1]), np.max(Epre_obj[2])],[np.mean(EWTmean), np.mean(EWFmean), np.max(EWFmax)])
    r = peneltyR
    return r

def cal_Reward6(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    epsilon = 1e-10
    penalty = 0
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_preEWTmean = (Epre_obj[0] - np.min(Epre_obj[0])) / (np.max(Epre_obj[0]) - np.min(Epre_obj[0]) + epsilon)
    normalized_preEWFmean = (Epre_obj[1] - np.min(Epre_obj[1])) / (np.max(Epre_obj[1]) - np.min(Epre_obj[1]) + epsilon)
    normalized_preEWTmax = (Epre_obj[2] - np.min(Epre_obj[2])) / (np.max(Epre_obj[2]) - np.min(Epre_obj[2]) + epsilon)

    peneltyR = penelty([np.mean(Epre_obj[0]), np.mean(Epre_obj[1]), np.max(Epre_obj[2])],[np.mean(EWTmean), np.mean(EWFmean), np.max(EWFmax)])

    if done:
        penalty = penalty_fun(tasks_List, pre_WFmean)

    r = peneltyR + reward + penalty
    return r

def cal_Reward7(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    epsilon = 1e-10
    penalty = 0
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_preEWTmean = (Epre_obj[0] - np.min(Epre_obj[0])) / (np.max(Epre_obj[0]) - np.min(Epre_obj[0]) + epsilon)
    normalized_preEWFmean = (Epre_obj[1] - np.min(Epre_obj[1])) / (np.max(Epre_obj[1]) - np.min(Epre_obj[1]) + epsilon)
    normalized_preEWTmax = (Epre_obj[2] - np.min(Epre_obj[2])) / (np.max(Epre_obj[2]) - np.min(Epre_obj[2]) + epsilon)

    peneltyR = penelty([np.mean(normalized_preEWTmean), np.mean(normalized_preEWFmean), np.max(normalized_preEWTmax)],[np.mean(normalized_EWTmean), np.mean(normalized_EWFmean), np.max(normalized_EWTmax)])

    if done:
        penalty = penalty_fun(tasks_List, pre_WFmean)

    r = peneltyR + reward + penalty
    return r

def penalty_fun(tasks_List, pre_WFmean):
    r = 0
    WTmean = WT_mean_func(tasks_List[0].jobsList)
    WFmean = WF_mean_func(tasks_List[1].jobsList)
    WTmax = WT_max_func(tasks_List[2].jobsList)

    if WTmean < 0:
        r += 1
    else:
        r -= 1
    if WTmax < 0:
        r += 1
    else:
        r -= 1

    if pre_WFmean > WFmean:
        r += 1
    else:
        if pre_WFmean < WFmean:
            r -= 1
    return r

def totalReward(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    penalty = 0
    reward = interval_time(tasks_List, last_PT)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)
    r1, r2, r3, r4 = 0, 0, 0, 0
    if EWTmean == 0:
        r1 += 1
    else:
        r1 -= 1   # EWTmean一定大于等于0

    if np.mean(EWFmean) < pre_WFmean:
        r2 += 1
    else:
        if np.mean(EWFmean) == pre_WFmean:
            r2 += 0
        else:
            r2 -= 1

    if np.max(EWFmax) == 0:
        r3 += 1
    else:
        r3 -= 1   # EWFmax一定大于等于0
    r4 = (r1 + r2 + r3)/3
    if done:
        penalty = penalty_fun(tasks_List, pre_WFmean)

    r1 = r1 + penalty + reward
    r2 = r2 + penalty + reward
    r3 = r3 + penalty + reward
    r4 = r4 + penalty + reward
    return [r1, r2, r3, r4]

################## use for ablation ##########################
def Ablation_Reward1(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    epsilon = 1e-10
    penalty = 0
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_preEWTmean = (Epre_obj[0] - np.min(Epre_obj[0])) / (np.max(Epre_obj[0]) - np.min(Epre_obj[0]) + epsilon)
    normalized_preEWFmean = (Epre_obj[1] - np.min(Epre_obj[1])) / (np.max(Epre_obj[1]) - np.min(Epre_obj[1]) + epsilon)
    normalized_preEWTmax = (Epre_obj[2] - np.min(Epre_obj[2])) / (np.max(Epre_obj[2]) - np.min(Epre_obj[2]) + epsilon)

    peneltyR = penelty([np.mean(normalized_preEWTmean), np.mean(normalized_preEWFmean), np.max(normalized_preEWTmax)],
                       [np.mean(normalized_EWTmean), np.mean(normalized_EWFmean), np.max(normalized_EWTmax)])

    r = peneltyR
    return r

def Ablation_Reward2(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    epsilon = 1e-10
    penalty = 0
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    r = reward
    return r

def Ablation_Reward3(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    epsilon = 1e-10
    penalty = 0
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_preEWTmean = (Epre_obj[0] - np.min(Epre_obj[0])) / (np.max(Epre_obj[0]) - np.min(Epre_obj[0]) + epsilon)
    normalized_preEWFmean = (Epre_obj[1] - np.min(Epre_obj[1])) / (np.max(Epre_obj[1]) - np.min(Epre_obj[1]) + epsilon)
    normalized_preEWTmax = (Epre_obj[2] - np.min(Epre_obj[2])) / (np.max(Epre_obj[2]) - np.min(Epre_obj[2]) + epsilon)

    peneltyR = penelty([np.mean(normalized_preEWTmean), np.mean(normalized_preEWFmean), np.max(normalized_preEWTmax)],[np.mean(normalized_EWTmean), np.mean(normalized_EWFmean), np.max(normalized_EWTmax)])

    if done:
        penalty = penalty_fun(tasks_List, pre_WFmean)

    r = peneltyR + penalty
    return r

def Ablation_Reward4(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    epsilon = 1e-10
    penalty = 0
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    if done:
        penalty = penalty_fun(tasks_List, pre_WFmean)

    r = reward + penalty
    return r

def Ablation_Reward5(tasks_List, machines, Epre_obj, last_PT, done, pre_WFmean):
    epsilon = 1e-10
    penalty = 0
    # reward = Sit_reward(jobs, pre_C_last)    # 可以收敛
    reward = interval_time(tasks_List, last_PT)
    EWTmean = estimated_WTmean(tasks_List[0].jobsList, machines)
    EWFmean = estimated_WFmean(tasks_List[1].jobsList, machines)
    EWFmax = estimated_WTmax(tasks_List[2].jobsList, machines)

    normalized_EWTmean = (EWTmean - np.min(EWTmean)) / (np.max(EWTmean) - np.min(EWTmean) + epsilon)
    normalized_EWFmean = (EWFmean - np.min(EWFmean)) / (np.max(EWFmean) - np.min(EWFmean) + epsilon)
    normalized_EWTmax = (EWFmax - np.min(EWFmax)) / (np.max(EWFmax) - np.min(EWFmax) + epsilon)

    normalized_preEWTmean = (Epre_obj[0] - np.min(Epre_obj[0])) / (np.max(Epre_obj[0]) - np.min(Epre_obj[0]) + epsilon)
    normalized_preEWFmean = (Epre_obj[1] - np.min(Epre_obj[1])) / (np.max(Epre_obj[1]) - np.min(Epre_obj[1]) + epsilon)
    normalized_preEWTmax = (Epre_obj[2] - np.min(Epre_obj[2])) / (np.max(Epre_obj[2]) - np.min(Epre_obj[2]) + epsilon)

    peneltyR = penelty([np.mean(normalized_preEWTmean), np.mean(normalized_preEWFmean), np.max(normalized_preEWTmax)],[np.mean(normalized_EWTmean), np.mean(normalized_EWFmean), np.max(normalized_EWTmax)])

    r = reward + peneltyR
    return r

def get_PAR(pre_objectives, taskList):
    current_obj = [0, 0, 0]
    current_obj[0] = WT_mean_func(taskList[0].jobsList)
    current_obj[1] = WF_mean_func(taskList[1].jobsList)
    current_obj[2] = WT_max_func(taskList[2].jobsList)
    R_p = 0
    for pre, cur in zip(pre_objectives, current_obj):
        D = max(0, cur - pre)
        R_p += np.tanh(D)
        # R_p += (np.exp(D) - np.exp(-D))/(np.exp(D) + np.exp(-D))
    return R_p

def get_IIR(all_C_last, taskList):
    ### the best one ###
    now_C_last = C_last(taskList)
    interval_time = []
    for i in range(len(taskList)):
        # intervals = []
        for now, old, job in zip(now_C_last[i], all_C_last[i], taskList[i].jobsList):
            interval = job.weight * (now - old)
            interval_time.append(interval)
    avg_interval_time = -np.mean(interval_time)
    IIR = np.exp(avg_interval_time)
    return IIR

def get_Re(taskList, pre_WF):
    WTmean = WT_mean_func(taskList[0].jobsList)
    WFmean = WF_mean_func(taskList[1].jobsList)
    WFmax = WT_max_func(taskList[2].jobsList)
    R_e = 0
    if WTmean == 0:
        R_e += 1
    else:
        R_e -= 1
    if WFmean < pre_WF:
        R_e += 1
    else:
        if WFmean == pre_WF:
            R_e += 0
        else:
            R_e -= 1
    if WFmax == 0:
        R_e += 1
    else:
        R_e -= 1
    return R_e

def get_reward_func(tasks_List, pre_obj, all_C_last, done):
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = PAR + IIR + Re
    else:
        reward = PAR + IIR
    return reward

def get_reward_func1(tasks_List, pre_obj, all_C_last, done):
    ## PAR + IIR
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    reward = PAR + IIR
    return reward

def get_reward_func2(tasks_List, pre_obj, all_C_last, done):
    ## PAR + Re
    PAR = get_PAR(pre_obj, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:

        reward = PAR + Re
    else:
        reward = PAR
    return reward

def get_reward_func3(tasks_List, pre_obj, all_C_last, done):
    ## IIR + Re
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = IIR + Re
    else:
        reward = IIR
    return reward

def get_reward_func4(tasks_List, pre_obj, all_C_last, done):
    ## PAR
    PAR = get_PAR(pre_obj, tasks_List)
    return PAR

def get_reward_func5(tasks_List, pre_obj, all_C_last, done):
    ## Re
    IIR = get_IIR(all_C_last, tasks_List)
    return IIR


################ 给不同的奖励函数设置不同的权重 ##################
def get_reward_func6(tasks_List, pre_obj, all_C_last, done):
    # PAR占比重
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = 2*PAR + IIR + Re
    else:
        reward = 2*PAR + IIR
    return reward

def get_reward_func7(tasks_List, pre_obj, all_C_last, done):
    # IIR占比重
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = PAR + 2*IIR + Re
    else:
        reward = PAR + 2*IIR
    return reward

def get_reward_func8(tasks_List, pre_obj, all_C_last, done):
    # Re占比重
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = PAR + IIR + 2*Re
    else:
        reward = PAR + IIR
    return reward

def get_reward_func9(tasks_List, pre_obj, all_C_last, done):
    # 2 * PAR + 2*IIR 
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = 2 * PAR + 2*IIR + Re
    else:
        reward = 2*PAR + 2*IIR
    return reward

def get_reward_func10(tasks_List, pre_obj, all_C_last, done):
    # 2 * PAR + 2*Re
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = 2 * PAR + IIR + 2*Re
    else:
        reward = 2*PAR + IIR
    return reward

def get_reward_func11(tasks_List, pre_obj, all_C_last, done):
    # 2 * IIR + 2*Re
    PAR = get_PAR(pre_obj, tasks_List)
    IIR = get_IIR(all_C_last, tasks_List)
    Re = get_Re(tasks_List, pre_obj[1])
    if done:
        reward = PAR + 2 * IIR + 2*Re
    else:
        reward = PAR + 2 * IIR
    return reward