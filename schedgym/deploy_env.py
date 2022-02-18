'''
TODO List:
    cluster.describe()      done
    reward                  done
    termination             done
    get_attr                done
    reward计算向量化
    Node的pods列表可以只存index
    step 返回的state要交给q_learner，应一致！本打算返回下个五分钟的所有状态，目前暂时返回self.t-1时刻的状态，即action做完之后立即状态
'''

import gym
import numpy as np
import pandas as pd
import copy

TOTAL_CPU = 100
TOTAL_MEM = 100
INITIAL_PODS = 5
DELETE = 1
NODE_OVER_UTILIZED_THRESHOLD = 0.8
NODE_UNDER_UTILIZED_THRESHOLD = 0.2
NODE_OVER_UTILIZED_PENALTY = 0.6
NODE_UNDER_UTILIZED_PENALTY = 0.4
POD_UNDER_REQUESTED_PENALTY = 1
MIGRATION_COST_PENALTY = 0.01

df = pd.read_csv('../vmagent/data/Huawei-East-1.csv')
df.head()

def getNextData(cur_step):
    '''
        - getNextData返回下一个5min内的请求
        - 下一个5min[cur_step*300,(cur_step+1)*300)
        - 忽略所有释放资源的请求
    '''
    start_time = cur_step*300
    end_time = (cur_step+1)*300
    next_df = df[(df['time']>=start_time) & (df['time']<end_time) & (df['type']!=1)]
    # print(f'[getNextData.start_time={start_time}.end_time={end_time}] len(next_df)={len(next_df)}, next_df :')
    next_df.head()
    request = []
    def converToArray(row):
        request.append([row['time'],row['cpu'],row['memory']])
        return
    next_df.apply(converToArray,axis=1)
    # print(f'[getNextData.start_time={start_time}.end_time={end_time}] len(request)={len(request)}, request={request}')
    return request

def hasNextData(cur_step):
    start_time = cur_step*300
    lastRequest = df.iloc[-1,:]
    # print(f'[hasNextData.start_time={start_time}] lastRequest={lastRequest}')
    if lastRequest['time']<start_time:
        return False
    return True

class Pod():
    def __init__(self, index, pod, node):
        self.index = index
        self.request_cpu = pod["request_cpu"]
        self.request_mem = pod["request_mem"]
        self.assigned_cpu = pod["assigned_cpu"]
        self.assigned_mem = pod["assigned_mem"]
        self.current_node = node

class Node():
    def __init__(self, node_index, node_value):
        self.index = node_index
        self.total_cpu = node_value["total_cpu"]
        self.total_mem = node_value["total_mem"]
        self.remain_cpu = node_value["remain_cpu"]
        self.remain_mem = node_value["remain_mem"]
        self.pods = []

    '''
        只将pod分配到该node上
        只有在self.update函数更新后node的状态才有效
    '''
    def append_pod(self,pod):
        self.pods.append(pod)
        pod.current_node = self.index

    '''
        根据node上搭载的pod更新node的资源使用情况
        pod更新规则：
            - 按照pod.request_cpu/node.total_cpu + pod.request_mem/node.total_mem升序的顺序排列，按顺序分配资源
    '''
    def update(self):
        def compare(a):
            return a.request_cpu/self.total_cpu+a.request_mem/self.total_mem
        self.pods.sort(key=compare)
        for pod in self.pods:
            if(self.remain_cpu>=pod.request_cpu):
                self.remain_cpu -= pod.request_cpu
                pod.assigned_cpu = pod.request_cpu
            else:
                pod.assigned_cpu = self.remain_cpu
                self.remain_cpu = 0
            if(self.remain_mem>=pod.request_mem):
                self.remain_mem -= pod.request_mem
                pod.assigned_mem = pod.request_mem
            else:
                pod.assigned_mem = self.remain_mem
                self.remain_mem = 0

    '''
        node当前状态返回列表： shape = (51,4)
        [[node.total_cpu,node.total_mem,node.remain_cpu,node.remain_mem],
         [pod1.request_cpu,pod1.request_mem,pod1.remain_cpu,pod1.remain_mem],
         [...],
         [...]]
    '''
    def describe(self):
        ret = [[self.total_cpu,self.total_mem,self.remain_cpu,self.remain_mem]]
        for pod in self.pods:
            ret.append([pod.request_cpu,pod.request_mem,pod.assigned_cpu,pod.assigned_mem])
        assert len(self.pods)<=50,len(self.pods)
        for _ in range(50-len(self.pods)):
            ret.append([0,0,0,0])
        return ret

class ClusterState():
    def __init__(self,nodes,pods):
        self.node_num = len(nodes)
        self.pod_num = len(pods)
        self.nodes = [Node(i,node) for i,node in enumerate(nodes)]
        self.pods = [Pod(i,pod,0) for i,pod in enumerate(pods)]
        node_index = 0
        for i,pod in enumerate(self.pods):
            self.nodes[node_index].append_pod(pod)
            if (i+1)%5==0:
                self.nodes[node_index].update()
                node_index = (node_index+1)%len(self.nodes)
    
    def handle_request(self,handle_pod,cpu,mem):
        self.pods[handle_pod].request_cpu += cpu
        self.pods[handle_pod].request_mem += mem
        self.nodes[self.pods[handle_pod].current_node].update()

    def handle_migration(self, action):
        assert action.shape==(50,), f'{action.shape}'
        for pod_index, pod_action in enumerate(action):
            assert pod_action>=0 and pod_action<=9,f'{pod_action}' 
            if self.pods[pod_index].current_node==pod_action:
                continue
            target_pod = self.pods[pod_index]
            from_node = target_pod.current_node
            self.nodes[from_node].pods.remove(target_pod)
            self.nodes[pod_action].append_pod(target_pod)

            self.nodes[from_node].update()
            self.nodes[pod_action].update()

    def migrationCost(self,action):
        assert action.shape==(50,), f'{action.shape}'
        migration_cost = 0
        for pod_index, pod_action in enumerate(action):
            assert pod_action>=0 and pod_action<=9,f'{pod_action}' 
            if self.pods[pod_index].current_node==pod_action:
                continue
            migration_cost += self.pods[pod_index].request_cpu+self.pods[pod_index].request_mem
        return migration_cost

    '''
        返回列表shape = (nodes数,51,4)
    '''
    def describe(self):
        ret = []
        for node in self.nodes:
            ret.append(node.describe())
        return ret

class Cluster():
    '''
        集群初始化，两个参数：nodes和pods
        - nodes(List) 每个元素为一个字典，记录nodes的total_cpu, total_mem, remain_cpu, remain_mem
        - pods(List) 每个元素为一个字典，记录pods的request_cpu, request_mem
    '''
    def __init__(self, time, nodes, pods):
        self.clusterState={time:ClusterState(nodes,pods)}
        self.latestTime = time
        self.n_pods = len(pods)
        self.handle_pod = 0
        self.latestMigrationCost = 0

    def reset(self, time, nodes, pods):
        self.clusterState={time:ClusterState(nodes,pods)}
        self.latestTime = time

    def scheduleRequest(self, request):
        for time,cpu,mem in request:
            if time in self.clusterState.keys():
                self.clusterState[time].handle_request(self.handle_pod,cpu,mem)
            else:
                assert time > self.latestTime, f'new request time is less than self.latestTime, time={time}, latestTime={self.latestTime}, clusterState={self.clusterState}'
                self.clusterState[time] = copy.deepcopy(self.clusterState[self.latestTime])
                self.clusterState[time].handle_request(self.handle_pod,cpu,mem)
                self.latestTime = time
            self.handle_pod=(self.handle_pod+1)%self.n_pods

    def handleMigrations(self, time, actions):
        assert time >= self.latestTime, f'migration time is less than self.latestTime, time={time}, latestTime={self.latestTime}, clusterState={self.clusterState}'
        if time > self.latestTime:
            self.clusterState[time] = copy.deepcopy(self.clusterState[self.latestTime])
            self.latestTime = time
            self.updateLastestMigrationCost(time,actions)
            self.clusterState[time].handle_migration(actions)
        else:
            self.updateLastestMigrationCost(time,actions)
            self.clusterState[time].handle_migration(actions)

    def updateLastestMigrationCost(self,time,actions):
        self.latestMigrationCost = self.clusterState[time].migrationCost(actions)
        

    '''
        返回列表shape = (当前时间段内数据突变次数，nodes数，51，4)
    '''  
    def describe(self, start_time, end_time=None):
        if end_time is None:
            # 只返回start_time时刻的状态
            if start_time > self.latestTime:
                # print(f'start_time({start_time}) > self.latestTime: return latestTime({self.latestTime}) state')
                return [self.clusterState[self.latestTime].describe()]
            return [self.clusterState[start_time].describe()]
        else:
            # 返回[start_time,end_time)这段时间内的状态
            ret = []
            for time, state in self.clusterState.items():
                if time>=start_time and time<end_time:
                    ret.append(state.describe())
            return ret
    
    def clean(self,start_time):
        keys = list(self.clusterState.keys())
        for time in keys:
            if(time<start_time):
                del self.clusterState[time]

class DeployEnv(gym.Env):
    def __init__(self, nodes, pods):
        super(DeployEnv, self).__init__()
        self.t = 0
        self.N_node = len(nodes)
        self.nodes = nodes
        self.pods = pods
        self.cluster = Cluster(self.t,nodes,pods)

    '''
        以step作为请求序列的开始请求，
        cluster reset到初始化的阶段，
        t reset到step后第一个请求分配的请求
    '''
    def reset(self, step, nodes, pods):
        self.t = step
        self.cluster.reset(self.t*300, nodes, pods)
        state = self.cluster.describe(self.t*300)
        return state

    def termination(self):
        if not hasNextData(self.t):
            return True
        # TODO: 没有考虑当前资源已经分配不了的情况
        # TODO: 如果内存不够会存在无法handle下一个请求的情况
        return False

    '''
        执行迁移action
        action是一个元素为[from,target,to]的列表
    '''
    def _step(self,actions):
        if len(actions)==0:
            return
        self.cluster.handleMigrations(self.t*300, actions)

    '''
        handle下一个五分钟内的请求
        - getNextData获取下一个五分钟内的请求
        - scheduleRequest做请求的调度，返回pod固定时刻的request data序列
    '''
    def handle_next_request(self):
        request = getNextData(self.t)
        self.cluster.scheduleRequest(request)

    '''
        1. 执行迁移action
        2. 计算迁移后的即时reward
        3. getNextData handle下一个五分钟内的请求，更改cluster状态
    '''
    def step(self, actions):
        self._step(actions)
        self.handle_next_request()
        self.t += 1
        reward = self.reward(actions)
        state = self.cluster.describe((self.t-1)*300)
        done = self.termination()
        self.cluster.clean((self.t-1)*300)
        return actions, state, reward, done

    def reward(self,actions):
        state = self.cluster.describe((self.t-1)*300,self.t*300)
        # 1. 资源考量：当前时间段内node上remain_cpu/total_cpu（或mem）<20%的次数或>80%的次数
        over = 0
        under = 0
        for clusterState in state:
            for node in clusterState:
                # assigned_cpu/total_cpu
                assert node[0][0]!=0 and node[0][1]!=0, f'{node[0]}'
                cpu_util = (node[0][0]-node[0][2])/node[0][0]
                mem_util = (node[0][1]-node[0][3])/node[0][1]
                if(cpu_util>NODE_OVER_UTILIZED_THRESHOLD or mem_util>NODE_OVER_UTILIZED_THRESHOLD):
                    over += 1
                if(cpu_util<NODE_UNDER_UTILIZED_THRESHOLD or mem_util<NODE_UNDER_UTILIZED_THRESHOLD):
                    under +=1
        resource_penalty = NODE_OVER_UTILIZED_PENALTY*over + NODE_UNDER_UTILIZED_PENALTY*under

        # 2. 性能考量：当前时间段内pod request_cpu（或mem）未满足的次数
        perf_count = 0
        for clusterState in state:
            for node in clusterState:
                for pod in node[1:]:
                    if(pod[0]>pod[2]):
                        perf_count+=1
                    if(pod[1]>pod[3]):
                        perf_count+=1
        perf_penalty = perf_count * POD_UNDER_REQUESTED_PENALTY

        # 3. 互补性考量
        # 4. 迁移成本考量：被迁移的pod的request_cpu和request_mem累加
        migration_cost = self.cluster.latestMigrationCost * MIGRATION_COST_PENALTY
        return -(resource_penalty+perf_penalty+migration_cost)
        
    def get_attr(self, attr_name):
        # request是还没有handle的下一阶段的请求
        if attr_name == 'obs':
            return self.cluster.describe(self.t*300)
        elif attr_name == 'req_step':
            return self.t
        return None

