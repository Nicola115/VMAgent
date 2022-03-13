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


NODE_OVER_UTILIZED_THRESHOLD = 0.8*64
NODE_UNDER_UTILIZED_THRESHOLD = 0.2*64
NODE_OVER_UTILIZED_PENALTY = 0.6
NODE_UNDER_UTILIZED_PENALTY = 0.4
POD_UNDER_REQUESTED_PENALTY = 1
MIGRATION_COST_PENALTY = 0.1

df = pd.read_csv('/data/clusterdata/cluster-trace-v2017/usage_data_small.csv')

def getNextData(step):
    '''
        - getNextData返回下一个5min内的请求
        - 下一个5min[cur_step*300,(cur_step+1)*300)
        - 忽略所有释放资源的请求
    '''
    request = df[df['start_time']==step]
    return request

def hasNextData(step):
    request = df[df['start_time']==step]
    return len(request)!=0

class Pod():
    def __init__(self, index, node_id):
        self.index = index
        self.current_node = node_id
        self.used_cpu = 0
        self.used_mem = 0

class Node():
    def __init__(self, node_index):
        self.index = node_index
        self.pods = set()
    

class Cluster():
    def __init__(self,nodes_num,pods_num,init_data):
        self.nodes_num = nodes_num
        self.pods_num = pods_num
        self.nodes = [Node(i) for i in range(nodes_num)]
        self.pods = [Pod(i,-1) for i in range(pods_num)]
        def initPod(row):
            pod_id = int(row['pod_id'])
            node_id = int(row['node_id'])
            self.pods[pod_id].used_cpu = row['used_cpu']
            self.pods[pod_id].used_mem = row['used_mem']
            self.pods[pod_id].current_node = node_id
            self.nodes[node_id].pods.add(pod_id)
        init_data.apply(initPod,axis=1)
    
    def reset(self, nodes_num,pods_num,init_data):
        self.__init__(nodes_num,pods_num,init_data)

    def update(self, request):
        def updatePod(row):
            pod_id = int(row['pod_id'])
            self.pods[pod_id].used_cpu = row['used_cpu']
            self.pods[pod_id].used_mem = row['used_mem']

        request.apply(updatePod,axis=1)

    def handle_migration(self, action):
        assert action.shape==(self.pods_num,), f'{action.shape}'
        cost = 0
        for pod_index, pod_action in enumerate(action):
            assert pod_action>=0 and pod_action<self.nodes_num,f'{pod_action}' 
            if self.pods[pod_index].current_node==pod_action:
                continue

            from_node = self.pods[pod_index].current_node
            self.nodes[from_node].pods.remove(pod_index)
            self.nodes[pod_action].pods.add(pod_index)
            cost+=1
        return cost

    def describe(self):
        ret = []
        for node in self.nodes:
            node_data = []
            for pod_index in node.pods:
                node_data.append([self.pods[pod_index].used_cpu,self.pods[pod_index].used_mem])
            for i in range(self.pods_num-len(node.pods)):
                node_data.append([0.0,0.0])
            ret.append(node_data)
        return np.array(ret)

class DeployEnvAlibaba(gym.Env):
    def __init__(self, nodes_num, pods_num, init_data):
        super(DeployEnvAlibaba, self).__init__()
        self.t = init_data['start_time'][0]
        self.start = self.t
        self.nodes_num = nodes_num
        self.pods_num = pods_num
        self.cluster = Cluster(nodes_num,pods_num,init_data)

    '''
        以step作为请求序列的开始请求，
        cluster reset到初始化的阶段，
        t reset到step后第一个请求分配的请求
    '''
    def reset(self, step,nodes_num,pods_num,init_data):
        self.t = step*300+self.start
        self.cluster.reset(nodes_num,pods_num,init_data)
        state = self.cluster.describe()
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
            return 0
        return self.cluster.handle_migration(actions)

    '''
        handle下一个五分钟内的请求
        - getNextData获取下一个五分钟内的请求
        - scheduleRequest做请求的调度，返回pod固定时刻的request data序列
    '''
    def handle_next_request(self):
        request = getNextData(self.t)
        self.cluster.update(request)

    '''
        1. 执行迁移action
        2. 计算迁移后的即时reward
        3. getNextData handle下一个五分钟内的请求，更改cluster状态
    '''
    def step(self, actions):
        migration_cost = self._step(actions)
        self.handle_next_request()
        self.t += 300
        reward = self.reward(migration_cost)
        state = self.cluster.describe()
        done = self.termination()
        return actions, state, reward, done

    def reward(self,migration_cost):
        state = self.cluster.describe()
        # 1. 资源考量：当前时间段内node上remain_cpu/total_cpu（或mem）<20%的次数或>80%的次数
        over = 0
        under = 0
        for node in state:
            # assigned_cpu/total_cpu
            cpu_util = node[:,0].sum()
            mem_util = node[:,1].sum()
            if cpu_util>NODE_OVER_UTILIZED_THRESHOLD:
                over += 1
            if mem_util>0.8:
                over += 1
            if cpu_util<NODE_UNDER_UTILIZED_THRESHOLD:
                under += 1
            if mem_util<0.2:
                under += 1
        resource_penalty = NODE_OVER_UTILIZED_PENALTY*over + NODE_UNDER_UTILIZED_PENALTY*under

        # 2. 性能考量

        # 3. 互补性考量

        # 4. 迁移成本考量：被迁移的pod的request_cpu和request_mem累加
        migration_penalty = MIGRATION_COST_PENALTY*migration_cost
        return -resource_penalty-migration_cost
        
    def get_attr(self, attr_name):
        # request是还没有handle的下一阶段的请求
        if attr_name == 'obs':
            return self.cluster.describe()
        elif attr_name == 'req_step':
            return self.t
        return None

