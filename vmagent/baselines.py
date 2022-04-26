import numpy as np
from config import Config
import pandas as pd
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from components import REGISTRY as mem_REGISTRY
from schedgym import REGISTRY as env_REGISTRY

class Args():
    def __init__(self,env,alg,memory):
        self.env = env
        self.alg = alg
        self.memory = memory

args = Args('deployenv_monitor_data','deploy_gumbelsoftmax','replay')
args = Config(args.env, args.alg)
args.env = 'deployenv_monitor_data'

# 环境初始化
init_data = pd.read_csv('/data/monitor_init_data_imbalanced.csv')
nodes_num = 2
pods_num = 6
env = env_REGISTRY[args.env](nodes_num,pods_num,init_data)
env.reset(0,nodes_num,pods_num,init_data)

# 内存回放初始化
mem = mem_REGISTRY[args.memory](args)
migrate_count = 0
cpu_over = 0
cpu_under = 0
mem_over = 0
mem_under = 0

def select_action(state):
    # 1. 计算original mapping
    # import pdb
    # pdb.set_trace()
    node = state['obs']
    node1 = node[:10,:,0]
    node2 = node[10:,:,0]
    node1_mean = node1.mean(0)
    node2_mean = node2.mean(0)
    original_mapping = (node1_mean==0).astype(int)

    # 2. 计算负载均衡性
    thegma = (node1_mean.sum()-node2_mean.sum())**2/2
    if(thegma<=0.5):
        return original_mapping

    # 3. 选择热点服务器
    node_target = []
    if node1_mean.sum()>0.6:
        node_target = node1
    elif node2_mean.sum()>0.6:
        node_target = node2
    else:
        return original_mapping

    # 4. 选择热点容器
    pri1 = np.argsort(node_target.mean(0))
    original_mapping[pri1[0]] = ~original_mapping[pri1[0]]
    migrate_count+=1
    return original_mapping
    

    

def run(env, step, mem):
    tot_reward = 0
    tot_lenth = 0
    done_flag = False
    global cpu_over
    global cpu_under
    global mem_over
    global mem_under
    while True:
        # get action
        step += 1
        print(step)

        if done_flag:
            return tot_reward, tot_lenth

        avail = env.get_attr('avail')
        feat = env.get_attr('req')
        obs = env.get_attr('obs')
        state = {'obs': obs, 'feat': feat, 'avail': avail}
        for node in obs:
            # import pdb;pdb.set_trace()
            cpu = node[:,0].sum()
            mem = node[:,1].sum()
            if cpu>0.8:
                cpu_over+=1
            if cpu!=0 and cpu<0.2:
                cpu_under+=1
            if mem>0.8:
                mem_over+=1
            if mem!=0 and mem<0.2:
                mem_under+=1
        
        action = select_action(state)
       
        action, next_obs, reward, done = env.step(action)
        if done==True:
            print(done)
            done_flag = done
        

        tot_reward += reward
        tot_lenth += 1


val_return, val_lenth = run(env, 0, mem)
val_metric = {
    'tot_reward': val_return,
    'tot_len': val_lenth,
}

print(val_return, val_lenth)
print(f"migrate_count:{migrate_count}, cpu_over:{cpu_over}, cpu_under:{cpu_under}, mem_over:{mem_over}, mem_under:{mem_under}")
