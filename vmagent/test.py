import numpy as np
import pandas as pd
from config import Config
import argparse
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from components import REGISTRY as mem_REGISTRY
from schedgym import REGISTRY as env_REGISTRY

parser = argparse.ArgumentParser(description='Sched More Servers')
parser.add_argument('--env', type=str)
parser.add_argument('--alg', type=str)
parser.add_argument('--gamma', type=float)
parser.add_argument('--lr', type=float)
conf = parser.parse_args()
args = Config(conf.env, conf.alg)
if conf.gamma is not None:
    args.gamma = conf.gamma 
if conf.lr is not None: 
    args.lr = conf.lr

if __name__ == "__main__":
    # 初始化env
    init_data = pd.read_csv('/data/monitor_init_data_imbalanced.csv')
    nodes_num = 2
    pods_num = 6
    args.node_num = nodes_num
    args.pod_num = pods_num
    env = env_REGISTRY["deployenv_monitor_data"](nodes_num,pods_num,init_data)

    #初始化mac
    mac = mac_REGISTRY["deploy_monitor_mac"](args)
    mac.load_models("models")

    env.reset(0,nodes_num,pods_num,init_data)
    done = env.termination()
    total_reward = 0
    total_len = 0
    cpu_over = 0
    cpu_under = 0
    mem_over = 0
    mem_under = 0
    migrate_count = 0
    while not done:
        feat = env.get_attr('req')
        obs = env.get_attr('obs')
        # sample by first fit
        avail = env.get_attr('avail')
        state = {'obs': obs, 'feat': feat, 'avail': avail}
        for node in obs:
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
        action, _ = mac.select_actions(state, 0)
        action = action[0]
        migrate_count += env.get_migrate_count(action)
        action, next_obs, reward, done = env.step(action)
        total_reward += reward
        total_len += 1
    print(total_reward)
    print(f"migrate_count:{migrate_count}, cpu_over:{cpu_over}, cpu_under:{cpu_under}, mem_over:{mem_over}, mem_under:{mem_under}")