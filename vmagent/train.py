from config import Config
import os
import copy
import numpy as np
from utils.rl_utils import linear_decay, time_format
import argparse
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from components import REGISTRY as mem_REGISTRY
from schedgym import REGISTRY as env_REGISTRY
from schedgym.mySubproc_vec_env import SubprocVecEnv
from runx.logx import logx
from hashlib import sha1
import time
import json
import pandas as pd

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

MAX_EPOCH = args.max_epoch
BATCH_SIZE = args.batch_size

logpath = '../log/search_'+str(args.learner)+conf.env+'/'+str(args.learner) + \
    str(args.gamma)+'_' + str(args.lr)+'/'

# reward discount
logx.initialize(logdir=logpath, coolname=True, tensorboard=True)

# N is the number of servers, cpu and mem are the attribute of the server
def make_env(N, cpu, mem, allow_release, double_thr=1e10):
    def _init():
        if args.env=="schedenv":
            env = env_REGISTRY[args.env](N, cpu, mem, DATA_PATH, render_path=None,
                        allow_release=allow_release, double_thr=double_thr)
        # env.seed(seed + rank)
            return env
        elif args.env=="deployenv":
            #TODO: nodes和pods的初始化
            with open('nodes.json','r')as f:
                nodes = json.load(f)
            with open('pods.json','r')as f:
                pods = json.load(f)
            env = env_REGISTRY[args.env](nodes,pods)
            return env
        elif args.env=="deployenv_alibaba":
            init_data = pd.read_csv('/data/clusterdata/cluster-trace-v2017/init_data_small.csv')
            nodes_num = len(init_data['node_id'].unique())
            pods_num = len(init_data['pod_id'].unique())
            env = env_REGISTRY[args.env](nodes_num,pods_num,init_data)
            return env
        elif args.env=="deployenv_monitor_data":
            init_data = pd.read_csv('/data/monitor_init_data.csv')
            nodes_num = len(init_data['node_id'].unique())
            pods_num = len(init_data['pod_id'].unique())
            env = env_REGISTRY[args.env](nodes_num,pods_num,init_data)
            return env
    # set_global_seeds(seed)
    return _init


def run(envs, step_list, mac, mem, learner, eps, args):
    tot_reward = np.array([0. for j in range(args.num_process)])
    tot_lenth = np.array([0. for j in range(args.num_process)])
    step = 0
    stop_idxs = np.array([0 for j in range(args.num_process)])
    while True:
        # get action
        step += 1
        envs.update_alives()

        alives = envs.get_alives().copy()
        if all(~alives):
            return tot_reward.mean(), tot_lenth.mean()

        if args.env=='deployenv':
            obs = envs.get_attr('obs')
            state = {'obs': obs}
        else:
            avail = envs.get_attr('avail')
            feat = envs.get_attr('req')
            obs = envs.get_attr('obs')
            state = {'obs': obs, 'feat': feat, 'avail': avail}
        
        action, raw_action = mac.select_actions(state, eps)
        
        action, next_obs, reward, done = envs.step(action)

        stop_idxs[alives] += 1

        tot_reward[alives] += reward
        tot_lenth[alives] += 1

        buf = {'obs': obs, 'action': raw_action,'reward': reward, 'next_obs': next_obs, 'done': done}
        mem.push(buf)


if __name__ == "__main__":
    # execution
    step_list = []
    # args.num_process is defined in default.yaml as 5
    for i in range(args.num_process):
        step_list.append(np.random.randint(0, 143))
    my_steps = np.array(step_list) # my_step is a 5 items' list, contains 5 random int from 1000-9999 

    if args.double_thr is None:
        double_thr = 1000
    else:
        double_thr = args.double_thr

    init_data = pd.read_csv('/data/monitor_init_data.csv')
    nodes_num = len(init_data['node_id'].unique())
    pods_num = len(init_data['pod_id'].unique())
    args.node_num = nodes_num
    args.pod_num = pods_num

    # SubprocVecEnv是一个多进程wrapper，使得原来的实现可以并发
    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(
        args.allow_release == 'True'), double_thr=double_thr) for i in range(args.num_process)])

    # 搭建模型
    mac = mac_REGISTRY[args.mac](args)
    # import pdb
    # pdb.set_trace()
    print(f'Sampling with {args.mac} for {MAX_EPOCH} epochs; Learn with {args.learner}')
    learner = le_REGISTRY[args.learner](mac, args)
    mem = mem_REGISTRY[args.memory](args)
    t_start = time.time()
    for x in range(MAX_EPOCH):
        eps = linear_decay(x, [0, int(
            MAX_EPOCH * 0.25),  int(MAX_EPOCH * 0.9), MAX_EPOCH], [0.9, 0.5, 0.2, 0.2])
        if args.env=='deployenv':
            envs.reset(my_steps,nodes,pods)
        elif args.env=='deployenv_alibaba' or args.env=='deployenv_monitor_data':
            envs.reset(my_steps,nodes_num,pods_num,init_data)
        else:
            envs.reset(my_steps) # set env.t and env.start to my_steps

        train_rew, train_len = run(
            envs, my_steps, mac, mem, learner, eps, args)
        actor_loss, critic_loss, critic1_loss, critic2_loss, alpha_loss = [0 for i in range(5)]

        # start optimization
        for i in range(args.train_n):
            batch = mem.sample(BATCH_SIZE)
            metrics = learner.train(batch)

        # log training curves
        metrics['eps'] = eps
        metrics['tot_reward'] = train_rew.mean()
        metrics['tot_len'] = train_len.mean()
        print(f'Epoch {x}/{MAX_EPOCH}; total_reward: {train_rew.mean()}, total_len: {train_len.mean()}, critic_loss: {metrics["critic_loss"]}, actor_loss: {metrics["actor_loss"]}')
        logx.metric('train', metrics, x)

        if x % args.test_interval == 0:
            envs.reset(my_steps, nodes_num, pods_num, init_data)
            val_return, val_lenth = run(
                envs, my_steps, mac, mem, learner, 0, args)
            val_metric = {
                'tot_reward': val_return.mean(),
                'tot_len': val_lenth.mean(),
            }

            logx.metric('val', val_metric, x)

            path = f'{logpath}/models/{args.N}server-{x}'
            

            if not os.path.exists(path):
                os.makedirs(path)

            learner.save_models(path)

            t_end = time.time()
            print(f'Epoch {x}/{MAX_EPOCH}; lasted %d hour, %d min, %d sec ' %
                  time_format(t_end - t_start))
            # print('remain %d hour, %d min, %d sec' % time_format(
                # (MAX_EPOCH-x)//args.test_interval * (t_end - t_start)))
            # t_start = t_end