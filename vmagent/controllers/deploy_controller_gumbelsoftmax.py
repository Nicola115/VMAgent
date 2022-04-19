'''
TODO List:
    1.  为了固定输入size，原计划针对过去一段时间的state作为observation
        但由于过去一段时间内发生突变的点数不定，简化为针对最后时刻的state作为observation
    
    2.  对齐obs_space,action_space和相应的输入，输出
'''

import torch as th
import torch.nn as nn
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np
import random

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class DeployGumbelSoftmaxMAC:
    def __init__(self, args):
        self.args = args
        self.node_num = args.node_num
        self.pod_num = args.node_num
        action_space = args.pod_num*args.node_num
        obs_space = (2,args.node_num*10,args.pod_num) # TODO: hard code: routine number 5;timestamp number 10
        self._build_agents(obs_space, action_space, args)
        self.agent.apply(init_weights)

    def select_actions(self, ep_batch, eps):
        agent_outputs = self.forward(ep_batch)
        return_action = agent_outputs.cpu().detach().numpy()
        actions = th.argmax(agent_outputs,dim=2)
        actions = actions.cpu().detach().numpy().astype(int)
        if random.random()<0.2:
            # import pdb;pdb.set_trace()
            random_actions = np.zeros(actions.shape,dtype=int)
            random_return_action = np.zeros(return_action.shape)
            for i in range(len(actions)):
                random_actions[i] = np.random.randint(2,size=6)
                for j in range(6):
                    random_return_action[i,j,random_actions[i,j]]=1.0
            # print(actions)
            # print(return_action)

            actions = random_actions
            return_action = random_return_action
        return actions,return_action

    def forward(self, ep_batch):
        agent_inputs = self._build_inputs(ep_batch)
        agent_outs = self.agent(agent_inputs)
        return agent_outs

    def _build_inputs(self, states):
        if type(states) is dict:
            # avail_actions = th.Tensor(states['avail']).cuda()
            # import pdb; pdb.set_trace()
            obs = th.Tensor(states['obs']).cuda()
            # feat = th.Tensor(states['feat']).cuda()
            return obs
        elif type(states) is list:
            return states
        else:
            return states

    def _build_agents(self, obs_space, action_space, args):
        self.agent = agent_REGISTRY[self.args.agent](obs_space, action_space, args).cuda()

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))