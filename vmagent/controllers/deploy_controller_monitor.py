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

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class DeployMonitorMAC:
    def __init__(self, args):
        self.args = args
        self.node_num = args.node_num
        self.pod_num = args.node_num
        action_space = args.pod_num
        obs_space = (2,args.node_num*10,args.pod_num)
        self._build_agents(obs_space, action_space, args)
        # TODO: adapted action_selector
        # self.action_selector = action_REGISTRY['softmax_pos'](args)
        self.agent.apply(init_weights)

    def select_actions(self, ep_batch, eps):
        agent_outputs = self.forward(ep_batch)
        x = agent_outputs
        x_min,_ = th.min(x,1)
        x_min = x_min.unsqueeze(1)
        x_max,_ = th.max(x,1)
        x_max = x_max.unsqueeze(1)
        x = th.sub(x,x_min)/(x_max-x_min)*(self.node_num-1)
        actions = th.round(x).cpu().detach().numpy().astype(int)# 9 and 0 has relatively low possibility to get
        return actions,agent_outputs.cpu().detach().numpy()

    def forward(self, ep_batch):
        agent_inputs = self._build_inputs(ep_batch)
        agent_outs = self.agent(agent_inputs)
        return agent_outs

    def _build_inputs(self, states):
        if type(states) is dict:
            # import pdb; pdb.set_trace()
            obs = th.Tensor(states['obs']).cuda()
            if obs.dim()==3:
                obs = obs.unsqueeze(0)
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