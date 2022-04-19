import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import copy
import torch.autograd as autograd
from einops import rearrange

'''
    action model
'''
class QmixAgentForDeployGumbelSoftmax(nn.Module):
    def __init__(self, state_space, act_space, args):
        super(QmixAgentForDeployGumbelSoftmax, self).__init__()
        # state_space = (2*5,node_num*10,pod_num)
        self.state_space = state_space
        self.resource_num = state_space[0]
        self.node_num = state_space[1]/10
        self.pod_num = state_space[2]
        self.act_space = act_space
        assert self.act_space == self.pod_num*self.node_num,f'wrong act_space: {self.act_space}, pod_num={self.pod_num}, node_num = {self.node_num}'

        self.features = nn.Sequential(
            nn.Conv2d(self.resource_num,32,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=2,stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(),512),
            nn.ReLU(),
            nn.Linear(512,self.act_space)
        )

    def feature_size(self):
        print(self.state_space)
        return self.features(autograd.Variable(th.zeros(1,*self.state_space))).view(1,-1).size(-1)
    
    def forward(self, x):
        # x = (5, node_num*10,pod_num,resource_type)
        assert x.dim()==4 or x.dim()==5,f'{x.dim()}'
        if x.dim() == 5:
            x = rearrange(x, 'b1 b2 n p c -> (b1 b2) c p n')
        elif x.dim() == 4:
            x = rearrange(x, 'b n p c -> b c p n')
            
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        logits = rearrange(x, 'b (p n) -> b p n',p=self.pod_num)
        action = F.gumbel_softmax(logits, tau=1, hard=True)
        return action

        