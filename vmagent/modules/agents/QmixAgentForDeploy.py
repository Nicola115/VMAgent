import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import copy
import torch.autograd as autograd

'''
    action model
'''
class QmixAgentForDeploy(nn.Module):
    def __init__(self, state_space, act_space, args):
        super(QmixAgentForDeploy, self).__init__()
        self.state_space = state_space
        self.act_space = act_space

        self.features = nn.Sequential(
            nn.Conv2d(self.state_space[0],32,kernel_size=2,stride=1),
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
        return self.features(autograd.Variable(th.zeros(1,*self.state_space))).view(1,-1).size(-1)
    
    def forward(self, x):
        x = th.transpose(x,2,4)
        x = x.reshape(-1,x.shape[2],x.shape[3],x.shape[4])
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        # TODO: action的放缩
        return x

        