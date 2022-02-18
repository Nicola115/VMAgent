import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import copy

def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            th.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

class QmixAgentForDeploy(nn.Module):
    '''
        Virtual Done Agent: Assume the req has been placed
        TODO: regularize the virtual done(i.e. negative input to -1 to indicate not appliable)
    '''
    def __init__(self, state_space, act_space, args):
        super(QmixAgentForDeploy, self).__init__()
        self.state_space = state_space
        self.obs_space = state_space[0]
        self.num_actions = act_space

        self.flat = nn.Sequential(
            nn.Flatten(),
        )

        self.fc0 = nn.Sequential(
            nn.Linear(51*4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.fc0 = nn.DataParallel(self.fc0)

        self.fc1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.fc1 = nn.DataParallel(self.fc1)

        self.fc2 = nn.Sequential(
            nn.Linear(64, self.obs_space[0]*50),
            nn.ReLU(),
        )
        self.fc2 = nn.DataParallel(self.fc2)
        self.fc_o = nn.Sequential(
            nn.Linear(51*4, 16),
            nn.ReLU()
        )

        self.fc_s = nn.Sequential(
            nn.Linear(4*self.obs_space[0]*51, 32),
            nn.ReLU()
        )

        self.fc_c = nn.Sequential(
            nn.Linear(16+32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.value = nn.Linear(self.obs_space[0]*50, 1)
        self.value = nn.DataParallel(self.value)
        
        self.softmax = nn.Softmax(dim=2)


    def forward(self, state):
        # import pdb;pdb.set_trace()
        # obs = (5, 1, server_num, 51, 4)
        obs = state[0]
        # obs = (5, server_num*51*4)
        h01 = self.flat(obs)
        # bs = 5
        bs = h01.shape[0]
        h00 = copy.deepcopy(h01)
        
        # h00 = (1, server_nums*50, 5, server_num*51*4) = (server_num*50*5, server_num*51*4)
        h00 = h00.repeat(1,self.obs_space[0]*50).reshape((-1,h01.shape[1]))
        # h01 = (5*server_num, 51*4) = (1,50,5*server_num, 51*4) = (server_num*50*5, 51*4)
        h01 = h01.reshape((-1, 51*4)).repeat(1, 50).reshape(-1, 51*4)
        
        h = h01
        s = h00
        
        # s = (server_num*50*5,32)
        s = self.fc_s(s)
        # o = (server_num*50*5,16)
        o = self.fc_o(h)
        # os = (server_num*50*5,48)
        os = th.cat([s,o], dim=-1)
        # weights = (server_num*50*5,1)
        weights = self.fc_c(os)

        # h = (server_num*50*5, 256)
        h= self.fc0(h)

        # h = (server_num*50*5, 64)
        h3 = self.fc1(h)
        # h = (server_num*50*5, server_num*2)
        h3 = self.fc2(h3)

        # q_values = (server_num*50*5, 1)
        q_values = self.value(h3)
        # TOBE: w_q_value = (5*500, 1)
        # w_q_values = (server_num*50*5, 1)
        w_q_values = q_values * weights
        # w_q_value = (5, 50, server_num)
        w_q_values = w_q_values.reshape((bs, 50, -1))
        w_q_values = self.softmax(w_q_values)
        w_actions = w_q_values
        # w_actions = w_q_values.argmax(2)
        if len(w_actions.shape) == 1:
            return w_actions.reshape(1, -1)
        else:
            return w_actions