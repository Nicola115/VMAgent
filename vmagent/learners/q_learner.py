import copy
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, Adam
import numpy as np
import torch.autograd as autograd
import time

class QLearner:
    def __init__(self, mac, args):
        self.args = args
        self.mac = mac

        self.params = list(mac.parameters())

        self.learn_cnt= 0

        self.optimiser = Adam(self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_param = list(self.target_mac.parameters())
        self.last_target_update_episode = 0
        self.tau = self.args.tau
        self.gpu_enable = True


    def get_td_error(self, batch):
        obs = batch['obs']
        next_obs = batch['next_obs']
        action_list = batch['action']
        reward_list = batch['reward']
        mask_list = 1 - batch['done']
        y_list = [0]

        obs = th.FloatTensor(np.array(obs))
        a = th.LongTensor(np.array(action_list))#batch.action))
        rew = th.FloatTensor(np.array(reward_list))##batch.reward))
        rew = rew.view(-1)#, 1)
        mask = th.LongTensor(np.array(mask_list))#batch.mask))
        mask = mask.view(-1)#, 1)
        next_obs = th.FloatTensor(np.array(next_obs))
        ind =  th.arange(a.shape[0])
        if self.gpu_enable:
            obs = obs.cuda()
            a = a.cuda()
            rew = rew.cuda()
            mask = mask.cuda()
            next_obs = next_obs.cuda()
            ind =  ind.cuda()
        # Calculate estimated Q-Values

        # import pdb; pdb.set_trace()
        mac_out, _ = self.mac.forward([obs])

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = mac_out[ind, a]  # Remove the last dim
        target_mac_out = self.target_mac.forward([[next_obs, next_feat], None])[0]


        # Max over target Q-Values
            # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = self.mac.forward([[next_obs, next_feat], next_avail])[0].detach()
        cur_max_actions = th.argmax(mac_out_detach, axis=1)
        cur_max_actions = cur_max_actions.reshape(-1)
        target_max_qvals = target_mac_out[ind, cur_max_actions]


        # Calculate 1-step Q-Learning targets
        targets = rew + self.args.gamma * mask * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        td_error_array=td_error.float().cpu().data.numpy()
        return td_error_array

    def train(self, batch):
        # Get the relevant quantities
        obs = batch['obs']
        feat = batch['feat']
        avail = batch['avail']
        action_list = batch['action']
        reward_list = batch['reward']
        next_obs = batch['next_obs']
        next_feat = batch['next_feat']
        mask_list = 1 - batch['done']
        next_avail = batch['next_avail']
        y_list = [0]

        obs = th.FloatTensor(obs)
        feat = th.FloatTensor(feat)
        avail = th.FloatTensor(avail)
        a = th.LongTensor(action_list)#batch.action))
        rew = th.FloatTensor(reward_list)##batch.reward))
        rew = rew.view(-1)#, 1)
        mask = th.LongTensor(mask_list)#batch.mask))
        mask = mask.view(-1)#, 1)
        next_obs = th.FloatTensor(next_obs)
        next_feat = th.FloatTensor(next_feat)
        next_avail = th.FloatTensor(next_avail)
        ind =  th.arange(a.shape[0])
        if self.gpu_enable:
            obs = obs.cuda()
            feat = feat.cuda()
            avail = avail.cuda()
            a = a.cuda()
            rew = rew.cuda()
            mask = mask.cuda()
            next_obs = next_obs.cuda()
            next_feat = next_feat.cuda()
            next_avail = next_avail.cuda()
            ind =  ind.cuda()
        # Calculate estimated Q-Values

        mac_out, _ = self.mac.forward([[obs, feat], avail])

        # Pick the Q-Values for the actions taken by each agent

        chosen_action_qvals = mac_out[ind, a]  # Remove the last dim
        target_mac_out = self.target_mac.forward([[next_obs, next_feat], next_avail])[0]


        # Max over target Q-Values
            # Get actions that maximise live Q (for double q-learning)
        mac_out_detach = self.mac.forward([[next_obs, next_feat], next_avail])[0].detach()
        cur_max_actions = th.argmax(mac_out_detach, axis=1)
        cur_max_actions = cur_max_actions.reshape(-1)
        target_max_qvals = target_mac_out[ind, cur_max_actions]


        # Calculate 1-step Q-Learning targets
        targets = rew + self.args.gamma * mask * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        td_error_array = td_error.float().cpu().data.numpy()


        # mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # loss = (weight * masked_td_error ** 2).sum() / mask.sum()
        # import pdb; pdb.set_trace()
        # loss = self.lamb2 * (weight * td_error ** 2).sum() / obs.shape[0] + self.lamb * (ep_error **2).sum()/obs.shape[0]
        # import pdb; pdb.set_trace()
        loss = (td_error ** 2).sum() / obs.shape[0]


        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.learn_cnt += 1
        if  self.learn_cnt / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.learn_cnt = 0
        return {
            'critic_loss': loss,
        }



    def _update_targets(self):
        for target_param, local_param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        # self.target_mac.load_state(self.mac)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()


class Critic(nn.Module):
    def __init__(self, state_space, act_space):
        super(Critic, self).__init__()
        self.state_space = state_space
        self.act_space = act_space

        self.features = nn.Sequential(
            nn.Conv2d(self.state_space[0],32,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=2,stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(),512),
            nn.ReLU(),
            nn.Linear(512,self.act_space)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2*self.act_space, 512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def feature_size(self):
        return self.features(autograd.Variable(th.zeros(1,*self.state_space))).view(1,-1).size(-1)
    
    def forward(self, state, action):
        # import pdb; pdb.set_trace()
        x = th.transpose(state,1,3)
        x = x.reshape(-1,x.shape[1],x.shape[2],x.shape[3])
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        # TODO: action在这里需不需要放缩，concat的dim
        # import pdb
        # pdb.set_trace()
        x = th.cat((x,action),1)
        x = self.fc2(x)
        return x


class QmixLearner:
    def __init__(self, actor, args):
        self.args = args

        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.actor_optimiser = Adam(self.actor.parameters(), lr=args.lr)
        self.target_actor_param = list(self.target_actor.parameters())

        self.critic = Critic((2,args.node_num,args.pod_num),args.pod_num).cuda() # TODO: avoiding hardcode, here should be state_space and act_space
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimiser = Adam(self.critic.parameters(), lr=args.lr)
        self.target_critic_param = list(self.target_critic.parameters())

        self.learn_cnt= 0
        self.tau = self.args.tau
        self.gpu_enable = True


    def train(self, batch):
        # 1. get history observation, action, reward and next observation
        obs = batch['obs']
        action_list = batch['action']
        reward_list = batch['reward']
        next_obs = batch['next_obs']
        mask_list = 1 - batch['done']
        obs = th.FloatTensor(obs)
        a = th.FloatTensor(action_list)
        rew = th.FloatTensor(reward_list)
        mask = th.LongTensor(mask_list)
        next_obs = th.FloatTensor(next_obs)
        
        if self.gpu_enable:
            obs = obs.cuda()
            a = a.cuda()
            rew = rew.cuda()
            mask = mask.cuda()
            next_obs = next_obs.cuda()

        # 2. update critic
        critic_out = self.critic.forward(obs,a)
        target_actor_out = self.target_actor.forward(next_obs)
        target_critic_out = self.target_critic.forward(next_obs,target_actor_out)
        y = rew + self.args.gamma * mask * target_critic_out
        td_error = (critic_out - y.detach())
        loss_for_critic = (td_error**2).sum() / obs.shape[0]
        self.critic_optimiser.zero_grad()
        loss_for_critic.backward()
        self.critic_optimiser.step()

        # 3. freeze critic and update actor
        for param in self.critic.parameters():
            param.requires_grad = False

        actor_out = self.actor.forward(obs)
        freeze_critic_out = self.critic.forward(obs, actor_out)
        loss_for_actor = - freeze_critic_out.sum() / obs.shape[0] #TODO: this loss function can be replaced by gan loss
        self.actor_optimiser.zero_grad()
        loss_for_actor.backward()
        self.actor_optimiser.step()

        for param in self.critic.parameters():
            param.requires_grad = True

        # 4. if meets update target interval,update target networks weights
        self.learn_cnt += 1
        if  self.learn_cnt / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.learn_cnt = 0
        return {
            'critic_loss': loss_for_critic.detach().cpu(),
            'actor_loss': loss_for_actor.detach().cpu()
        }

    def _update_targets(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def cuda(self):
        self.critic.cuda()
        self.target_critic.cuda()
        self.actor.cuda()
        self.target_actor.cuda()

    def save_models(self, path):
        self.actor.save_models(path)
        th.save(self.actor_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.actor.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_actor.load_models(path)
        self.actor_optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))