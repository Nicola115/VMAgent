'''
TODO List:
    1.  为了固定输入size，原计划针对过去一段时间的state作为observation
        但由于过去一段时间内发生突变的点数不定，简化为针对最后时刻的state作为observation
    
    2.  对齐obs_space,action_space和相应的输入，输出
'''

import torch as th
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY

class DeployMAC:
    def __init__(self, args):
        self.args = args
        self.node_num = args.node_num
        # TODO: build agents according to action space and obs space
        action_space = [args.node_num*5, args.node_num]
        obs_space = args.node_num*51*4
        self._build_agents(obs_space, action_space, args)
        # TODO: adapted action_selector
        self.action_selector = action_REGISTRY['epsilon_greedy'](args)

    def select_actions(self, ep_batch, eps):
        agent_outputs = self.forward(ep_batch)
        avail_actions = []
        for i in range(5*self.node_num):
            for j in range(self.node_num):
                avail_actions.append([i,j])
        chosen_actions = self.action_selector.select_action(agent_outputs, eps,avail_actions, 0)
        try:
            chosen_actions.cpu().numpy()
        except:
            import pdb; pdb.set_trace()
        return  chosen_actions.cpu().numpy()

    def forward(self, ep_batch):
        agent_inputs = self._build_inputs(ep_batch)
        agent_outs = self.agent(agent_inputs)
        return agent_outs

    def _build_inputs(self, states):
        if type(states) is dict:
            # avail_actions = th.Tensor(states['avail']).cuda()
            # import pdb; pdb.set_trace()
            obs = th.Tensor(states['obs']).cuda()
            feat = th.Tensor(states['feat']).cuda()
            return [obs, feat]
        else:
            return states[0], states[1]

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