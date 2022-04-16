from .q_learner import QLearner
from .q_learner import QmixLearner
from .q_learner import QmixLearnerGumbelSoftmax
from .sac_learner import SACLearner
from .ppo_learner import PPOLearner
from .a2c_learner import A2CLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qmix_learner"] = QmixLearner
REGISTRY["qmix_learner_gumbelsoftmax"] = QmixLearnerGumbelSoftmax
REGISTRY["sac_learner"] = SACLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["a2c_learner"] = A2CLearner
