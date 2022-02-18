from .ppo_controller import PPOMAC
from .sac_controller import SACMAC
from .basic_controller import VectorMAC
from .deploy_controller import DeployMAC
REGISTRY = {}


REGISTRY["vectormac"] = VectorMAC
REGISTRY["deploymac"] = DeployMAC
REGISTRY["sacmac"] = SACMAC
REGISTRY["ppomac"] = PPOMAC
