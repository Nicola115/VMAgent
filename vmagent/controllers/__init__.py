from .ppo_controller import PPOMAC
from .sac_controller import SACMAC
from .basic_controller import VectorMAC
from .deploy_controller import DeployMAC
from .deploy_controller_monitor import DeployMonitorMAC
from .deploy_controller_gumbelsoftmax import DeployGumbelSoftmaxMAC
REGISTRY = {}


REGISTRY["vectormac"] = VectorMAC
REGISTRY["deploymac"] = DeployMAC
REGISTRY["deploy_monitor_mac"] = DeployMonitorMAC
REGISTRY["sacmac"] = SACMAC
REGISTRY["ppomac"] = PPOMAC
REGISTRY["deploy_gumbelsoftmax_mac"] = DeployGumbelSoftmaxMAC
