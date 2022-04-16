from .deploy_env import DeployEnv
from .sched_env import SchedEnv
from .deploy_env_alibaba import DeployEnvAlibaba
from .deploy_env_monitor_data import DeployEnvMonitorData
REGISTRY = {}
REGISTRY["deployenv"] = DeployEnv
REGISTRY["schedenv"] = SchedEnv
REGISTRY["deployenv_alibaba"] = DeployEnvAlibaba
REGISTRY["deployenv_monitor_data"] = DeployEnvMonitorData