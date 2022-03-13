from .deploy_env import DeployEnv
from .sched_env import SchedEnv
from .deploy_env_alibaba import DeployEnvAlibaba
REGISTRY = {}
REGISTRY["deployenv"] = DeployEnv
REGISTRY["schedenv"] = SchedEnv
REGISTRY["deployenv_alibaba"] = DeployEnvAlibaba