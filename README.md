# [![VMAgent LOGO](./docs/source/images/logo.svg)](https://vmagent.readthedocs.io/en/latest/)

VMAgent is a platform for exploiting Reinforcement Learning (RL) on Virtual Machine (VM)Scheduling tasks.
It collects one month real scheduling data from **huawei cloud** and contains multiple practicle VM scheduling scenarios (such as Fading, Rcovering, etc).
These scenarios also correspond to the challanges in the RL.
Exploiting the design of RL methods in these secenarios help both the RL and VM scheduling community.

Key Components of VMAgent:
* SchedGym (Simulator): it provides many practical scenarios and flexible configurations to define custom scenarios.
* SchedAgent (Algorithms): it provides many popular RL methods as the baselines.
* SchedVis (Visulization): it provides the visualization of schedlueing dynamics on many metrics.
## Scenarios and Baselines

The VMAgent provides multiple practical scenarios: 
| Scenario     | Allow-Deletion | Allow-Expansion | Server Num |
|--------------|----------------|-----------------|------------|
| Fading       | False          | False           | Small      |
| Recovering   | True           | False           | Small      |
| Expanding    | True           | True            | Small      |
| Recovering-L | True           | False           | Large      |

Researchers can also flexibly customized their scenarios in the `vmagent/config/` folder.


Besides, we provides many baselines for quick startups.
It includes FirstFit, BestFit, DQN, PPO, A2C and SAC.
More baselines is coming.
## Installation 

### Install from PyPI

TBA

### Install from Source

```
git clone git@github.com:mail-ecnu/VMAgent.git
cd VMAgent
conda env create -f conda_env.yml
conda activate VMAgent-dev
python3 setup.py develop
```

## Quick Examples

In this quick example, we show how to train a dqn agent in a fading scenario. 
For more examples and the configurations' concrete definitions, we refer readers to our [docs](https://VNAgent.readthedocs.io/en/latest/).

config/fading.yaml:
```yaml
N: 5
cpu: 40 
mem: 90
allow_release: False
```
config/algs/dqn.yaml:
```yaml
mac: 'vectormac'
learner: 'q_learner'
agent: 'DQNAgent'
```
Then 
```sh
python train.py --env=fading --alg=dqn
```

It provides the first VM scheudling simulator based on the one month east china data in huawei cloud.
It includes three scenarios in practical cloud: Recovering, Fading and Expansion.
Our video is at [video](https://drive.google.com/file/d/14EkVzUnEXM7b8YNJiZ6cxLxhcj5yW4V_/view?usp=sharing).
Some demonstrations are listed:

## Docs

For more information of our VMAgent, we refer the readers to the [document](https://vmagent.readthedocs.io/en/latest/).
It describes the detail of SchedGym, SchedAgent and SchedVis.

## Data 

We collect one month scheduling data in east china region of huawei cloud.
The format and the stastical analysis of the data are presented in the docs.
one month east china data in huawei cloud.

## Visualization
<img src="./docs/source/images/rec-small.gif" width="250"><img src="./docs/source/images/rec-large.gif" width="250"><img src="./docs/source/images/exp-large.gif" width="250">

For visualization, see the [`schedvis`](./schedvis) directory in detail.
