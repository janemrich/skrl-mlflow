import yaml
import gymnasium as gym
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.runner.torch.runner import Runner
import os

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "agent.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.load(f, Loader=yaml.UnsafeLoader)


env = gym.make("Pendulum-v1")
env = wrap_env(env)
device = env.device

runner = Runner(env=env, cfg=cfg)
runner.run()
