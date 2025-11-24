import argparse
import gymnasium as gym
import os
from dotenv import load_dotenv
import time

import torch
import torch.nn as nn

from packaging import version
import yaml

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import mlflow, set_seed


import os, re, yaml
from mlflow import MlflowClient

def upload_agent_yaml_to_existing_run(load_uri: str, params_config: dict, yaml_name="agent.yaml"):
    """
    Добавляет agent.yaml в тот же MLflow run, откуда взят checkpoint.
    Пример:
        mlflow-artifacts:/50/90a2668961164475a08682ed44533ac5/artifacts/checkpoints/agent_9000.pt
    """
    if not load_uri.startswith("mlflow-artifacts:/"):
        print("❌ Не MLflow путь, пропускаю загрузку параметров.")
        return

    # достаём run_id из URI
    match = re.search(r"mlflow-artifacts:/\d+/([0-9a-f]+?)/artifacts", load_uri)
    if not match:
        print(f"❌ Не удалось извлечь run_id из {load_uri}")
        return
    run_id = match.group(1)

    # локальный YAML
    yaml_path = os.path.join("tmp_params", yaml_name)
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(params_config, f, sort_keys=False, allow_unicode=True)

    # логируем в существующий run
    client = MlflowClient()
    client.log_artifact(run_id=run_id, local_path=yaml_path, artifact_path="params")
    print(f"✅ Uploaded {yaml_name} to existing MLflow run {run_id}")


# ---------------------------------------------------------------------
# 1. CLI аргументы
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train or load PPO agent with skrl.")
parser.add_argument("--load", type=str, default=None,
                    help="Path or MLflow URI to the checkpoint to load (optional).")
args = parser.parse_args()

# ---------------------------------------------------------------------
# 2. Загружаем .env и задаем seed
# ---------------------------------------------------------------------
load_dotenv()
set_seed()

# ---------------------------------------------------------------------
# 3. Модели
# ---------------------------------------------------------------------
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ---------------------------------------------------------------------
# 4. Среда
# ---------------------------------------------------------------------
try:
    env = gym.make_vec("Pendulum-v1", num_envs=4, render_mode="rgb_array")
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("Pendulum-v-")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make_vec(env_id, num_envs=4, render_mode="rgb_array")

log_dir = "logs/test-skrl-mlflow-" + str(time.time())
video_kwargs = {
    "video_folder": os.path.join(log_dir, "videos"),
    "name_prefix": "ppo-pendulum",
    "step_trigger": lambda step: step % 1000 == 0,
    "video_length": 200
}
env = gym.wrappers.vector.RecordVideo(env, **video_kwargs)
env = wrap_env(env)
device = env.device

# ---------------------------------------------------------------------
# 5. Память и модели
# ---------------------------------------------------------------------
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models = {
    "policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
    "value": Value(env.observation_space, env.action_space, device),
}

# ---------------------------------------------------------------------
# 6. Конфигурация агента PPO
# ---------------------------------------------------------------------
cfg = PPO_DEFAULT_CONFIG.copy()
params_config = {
    "rollouts": 1024,
    "learning_epochs": 10,
    "mini_batches": 16,
    "discount_factor": 0.9,
    "lambda": 0.95,
    "learning_rate": 1e-4,
    "learning_rate_scheduler": KLAdaptiveRL,
    "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
    "grad_norm_clip": 0.5,
    "ratio_clip": 0.2,
    "value_clip": 0.2,
    "clip_predicted_values": False,
    "entropy_loss_scale": 0.0001,
    "value_loss_scale": 0.5,
    "kl_threshold": 0,
    "state_preprocessor": RunningStandardScaler,
    "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
    "value_preprocessor": RunningStandardScaler,
    "value_preprocessor_kwargs": {"size": 1, "device": device},
    "experiment": {
        "directory": os.path.abspath(log_dir),
        "mlflow": True,
        "mlflow_kwargs": {"experiment_name": "test-skrl-mlflow"},
        "video_kwargs": video_kwargs
    }
}
cfg.update(params_config)
# ---------------------------------------------------------------------
# 7. Инициализация агента
# ---------------------------------------------------------------------
agent = PPO(models=models, memory=memory, cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space, device=device)

# ---------------------------------------------------------------------
# 8. Если указан путь, загружаем чекпоинт
# ---------------------------------------------------------------------
if args.load:
    print(f"Loading checkpoint from: {args.load}")
    upload_agent_yaml_to_existing_run(args.load, params_config)
    agent.load(args.load)


# ---------------------------------------------------------------------
# 9. Тренировка
# ---------------------------------------------------------------------
cfg_trainer = {"timesteps": 10000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
trainer.train()

# ---------------------------------------------------------------------
# 10. Завершение MLflow run
# ---------------------------------------------------------------------
if agent.cfg["experiment"]["mlflow"]:
    mlflow.end_run()
