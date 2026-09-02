"""Nominal and obstacle-aware Unitree G1 walking tasks."""

from copy import deepcopy

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from cbf_rl_humanoid.config import (
  unitree_g1_nominal_env_cfg,
  unitree_g1_obstacle_cbf_env_cfg,
)


NOMINAL_TASK_ID = "Velocity-Flat-Unitree-G1"
OBSTACLE_TASK_ID = "Velocity-Flat-Unitree-G1-Obstacle-CBF"


def _runner_cfg(experiment_name: str):
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.hidden_dims = (512, 256, 128)
  cfg.actor.obs_normalization = True
  cfg.critic.hidden_dims = (512, 256, 128)
  cfg.critic.obs_normalization = True
  cfg.algorithm.entropy_coef = 0.005
  cfg.max_iterations = 20_001
  cfg.save_interval = 1_000
  cfg.experiment_name = experiment_name
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  return cfg


nominal_runner_cfg = _runner_cfg("g1_velocity_flat")
obstacle_runner_cfg = deepcopy(nominal_runner_cfg)
obstacle_runner_cfg.experiment_name = "g1_velocity_flat_obstacle_cbf"

register_mjlab_task(
  task_id=NOMINAL_TASK_ID,
  env_cfg=unitree_g1_nominal_env_cfg(),
  play_env_cfg=unitree_g1_nominal_env_cfg(play=True),
  rl_cfg=nominal_runner_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=OBSTACLE_TASK_ID,
  env_cfg=unitree_g1_obstacle_cbf_env_cfg(),
  play_env_cfg=unitree_g1_obstacle_cbf_env_cfg(play=True),
  rl_cfg=obstacle_runner_cfg,
  runner_cls=VelocityOnPolicyRunner,
)

__all__ = [
  "NOMINAL_TASK_ID",
  "OBSTACLE_TASK_ID",
  "unitree_g1_nominal_env_cfg",
  "unitree_g1_obstacle_cbf_env_cfg",
]
