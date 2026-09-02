"""Configuration checks for the nominal and obstacle-aware walking tasks."""

from __future__ import annotations

from dataclasses import fields
import math

import mujoco
import torch

from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import cbf_rl_humanoid
from cbf_rl_humanoid.actions import (
  BODY_JOINT_NAMES,
  DefaultArmLocomotionActionCfg,
)
from cbf_rl_humanoid.commands import (
  ObstacleVelocityCommandCfg,
  obstacle_positions_from_velocity_command,
  planar_velocity_toward_obstacle,
)
from cbf_rl_humanoid.config import (
  unitree_g1_nominal_env_cfg,
  unitree_g1_obstacle_cbf_env_cfg,
)
from cbf_rl_humanoid.observations import (
  locomotion_phase_features,
  relative_obstacle_position_b,
)
from cbf_rl_humanoid.obstacle import (
  DEFAULT_OBSTACLE_POSITION,
  DEFAULT_OBSTACLE_RADIUS,
  DEFAULT_OBSTACLE_SIZE,
)
from cbf_rl_humanoid.rewards import (
  obstacle_cbf_rl_reward,
  track_safe_linear_velocity,
)
from cbf_rl_humanoid.terminations import obstacle_collision


def test_only_the_two_project_tasks_are_registered() -> None:
  registered = list_tasks()
  assert cbf_rl_humanoid.NOMINAL_TASK_ID in registered
  assert cbf_rl_humanoid.OBSTACLE_TASK_ID in registered
  assert set(cbf_rl_humanoid.__all__[:2]) == {
    "NOMINAL_TASK_ID",
    "OBSTACLE_TASK_ID",
  }
  legacy_ids = {
    "Mjlab-Velocity-Flat-Unitree-G1-Obstacle",
    "Mjlab-Velocity-Flat-Unitree-G1-Obstacle-Xbox",
    "Mjlab-Velocity-Flat-Unitree-G1-Obstacle-IsaacLab",
    "Mjlab-Velocity-Flat-Unitree-G1-Obstacle-IsaacLab-Xbox",
    "Mjlab-Velocity-Flat-Unitree-G1-Obstacle-IsaacLab-Phase",
    "Mjlab-Velocity-Flat-Unitree-G1-Obstacle-IsaacLab-Phase-Xbox",
  }
  assert legacy_ids.isdisjoint(registered)


def test_both_tasks_use_the_expected_runner() -> None:
  nominal = load_rl_cfg(cbf_rl_humanoid.NOMINAL_TASK_ID)
  obstacle = load_rl_cfg(cbf_rl_humanoid.OBSTACLE_TASK_ID)
  for cfg in (nominal, obstacle):
    assert cfg.logger == "tensorboard"
    assert cfg.upload_model is False
    assert cfg.actor.class_name == "MLPModel"
    assert cfg.critic.class_name == "MLPModel"
    assert cfg.actor.hidden_dims == (512, 256, 128)
    assert cfg.critic.hidden_dims == (512, 256, 128)
    assert cfg.actor.obs_normalization is True
    assert cfg.critic.obs_normalization is True
    assert cfg.algorithm.entropy_coef == 0.005
    assert cfg.max_iterations == 20_001
    assert cfg.save_interval == 1_000
  assert nominal.experiment_name == "g1_velocity_flat"
  assert obstacle.experiment_name == "g1_velocity_flat_obstacle_cbf"


def test_nominal_policy_keeps_direct_yaw_and_fixed_arms() -> None:
  cfg = unitree_g1_nominal_env_cfg()
  command = cfg.commands["twist"]
  assert command.resampling_time_range == (3.0, 8.0)
  assert command.heading_command is True
  assert command.rel_heading_envs == 0.3
  assert command.rel_forward_envs == 0.2
  assert command.rel_standing_envs == 0.1
  assert command.ranges.lin_vel_x == (-1.0, 1.0)
  assert command.ranges.lin_vel_y == (-1.0, 1.0)
  assert command.ranges.ang_vel_z == (-1.0, 1.0)

  stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert stages[0]["step"] == 0
  assert stages[0]["ang_vel_z"] == (-0.5, 0.5)
  assert stages[1]["step"] == 5_000 * 24
  assert stages[1]["ang_vel_z"] == (-1.0, 1.0)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, DefaultArmLocomotionActionCfg)
  assert action.body_joint_names == BODY_JOINT_NAMES
  assert not hasattr(action, "motion_file")
  assert set(cfg.scene.entities) == {"robot"}
  assert "obstacle_cbf_rl" not in cfg.rewards
  assert "obstacle_collision" not in cfg.terminations

  assert list(cfg.observations["actor"].terms) == [
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
    "phase",
  ]
  assert list(cfg.observations["critic"].terms) == [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
    "phase",
  ]
  actor = cfg.observations["actor"].terms
  assert actor["command"].scale is None
  assert actor["phase"].func is locomotion_phase_features
  assert actor["phase"].params == {
    "command_name": "twist",
    "gait_period": 1.0,
    "gait_offset": 0.5,
  }
  assert cfg.rewards["track_angular_velocity"].weight == 2.0
  assert cfg.rewards["track_angular_velocity"].params["std"] == math.sqrt(0.5)


def test_obstacle_task_is_exact_nominal_extension() -> None:
  nominal = unitree_g1_nominal_env_cfg()
  obstacle = unitree_g1_obstacle_cbf_env_cfg()
  assert set(obstacle.scene.entities) == {"robot", "obstacle"}
  assert obstacle.actions == nominal.actions
  assert obstacle.events == nominal.events
  assert set(obstacle.terminations) == set(nominal.terminations) | {
    "obstacle_collision"
  }
  collision = obstacle.terminations["obstacle_collision"]
  assert collision.func is obstacle_collision
  assert collision.params == {"command_name": "twist"}

  for group_name in ("actor", "critic"):
    nominal_terms = nominal.observations[group_name].terms
    obstacle_terms = obstacle.observations[group_name].terms
    assert set(obstacle_terms) == set(nominal_terms) | {"obstacle_position_b"}
    for name, term in nominal_terms.items():
      assert obstacle_terms[name] == term
    relative = obstacle_terms["obstacle_position_b"]
    assert relative.func is relative_obstacle_position_b
    assert relative.params == {
      "robot_name": "robot",
      "obstacle_name": "obstacle",
    }

  nominal_command = nominal.commands["twist"]
  obstacle_command = obstacle.commands["twist"]
  assert isinstance(obstacle_command, ObstacleVelocityCommandCfg)
  for field in fields(nominal_command):
    assert getattr(obstacle_command, field.name) == getattr(nominal_command, field.name)
  assert obstacle_command.robot_radius == 0.35
  assert obstacle_command.obstacle_radius == DEFAULT_OBSTACLE_RADIUS
  assert obstacle_command.cbf_alpha == 1.5
  assert obstacle_command.obstacle_env_fraction == 0.5
  assert obstacle_command.obstacle_attractor_env_fraction == 0.25

  assert obstacle.rewards["track_linear_velocity"].func is track_safe_linear_velocity
  assert obstacle.rewards["track_linear_velocity"].weight == 2.0
  assert obstacle.rewards["obstacle_cbf_rl"].func is obstacle_cbf_rl_reward
  assert obstacle.rewards["obstacle_cbf_rl"].weight == 5.0
  assert "obstacle_position_b" in obstacle.observations["actor"].terms

  play_command = unitree_g1_obstacle_cbf_env_cfg(
    play=True
  ).commands["twist"]
  assert play_command.obstacle_env_fraction == 1.0
  assert play_command.obstacle_attractor_env_fraction == 0.0


def test_obstacle_geometry_and_scene_compile() -> None:
  cfg = unitree_g1_obstacle_cbf_env_cfg(play=True)
  obstacle_cfg = cfg.scene.entities["obstacle"]
  assert obstacle_cfg.init_state.pos == DEFAULT_OBSTACLE_POSITION
  assert obstacle_cfg.init_state.pos[2] == DEFAULT_OBSTACLE_SIZE[2] / 2.0

  model = Scene(cfg.scene, device="cpu").compile()
  geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "obstacle/obstacle_collision",
  )
  assert geom_id >= 0
  assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
  assert tuple(model.geom_size[geom_id]) == tuple(
    dimension / 2.0 for dimension in DEFAULT_OBSTACLE_SIZE
  )


def test_obstacle_position_follows_body_and_world_commands() -> None:
  half_yaw = math.pi / 4.0
  root_pose_w = torch.tensor(
    [
      [0.0, 0.0, 0.8, math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
      [2.0, 3.0, 0.8, 1.0, 0.0, 0.0, 0.0],
      [4.0, 5.0, 0.8, 1.0, 0.0, 0.0, 0.0],
    ]
  )
  command_b = torch.tensor(
    [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  )
  command_w = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]]
  )
  positions = obstacle_positions_from_velocity_command(
    root_pose_w,
    command_b,
    command_w,
    torch.tensor([False, True, False]),
    torch.zeros(3),
    distance=1.5,
    half_height=DEFAULT_OBSTACLE_SIZE[2] / 2.0,
    min_linear_speed=1.0e-4,
  )
  assert torch.allclose(
    positions,
    torch.tensor([[0.0, 1.5, 0.65], [2.0, 1.5, 0.65], [5.5, 5.0, 0.65]]),
    atol=1.0e-6,
  )


def test_attractor_command_continually_points_at_obstacle() -> None:
  half_yaw = math.pi / 4.0
  root_pose_w = torch.tensor(
    [
      [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
      [2.0, 3.0, 0.8, math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
    ]
  )
  obstacle_pos_w = torch.tensor([[-1.5, 0.0, 0.65], [0.5, 3.0, 0.65]])
  velocity_b = planar_velocity_toward_obstacle(
    root_pose_w, obstacle_pos_w, speed=torch.tensor([0.25, 0.75])
  )
  assert torch.allclose(
    velocity_b,
    torch.tensor([[-0.25, 0.0], [0.0, 0.75]]),
    atol=1.0e-6,
  )
