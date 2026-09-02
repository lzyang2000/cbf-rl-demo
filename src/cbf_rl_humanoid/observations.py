"""Gait-phase and obstacle observations for the two walking tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def locomotion_leg_phase(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gait_period: float = 1.0,
  gait_offset: float = 0.5,
) -> torch.Tensor:
  """Return left/right gait phase, frozen in commanded standing environments."""

  if gait_period <= 0.0:
    raise ValueError(f"gait_period must be positive, got {gait_period}")
  walk_phase = torch.remainder(
    env.episode_length_buf.to(torch.float32) * env.step_dt,
    gait_period,
  ) / gait_period
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, UniformVelocityCommand):
    raise TypeError(
      f"Command '{command_name}' must be UniformVelocityCommand, "
      f"got {type(command).__name__}"
    )
  zeros = torch.zeros_like(walk_phase)
  left = torch.where(command.is_standing_env, zeros, walk_phase)
  right = torch.where(
    command.is_standing_env,
    zeros,
    torch.remainder(walk_phase + gait_offset, 1.0),
  )
  return torch.stack((left, right), dim=-1)


def locomotion_phase_features(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gait_period: float = 1.0,
  gait_offset: float = 0.5,
) -> torch.Tensor:
  """Return four left/right sinusoidal gait-clock features."""

  phase = locomotion_leg_phase(
    env,
    command_name=command_name,
    gait_period=gait_period,
    gait_offset=gait_offset,
  )
  angle = 2.0 * torch.pi * phase
  return torch.stack(
    (
      torch.sin(angle[:, 0]),
      torch.cos(angle[:, 0]),
      torch.sin(angle[:, 1]),
      torch.cos(angle[:, 1]),
    ),
    dim=-1,
  )


def relative_position_in_yaw_frame(
  robot_pose_w: torch.Tensor,
  obstacle_pos_w: torch.Tensor,
) -> torch.Tensor:
  """Return robot-to-obstacle XY position in the robot's yaw frame."""

  if robot_pose_w.ndim != 2 or robot_pose_w.shape[1] != 7:
    raise ValueError(f"Expected robot_pose_w shape (N, 7), got {robot_pose_w.shape}")
  if obstacle_pos_w.shape != (robot_pose_w.shape[0], 3):
    raise ValueError(
      f"Expected obstacle_pos_w shape {(robot_pose_w.shape[0], 3)}, "
      f"got {obstacle_pos_w.shape}"
    )
  relative_w = obstacle_pos_w - robot_pose_w[:, :3]
  relative_b = quat_apply_inverse(yaw_quat(robot_pose_w[:, 3:7]), relative_w)
  return relative_b[:, :2]


def mocap_position_w(obstacle: Entity) -> torch.Tensor:
  """Read the current mocap position without requiring another sim forward."""

  mocap_id = obstacle.indexing.mocap_id
  if mocap_id is None:
    raise ValueError("Obstacle entity must have a mocap body")
  return obstacle.data.data.mocap_pos[:, mocap_id]


def relative_obstacle_position_b(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  obstacle_name: str = "obstacle",
) -> torch.Tensor:
  """Observe planar robot-to-obstacle position in the robot's yaw frame."""

  robot: Entity = env.scene[robot_name]
  obstacle: Entity = env.scene[obstacle_name]
  return relative_position_in_yaw_frame(
    robot.data.root_link_pose_w,
    mocap_position_w(obstacle),
  )
