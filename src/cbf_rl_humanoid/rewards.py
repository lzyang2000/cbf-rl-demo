"""Gait-shaping and CBF-RL obstacle rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from cbf_rl_humanoid.commands import ObstacleVelocityCommand
from cbf_rl_humanoid.observations import locomotion_leg_phase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _obstacle_command_term(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> ObstacleVelocityCommand:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, ObstacleVelocityCommand):
    raise TypeError(
      f"Command '{command_name}' must be ObstacleVelocityCommand, "
      f"got {type(term).__name__}"
    )
  return term


def track_safe_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Stock velocity tracking with the private CBF-safe linear target."""

  asset: Entity = env.scene[asset_cfg.name]
  command_term = _obstacle_command_term(env, command_name)
  target_w = torch.zeros(
    (env.num_envs, 3),
    device=command_term._cbf_safe_velocity_w.device,
    dtype=command_term._cbf_safe_velocity_w.dtype,
  )
  target_w[:, :2] = command_term._cbf_safe_velocity_w
  target_b = quat_apply_inverse(asset.data.root_link_quat_w, target_w)

  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(target_b[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return torch.exp(-(xy_error + z_error) / std**2)


def stand_pose_l2(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Default-pose cost active only in sampled standing environments."""

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  error = torch.sum(
    torch.square(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )
  command = env.command_manager.get_term(command_name)
  standing = getattr(command, "is_standing_env", None)
  if standing is None:
    raise TypeError(f"Command '{command_name}' has no standing-environment mask")
  return error * standing.float()


def flat_foot_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize roll/pitch foot tilt while each foot is in contact."""

  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = (
    torch.linalg.vector_norm(force[..., :3], dim=-1) > contact_force_threshold
  ).float()
  foot_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  gravity = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, foot_quat.shape[1], -1)
  projected = quat_apply_inverse(
    foot_quat.reshape(-1, 4), gravity.reshape(-1, 3)
  ).reshape(env.num_envs, foot_quat.shape[1], 3)
  tilt_error = torch.sum(torch.square(projected[..., :2]), dim=-1)
  return torch.sum(tilt_error * contact, dim=-1)


def gait_phase_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str = "twist",
  gait_period: float = 1.0,
  gait_offset: float = 0.5,
  stance_ratio: float = 0.55,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Reward left/right contact agreement with the gait phase."""

  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = torch.linalg.vector_norm(force[..., :3], dim=-1) > contact_force_threshold
  expected_stance = locomotion_leg_phase(
    env,
    command_name=command_name,
    gait_period=gait_period,
    gait_offset=gait_offset,
  ) < stance_ratio
  return (~(contact ^ expected_stance)).float().sum(dim=-1)


def feet_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  """Reward keeping the two feet within a lateral-width band."""

  asset: Entity = env.scene[asset_cfg.name]
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  body_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  relative = body_pos - root_pos.unsqueeze(1)
  body_pos_b = quat_apply_inverse(
    root_quat.unsqueeze(1).expand(-1, body_pos.shape[1], -1).reshape(-1, 4),
    relative.reshape(-1, 3),
  ).reshape(env.num_envs, body_pos.shape[1], 3)
  lateral = torch.abs(body_pos_b[:, 0, 1] - body_pos_b[:, 1, 1])
  too_close = torch.clamp(lateral - min_distance, max=0.0)
  too_far = torch.clamp(-lateral + max_distance, max=0.0)
  return too_close + too_far


def knee_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  """Reward maintaining lateral knee and hip separation."""

  asset: Entity = env.scene[asset_cfg.name]
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  body_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  relative = body_pos - root_pos.unsqueeze(1)
  body_pos_b = quat_apply_inverse(
    root_quat.unsqueeze(1).expand(-1, body_pos.shape[1], -1).reshape(-1, 4),
    relative.reshape(-1, 3),
  ).reshape(env.num_envs, body_pos.shape[1], 3)
  lateral = torch.abs(body_pos_b[:, 0, 1] - body_pos_b[:, 2, 1])
  lateral += torch.abs(body_pos_b[:, 1, 1] - body_pos_b[:, 3, 1])
  too_close = torch.clamp(lateral - 2.0 * min_distance, max=0.0)
  too_far = torch.clamp(-lateral + 2.0 * max_distance, max=0.0)
  return too_close + too_far


def cbf_rl_reward_from_state(
  actual_velocity_w: torch.Tensor,
  safe_velocity_w: torch.Tensor,
  outward_gradient_w: torch.Tensor,
  h: torch.Tensor,
  *,
  alpha: float,
  sigma: float,
) -> torch.Tensor:
  """Evaluate the paper's CBF-condition and intervention reward terms."""

  if sigma <= 0.0:
    raise ValueError(f"sigma must be positive, got {sigma}")
  if alpha <= 0.0:
    raise ValueError(f"alpha must be positive, got {alpha}")
  if actual_velocity_w.shape != safe_velocity_w.shape:
    raise ValueError(
      "actual_velocity_w and safe_velocity_w must have matching shapes, got "
      f"{actual_velocity_w.shape} and {safe_velocity_w.shape}"
    )
  if outward_gradient_w.shape != actual_velocity_w.shape:
    raise ValueError(
      "outward_gradient_w must match velocity shape, got "
      f"{outward_gradient_w.shape} and {actual_velocity_w.shape}"
    )
  if h.shape != actual_velocity_w.shape[:-1]:
    raise ValueError(
      f"h must have shape {actual_velocity_w.shape[:-1]}, got {h.shape}"
    )

  actual_condition = (
    torch.sum(outward_gradient_w * actual_velocity_w, dim=-1) + alpha * h
  )
  cbf_violation = torch.minimum(actual_condition, torch.zeros_like(actual_condition))
  intervention_error = torch.sum(
    torch.square(actual_velocity_w - safe_velocity_w), dim=-1
  )
  intervention_reward = torch.exp(-intervention_error / sigma**2) - 1.0
  return cbf_violation + intervention_reward


def obstacle_cbf_rl_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.5,
) -> torch.Tensor:
  """Reward actual root velocity for satisfying and imitating the CBF target."""

  command_term = _obstacle_command_term(env, command_name)
  robot: Entity = env.scene[command_term.cfg.entity_name]
  reward = cbf_rl_reward_from_state(
    robot.data.root_link_lin_vel_w[:, :2],
    command_term._cbf_safe_velocity_w,
    command_term._cbf_outward_gradient_w,
    command_term._cbf_h,
    alpha=command_term.cfg.cbf_alpha,
    sigma=sigma,
  )
  return reward * command_term._obstacle_active
