"""Termination conditions for obstacle avoidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from cbf_rl_humanoid.commands import ObstacleVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def obstacle_collision(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Terminate when the circularized robot and obstacle footprints overlap."""

  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ObstacleVelocityCommand):
    raise TypeError(
      f"Command '{command_name}' must be ObstacleVelocityCommand, "
      f"got {type(command).__name__}"
    )
  return command._obstacle_active & (command._cbf_h < 0.0)


def foot_overlap(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  """Terminate crossed-foot states during double support."""

  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = torch.linalg.vector_norm(force[..., :3], dim=-1) > contact_force_threshold
  both_in_contact = torch.all(contact, dim=-1)
  foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  relative = foot_pos - asset.data.root_link_pos_w.unsqueeze(1)
  foot_pos_b = quat_apply_inverse(
    asset.data.root_link_quat_w.unsqueeze(1)
    .expand(-1, foot_pos.shape[1], -1)
    .reshape(-1, 4),
    relative.reshape(-1, 3),
  ).reshape(env.num_envs, foot_pos.shape[1], 3)
  lateral = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
  return (lateral < threshold) & both_in_contact
