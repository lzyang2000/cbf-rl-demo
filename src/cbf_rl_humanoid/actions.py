"""Locomotion-only action term for the Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


BODY_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)

ARM_JOINT_NAMES: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


@dataclass(kw_only=True)
class DefaultArmLocomotionActionCfg(ActionTermCfg):
  """Expose 15 body actions while holding all arm joints fixed."""

  body_joint_names: tuple[str, ...] = BODY_JOINT_NAMES
  arm_joint_names: tuple[str, ...] = ARM_JOINT_NAMES
  scale: float | dict[str, float] = field(
    default_factory=lambda: dict(G1_ACTION_SCALE)
  )
  use_default_offset: bool = True

  def build(self, env) -> DefaultArmLocomotionAction:
    return DefaultArmLocomotionAction(self, env)


class DefaultArmLocomotionAction(ActionTerm):
  """Expand 15 learned body targets to the full G1 with default arm targets.

  The arms remain at their default joint positions while the policy controls
  the legs and waist with the standard per-joint G1 action scales.
  """

  cfg: DefaultArmLocomotionActionCfg

  def __init__(self, cfg: DefaultArmLocomotionActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    body_ids, body_names = self._entity.find_joints(
      cfg.body_joint_names, preserve_order=True
    )
    arm_ids, _ = self._entity.find_joints(cfg.arm_joint_names, preserve_order=True)
    self._body_joint_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
    self._arm_joint_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
    self._joint_ids = torch.cat((self._body_joint_ids, self._arm_joint_ids))
    self._action_dim = len(body_ids)
    self._raw_actions = torch.zeros(
      self.num_envs, self._action_dim, device=self.device
    )
    default_joint_pos = self._entity.data.default_joint_pos
    assert default_joint_pos is not None
    self._body_default_pos = default_joint_pos[:, self._body_joint_ids].clone()
    self._arm_default_pos = default_joint_pos[:, self._arm_joint_ids].clone()
    self._processed_body_targets = self._body_default_pos.clone()
    self._scale = self._resolve_scale(cfg.scale, body_names)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    offset = self._body_default_pos if self.cfg.use_default_offset else 0.0
    self._processed_body_targets = offset + actions * self._scale

  def apply_actions(self) -> None:
    target = torch.cat(
      (self._processed_body_targets, self._arm_default_pos), dim=-1
    )
    encoder_bias = self._entity.data.encoder_bias[:, self._joint_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias,
      joint_ids=self._joint_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_body_targets[env_ids] = self._body_default_pos[env_ids]

  def _resolve_scale(
    self,
    scale: float | dict[str, float],
    joint_names: list[str],
  ) -> torch.Tensor:
    if isinstance(scale, (float, int)):
      return torch.full(
        (self.num_envs, self._action_dim), float(scale), device=self.device
      )
    values = torch.ones(self._action_dim, device=self.device)
    for pattern, value in scale.items():
      for idx, joint_name in enumerate(joint_names):
        if re.fullmatch(pattern, joint_name):
          values[idx] = float(value)
    return values.unsqueeze(0).repeat(self.num_envs, 1)
