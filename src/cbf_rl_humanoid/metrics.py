"""Safety metrics for the obstacle walking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

from cbf_rl_humanoid.commands import ObstacleVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _term(env: ManagerBasedRlEnv, command_name: str) -> ObstacleVelocityCommand:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, ObstacleVelocityCommand):
    raise TypeError(
      f"Command '{command_name}' must be ObstacleVelocityCommand, "
      f"got {type(term).__name__}"
    )
  return term


def cbf_intervention(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Whether the nominal user command required closed-form filtering."""

  command = _term(env, command_name)
  return (command._obstacle_active & command._cbf_intervened).float()


def cbf_clearance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Signed clearance to the circularized robot-obstacle boundary."""

  command = _term(env, command_name)
  return torch.where(
    command._obstacle_active,
    command._cbf_h,
    torch.zeros_like(command._cbf_h),
  )


def cbf_collision(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Whether circularized robot and obstacle footprints overlap."""

  command = _term(env, command_name)
  return (command._obstacle_active & (command._cbf_h < 0.0)).float()


def cbf_violation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Whether the actual planar root velocity violates the CBF condition."""

  command_term = _term(env, command_name)
  robot: Entity = env.scene[command_term.cfg.entity_name]
  actual_condition = (
    torch.sum(
      command_term._cbf_outward_gradient_w
      * robot.data.root_link_lin_vel_w[:, :2],
      dim=-1,
    )
    + command_term.cfg.cbf_alpha * command_term._cbf_h
  )
  return (command_term._obstacle_active & (actual_condition < 0.0)).float()
