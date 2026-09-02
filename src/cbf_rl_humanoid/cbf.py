"""Closed-form planar control-barrier velocity projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanarCbfState:
  """Batched state of one circular robot-to-obstacle CBF constraint."""

  h: torch.Tensor
  outward_gradient_w: torch.Tensor
  nominal_condition: torch.Tensor
  safe_velocity_w: torch.Tensor
  intervened: torch.Tensor


def planar_obstacle_cbf(
  nominal_velocity_w: torch.Tensor,
  robot_pos_w: torch.Tensor,
  obstacle_pos_w: torch.Tensor,
  *,
  robot_radius: float,
  obstacle_radius: float,
  alpha: float,
  eps: float = 1.0e-6,
) -> PlanarCbfState:
  """Project a nominal planar velocity onto one CBF half-space.

  The safe set is ``h = ||p_robot - p_obstacle|| - (r_robot +
  r_obstacle) >= 0``. Its gradient points away from the obstacle, so the CBF
  condition is ``grad(h)^T v + alpha * h >= 0``. With one constraint, the
  Euclidean projection has the closed form used here and needs no QP solver.
  """

  if nominal_velocity_w.ndim != 2 or nominal_velocity_w.shape[1] != 2:
    raise ValueError(
      "Expected nominal_velocity_w shape (N, 2), "
      f"got {nominal_velocity_w.shape}"
    )
  expected_position_shape = (nominal_velocity_w.shape[0], 2)
  if robot_pos_w.shape != expected_position_shape:
    raise ValueError(
      f"Expected robot_pos_w shape {expected_position_shape}, got {robot_pos_w.shape}"
    )
  if obstacle_pos_w.shape != expected_position_shape:
    raise ValueError(
      "Expected obstacle_pos_w shape "
      f"{expected_position_shape}, got {obstacle_pos_w.shape}"
    )
  if robot_radius <= 0.0:
    raise ValueError(f"robot_radius must be positive, got {robot_radius}")
  if obstacle_radius <= 0.0:
    raise ValueError(f"obstacle_radius must be positive, got {obstacle_radius}")
  if alpha <= 0.0:
    raise ValueError(f"alpha must be positive, got {alpha}")
  if eps <= 0.0:
    raise ValueError(f"eps must be positive, got {eps}")

  displacement_w = robot_pos_w - obstacle_pos_w
  distance = torch.linalg.vector_norm(displacement_w, dim=-1)

  # The gradient is undefined only at the obstacle center. Use the direction
  # opposite the nominal motion there, with +x as the zero-command fallback.
  outward_gradient_w = displacement_w / distance.clamp_min(eps).unsqueeze(-1)
  nominal_speed = torch.linalg.vector_norm(nominal_velocity_w, dim=-1)
  fallback = -nominal_velocity_w / nominal_speed.clamp_min(eps).unsqueeze(-1)
  zero_fallback = torch.zeros_like(fallback)
  zero_fallback[:, 0] = 1.0
  fallback = torch.where(
    (nominal_speed > eps).unsqueeze(-1), fallback, zero_fallback
  )
  outward_gradient_w = torch.where(
    (distance > eps).unsqueeze(-1), outward_gradient_w, fallback
  )

  h = distance - (robot_radius + obstacle_radius)
  nominal_condition = (
    torch.sum(outward_gradient_w * nominal_velocity_w, dim=-1) + alpha * h
  )
  correction = torch.clamp_min(-nominal_condition, 0.0).unsqueeze(-1)
  safe_velocity_w = nominal_velocity_w + correction * outward_gradient_w

  return PlanarCbfState(
    h=h,
    outward_gradient_w=outward_gradient_w,
    nominal_condition=nominal_condition,
    safe_velocity_w=safe_velocity_w,
    intervened=nominal_condition < 0.0,
  )
