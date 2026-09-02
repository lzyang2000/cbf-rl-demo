"""Pure tensor checks for the closed-form obstacle CBF and reward."""

from __future__ import annotations

import math

import torch

from cbf_rl_humanoid.cbf import planar_obstacle_cbf
from cbf_rl_humanoid.observations import relative_position_in_yaw_frame
from cbf_rl_humanoid.rewards import cbf_rl_reward_from_state


def _state(nominal_velocity_w: torch.Tensor):
  return planar_obstacle_cbf(
    nominal_velocity_w,
    nominal_velocity_w.new_tensor([[0.0, 0.0]]),
    nominal_velocity_w.new_tensor([[1.0, 0.0]]),
    robot_radius=0.3,
    obstacle_radius=0.2,
    alpha=1.0,
  )


def test_moving_away_does_not_filter_nominal_velocity() -> None:
  nominal = torch.tensor([[-1.0, 0.0]])
  state = _state(nominal)

  assert state.h.item() == 0.5
  assert torch.equal(state.outward_gradient_w, torch.tensor([[-1.0, 0.0]]))
  assert not state.intervened.item()
  assert torch.equal(state.safe_velocity_w, nominal)


def test_driving_toward_obstacle_uses_closed_form_projection() -> None:
  nominal = torch.tensor([[1.0, 0.0]])
  nominal_before = nominal.clone()
  state = _state(nominal)

  assert torch.equal(nominal, nominal_before)
  assert state.intervened.item()
  assert torch.allclose(state.safe_velocity_w, torch.tensor([[0.5, 0.0]]))
  safe_condition = (
    torch.sum(state.outward_gradient_w * state.safe_velocity_w, dim=-1)
    + state.h
  )
  assert torch.all(safe_condition >= -1.0e-7)


def test_projection_preserves_tangential_component_and_dtype() -> None:
  nominal = torch.tensor([[1.0, 0.7]], dtype=torch.float64)
  state = _state(nominal)

  assert state.safe_velocity_w.dtype == torch.float64
  assert torch.allclose(
    state.safe_velocity_w,
    torch.tensor([[0.5, 0.7]], dtype=torch.float64),
  )


def test_cbf_reward_matches_condition_and_intervention_terms() -> None:
  actual = torch.tensor([[0.8, 0.1], [-0.2, 0.0]])
  safe = torch.tensor([[0.5, 0.0], [-0.2, 0.0]])
  gradient = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])
  h = torch.tensor([0.5, 0.5])
  sigma = 0.5

  reward = cbf_rl_reward_from_state(
    actual,
    safe,
    gradient,
    h,
    alpha=1.0,
    sigma=sigma,
  )
  condition = torch.tensor([-0.3, 0.7])
  squared_error = torch.tensor([0.1, 0.0])
  expected = torch.minimum(condition, torch.zeros_like(condition)) + torch.exp(
    -squared_error / sigma**2
  ) - 1.0
  assert torch.allclose(reward, expected)


def test_relative_obstacle_position_uses_robot_yaw_frame() -> None:
  half_yaw = math.pi / 4.0
  robot_pose_w = torch.tensor(
    [[2.0, 3.0, 0.8, math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]]
  )
  obstacle_pos_w = torch.tensor([[2.0, 4.5, 0.65]])

  relative_b = relative_position_in_yaw_frame(robot_pose_w, obstacle_pos_w)

  assert torch.allclose(relative_b, torch.tensor([[1.5, 0.0]]), atol=1.0e-6)
