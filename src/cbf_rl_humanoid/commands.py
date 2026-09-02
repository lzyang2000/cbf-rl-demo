"""Velocity command that places an obstacle along the commanded path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from cbf_rl_humanoid.cbf import planar_obstacle_cbf
from cbf_rl_humanoid.observations import mocap_position_w


def obstacle_positions_from_velocity_command(
  root_pose_w: torch.Tensor,
  vel_command_b: torch.Tensor,
  vel_command_w: torch.Tensor,
  is_world_env: torch.Tensor,
  ground_z: torch.Tensor,
  *,
  distance: float,
  half_height: float,
  min_linear_speed: float,
) -> torch.Tensor:
  """Compute obstacle centers along each robot's commanded travel direction.

  Body-frame commands are rotated into the world using the current robot pose.
  MJLab's world-frame command environments already expose their world direction
  through ``vel_command_w``. A zero linear command has no path, so the obstacle
  falls back to the robot's current forward direction.
  """

  if root_pose_w.ndim != 2 or root_pose_w.shape[1] != 7:
    raise ValueError(f"Expected root_pose_w shape (N, 7), got {root_pose_w.shape}")
  if vel_command_b.shape != (root_pose_w.shape[0], 3):
    raise ValueError(
      f"Expected vel_command_b shape {(root_pose_w.shape[0], 3)}, "
      f"got {vel_command_b.shape}"
    )
  if vel_command_w.shape != vel_command_b.shape:
    raise ValueError(
      f"Expected vel_command_w shape {vel_command_b.shape}, got {vel_command_w.shape}"
    )
  if distance <= 0.0:
    raise ValueError(f"Obstacle distance must be positive, got {distance}")

  # Velocity commands are planar heading-frame quantities. Ignore transient
  # roll/pitch so both obstacle placement and the CBF use the same XY path.
  root_quat_w = yaw_quat(root_pose_w[:, 3:7])
  command_3d_b = torch.cat(
    (vel_command_b[:, :2], torch.zeros_like(vel_command_b[:, :1])), dim=-1
  )
  command_3d_w = quat_apply(root_quat_w, command_3d_b)
  command_xy_w = command_3d_w[:, :2]
  command_xy_w = torch.where(
    is_world_env[:, None], vel_command_w[:, :2], command_xy_w
  )

  forward_b = torch.zeros_like(command_3d_b)
  forward_b[:, 0] = 1.0
  forward_xy_w = quat_apply(root_quat_w, forward_b)[:, :2]
  forward_xy_w = forward_xy_w / torch.linalg.vector_norm(
    forward_xy_w, dim=-1, keepdim=True
  ).clamp_min(1.0e-6)

  linear_speed = torch.linalg.vector_norm(command_xy_w, dim=-1, keepdim=True)
  command_direction = command_xy_w / linear_speed.clamp_min(min_linear_speed)
  direction_xy_w = torch.where(
    linear_speed > min_linear_speed,
    command_direction,
    forward_xy_w,
  )

  obstacle_pos_w = root_pose_w[:, :3].clone()
  obstacle_pos_w[:, :2] += distance * direction_xy_w
  obstacle_pos_w[:, 2] = ground_z + half_height
  return obstacle_pos_w


def planar_velocity_toward_obstacle(
  root_pose_w: torch.Tensor,
  obstacle_pos_w: torch.Tensor,
  *,
  speed: float | torch.Tensor,
) -> torch.Tensor:
  """Return a body-frame XY command aimed at a fixed world-frame obstacle."""

  if root_pose_w.ndim != 2 or root_pose_w.shape[1] != 7:
    raise ValueError(f"Expected root_pose_w shape (N, 7), got {root_pose_w.shape}")
  if obstacle_pos_w.shape != (root_pose_w.shape[0], 3):
    raise ValueError(
      f"Expected obstacle_pos_w shape {(root_pose_w.shape[0], 3)}, "
      f"got {obstacle_pos_w.shape}"
    )
  if isinstance(speed, torch.Tensor):
    if speed.shape not in ((root_pose_w.shape[0],), (root_pose_w.shape[0], 1)):
      raise ValueError(
        "Expected attractor speed shape "
        f"{(root_pose_w.shape[0],)} or {(root_pose_w.shape[0], 1)}, "
        f"got {speed.shape}"
      )
    if torch.any(speed <= 0.0):
      raise ValueError("Attractor speeds must be positive")
    speed_column = speed.reshape(-1, 1)
  else:
    if speed <= 0.0:
      raise ValueError(f"Attractor speed must be positive, got {speed}")
    speed_column = speed

  relative_w = obstacle_pos_w - root_pose_w[:, :3]
  relative_b = quat_apply_inverse(yaw_quat(root_pose_w[:, 3:7]), relative_w)
  direction_b = relative_b[:, :2] / torch.linalg.vector_norm(
    relative_b[:, :2], dim=-1, keepdim=True
  ).clamp_min(1.0e-6)
  return speed_column * direction_b


class ObstacleVelocityCommand(UniformVelocityCommand):
  """Stock velocity sampler plus command-conditioned obstacle placement."""

  cfg: "ObstacleVelocityCommandCfg"

  def __init__(self, cfg: "ObstacleVelocityCommandCfg", env):
    super().__init__(cfg, env)
    self.obstacle: Entity = env.scene[cfg.obstacle_name]
    if not self.obstacle.is_fixed_base or not self.obstacle.is_mocap:
      raise ValueError(
        f"Obstacle '{cfg.obstacle_name}' must be a fixed-base mocap entity"
      )
    if not 0.0 <= cfg.obstacle_env_fraction <= 1.0:
      raise ValueError(
        "obstacle_env_fraction must be in [0, 1], got "
        f"{cfg.obstacle_env_fraction}"
      )
    if not 0.0 <= cfg.obstacle_attractor_env_fraction <= cfg.obstacle_env_fraction:
      raise ValueError(
        "obstacle_attractor_env_fraction must be in [0, "
        "obstacle_env_fraction], got "
        f"{cfg.obstacle_attractor_env_fraction} > {cfg.obstacle_env_fraction}"
      )
    if cfg.inactive_obstacle_distance <= cfg.obstacle_distance:
      raise ValueError(
        "inactive_obstacle_distance must exceed obstacle_distance, got "
        f"{cfg.inactive_obstacle_distance} <= {cfg.obstacle_distance}"
      )
    if cfg.inactive_obstacle_depth <= cfg.obstacle_half_height:
      raise ValueError(
        "inactive_obstacle_depth must exceed obstacle_half_height, got "
        f"{cfg.inactive_obstacle_depth} <= {cfg.obstacle_half_height}"
      )
    if cfg.adversarial_obstacle_direction_b is not None:
      direction = torch.tensor(cfg.adversarial_obstacle_direction_b)
      if torch.linalg.vector_norm(direction).item() <= 0.0:
        raise ValueError("adversarial_obstacle_direction_b must be nonzero")
      if cfg.adversarial_speed <= 0.0:
        raise ValueError(
          f"adversarial_speed must be positive, got {cfg.adversarial_speed}"
        )
    self._obstacle_active = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._obstacle_attractor = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._obstacle_attractor_speed = torch.ones(
      self.num_envs, device=self.device, dtype=self.vel_command_b.dtype
    )
    self._obstacle_reposition_pending = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._cbf_h = torch.zeros(self.num_envs, device=self.device)
    self._cbf_outward_gradient_w = torch.zeros(
      (self.num_envs, 2), device=self.device
    )
    self._cbf_nominal_condition = torch.zeros(self.num_envs, device=self.device)
    self._cbf_safe_velocity_w = torch.zeros(
      (self.num_envs, 2), device=self.device
    )
    self._cbf_intervened = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._obstacle_attractor_speed[env_ids] = torch.linalg.vector_norm(
      self.vel_command_b[env_ids, :2], dim=-1
    ).clamp_min(self.cfg.min_linear_speed)
    self._obstacle_attractor[env_ids] = False
    fraction = self.cfg.obstacle_env_fraction
    if fraction <= 0.0:
      self._obstacle_active[env_ids] = False
    elif fraction >= 1.0:
      self._obstacle_active[env_ids] = True
    else:
      self._obstacle_active[env_ids] = (
        torch.rand(len(env_ids), device=self.device) < fraction
      )
    attractor_fraction = self.cfg.obstacle_attractor_env_fraction
    if attractor_fraction > 0.0:
      # This fraction is expressed over the full environment population. Draw
      # only from active-obstacle environments so, for example, 0.25 attractor
      # plus 0.25 stock-command obstacle gives the desired 50% obstacle split.
      conditional_fraction = attractor_fraction / fraction
      self._obstacle_attractor[env_ids] = self._obstacle_active[env_ids] & (
        torch.rand(len(env_ids), device=self.device) < conditional_fraction
      )
    if self.cfg.adversarial_obstacle_direction_b is not None:
      direction_b = torch.tensor(
        self.cfg.adversarial_obstacle_direction_b,
        device=self.device,
        dtype=self.vel_command_b.dtype,
      )
      direction_b /= torch.linalg.vector_norm(direction_b)
      self.vel_command_b[env_ids] = 0.0
      self.vel_command_b[env_ids, :2] = self.cfg.adversarial_speed * direction_b
      self._obstacle_attractor_speed[env_ids] = self.cfg.adversarial_speed
      self.vel_command_w[env_ids] = 0.0
      self.is_heading_env[env_ids] = False
      self.is_standing_env[env_ids] = False
      self.is_world_env[env_ids] = False
      self.is_forward_env[env_ids] = False
      self._obstacle_active[env_ids] = True
      self._obstacle_attractor[env_ids] = True
    attractor_ids = env_ids[self._obstacle_attractor[env_ids]]
    if len(attractor_ids) > 0:
      # Re-aim only planar translation. A zero yaw target prevents the
      # attractor cohort from becoming an accidental turning curriculum.
      self.vel_command_b[attractor_ids, 2] = 0.0
      self.vel_command_w[attractor_ids] = 0.0
      self.is_heading_env[attractor_ids] = False
      self.is_standing_env[attractor_ids] = False
      self.is_world_env[attractor_ids] = False
      self.is_forward_env[attractor_ids] = False
    # Defer placement until compute(), which runs after sim.forward(). This
    # makes the robot pose current even when reset events just changed it.
    self._obstacle_reposition_pending[env_ids] = True

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)

    if env_ids is None:
      selected_ids = torch.arange(self.num_envs, device=self.device)
    else:
      selected_ids = env_ids
    attractor_ids = selected_ids[self._obstacle_attractor[selected_ids]]
    if len(attractor_ids) == 0:
      return
    self.time_left[attractor_ids] = torch.inf
    self.is_heading_env[attractor_ids] = False
    self.is_standing_env[attractor_ids] = False
    self.is_world_env[attractor_ids] = False
    self.is_forward_env[attractor_ids] = False

    # During reset, retain the configured direction until compute() has placed
    # the obstacle. On every later step, aim the nominal command at its fixed
    # world position without moving the obstacle itself.
    tracking_ids = attractor_ids[
      ~self._obstacle_reposition_pending[attractor_ids]
    ]
    if len(tracking_ids) == 0:
      return
    velocity_b = planar_velocity_toward_obstacle(
      self.robot.data.root_link_pose_w[tracking_ids],
      mocap_position_w(self.obstacle)[tracking_ids],
      speed=self._obstacle_attractor_speed[tracking_ids],
    )
    self.vel_command_b[tracking_ids] = 0.0
    self.vel_command_b[tracking_ids, :2] = velocity_b
    self.vel_command_w[tracking_ids] = 0.0

  def compute(
    self,
    dt: float | torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Advance commands and refresh obstacle/CBF state.

    MJLab 1.6 passes ``env_ids`` when computing commands during a scoped
    reset. The obstacle placement remains driven by the pending resamples,
    while the closed-form CBF buffers are pure functions of current state and
    can safely be refreshed for the full batch.
    """

    super().compute(dt, env_ids)
    env_ids = self._obstacle_reposition_pending.nonzero(as_tuple=False).flatten()
    if len(env_ids) > 0:
      obstacle_pos_w = obstacle_positions_from_velocity_command(
        self.robot.data.root_link_pose_w[env_ids],
        self.vel_command_b[env_ids],
        self.vel_command_w[env_ids],
        self.is_world_env[env_ids],
        self._env.scene.env_origins[env_ids, 2],
        distance=self.cfg.obstacle_distance,
        half_height=self.cfg.obstacle_half_height,
        min_linear_speed=self.cfg.min_linear_speed,
      )
      active = self._obstacle_active[env_ids]
      inactive = ~active
      if inactive.any():
        root_pos_w = self.robot.data.root_link_pose_w[env_ids, :3]
        direction_w = obstacle_pos_w[:, :2] - root_pos_w[:, :2]
        direction_w /= torch.linalg.vector_norm(
          direction_w, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        obstacle_pos_w[inactive, :2] = (
          root_pos_w[inactive, :2]
          + self.cfg.inactive_obstacle_distance * direction_w[inactive]
        )
        obstacle_pos_w[inactive, 2] = (
          self._env.scene.env_origins[env_ids[inactive], 2]
          - self.cfg.inactive_obstacle_depth
        )
      obstacle_pose_w = torch.zeros(
        (len(env_ids), 7), device=self.device, dtype=obstacle_pos_w.dtype
      )
      obstacle_pose_w[:, :3] = obstacle_pos_w
      obstacle_pose_w[:, 3] = 1.0
      self.obstacle.write_mocap_pose_to_sim(obstacle_pose_w, env_ids=env_ids)
      self._obstacle_reposition_pending[env_ids] = False

    # The command property stays nominal. The filtered target is private and is
    # refreshed separately for the tracking reward and CBF reward/metrics.
    root_pose_w = self.robot.data.root_link_pose_w
    nominal_velocity_3d_b = torch.zeros(
      (self.num_envs, 3), device=self.device, dtype=self.vel_command_b.dtype
    )
    nominal_velocity_3d_b[:, :2] = self.vel_command_b[:, :2]
    nominal_velocity_w = quat_apply(
      yaw_quat(root_pose_w[:, 3:7]), nominal_velocity_3d_b
    )[:, :2]
    cbf_state = planar_obstacle_cbf(
      nominal_velocity_w,
      root_pose_w[:, :2],
      mocap_position_w(self.obstacle)[:, :2],
      robot_radius=self.cfg.robot_radius,
      obstacle_radius=self.cfg.obstacle_radius,
      alpha=self.cfg.cbf_alpha,
    )
    self._cbf_h.copy_(cbf_state.h)
    self._cbf_outward_gradient_w.copy_(cbf_state.outward_gradient_w)
    self._cbf_nominal_condition.copy_(cbf_state.nominal_condition)
    self._cbf_safe_velocity_w.copy_(cbf_state.safe_velocity_w)
    self._cbf_intervened.copy_(cbf_state.intervened)
    inactive = ~self._obstacle_active
    self._cbf_safe_velocity_w[inactive] = nominal_velocity_w[inactive]
    self._cbf_intervened[inactive] = False


@dataclass(kw_only=True)
class ObstacleVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for command-conditioned obstacle placement."""

  obstacle_name: str = "obstacle"
  obstacle_distance: float = 1.5
  obstacle_half_height: float = 0.65
  min_linear_speed: float = 1.0e-4
  robot_radius: float = 0.35
  obstacle_radius: float = 0.21213203435596426
  cbf_alpha: float = 1.5
  obstacle_env_fraction: float = 1.0
  obstacle_attractor_env_fraction: float = 0.0
  inactive_obstacle_distance: float = 10.0
  inactive_obstacle_depth: float = 10.0
  adversarial_obstacle_direction_b: tuple[float, float] | None = None
  adversarial_speed: float = 1.0

  def build(self, env) -> ObstacleVelocityCommand:
    return ObstacleVelocityCommand(self, env)
