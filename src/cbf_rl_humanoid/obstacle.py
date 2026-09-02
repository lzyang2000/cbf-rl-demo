"""Obstacle assets for the MJLab walking task."""

from __future__ import annotations

import math

import mujoco

from mjlab.entity import EntityCfg

DEFAULT_OBSTACLE_POSITION = (1.5, 0.0, 0.65)
"""Obstacle center in the robot's initial frame (+x is forward)."""

DEFAULT_OBSTACLE_SIZE = (0.3, 0.3, 1.3)
"""Full obstacle dimensions in meters (length, width, height)."""

DEFAULT_OBSTACLE_RADIUS = math.hypot(
  DEFAULT_OBSTACLE_SIZE[0] / 2.0,
  DEFAULT_OBSTACLE_SIZE[1] / 2.0,
)
"""Conservative circumscribed radius of the square obstacle footprint."""

DEFAULT_OBSTACLE_RGBA = (0.9, 0.35, 0.08, 1.0)


def _make_box_spec(
  size: tuple[float, float, float],
  rgba: tuple[float, float, float, float],
):
  """Return a spec factory for a fixed, collidable box."""

  if any(dimension <= 0.0 for dimension in size):
    raise ValueError(f"Obstacle dimensions must be positive, got {size}")

  half_size = tuple(dimension / 2.0 for dimension in size)

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="obstacle")
    geom = body.add_geom(name="obstacle_collision")
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = half_size
    geom.rgba = rgba
    # Match the default terrain's high-friction contact behavior. MuJoCo's
    # default collision masks remain enabled, so this contacts both robot and floor.
    geom.friction = (1.0, 0.005, 0.0001)
    return spec

  return spec_fn


def get_box_obstacle_cfg(
  *,
  position: tuple[float, float, float] = DEFAULT_OBSTACLE_POSITION,
  size: tuple[float, float, float] = DEFAULT_OBSTACLE_SIZE,
  rgba: tuple[float, float, float, float] = DEFAULT_OBSTACLE_RGBA,
) -> EntityCfg:
  """Create the fixed box placed directly along the robot's forward path."""

  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=position),
    spec_fn=_make_box_spec(size, rgba),
  )
