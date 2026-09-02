"""Configuration for nominal and obstacle-aware Unitree G1 locomotion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from cbf_rl_humanoid.actions import (
  BODY_JOINT_NAMES,
  DefaultArmLocomotionActionCfg,
)
from cbf_rl_humanoid.commands import ObstacleVelocityCommandCfg
from cbf_rl_humanoid.metrics import (
  cbf_clearance,
  cbf_collision,
  cbf_intervention,
  cbf_violation,
)
from cbf_rl_humanoid.observations import (
  locomotion_phase_features,
  relative_obstacle_position_b,
)
from cbf_rl_humanoid.obstacle import (
  DEFAULT_OBSTACLE_RADIUS,
  DEFAULT_OBSTACLE_SIZE,
  get_box_obstacle_cfg,
)
from cbf_rl_humanoid.rewards import (
  feet_distance_lateral,
  flat_foot_l2,
  gait_phase_contact_reward,
  knee_distance_lateral,
  obstacle_cbf_rl_reward,
  stand_pose_l2,
  track_safe_linear_velocity,
)
from cbf_rl_humanoid.terminations import foot_overlap, obstacle_collision


_BODY_STD_WALKING = {
  r".*hip_pitch.*": 0.3,
  r".*hip_roll.*": 0.15,
  r".*hip_yaw.*": 0.15,
  r".*knee.*": 0.35,
  r".*ankle_pitch.*": 0.25,
  r".*ankle_roll.*": 0.1,
  r".*waist_yaw.*": 0.2,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.1,
}

_BODY_STD_RUNNING = {
  r".*hip_pitch.*": 0.5,
  r".*hip_roll.*": 0.2,
  r".*hip_yaw.*": 0.2,
  r".*knee.*": 0.6,
  r".*ankle_pitch.*": 0.35,
  r".*ankle_roll.*": 0.15,
  r".*waist_yaw.*": 0.3,
  r".*waist_roll.*": 0.08,
  r".*waist_pitch.*": 0.2,
}


def unitree_g1_nominal_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Return the nominal fixed-arm G1 locomotion environment."""

  cfg = unitree_g1_flat_env_cfg(play=play)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.resampling_time_range = (3.0, 8.0)
  command.heading_command = True
  command.heading_control_stiffness = 0.5
  command.rel_standing_envs = 0.1
  command.rel_heading_envs = 0.3
  command.rel_world_envs = 0.0
  command.rel_forward_envs = 0.2
  command.ranges.lin_vel_x = (-1.0, 1.0)
  command.ranges.lin_vel_y = (-1.0, 1.0)
  command.ranges.ang_vel_z = (-1.0, 1.0)
  command.ranges.heading = (-math.pi, math.pi)
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=velocity_mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": (-0.5, 0.5),
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.5, 0.5),
        },
        {
          "step": 5_000 * 24,
          "lin_vel_x": (-1.0, 1.0),
          "lin_vel_y": (-1.0, 1.0),
          "ang_vel_z": (-1.0, 1.0),
        },
      ],
    },
  )

  # Learn only the 15 leg/waist targets. Arms remain at the default G1 pose.
  cfg.actions["joint_pos"] = DefaultArmLocomotionActionCfg(entity_name="robot")

  source_terms = cfg.observations["actor"].terms
  actor_terms = {
    "base_ang_vel": deepcopy(source_terms["base_ang_vel"]),
    "projected_gravity": deepcopy(source_terms["projected_gravity"]),
    "command": deepcopy(source_terms["command"]),
    "joint_pos": deepcopy(source_terms["joint_pos"]),
    "joint_vel": deepcopy(source_terms["joint_vel"]),
    "actions": deepcopy(source_terms["actions"]),
    "phase": ObservationTermCfg(
      func=locomotion_phase_features,
      params={
        "command_name": "twist",
        "gait_period": 1.0,
        "gait_offset": 0.5,
      },
    ),
  }
  actor_terms["base_ang_vel"].scale = 0.25
  actor_terms["command"].scale = None
  actor_terms["joint_pos"].scale = 1.0
  actor_terms["joint_vel"].scale = 0.05
  cfg.observations["actor"].terms = actor_terms
  cfg.observations["actor"].enable_corruption = not play

  base_lin_vel = deepcopy(source_terms["base_lin_vel"])
  base_lin_vel.scale = 2.0
  base_lin_vel.noise = None
  critic_terms = {"base_lin_vel": base_lin_vel, **deepcopy(actor_terms)}
  for term in critic_terms.values():
    term.noise = None
  cfg.observations["critic"].terms = critic_terms
  cfg.observations["critic"].enable_corruption = False

  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=BODY_JOINT_NAMES
  )
  cfg.rewards["pose"].params["std_walking"] = dict(_BODY_STD_WALKING)
  cfg.rewards["pose"].params["std_running"] = dict(_BODY_STD_RUNNING)
  cfg.rewards["foot_clearance"].weight = -6.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.05
  cfg.rewards["foot_swing_height"].weight = -0.75
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.08

  feet = ("left_ankle_roll_link", "right_ankle_roll_link")
  knees_and_hips = (
    "left_knee_link",
    "left_hip_yaw_link",
    "right_knee_link",
    "right_hip_yaw_link",
  )
  cfg.rewards["stand_pose"] = RewardTermCfg(
    func=stand_pose_l2,
    weight=-5.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  cfg.rewards["flat_foot"] = RewardTermCfg(
    func=flat_foot_l2,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", body_names=feet),
    },
  )
  cfg.rewards["gait_phase_contact"] = RewardTermCfg(
    func=gait_phase_contact_reward,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "gait_period": 1.0,
      "gait_offset": 0.5,
      "stance_ratio": 0.55,
    },
  )
  cfg.rewards["feet_distance_lateral"] = RewardTermCfg(
    func=feet_distance_lateral,
    weight=0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=feet),
      "min_distance": 0.2,
      "max_distance": 0.35,
    },
  )
  cfg.rewards["knee_distance_lateral"] = RewardTermCfg(
    func=knee_distance_lateral,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=knees_and_hips),
      "min_distance": 0.2,
      "max_distance": 0.35,
    },
  )
  cfg.terminations["foot_overlap"] = TerminationTermCfg(
    func=foot_overlap,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", body_names=feet),
    },
  )

  if not play:
    cfg.events["base_mass"] = EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": (-3.0, 3.0),
      },
    )
  return cfg


def _add_obstacle_avoidance(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> ManagerBasedRlEnvCfg:
  """Add obstacle sensing and CBF-RL training without changing the user command."""

  cfg.scene.entities["obstacle"] = get_box_obstacle_cfg()
  stock_command = cfg.commands["twist"]
  assert isinstance(stock_command, UniformVelocityCommandCfg)
  stock_fields = {
    field.name: getattr(stock_command, field.name) for field in fields(stock_command)
  }
  cfg.commands["twist"] = ObstacleVelocityCommandCfg(
    **stock_fields,
    obstacle_name="obstacle",
    obstacle_distance=1.5,
    obstacle_half_height=DEFAULT_OBSTACLE_SIZE[2] / 2.0,
    robot_radius=0.35,
    obstacle_radius=DEFAULT_OBSTACLE_RADIUS,
    cbf_alpha=1.5,
    obstacle_env_fraction=1.0 if play else 0.5,
    obstacle_attractor_env_fraction=0.0 if play else 0.25,
  )

  cfg.observations["actor"].terms["obstacle_position_b"] = ObservationTermCfg(
    func=relative_obstacle_position_b,
    params={"robot_name": "robot", "obstacle_name": "obstacle"},
    noise=Unoise(n_min=-0.02, n_max=0.02),
    clip=(-3.0, 3.0),
  )
  cfg.observations["critic"].terms["obstacle_position_b"] = ObservationTermCfg(
    func=relative_obstacle_position_b,
    params={"robot_name": "robot", "obstacle_name": "obstacle"},
    clip=(-3.0, 3.0),
  )

  # The actor still observes the nominal [vx, vy, yaw] command. The closed-form
  # safe linear velocity is private to these two reward targets.
  cfg.rewards["track_linear_velocity"].func = track_safe_linear_velocity
  cfg.rewards["obstacle_cbf_rl"] = RewardTermCfg(
    func=obstacle_cbf_rl_reward,
    weight=5.0,
    params={"command_name": "twist", "sigma": 0.5},
  )
  cfg.terminations["obstacle_collision"] = TerminationTermCfg(
    func=obstacle_collision,
    params={"command_name": "twist"},
  )

  metric_params = {"command_name": "twist"}
  cfg.metrics["cbf_intervention"] = MetricsTermCfg(
    func=cbf_intervention, params=metric_params
  )
  cfg.metrics["cbf_clearance"] = MetricsTermCfg(
    func=cbf_clearance, params=metric_params
  )
  cfg.metrics["cbf_collision"] = MetricsTermCfg(
    func=cbf_collision, params=metric_params
  )
  cfg.metrics["cbf_violation"] = MetricsTermCfg(
    func=cbf_violation, params=metric_params
  )
  return cfg


def unitree_g1_obstacle_cbf_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Return G1 locomotion with obstacle observation and CBF-RL rewards."""

  return _add_obstacle_avoidance(
    unitree_g1_nominal_env_cfg(play=play),
    play=play,
  )
