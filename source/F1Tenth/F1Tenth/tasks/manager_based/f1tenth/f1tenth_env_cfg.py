# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import numpy as np
from scipy.interpolate import splprep, splev
import torch


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
# from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.markers import VisualizationMarkersCfg
from . import mdp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.assets import SceneEntityCfg

##
# Pre-defined configs
##

from wheeledlab_assets.f1tenth import F1TENTH_CFG
from wheeledlab_tasks.common import F1Tenth4WDActionCfg
from wheeledlab_tasks.drifting.f1tenth_drift_env_cfg import (
    disable_all_lidars,
)

##
# Scene definition
##

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

centerline_path = os.path.join(
    WORKSPACE_ROOT, "custom_assets", "centerline.csv"
)

centerline = np.loadtxt(centerline_path, delimiter=",", skiprows=1)

# offset from 0,0 to q1 positioning
centerline += np.array([10.68704, -72.58866])
boundary = mdp._compute_boundaries(centerline, width=0.5)

c_tck, u_dense = splprep(
    [centerline[:, 0], centerline[:, 1]],
    s=0.0,
)

# Convert c_tck from numpy to native
c_tck = (
    c_tck[0].tolist(),
    [c.tolist() for c in c_tck[1]],
    c_tck[2],
)
# Remapping types to native for omega
u_dense = u_dense.tolist()
centerline = centerline.tolist()


# @configclass
# class RaceTrackTerrainImporterCfg(TerrainImporterCfg):
#     prim_path = "/World/track"
#     terrain_type = "usd"
#     usd_path = os.path.join(WORKSPACE_ROOT, "custom_assets", "Track.usd")
#     collision_group = -1
#     physics_material = sim_utils.RigidBodyMaterialCfg(
#         friction_combine_mode="multiply",
#         restitution_combine_mode="multiply",
#         static_friction=1.0,
#         dynamic_friction=1.0,
#     )
#     debug_vis = True


@configclass
class F1tenthSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
        collision_group=-1,
    )

    track = AssetBaseCfg(
        prim_path="/World/track",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(WORKSPACE_ROOT, "custom_assets", "Track-trimmed.usd"),
        ),
    )

    # robot
    robot: ArticulationCfg = F1TENTH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["/World/track/.*"],  # only wall contacts count
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    # imu = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Imu_Sensor", gravity_bias=(0, 0, 0)
    # )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # # Displaying visual cones along the centerline for debugging

    # boundary_cone_1 = AssetBaseCfg(
    #     prim_path="/World/boundary_cone_1",
    #     spawn=sim_utils.ConeCfg(
    #         radius=0.05,
    #         height=0.2,
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 1.0, 0.0)
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(centerline[0][0], centerline[0][1], 0.1),
    #     )
    # )
    # for idx in range(0, len(centerline), 5):
    #     locals()[f"boundary_cone_{idx}"] = AssetBaseCfg(
    #         prim_path=f"/World/boundary_cone_{idx}",
    #         spawn=sim_utils.ConeCfg(
    #             radius=0.05,
    #             height=0.2,
    #             visual_material=sim_utils.PreviewSurfaceCfg(
    #                 diffuse_color=(0.0, 1.0, 0.0)
    #             ),
    #         ),
    #         init_state=AssetBaseCfg.InitialStateCfg(
    #             pos=(centerline[idx][0], centerline[idx][1], 0.1),
    #         )
    #     )
    
    # del idx
    
    def __post_init__(self) -> None:
        """Post initialization."""
        # robot
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(
            pos=(36.68704, -90.58866, 0.0),
            #      Quaternion rotation (w, x, y, z) of the root in simulation world frame. Defaults to (1.0, 0.0, 0.0, 0.0).
            rot=(0.9238795, 0.0, 0.0, -0.3826834),  # -45 degrees around Z
        )


##
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)

        #
        #   VEHICLE STATE
        #

        # Removed b/c all positions are relative
        # root_pos_w_term = ObsTerm(
        #     func=mdp.root_pos_w,
        #     noise=Gnoise(mean=0.0, std=0.1),
        # )

        # Removed b/c all positions are relative
        # root_euler_xyz_term = ObsTerm(
        #     func=mdp.root_euler_xyz,
        #     noise=Gnoise(mean=0.0, std=0.1),
        # )

        # base_lin_vel_term = ObsTerm(
        #     func=mdp.base_lin_vel,
        #     noise=Gnoise(mean=0.0, std=0.5),
        # )

        # base_ang_vel_term = ObsTerm(
        #     func=mdp.base_ang_vel,
        #     noise=Gnoise(std=0.4),
        # )

        # # TODO: Implement the IMU
        # # base_lin_acc_term = ObsTerm(
        # #     func=mdp.imu_lin_acc,
        # #     noise=Gnoise(std=0.5),
        # # )

        # # TODO: See if we need this?
        # last_action_term = ObsTerm(
        #     func=mdp.last_action,
        #     clip=(-1.0, 1.0),  # TODO: get from ClipAction wrapper or action space
        # )

        # track_progress_term = ObsTerm(
        #     func=mdp.track_progress,
        #     params={"centerline": centerline},
        # )

        centerline_angle_term = ObsTerm(
            func=mdp.centerline_angle,
            params={"centerline": centerline},
        )

        #
        #   TRACK INFORMATION
        #

        #
        # future_track_points = ObsTerm(
        #     func=mdp.future_track_points,
        #     params={"centerline": centerline},
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.1, 0.2),
                "y": (0.1, 0.2),
            },
        },
    )

    kill_lidar = EventTerm(func=disable_all_lidars, mode="startup", params={})


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Progress reward
    # progress = RewTerm(
    #     func=mdp.track_progress_reward,
    #     params={
    #         "centerline": centerline,
    #     },
    #     weight=1.0,
    # )
    # (3) Speed reward
    speed = RewTerm(
        func=mdp.vel_dist,
        weight=-5.0,
    )
    # (4) Centerline distance penalty
    # centerline_distance = RewTerm(
    #     func=mdp.centerline_dist_reward,
    #     params={
    #         "centerline": centerline,
    #     },
    #     weight=-0.5,
    # )

    # (5) Collision penalty
    collision = RewTerm(
        func=mdp.contact_forces,
        params={
            "threshold": 200.0,
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
        weight=-10.0,
    )

    wall_stick = RewTerm(
        func=mdp.contact_duration_penalty,
        weight=1.0,  # RewardManager multiplies "value * weight * dt"
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 50.0,  # N (tune)
            "k": 2.0,  # scale (tune)
            "exponent": 1.5,  # ramp severity (tune)
            "debounce_steps": 2,  # tolerate brief flicker
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Collision
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 800.0,
        },
    )

    not_moving = DoneTerm(
        func=mdp.not_moving,
        params={
            "speed_threshold": 0.05,
            "time_threshold": 1.0,
        },
    )

    # illegal_contact_duration = DoneTerm(
    #     func=mdp.illegal_contact_duration,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces"),
    #         "threshold": 500.0,
    #     },
    # )


##
# Environment configuration
##


@configclass
class F1tenthEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: F1tenthSceneCfg = F1tenthSceneCfg(num_envs=4096, env_spacing=0.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: F1Tenth4WDActionCfg = F1Tenth4WDActionCfg()  # 4WD throttle/steer actions
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
