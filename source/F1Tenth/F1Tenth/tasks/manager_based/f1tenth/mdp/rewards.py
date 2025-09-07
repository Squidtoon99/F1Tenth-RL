# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from isaaclab.envs import mdp as _mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

MAX_SPEED = 5.0 # m/s

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def vel_dist(env, speed_target: float = MAX_SPEED, offset: float = -(MAX_SPEED**2)):
    lin_vel = _mdp.base_lin_vel(env)
    ground_speed = torch.norm(lin_vel[..., :2], dim=-1)
    speed_dist = (ground_speed - speed_target) ** 2 + offset
    return speed_dist  # speed target

# rewards_contact.py
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def contact_duration_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,  # N; treat as "in contact" above this net normal force
    k: float = 1.0,  # scale of the penalty rate
    exponent: float = 1.0,  # 1.0 = linear ramp; 2.0 = quadratic; >1 punishes sticking harder
    debounce_steps: int = 2,  # tolerate brief breaks (contact flicker)
) -> torch.Tensor:
    # Get the sensor
    sensor = env.scene.sensors[sensor_cfg.name]  # ContactSensor
    # net normal forces per body: (num_envs, num_bodies, 3) -> magnitudes (num_envs, num_bodies)
    f_mag = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)
    # reduce across the car bodies being monitored
    f_env = f_mag.amax(dim=-1)  # (num_envs,)
    in_contact = f_env > threshold

    # --- persistent state on the env ---
    if not hasattr(env, "_contact_timer_s"):
        env._contact_timer_s = torch.zeros(env.num_envs, device=env.device)
        env._no_contact_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    # debounce small gaps so the timer doesn't reset for 1–2 missed frames
    env._no_contact_count = torch.where(
        in_contact, torch.zeros_like(env._no_contact_count), env._no_contact_count + 1
    )
    still_sticking = in_contact | (env._no_contact_count <= debounce_steps)

    # advance / reset timer
    dt = env.step_dt  # environment step size (physics_dt * decimation)
    env._contact_timer_s = torch.where(
        still_sticking,
        env._contact_timer_s + dt,
        torch.zeros_like(env._contact_timer_s),
    )

    # penalty *rate* (RewardManager multiplies by dt)
    penalty_rate = k * (env._contact_timer_s**exponent)
    return -penalty_rate
