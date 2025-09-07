# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

import numpy as np
from isaaclab.envs.mdp import *  # noqa: F401, F403
from scipy.interpolate import splev
from .rewards import *  # noqa: F401, F403
from isaaclab.utils import math
from isaaclab.utils.math import wrap_to_pi
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import torch

def root_euler_xyz(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root Euler XYZ angles in the environment frame."""
    xyz_tuple = math.euler_xyz_from_quat(root_quat_w(env, asset_cfg))
    return torch.stack(xyz_tuple, dim=-1)


def distance_from_start(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    centerline: list = None,
) -> torch.Tensor:
    robot_pos = root_pos_w(env, asset_cfg)[:, :2]  # (1, 2)
    centerline = torch.tensor(centerline).to(robot_pos.device)

    points = centerline - robot_pos.unsqueeze(1)  # (N, 2)
    dists = torch.linalg.vector_norm(points, dim=-1)  # (N,)
    curr_idx = torch.argmin(dists, dim=-1)  # (1,)

    return curr_idx.float() / (len(centerline) - 1)  # normalized progress [0, 1]


def track_progress(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    centerline: list = None,
) -> torch.Tensor:
    """Track progress of the robot in the environment."""
    robot_pos = root_pos_w(env, asset_cfg)
    centerline = torch.tensor(centerline).to(robot_pos.device)
    # Get the closest point on the centerline to the robot position
    points = centerline - robot_pos[:, :2].unsqueeze(1)
    distances = torch.norm(points, dim=-1)
    closest_point_index = torch.argmin(distances, dim=-1)

    # applying 2/len(centerline) to normalize the progress so there are no large jumps
    # when the robot is at the end of the track
    progress_ratio = closest_point_index / (
        len(centerline) - 1
    )  # normalized progress [0, 1]

    angle = 2 * np.pi * progress_ratio
    return torch.stack(
        [torch.cos(angle), torch.sin(angle)], dim=-1
    )  # [cos, sin] representation


def track_progress_reward(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    centerline: list = None,
):
    robot_pos = root_pos_w(env, asset_cfg)
    centerline = torch.tensor(centerline).to(robot_pos.device)
    # Get the closest point on the centerline to the robot position
    points = centerline - robot_pos[:, :2].unsqueeze(1)
    distances = torch.norm(points, dim=-1)
    closest_point_index = torch.argmin(distances, dim=-1)

    # applying 2/len(centerline) to normalize the progress so there are no large jumps
    # when the robot is at the end of the track
    progress_ratio = closest_point_index / (
        len(centerline) - 1
    )  # normalized progress [0, 1]
    return progress_ratio.float()  # normalized progress [0, 1]


def _compute_boundaries(centerline: list, width: float = 0.5) -> np.ndarray:
    """Compute the left and right boundaries of the track."""
    # Compute the tangent vectors
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    tangents = np.vstack((dx, dy)).T
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    # Compute the normal vectors
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    # Compute the left and right boundaries
    left_boundary = centerline + width / 2 * normals
    right_boundary = centerline - width / 2 * normals

    return left_boundary, right_boundary


import numpy as np
from scipy.interpolate import splev


def _compute_spline_arc_length(tck, num_points=1000):
    """
    Approximates the arc length of a 2D spline by sampling `num_points` along it.

    Returns:
        u_values: array of shape (num_points,)
        arc_lengths: array of shape (num_points,), cumulative length along the spline
    """
    # 1. Sample u values evenly across the spline
    u_values = np.linspace(0, 1, num_points)

    # 2. Get the (x, y) coordinates of each u
    x, y = splev(u_values, tck)
    points = np.stack([x, y], axis=1)  # (num_points, 2)

    # 3. Compute distance between each consecutive point
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)

    # 4. Compute cumulative arc length
    arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)  # shape: (num_points,)

    return arc_lengths


def future_track_points(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    centerline: list = None,
    track_width: float = 2.2,
) -> torch.Tensor:
    """Return 60 egocentric points ahead along the track (center/left/right).

    For each env, sample 60 points along the centerline over a ~6s horizon
    using equal arc-length spacing determined by the current speed. Compute
    left/right boundaries by offsetting the centerline with the local normal,
    then transform all points into the robot's egocentric frame.

    Output shape: (num_envs, 360) = 3 curves x 60 points x (x, y).
    """
    # Extract state
    robot_pos = root_pos_w(env, asset_cfg)[:, :2]  # (B, 2)
    lin_vel = base_lin_vel(env, asset_cfg)[:, :2]   # (B, 2)
    yaw = root_euler_xyz(env, asset_cfg)[:, 2]      # (B,)

    device = robot_pos.device
    dtype = robot_pos.dtype

    # Centerline (N, 2) on device
    cl = torch.as_tensor(centerline, device=device, dtype=dtype)
    N = cl.shape[0]

    # Polyline arc-length (N,)
    seg = cl[1:] - cl[:-1]                                # (N-1, 2)
    seg_len = torch.linalg.vector_norm(seg, dim=-1)       # (N-1,)
    cumlen = torch.cat([torch.zeros(1, device=device, dtype=dtype), torch.cumsum(seg_len, dim=0)], dim=0)  # (N,)
    total_len = cumlen[-1].clamp(min=1e-6)

    B = robot_pos.shape[0]
    S = 60
    horizon_s = 6.0

    # Find closest centerline index to each robot
    # dists: (B, N)
    dists = torch.linalg.vector_norm(cl.unsqueeze(0) - robot_pos.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)             # (B,)
    s0 = cumlen[closest_idx]                              # (B,)

    # Lookahead distance per env based on speed
    speed = torch.linalg.vector_norm(lin_vel, dim=-1)     # (B,)
    lookahead = speed * horizon_s                         # (B,)

    # Target arc-lengths ahead (exclude current point: 1..S)
    steps = torch.arange(1, S + 1, device=device, dtype=dtype) / S  # (S,)
    s_targets = s0.unsqueeze(1) + lookahead.unsqueeze(1) * steps.unsqueeze(0)  # (B, S)

    # Wrap around track (assume closed loop). For open tracks, remove the modulo
    s_targets = torch.remainder(s_targets, total_len)

    # Locate segments for each target arc-length
    # searchsorted expects ascending cumlen; returns indices in [0..N]
    seg_idx = torch.searchsorted(cumlen, s_targets, right=True) - 1  # (B, S), in [0..N-2]
    seg_idx = seg_idx.clamp(min=0, max=N - 2)

    # Gather segment endpoints
    seg_idx_flat = seg_idx.reshape(-1)                    # (B*S,)
    p0 = cl[seg_idx_flat]                                 # (B*S, 2)
    p1 = cl[(seg_idx_flat + 1)]                           # (B*S, 2)

    # Segment lengths for selected segments
    seg_len_sel = seg_len[seg_idx_flat].clamp(min=1e-8)   # (B*S,)
    s_base = cumlen[seg_idx_flat]                         # (B*S,)
    alpha = ((s_targets.reshape(-1) - s_base) / seg_len_sel).unsqueeze(-1)  # (B*S, 1)

    # Interpolated centerline points and tangents
    center_pts = p0 + alpha * (p1 - p0)                   # (B*S, 2)
    tangents = (p1 - p0) / seg_len_sel.unsqueeze(-1)      # (B*S, 2)
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)  # (B*S, 2)

    half_w = float(track_width) * 0.5
    left_pts = center_pts + half_w * normals
    right_pts = center_pts - half_w * normals

    # Reshape to (B, S, 2)
    center_pts = center_pts.view(B, S, 2)
    left_pts = left_pts.view(B, S, 2)
    right_pts = right_pts.view(B, S, 2)

    # Egocentric transform per env: translate by robot_pos, rotate by -yaw
    # x' =  cos(yaw)*(x - xr) + sin(yaw)*(y - yr)
    # y' = -sin(yaw)*(x - xr) + cos(yaw)*(y - yr)
    cos_y = torch.cos(yaw).view(B, 1, 1)
    sin_y = torch.sin(yaw).view(B, 1, 1)

    def world_to_ego(points: torch.Tensor) -> torch.Tensor:
        d = points - robot_pos.unsqueeze(1)               # (B, S, 2)
        x = d[..., 0:1]
        y = d[..., 1:2]
        x_p =  cos_y * x + sin_y * y
        y_p = -sin_y * x + cos_y * y
        return torch.cat([x_p, y_p], dim=-1)              # (B, S, 2)

    center_ego = world_to_ego(center_pts)
    left_ego = world_to_ego(left_pts)
    right_ego = world_to_ego(right_pts)

    # Concatenate curves and flatten: (B, 3, S, 2) -> (B, 360)
    all_ego = torch.stack([center_ego, left_ego, right_ego], dim=1)  # (B, 3, S, 2)
    return all_ego.reshape(B, -1)                                    # (B, 360)


def convert_to_egocentric(yaw, centerline, robot_pos, left_boundary, right_boundary):
    # Convert to egocentric frame
    rotation_matrix = torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw)],
            [torch.sin(yaw), torch.cos(yaw)],
        ],
        device=centerline.device,
        dtype=centerline.dtype,
    )
    centerline_egocentric = torch.matmul(
        centerline - robot_pos.unsqueeze(1), rotation_matrix.T
    )  # (N, 2)
    left_boundary_egocentric = torch.matmul(
        left_boundary - robot_pos.unsqueeze(1), rotation_matrix.T
    )  # (N, 2)
    right_boundary_egocentric = torch.matmul(
        right_boundary - robot_pos.unsqueeze(1), rotation_matrix.T
    )  # (N, 2)
    return torch.cat(
        [
            centerline_egocentric.unsqueeze(0),
            left_boundary_egocentric.unsqueeze(0),
            right_boundary_egocentric.unsqueeze(0),
        ],
        dim=0,
    )


def centerline_dist_reward(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    centerline: list = None,
) -> torch.Tensor:
    robot_pos = root_pos_w(env, asset_cfg)[:, :2]  # (500, 2)
    centerline = torch.tensor(centerline).to(robot_pos.device) # (862, 2)
    
    # For each robot position, compute the distance to each point on the centerline
    points = centerline - robot_pos.unsqueeze(1)  # (500, 862, 2)
    dists = torch.linalg.vector_norm(points, dim=-1)  # (500, 862)
    # Get the minimum distance for each robot position
    min_dist = torch.min(dists, dim=-1).values  # (500,)
    print(f"min_dist: {min_dist}")
    return min_dist.unsqueeze(1)  # (500, 1)

def centerline_angle(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    centerline: list = None,
) -> torch.Tensor:
    """Orientation error of the robot relative to the centerline direction.

    Concept: θ_error = θ_car − θ_track, where θ_track is the tangent direction
    of the centerline at the closest centerline point to the robot. The result
    is wrapped to [-pi, pi].

    Args:
        env (ManagerBasedEnv): RL Environment.
        asset_cfg (SceneEntityCfg, optional): Asset Config of Robot. Defaults to SceneEntityCfg("robot").
        centerline (list, optional): List of evenly distributed (x, y) points along the centerline. Defaults to None.

    Returns:
        torch.Tensor: Orientation error per env of shape (num_envs, 1).
    """

    # Robot yaw and position
    robot_yaw = root_euler_xyz(env, asset_cfg)[:, 2]  # (B,)
    robot_pos = root_pos_w(env, asset_cfg)[:, :2]  # (B, 2)

    # Centerline points on the same device/dtype
    cl = torch.as_tensor(centerline, device=robot_pos.device, dtype=robot_pos.dtype)  # (N, 2)
    B = robot_pos.shape[0]
    N = cl.shape[0]

    # Find closest centerline index for each robot
    diffs = cl.unsqueeze(0) - robot_pos.unsqueeze(1)        # (B, N, 2)
    dists = torch.linalg.vector_norm(diffs, dim=-1)         # (B, N)
    closest_idx = torch.argmin(dists, dim=-1)               # (B,)

    # Compute tangent using neighbor points around closest index
    # Use clamped neighbors to support open polylines; change to modulo for closed tracks if desired
    prev_idx = torch.clamp(closest_idx - 1, min=0)
    next_idx = torch.clamp(closest_idx + 1, max=N - 1)

    p_prev = cl[prev_idx]  # (B, 2)
    p_next = cl[next_idx]  # (B, 2)
    tangent = p_next - p_prev
    # Normalize to get direction; guard against zero-length segments
    tangent_norm = torch.clamp(torch.linalg.vector_norm(tangent, dim=-1, keepdim=True), min=1e-8)
    tangent_dir = tangent / tangent_norm

    # Track direction angle at closest point
    track_angle = torch.atan2(tangent_dir[:, 1], tangent_dir[:, 0])  # (B,)

    # Orientation error (robot heading minus track direction), wrapped to [-pi, pi]
    theta_error = wrap_to_pi(robot_yaw - track_angle)

    return theta_error.unsqueeze(1)  # (B, 1)
