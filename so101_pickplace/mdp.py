"""MDP terms (observations, terminations) for the SO-101 block pick-and-place and Tower of Hanoi tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def joint_pos_target(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint position targets currently commanded to the actuators [rad], shape (N, J)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos_target.torch.clone()


def ee_pose_in_base(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Grasp-point pose in the robot base frame: ``[x, y, z, qx, qy, qz, qw]`` [m], shape (N, 7)."""
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    return torch.cat((ee.data.target_pos_source.torch[:, 0, :], ee.data.target_quat_source.torch[:, 0, :]), dim=1)


def object_pose_in_base(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Object root pose in the robot base frame: ``[x, y, z, qx, qy, qz, qw]`` [m], shape (N, 7)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
    )
    return torch.cat((pos_b, quat_b), dim=1)


def gripper_opening(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gripper jaw joint position [rad], shape (N, 1)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ids, _ = robot.find_joints(["gripper"])
    return robot.data.joint_pos.torch[:, ids[0]].unsqueeze(1)


def object_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    diff_threshold: float = 0.035,
    closed_below_rad: float = 1.2,
) -> torch.Tensor:
    """True when the grasp point is at the object and the jaw is closed on it."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.linalg.vector_norm(obj.data.root_pos_w.torch - ee.data.target_pos_w.torch[:, 0, :], dim=1)
    jaw = gripper_opening(env, robot_cfg).squeeze(1)
    return torch.logical_and(dist < diff_threshold, jaw < closed_below_rad)


def cube_placed(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    xy_tol: float = 0.035,
    z_tol: float = 0.03,
    max_speed: float = 0.05,
    gripper_open_above_rad: float = 1.0,
    ee_clear_dist: float = 0.05,
) -> torch.Tensor:
    """Success: cube resting on the target pad, released (jaw open, grasp point clear of the cube)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    d = cube.data.root_pos_w.torch - target.data.root_pos_w.torch
    on_pad_xy = torch.linalg.vector_norm(d[:, :2], dim=1) < xy_tol
    on_pad_z = torch.abs(d[:, 2]) < z_tol + 0.02  # cube centre sits ~half a cube above the thin pad
    at_rest = torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch, dim=1) < max_speed
    jaw_open = gripper_opening(env, robot_cfg).squeeze(1) > gripper_open_above_rad
    ee_clear = torch.linalg.vector_norm(cube.data.root_pos_w.torch - ee.data.target_pos_w.torch[:, 0, :], dim=1) > ee_clear_dist
    return on_pad_xy & on_pad_z & at_rest & jaw_open & ee_clear


def cube_stacked(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    xy_tol: float = 0.03,
    max_speed: float = 0.05,
    gripper_open_above_rad: float = 1.0,
    ee_clear_dist: float = 0.05,
) -> torch.Tensor:
    """Success for stacking: ``cube`` resting on top of the target cube, released and clear.

    For the printable cube set (25-40 mm) the centres of a stacked pair sit ~25-80 mm apart in z;
    the 15-50 mm window accepts that while rejecting a cube merely standing on the table next to
    the target.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    d = cube.data.root_pos_w.torch - target.data.root_pos_w.torch
    on_top_xy = torch.linalg.vector_norm(d[:, :2], dim=1) < xy_tol
    on_top_z = (d[:, 2] > 0.015) & (d[:, 2] < 0.05)
    at_rest = torch.linalg.vector_norm(cube.data.root_lin_vel_w.torch, dim=1) < max_speed
    jaw_open = gripper_opening(env, robot_cfg).squeeze(1) > gripper_open_above_rad
    ee_clear = torch.linalg.vector_norm(cube.data.root_pos_w.torch - ee.data.target_pos_w.torch[:, 0, :], dim=1) > ee_clear_dist
    return on_top_xy & on_top_z & at_rest & jaw_open & ee_clear


def cube_dropped(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"), min_height: float = -0.05) -> torch.Tensor:
    """True when the cube fell off the table."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_pos_w.torch[:, 2] < min_height


def disks_on_peg(
    env: ManagerBasedRLEnv,
    disk_cfgs: list[SceneEntityCfg],
    peg_pos: tuple[float, float, float],
    xy_tol: float = 0.006,
    max_height: float = 0.04,
) -> torch.Tensor:
    """Per-disk flag ``(N, D)``: disk threaded on the peg at ``peg_pos`` (env frame, board top surface level)."""
    peg_w = env.scene.env_origins + torch.as_tensor(peg_pos, device=env.device, dtype=torch.float32)
    flags = []
    for cfg in disk_cfgs:
        d = env.scene[cfg.name].data.root_pos_w.torch - peg_w
        flags.append((torch.linalg.vector_norm(d[:, :2], dim=1) < xy_tol) & (d[:, 2] < max_height))
    return torch.stack(flags, dim=1)


def hanoi_solved(
    env: ManagerBasedRLEnv,
    disk_cfgs: list[SceneEntityCfg],
    target_peg_pos: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    max_speed: float = 0.05,
    gripper_open_above_rad: float = 1.0,
    ee_clear_dist: float = 0.05,
) -> torch.Tensor:
    """Success: every disk threaded on the target peg, at rest, jaw open and grasp point clear of the disks."""
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee.data.target_pos_w.torch[:, 0, :]
    disks = [env.scene[cfg.name].data for cfg in disk_cfgs]
    all_on = disks_on_peg(env, disk_cfgs, target_peg_pos).all(dim=1)
    at_rest = torch.stack([torch.linalg.vector_norm(d.root_lin_vel_w.torch, dim=1) < max_speed for d in disks], dim=1).all(dim=1)
    ee_clear = torch.stack([torch.linalg.vector_norm(d.root_pos_w.torch - ee_pos, dim=1) > ee_clear_dist for d in disks], dim=1).all(dim=1)
    jaw_open = gripper_opening(env, robot_cfg).squeeze(1) > gripper_open_above_rad
    return all_on & at_rest & ee_clear & jaw_open


def disk_dropped(env: ManagerBasedRLEnv, disk_cfgs: list[SceneEntityCfg], min_height: float = -0.05) -> torch.Tensor:
    """True when any disk fell off the table."""
    return torch.stack([env.scene[cfg.name].data.root_pos_w.torch[:, 2] < min_height for cfg in disk_cfgs], dim=1).any(dim=1)
