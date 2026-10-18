"""SO-101 Tower of Hanoi scene: the cube/pad of the pick-and-place task replaced by a 3-peg board.

Same robot, table, front camera and joint-space action as :class:`SO101PickPlaceJointEnvCfg`. The
board (``assets/Tower_of_Hanoi/peg_board_3disks.usda``, static collider) sits in front of the robot
with its pegs in a row along env Y; the three disks (44 / 36 / 28 mm, largest at the bottom) start
stacked on the start peg. Pegs are 8 mm diameter and 40 mm tall, the disk holes 12 mm. The disks are
spawned from ``assets/Tower_of_Hanoi/sim/`` (``scripts/make_hanoi_collision.py``): the original USDs with
compound colliders that keep the hole open, a grippy outer ring and a slippery bore lining.

Seen from the front camera, +Y is image right, so peg 0 (Y = -50 mm) is the left peg.
"""

from __future__ import annotations

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.stack.mdp import stack_events

from . import mdp
from .pickplace_env_cfg import ASSETS_DIR, PickPlaceSceneCfg, SO101PickPlaceIKEnvCfg, SO101PickPlaceJointEnvCfg

HANOI_DIR = os.path.join(ASSETS_DIR, "Tower_of_Hanoi")

# Board origin (centre of its underside) in the env frame; the table top is at z = 0.
HANOI_BOARD_POS = (0.22, 0.0, 0.0)
BOARD_THICKNESS = 0.005
PEG_Y = (-0.05, 0.0, 0.05)
PEG_POS = tuple((HANOI_BOARD_POS[0], HANOI_BOARD_POS[1] + y, HANOI_BOARD_POS[2] + BOARD_THICKNESS) for y in PEG_Y)
START_PEG = 0
TARGET_PEG = 2

DISK_THICKNESS = 0.008
DISK_RADII = (0.022, 0.018, 0.014)
# Scene names, largest first (bottom of the stack); disk USD origins are at their bottom face.
DISKS = (("disk_large", "disk_44mm_blue"), ("disk_medium", "disk_36mm_green"), ("disk_small", "disk_28mm_red"))
DISK_CFGS = [SceneEntityCfg(name) for name, _ in DISKS]

HANOI_TASK = "move all the disks from the left peg to the right peg"
HANOI_SUCCESS_PARAMS = {"disk_cfgs": DISK_CFGS, "target_peg_pos": PEG_POS[TARGET_PEG]}
HANOI_DROP_PARAMS = {"disk_cfgs": DISK_CFGS}


def _disk_cfg(index: int, usd_name: str) -> RigidObjectCfg:
    x, y, z = PEG_POS[START_PEG]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{DISKS[index][0].title().replace('_', '')}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[x, y, z + index * DISK_THICKNESS + 0.0005], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(HANOI_DIR, "sim", f"{usd_name}.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=8,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            semantic_tags=[("class", "disk")],
        ),
    )


@configclass
class HanoiSceneCfg(PickPlaceSceneCfg):
    """Table, SO-101, Tower of Hanoi board with three disks, front camera."""

    cube: RigidObjectCfg | None = None
    target: RigidObjectCfg | None = None

    board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HanoiBoard",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(HANOI_BOARD_POS), rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(usd_path=os.path.join(HANOI_DIR, "peg_board_3disks.usda"), semantic_tags=[("class", "board")]),
    )
    disk_large = _disk_cfg(0, DISKS[0][1])
    disk_medium = _disk_cfg(1, DISKS[1][1])
    disk_small = _disk_cfg(2, DISKS[2][1])


@configclass
class HanoiEventCfg:
    """Reset: disks back to their start stack, then small joint noise (terms run in this order)."""

    reset_disks = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class HanoiObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target)
        ee_pose_b = ObsTerm(func=mdp.ee_pose_in_base)
        disk_large_pose_b = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("disk_large")})
        disk_medium_pose_b = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("disk_medium")})
        disk_small_pose_b = ObsTerm(func=mdp.object_pose_in_base, params={"object_cfg": SceneEntityCfg("disk_small")})
        front = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class HanoiTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    disk_dropped = DoneTerm(func=mdp.disk_dropped, params=HANOI_DROP_PARAMS)
    success = DoneTerm(func=mdp.hanoi_solved, params=HANOI_SUCCESS_PARAMS)


def _configure_hanoi_contacts(cfg):
    """Resolve contacts for 8 mm thick, 2–5 g disks without driving the fingers through them."""
    cfg.sim.dt = 1 / 120
    cfg.decimation = 4  # Keep the observation / action rate at 30 Hz.
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics.friction_correlation_distance = 0.001
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 64
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 8
    cfg.scene.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0)
    cfg.scene.robot.actuators["gripper"].joint_effort_limit = 0.05
    cfg.scene.robot.actuators["gripper"].joint_velocity_limit = 2.0


@configclass
class SO101HanoiJointEnvCfg(SO101PickPlaceJointEnvCfg):
    """Joint-space Tower of Hanoi: action = absolute joint targets [rad] in :data:`JOINTS` order."""

    scene: HanoiSceneCfg = HanoiSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: HanoiObservationsCfg = HanoiObservationsCfg()
    events: HanoiEventCfg = HanoiEventCfg()
    terminations: HanoiTerminationsCfg = HanoiTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_hanoi_contacts(self)
        self.episode_length_s = 120.0


@configclass
class SO101HanoiIKEnvCfg(SO101PickPlaceIKEnvCfg):
    """Hanoi scene controlled by Cartesian gripper waypoints, without a learned policy."""

    scene: HanoiSceneCfg = HanoiSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: HanoiObservationsCfg = HanoiObservationsCfg()
    events: HanoiEventCfg = HanoiEventCfg()
    terminations: HanoiTerminationsCfg = HanoiTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_hanoi_contacts(self)
        self.episode_length_s = 600.0
        self.events.randomize_joint_state.params["std"] = 0.0
        # The scripted controller uses object poses. Keep the camera for snapshots, but
        # avoid copying an RGB observation from the GPU on every control step.
        self.observations.policy.front = None
        # This frame is 42 mm above the fixed fingertip. The controller places it 44 mm
        # above the disk bottom so the tip engages only the top disk, not the stack below.
        self.actions.arm_action.body_offset.pos = (0.0, 0.0, -0.062)
        self.scene.ee_frame.target_frames[0].offset.pos = (0.0, 0.0, -0.062)
        # All five joints must serve the pose task to keep the jaws vertical along the
        # peg. The cube controller's wrist-only orientation mask leaves lateral drift.
        self.actions.arm_action.controller.orientation_joint_names = None
        self.actions.arm_action.controller.joint_limit_avoidance_gain = 0.0
