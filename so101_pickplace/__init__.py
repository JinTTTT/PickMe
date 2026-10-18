"""SO-101 block pick-and-place tasks for GR00T N1.7 post-training (Isaac Lab 3.0)."""

import gymnasium as gym

TASK_IK = "Isaac-PickPlace-Cube-SO101-IK-v0"
TASK_JOINT = "Isaac-PickPlace-Cube-SO101-Joint-v0"
TASK_TELEOP = "Isaac-PickPlace-Cube-SO101-Joint-Teleop-v0"
TASK_STACK_IK = "Isaac-Stack-Cube-SO101-IK-v0"
TASK_STACK_JOINT = "Isaac-Stack-Cube-SO101-Joint-v0"
TASK_HANOI_JOINT = "Isaac-Hanoi-SO101-Joint-v0"
TASK_HANOI_IK = "Isaac-Hanoi-SO101-IK-v0"

_REGISTERED = False


def register_tasks():
    """Register the gym ids (idempotent). Also usable as ``record_demos.py --external_callback``."""
    global _REGISTERED
    if _REGISTERED:
        return None
    _REGISTERED = True
    gym.register(
        id=TASK_IK,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:SO101PickPlaceIKEnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id=TASK_JOINT,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:SO101PickPlaceJointEnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id=TASK_STACK_IK,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:SO101PickPlaceStackIKEnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id=TASK_STACK_JOINT,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:SO101PickPlaceStackJointEnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id=TASK_TELEOP,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:SO101PickPlaceTeleopEnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id=TASK_HANOI_JOINT,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.hanoi_env_cfg:SO101HanoiJointEnvCfg"},
        disable_env_checker=True,
    )
    gym.register(
        id=TASK_HANOI_IK,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": f"{__name__}.hanoi_env_cfg:SO101HanoiIKEnvCfg"},
        disable_env_checker=True,
    )
    return None


register_tasks()
