"""Solve the three-disk Hanoi scene using deterministic waypoints and inverse kinematics.

Run with the Isaac Lab interpreter: scripts/solve_hanoi.py --viz kit --keep_open
No policy server or model is used. Only robot joint targets are commanded: disks move through
finger contact and gravity. Ground-truth poses provide waypoint and placement feedback.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from _common import build_env_cfg, launch_simulation, make_parser, parse_args

parser = make_parser(__doc__)
parser.add_argument("--keep_open", action="store_true", help="Hold the final scene until Ctrl-C.")
parser.add_argument("--out_dir", default="logs/hanoi-scripted", help="Directory for result.json and stage images.")
parser.add_argument("--moves", type=int, default=7, choices=range(1, 8), help="Run the first N moves (7 solves the puzzle).")
parser.add_argument("--grasp_height", type=float, default=0.044, help="Grasp frame height above disk bottom [m].")
parser.add_argument("--grasp_x", type=float, default=-0.004, help="X offset for the small disk; scaled by radius for the others [m].")
args = parse_args(parser)

from so101_pickplace.hanoi_env_cfg import (
    DISKS, DISK_RADII, DISK_THICKNESS, HANOI_SUCCESS_PARAMS, PEG_POS, SO101HanoiIKEnvCfg,
)

cfg = build_env_cfg(SO101HanoiIKEnvCfg, args)
if cfg.scene.num_envs != 1:
    parser.error("This controller supports one robot.")

with launch_simulation(cfg, args):
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import isaaclab.utils.math as math_utils
    from so101_pickplace import TASK_HANOI_IK, mdp
    from so101_pickplace.scripted_policy import ScriptedPickPlace

    env = gym.make(TASK_HANOI_IK, cfg=cfg).unwrapped
    env.sim.set_camera_view(eye=(0.55, 0.38, 0.35), target=(0.18, 0.0, 0.04))
    env.reset(seed=args.seed)
    robot = env.scene["robot"]
    ee = env.scene["ee_frame"]
    origin = env.scene.env_origins[0]
    quat = ScriptedPickPlace._down_quat(ee.data.target_quat_source.torch[:, 0, :]).clone()
    command = ee.data.target_pos_source.torch[:, 0, :].clone()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_steps = 0

    def disk_pos(index):
        return (env.scene[DISKS[index][0]].data.root_pos_w.torch[0] - origin).clone()

    def snapshot(name):
        image = env.scene["front_cam"].data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        imageio.imwrite(out / f"{name}.png", image)

    def goto(pos, jaw, label, steps=180, tolerance=0.002, speed=0.012):
        global total_steps
        target_w = torch.as_tensor(pos, dtype=torch.float32, device=env.device).reshape(1, 3) + origin
        target, _ = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, target_w
        )
        for t in range(steps):
            delta = target - command
            command.add_(delta * (speed / delta.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp(max=1))
            env.step(torch.cat((command, quat, torch.tensor([[jaw]], device=env.device)), dim=1))
            total_steps += 1
            error = (ee.data.target_pos_source.torch[:, 0, :] - target).norm().item()
            if t >= 29 and error < tolerance:
                break
        print(f"[hanoi] {label}: position error={error*1000:.1f} mm, steps={t+1}", flush=True)
        snapshot(label)
        if tolerance and error >= tolerance:
            raise RuntimeError(f"{label}: waypoint not reached after {steps} steps ({error*1000:.1f} mm error).")

    def verify_stacks(stacks):
        """Check measured poses, ordering, orientation and rest state, not just the move counter."""
        for peg, stack in enumerate(stacks):
            for level, disk in enumerate(stack):
                data = env.scene[DISKS[disk][0]].data
                expected = torch.tensor(PEG_POS[peg], device=env.device)
                expected[2] += level * DISK_THICKNESS
                error = disk_pos(disk) - expected
                up = math_utils.quat_apply(
                    data.root_quat_w.torch, torch.tensor([[0., 0., 1.]], device=env.device)
                )[0, 2].item()
                at_rest = data.root_lin_vel_w.torch.norm().item() < 0.01
                if error[:2].norm().item() > 0.003 or abs(error[2].item()) > 0.002 or up < 0.98 or not at_rest:
                    raise RuntimeError(
                        f"{DISKS[disk][0]} did not settle at level {level+1} on peg {peg+1}: "
                        f"position error={error.tolist()}, upright={up:.3f}, at_rest={at_rest}"
                    )

    # Disk indices are largest=0, middle=1, smallest=2.
    plan = [(2, 0, 2), (1, 0, 1), (2, 2, 1), (0, 0, 2),
            (2, 1, 0), (1, 1, 2), (2, 0, 2)]
    stacks = [[0, 1, 2], [], []]
    completed = []
    failure = None
    solved = False
    interrupted = False
    started = time.monotonic()
    try:
        goto((0.16, 0.0, 0.11), 1.5, "home")
        verify_stacks(stacks)
        for move, (disk, source, dest) in enumerate(plan[:args.moves], 1):
            assert stacks[source][-1] == disk
            assert not stacks[dest] or stacks[dest][-1] < disk
            p = disk_pos(disk)
            source_xy = p[:2].tolist()
            source_xy[0] -= DISK_RADII[disk] - DISK_RADII[2]
            source_xy[0] += args.grasp_x
            grasp_z = p[2].item() + args.grasp_height
            prefix = f"{move:02d}"
            print(f"[hanoi] MOVE {move}/7: {DISKS[disk][0]} peg {source+1} -> {dest+1}", flush=True)
            goto((*source_xy, 0.11), 1.5, prefix + "-approach")
            goto((*source_xy, grasp_z), 1.5, prefix + "-descend", speed=0.002)
            goto((*source_xy, grasp_z), -0.1, prefix + "-grasp", steps=36, tolerance=0)
            other_heights = {i: disk_pos(i)[2].item() for i in range(3) if i != disk}
            goto((*source_xy, 0.11), -0.1, prefix + "-lift", speed=0.002)
            if disk_pos(disk)[2].item() < 0.055:
                raise RuntimeError(f"Move {move}: {DISKS[disk][0]} was not lifted above the peg.")
            if any(abs(disk_pos(i)[2].item() - z) > 0.003 for i, z in other_heights.items()):
                raise RuntimeError(f"Move {move}: another disk was displaced during the lift.")
            # Use the measured grasp offset to centre the ring over the destination.
            offset = ee.data.target_pos_w.torch[0, 0] - origin - disk_pos(disk)
            target = torch.tensor(PEG_POS[dest], device=env.device)
            target[2] += len(stacks[dest]) * DISK_THICKNESS + 0.001
            place = target + offset
            goto((place[0].item(), place[1].item(), 0.11), -0.1, prefix + "-transfer")
            goto(place.tolist(), -0.1, prefix + "-lower", speed=0.002)
            goto(place.tolist(), 1.5, prefix + "-release", steps=36, tolerance=0)
            goto((place[0].item(), place[1].item(), 0.11), 1.5, prefix + "-retreat")
            stacks[source].pop()
            stacks[dest].append(disk)
            verify_stacks(stacks)
            completed.append({"disk": DISKS[disk][0], "source": source, "destination": dest})
            print(f"[hanoi] Move {move} verified.", flush=True)
        solved = len(completed) == 7 and bool(mdp.hanoi_solved(env, **HANOI_SUCCESS_PARAMS)[0])
        if args.moves == 7 and not solved:
            raise RuntimeError("Final scene did not pass the Hanoi success check.")
        print(f"HANOI_SOLVED={solved}", flush=True)
    except RuntimeError as exc:
        failure = str(exc)
        print(f"HANOI_FAILED: {failure}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        failure = "Interrupted before completing the requested moves."
    finally:
        snapshot("final")
        (out / "result.json").write_text(json.dumps({
            "controller": "scripted_ik", "solved": solved, "completed_moves": completed, "failure": failure,
            "seed": args.seed, "steps": total_steps, "elapsed_seconds": time.monotonic() - started,
            "disk_positions": {name: disk_pos(i).tolist() for i, (name, _) in enumerate(DISKS)},
        }, indent=2))
        if args.keep_open and not interrupted:
            print("HANOI_VIEWER_READY — press Ctrl-C to close.", flush=True)
            try:
                while True:
                    env.sim.render()
                    time.sleep(1 / 30)
            except KeyboardInterrupt:
                pass
        env.close()
    if failure:
        raise SystemExit(1)