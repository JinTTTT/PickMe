"""Closed-loop evaluation of a fine-tuned GR00T N1.7 checkpoint in the Isaac Lab scene.

Start the policy server first (``start_server.sh``), then:

    python scripts/eval_in_sim.py --num_episodes 20 --task_text "..." [--viz kit] [--record_dir logs/eval]
    python scripts/eval_in_sim.py --scene hanoi --viz kit --keep_open   # Tower of Hanoi board instead of the cube

Observations are built exactly like the dataset (front 640x480 RGB, state in LeRobot
normalised units); the returned action chunk (``single_arm`` 16x5, ``gripper`` 16x1) is converted
back to radians and executed on the joint-space env for ``--action_horizon`` steps per query.
"""

from __future__ import annotations

import os
import json
import time

from _common import PROJECT_ROOT, build_env_cfg, launch_simulation, make_parser, parse_args

parser = make_parser(__doc__)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--scene", choices=("cube", "hanoi"), default="cube", help="cube pick-and-place or Tower of Hanoi board.")
parser.add_argument("--task_text", type=str, default=None, help="Language instruction (default depends on --scene).")
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=5555)
parser.add_argument("--action_horizon", type=int, default=8, help="Predicted steps executed per policy query.")
parser.add_argument("--max_steps", type=int, default=None, help="Steps per episode (default: 600 cube, 3600 hanoi).")
parser.add_argument("--success_steps", type=int, default=15)
parser.add_argument("--record_dir", type=str, default="", help="Save a front-camera mp4 per episode here.")
parser.add_argument("--results_file", type=str, default="", help="Save evaluation outcomes and inference latency as JSON.")
parser.add_argument("--keep_open", action="store_true", help="Keep the final scene visible after evaluation.")
args = parse_args(parser)

from so101_pickplace.pickplace_env_cfg import IMAGE_HEIGHT, IMAGE_WIDTH, SO101PickPlaceJointEnvCfg  # noqa: E402

if args.scene == "hanoi":
    from so101_pickplace.hanoi_env_cfg import HANOI_DROP_PARAMS, HANOI_SUCCESS_PARAMS, HANOI_TASK, SO101HanoiJointEnvCfg  # noqa: E402

    env_cfg_cls, default_task, default_max_steps = SO101HanoiJointEnvCfg, HANOI_TASK, 3600
else:
    env_cfg_cls, default_task, default_max_steps = SO101PickPlaceJointEnvCfg, "pick up the red block and place it on the green pad", 600
args.task_text = args.task_text or default_task
args.max_steps = args.max_steps or default_max_steps
env_cfg = build_env_cfg(env_cfg_cls, args)
assert env_cfg.scene.num_envs == 1

with launch_simulation(env_cfg, args):
    import gymnasium as gym
    import numpy as np
    import torch

    import so101_pickplace  # noqa: F401
    from so101_pickplace import TASK_HANOI_JOINT, TASK_JOINT, mdp
    from so101_pickplace.gr00t_client import Gr00tClient, add_batch_time_dims
    from so101_pickplace.so101_units import JOINTS, norm_to_rad, rad_to_norm

    if args.scene == "hanoi":
        task_id = TASK_HANOI_JOINT
        is_dropped = lambda env: mdp.disk_dropped(env, **HANOI_DROP_PARAMS)  # noqa: E731
        is_success = lambda env: mdp.hanoi_solved(env, **HANOI_SUCCESS_PARAMS)  # noqa: E731
    else:
        task_id, is_dropped, is_success = TASK_JOINT, mdp.cube_dropped, mdp.cube_placed

    client = Gr00tClient(args.policy_host, args.policy_port)
    if not client.ping():
        raise SystemExit(f"policy server not reachable at {args.policy_host}:{args.policy_port} (run start_server.sh)")
    mc = client.get_modality_config()
    print("[eval] server modality config:", {k: v.get("modality_keys") for k, v in mc.items()})
    expected = {"video": ["front"], "state": ["single_arm", "gripper"], "action": ["single_arm", "gripper"]}
    got = {k: list(mc[k]["modality_keys"]) for k in expected}
    if got != expected:
        raise SystemExit(f"checkpoint modality layout {got} != {expected}")

    env = gym.make(task_id, cfg=env_cfg).unwrapped
    env.sim.set_camera_view(eye=(0.65, 0.45, 0.45), target=(0.18, 0.0, 0.04))
    env.reset(seed=args.seed)
    robot = env.scene["robot"]
    jidx, _ = robot.find_joints(JOINTS, preserve_order=True)
    jidx = torch.as_tensor(jidx, device=env.device)
    cams = {"front": env.scene["front_cam"]}
    rec = None
    if args.record_dir:
        import subprocess

        os.makedirs(args.record_dir, exist_ok=True)

    def observation():
        state = rad_to_norm(robot.data.joint_pos.torch[0, jidx].cpu().numpy())
        imgs = {k: c.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8) for k, c in cams.items()}
        obs = {
            "video": {"front": np.ascontiguousarray(imgs["front"])},
            "state": {"single_arm": state[:5].astype(np.float32), "gripper": state[5:6].astype(np.float32)},
            "language": {"annotation.human.task_description": args.task_text},
        }
        return add_batch_time_dims(obs), imgs["front"]

    results = []
    latencies = []
    for ep in range(args.num_episodes):
        client.reset()
        success_count = 0
        outcome = "timeout"
        t = 0
        if args.record_dir:
            path = os.path.join(args.record_dir, f"episode_{ep:03d}.mp4")
            rec = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
                 "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path],
                stdin=subprocess.PIPE,
            )
        while t < args.max_steps and outcome == "timeout":
            obs, front = observation()
            t0 = time.perf_counter()
            action, _ = client.get_action(obs)
            latencies.append(1e3 * (time.perf_counter() - t0))
            chunk = np.concatenate([action["single_arm"][0], action["gripper"][0]], axis=-1)  # (H, 6) normalised
            if chunk.ndim != 2 or chunk.shape[1] != 6 or not np.isfinite(chunk).all():
                raise ValueError(f"Policy returned invalid joint targets: shape={chunk.shape}")
            horizon = min(args.action_horizon, chunk.shape[0])
            for h in range(horizon):
                q = torch.as_tensor(norm_to_rad(chunk[h]), device=env.device).unsqueeze(0)
                env.step(q)
                if rec is not None:
                    rec.stdin.write(np.ascontiguousarray(cams["front"].data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)).tobytes())
                t += 1
                if t % 100 == 0:
                    print(f"[eval] episode {ep + 1}: step {t}/{args.max_steps}, inference avg {np.mean(latencies):.0f} ms", flush=True)
                if bool(is_dropped(env)[0]):
                    outcome = "dropped"
                    break
                if bool(is_success(env)[0]):
                    success_count += 1
                    if success_count >= args.success_steps:
                        outcome = "success"
                        break
                else:
                    success_count = 0
                if t >= args.max_steps:
                    break
        if rec is not None:
            rec.stdin.close()
            rec.wait()
        results.append(outcome)
        print(f"[eval] episode {ep + 1}/{args.num_episodes}: {outcome} after {t} steps "
              f"(success so far {results.count('success')}/{len(results)}, inference avg {np.mean(latencies):.0f} ms)", flush=True)
        if ep + 1 < args.num_episodes:
            env.reset()

    print(f"[eval] success rate: {results.count('success')}/{len(results)} = {results.count('success') / max(len(results), 1):.2f}")
    print(f"[eval] outcomes: { {o: results.count(o) for o in set(results)} }")
    if args.results_file:
        path = os.path.abspath(args.results_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            json.dump({
                "scene": args.scene,
                "task": args.task_text,
                "seed": args.seed,
                "action_horizon": args.action_horizon,
                "max_steps": args.max_steps,
                "success_steps": args.success_steps,
                "episodes": len(results),
                "successes": results.count("success"),
                "outcomes": results,
                "mean_inference_ms": float(np.mean(latencies)) if latencies else None,
            }, file, indent=2)
    client.close()
    if args.keep_open:
        print("LEARNED_CONTROL_VIEWER_READY", flush=True)
        while True:
            env.sim.render()
            time.sleep(1 / 30)
    env.close()
