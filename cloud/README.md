# PickMe cloud workstation

VM: `ubuntu@81.85.1.206` (Nebius, RTX PRO 6000 96 GB). Repository: `/home/ubuntu/work/PickMe`.

All simulator, dataset, and training processes run on the VM. Your Mac provides SSH and a browser.

From the Mac copy of this fork:

```bash
bash cloud.sh ui
```

Keep that terminal open and visit <http://localhost:6080/vnc.html?autoconnect=true&resize=scale>.
The desktop's ports are bound to loopback; SSH carries the browser connection.

Start the robot scene from another terminal:

```bash
bash cloud.sh view
```

View logs with `bash cloud.sh logs`, or open an SSH shell with `bash cloud.sh ssh`.
Cloud jobs run in tmux sessions `setup`, `desktop`, and `viewer`. Attach on the VM with `tmux attach -t viewer`; detach with Ctrl-b then d.

## Stack and installation

Use the Ubuntu 24.04 CUDA image. Run `bash nebius/setup_vm.sh` from the fork on the VM.
The installer pins Isaac Sim 6.1.0, Isaac Lab v3.0.0-EA, and the GR00T N1.7 revision recorded in `logs/stack-versions.txt`.
Environments are separate: `~/work/env_isaaclab` and `~/work/Isaac-GR00T/.venv`.
Shared paths in `env.sh` use `~/work`, with no Mac GPU requirement.

The browser desktop requires `bash cloud/desktop.sh` on the VM; `cloud.sh ui` starts it automatically.
The viewer holds its final robot scene open after running the scripted policy.

## Scripted Tower of Hanoi

With the browser UI connected, run this from the Mac's `results/PickMe` directory:

```bash
bash cloud.sh hanoi
```

This runs `scripts/solve_hanoi.py` on the VM. It uses inverse kinematics and simulation poses
to execute and verify seven legal moves, without a model server. The gripper and travel
between pegs run at normal speed; motion through the peg holes is slower for alignment.
The final scene stays open until Ctrl-C. Stop it before issuing another controller command.

Results and snapshots are saved to `~/work/PickMe/logs/hanoi-scripted/`.
For just the first transfer, use `bash cloud.sh hanoi --moves 1`.

## Model access

Accept access to <https://huggingface.co/nvidia/Cosmos-Reason2-2B> with your Hugging Face account.
Authenticate in the VM's private SSH terminal, never in a chat message:

```bash
cd ~/work/Isaac-GR00T
source .venv/bin/activate
hf auth login
hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
hf download nvidia/Cosmos-Reason2-2B config.json
```

The simulator and scripted demonstration collection do not require model authentication.
After the model is available, use the repository's dataset preparation, fine-tuning, and policy server scripts.

For a small isolated collection test on the VM:

```bash
cd ~/work/PickMe
DATASET_NAME=cloud_smoke NUM_EPISODES=3 bash collect.sh
DATASET_NAME=cloud_smoke bash prepare_dataset.sh
```

The VM remains running until explicitly stopped. Stopping compute retains the managed boot disk; deleting the VM deletes that disk.

## Learned robot control

GR00T N1.7 uses the Cosmos-Reason2-2B backbone to process the two camera images,
robot joint state, and your instruction, then predict arm and gripper actions.
The current training task is picking up the red block and placing it on the green pad.
Other skills require suitable demonstrations and validation.

From the original Mac project directory, check progress with:

```bash
bash results/PickMe/cloud.sh progress
```

After a checkpoint passes held-out robot evaluation and the policy server is running:

```bash
bash results/PickMe/cloud.sh control "pick up the red block and place it on the green pad"
```

Watch the browser desktop. This command replaces the automatic demonstration window
and runs one learned-control episode in a new scene, then keeps it visible. Press Ctrl-C
in that terminal before issuing another command. Keep the `cloud.sh ui` tunnel open.

On the VM, `logs/control-best.json` records measured outcomes for the selected checkpoint,
`logs/control-checkpoint.txt` identifies its weights, and `logs/live-control.json`
records the latest browser demonstration. The model server listens on loopback port 5555.
Training defaults for this first controller are in `cloud/control.env`.
