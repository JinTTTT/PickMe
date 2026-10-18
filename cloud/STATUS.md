# Verified deployment — 2026-10-18 13:06 UTC

- VM: `computeinstance-e05y9q7nf0cr5sbn2e`, `ubuntu@81.85.1.206`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB; driver 580.173.02.
- Managed boot disk: 300 GiB, expanded in the guest.
- Fork: `https://github.com/JinTTTT/PickMe.git`, base commit `0f99775` with deployment changes.
- Remote project: `/home/ubuntu/work/PickMe`; upstream libraries and environments: `/home/ubuntu/work`.
- Isaac Sim 6.1.0.0 and Isaac Lab v3.0.0-EA installed; CUDA matrix computation passed with torch 2.11.0+cu128.
- GR00T pinned at `dac54b313160c201c450f97df0fd74ebd379c892`; CUDA computation and torchcodec import passed with torch 2.9.0+cu128.
- Public GR00T-N1.7-3B weights downloaded; installer and downloader both exited 0.
- Browser desktop: HTTP 200 and VNC WebSocket upgrade 101 through the Mac's SSH tunnel. Remote ports bind to loopback only.
- GUI viewer: running in tmux `viewer`, both cameras rendered, scripted policy finished with `placed=True`.
- Headless collection: 3 successful episodes in 3 attempts, 917 frames, 30 fps H.264 videos.
- GR00T stats, relative stats, and loader checks passed for both camera streams (`DATASET OK`, `SMOKE_PASSED`, exit 0).
- Test dataset: `/home/ubuntu/work/datasets/local/cloud_smoke_20261018_125851`.
- Remaining model-access requirement: the authenticated Hugging Face account/token receives HTTP 403 for `nvidia/Cosmos-Reason2-2B`. Request access on its model page before training or policy inference.
- No training job was started. The VM and browser viewer remain running.

See `../logs/doctor.log`, `../logs/smoke.log`, and `../logs/view/final_front.png` for verification evidence.
