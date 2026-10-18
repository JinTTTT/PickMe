"""Serve Cosmos-Reason2 as a high-level planner/verifier for the GR00T eval loop.

Run this inside the Isaac-GR00T uv venv (it has torch + transformers and access to the gated
``nvidia/Cosmos-Reason2-2B`` weights), next to ``start_server.sh``:

    cd $GR00T_REPO && uv run python $PROJECT_DIR/scripts/cosmos_planner_server.py --port 5556

The wire format is the same ZeroMQ REQ + msgpack/msgpack_numpy protocol as the GR00T policy
server, so ``so101_pickplace.cosmos_planner.CosmosPlannerClient`` (which reuses ``Gr00tClient``)
can talk to it from the Isaac Lab venv without new dependencies.

Two endpoints:
  * ``plan``   : front/wrist camera frame + overall task text + completed subtasks
                 -> {"subtask": "<next instruction for GR00T>", "done": false}
  * ``verify`` : front camera frame + subtask text
                 -> {"done": true|false, "reason": "..."}
"""

from __future__ import annotations

import argparse
import json
import re

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason2-2B")
parser.add_argument("--host", type=str, default="*")
parser.add_argument("--port", type=int, default=5556)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--dtype", type=str, default="auto")
args = parser.parse_args()

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

PLAN_PROMPT = """\
You are the high-level planner for a robot arm doing a tabletop manipulation task.
Overall task: "{task}"

You are given the front and wrist camera views. Completed subtasks so far: {history}.

Decide the NEXT single subtask the low-level policy should execute. Rules:
- Each subtask must be a short imperative describing ONE pick-or-place action
  (e.g. "pick up the red block", "place the red block on top of the green block").
- If the overall task is already finished, say so.

Answer with ONLY a JSON object: {{"subtask": "...", "done": true|false}}"""

VERIFY_PROMPT = """\
Look at this front-camera view of a robot arm workspace.
The robot was asked to: "{subtask}"

Has this subtask been completed? Answer with ONLY a JSON object:
{{"done": true|false, "reason": "<one short sentence>"}}"""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in model output: {text!r}")
    return json.loads(m.group(0))


class CosmosPlanner:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=args.dtype, device_map=args.device
        )
        self.model.eval()

    def _generate(self, prompt: str, images: list[Image.Image]) -> dict:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return _extract_json(reply)

    def plan(self, front: np.ndarray, wrist: np.ndarray, task: str, history: list[str]) -> dict:
        prompt = PLAN_PROMPT.format(task=task, history=", ".join(history) if history else "none")
        try:
            res = self._generate(prompt, [Image.fromarray(front), Image.fromarray(wrist)])
            return {"subtask": str(res.get("subtask", "")), "done": bool(res.get("done", False)), "raw": res}
        except Exception as e:  # never kill the eval on a bad planner reply
            return {"subtask": task, "done": False, "error": str(e)}

    def verify(self, front: np.ndarray, subtask: str) -> dict:
        try:
            res = self._generate(VERIFY_PROMPT.format(subtask=subtask), [Image.fromarray(front)])
            return {"done": bool(res.get("done", False)), "reason": str(res.get("reason", "")), "raw": res}
        except Exception as e:
            return {"done": False, "reason": "", "error": str(e)}


planner = CosmosPlanner()
ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind(f"tcp://{args.host}:{args.port}")
print(f"[cosmos-planner] {args.model} serving on tcp://{args.host}:{args.port}", flush=True)

while True:
    req = msgpack.unpackb(sock.recv(), object_hook=mnp.decode, raw=False)
    endpoint, data = req.get("endpoint"), req.get("data", {})
    try:
        if endpoint == "ping":
            resp = {"ok": True}
        elif endpoint == "plan":
            resp = planner.plan(data["front"], data["wrist"], data["task"], data.get("history", []))
        elif endpoint == "verify":
            resp = planner.verify(data["front"], data["subtask"])
        else:
            resp = {"error": f"unknown endpoint {endpoint!r}"}
    except Exception as e:
        resp = {"error": str(e)}
    sock.send(msgpack.packb(resp, default=mnp.encode))
