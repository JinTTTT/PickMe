"""Client for the Cosmos-Reason2 planner server (``scripts/cosmos_planner_server.py``).

Reuses the GR00T policy wire format, so it runs inside the Isaac Lab venv with no extra
dependencies. The planner sits above GR00T: it turns an overall task ("stack the red block on
the green block" today, "move the chess pieces to the right squares" later) into the next
pick/place subtask GR00T should execute, and visually verifies subtask completion.
"""

from __future__ import annotations

import numpy as np

from .gr00t_client import Gr00tClient


class CosmosPlannerClient(Gr00tClient):
    """``plan``/``verify`` against ``cosmos_planner_server.py`` (same REQ/REP socket, port 5556)."""

    def plan(self, front: np.ndarray, wrist: np.ndarray, task: str, history: list[str]) -> dict:
        """-> {"subtask": str, "done": bool} — next instruction for GR00T, or done when finished."""
        return self.call("plan", {"front": front, "wrist": wrist, "task": task, "history": history})

    def verify(self, front: np.ndarray, subtask: str) -> dict:
        """-> {"done": bool, "reason": str} — whether the camera frame shows the subtask complete."""
        return self.call("verify", {"front": front, "subtask": subtask})
