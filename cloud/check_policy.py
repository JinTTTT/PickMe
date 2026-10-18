"""Check the Isaac client against a policy server without starting the simulator."""
from pathlib import Path
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_pickplace.gr00t_client import Gr00tClient

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--port', type=int, default=5555)
parser.add_argument('--stop-server', action='store_true')
args = parser.parse_args()
client = Gr00tClient('127.0.0.1', args.port, timeout_ms=2000)
try:
    for attempt in range(10):
        if client.ping():
            break
        time.sleep(0.5)
    else:
        raise RuntimeError('Policy server did not become ready')
    modalities = client.get_modality_config()
    assert modalities['video']['modality_keys'] == ['front'], modalities
    assert modalities['action']['modality_keys'] == ['single_arm', 'gripper'], modalities
    client.reset()
    obs = {
        'video': {'front': np.zeros((1, 1, 480, 640, 3), dtype=np.uint8)},
        'state': {'single_arm': np.zeros((1, 1, 5), dtype=np.float32), 'gripper': np.zeros((1, 1, 1), dtype=np.float32)},
        'language': {'annotation.human.task_description': [['pick up the red block and place it on the green pad']]},
    }
    actions, _ = client.get_action(obs)
    for key, size in [('single_arm', 5), ('gripper', 1)]:
        value = np.asarray(actions[key])
        assert value.shape == (1, 16, size), (key, value.shape)
        assert np.isfinite(value).all(), key
        print(key, value.shape, 'finite')
    if args.stop_server:
        client.call('kill', requires_input=False)
    print('POLICY_PROTOCOL_OK')
finally:
    client.close()
