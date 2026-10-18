"""Freeze completed episodes from an active collector without modifying its dataset."""
import argparse
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('destination', type=Path)
parser.add_argument('--episodes', type=int, default=60)
args = parser.parse_args()
assert args.episodes > 0
assert not args.destination.exists(), args.destination
for attempt in range(10):
    try:
        meta = args.source / 'meta'
        info = json.loads((meta / 'info.json').read_text())
        episodes = [json.loads(line) for line in (meta / 'episodes.jsonl').read_text().splitlines()][:args.episodes]
        stats = [json.loads(line) for line in (meta / 'episodes_stats.jsonl').read_text().splitlines()][:args.episodes]
        tasks = (meta / 'tasks.jsonl').read_text()
        assert len(episodes) == len(stats) == args.episodes
        assert [item['episode_index'] for item in episodes] == list(range(args.episodes))
        assert [item['episode_index'] for item in stats] == list(range(args.episodes))
        break
    except (ValueError, AssertionError):
        if attempt == 9:
            raise
        time.sleep(0.2)
(args.destination / 'meta').mkdir(parents=True)
frames = 0
for episode in episodes:
    index = episode['episode_index']
    substitutions = {'episode_chunk': index // info['chunks_size'], 'episode_index': index}
    relative = Path(info['data_path'].format(**substitutions))
    assert pq.read_metadata(args.source / relative).num_rows == episode['length']
    frames += episode['length']
    files = [relative]
    for key, feature in info['features'].items():
        if feature['dtype'] == 'video':
            files.append(Path(info['video_path'].format(video_key=key, **substitutions)))
    for relative in files:
        destination = args.destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(args.source / relative, destination)
info.update(total_episodes=args.episodes, total_frames=frames,
            total_videos=args.episodes * sum(v['dtype'] == 'video' for v in info['features'].values()),
            total_chunks=(args.episodes - 1) // info['chunks_size'] + 1,
            splits={'train': f'0:{args.episodes}'})
meta = args.destination / 'meta'
(meta / 'info.json').write_text(json.dumps(info, indent=2))
(meta / 'tasks.jsonl').write_text(tasks)
for name, values in [('episodes.jsonl', episodes), ('episodes_stats.jsonl', stats)]:
    (meta / name).write_text(''.join(json.dumps(value) + '\n' for value in values))
(meta / 'snapshot.json').write_text(json.dumps({'source': str(args.source), 'episodes': args.episodes}, indent=2))
print('SNAPSHOT_READY', args.destination, args.episodes, frames)
