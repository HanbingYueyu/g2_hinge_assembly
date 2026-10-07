"""Save fresh right-hand RGB frames from the existing DDS preview; no motion."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5001/get_all_images?indices=18')
    parser.add_argument('--count', type=int, default=120)
    parser.add_argument('--interval', type=float, default=3.)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.interval < 0:
        parser.error('count must be positive and interval nonnegative')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    previous = None
    for index in range(args.count):
        start = time.monotonic()
        deadline = start + 15
        while True:
            with urlopen(args.url, timeout=4) as response:
                document = json.load(response)
            candidates = [item for item in document['images']
                          if item.get('topic') == '/camera/hand_right_color']
            if len(candidates) != 1:
                raise RuntimeError('Expected exactly the right-hand color camera')
            item = candidates[0]
            stamp = item.get('source_timestamp')
            if (item.get('success') and item.get('available')
                    and item.get('age_ms', 1e9) < 500 and stamp != previous):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('No fresh right-hand image')
            time.sleep(.1)
        encoded = base64.b64decode(item['image'].split(',', 1)[1])
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (1056, 1280):
            raise RuntimeError('Expected the 1280x1056 reference view')
        name = f'frame_{index:04d}.jpg'
        (args.out_dir / name).write_bytes(encoded)
        metadata = {key: value for key, value in item.items() if key != 'image'}
        metadata.update(file=name, captured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                        captured_monotonic=time.monotonic(), crop_xyxy=[0, 0, 1280, 1056])
        with (args.out_dir / 'frames.jsonl').open('a') as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        previous = stamp
        print(f'{index+1}/{args.count} saved {name}', flush=True)
        if index + 1 < args.count:
            time.sleep(max(0, args.interval - (time.monotonic() - start)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
