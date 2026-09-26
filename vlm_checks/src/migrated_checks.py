"""Image preparation and decisions ported from the 20260923 reference packages."""
from collections import Counter
import subprocess

import cv2
import numpy as np


def decode_original(data):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (1056, 1280):
        raise ValueError("迁移包要求 1280×1056 原始相机图，不能使用叠加图或已裁剪图片")
    return frame


def encode(frame):
    ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise ValueError('Cannot encode VLM input')
    return data.tobytes()


def alignment_roi(data):
    decode_original(data)
    # Right-camera installation ROI. The source package's LEFT-camera ROI
    # (400, 445, 500, 484) must not be reused for this viewpoint.
    result = subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
        '-frames:v', '1', '-vf', 'crop=500:400:400:200,scale=900:713:flags=lanczos',
        '-q:v', '2', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1',
    ], input=data, capture_output=True, check=True, timeout=15)
    return result.stdout


def fixture_board(images, interval_s=1.5):
    if len(images) != 4:
        raise ValueError('V47 requires four chronological original right frames')
    tiles = []
    for data in images:
        frame = decode_original(data)
        wide = cv2.resize(frame[100:1056, 120:1100], (980, 956), interpolation=cv2.INTER_CUBIC)
        tiles.append(cv2.resize(wide[330:900, 155:835], (1020, 855), interpolation=cv2.INTER_CUBIC))
    canvas = cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])
    for index, (x, y) in enumerate(((15, 36), (1035, 36), (15, 891), (1035, 891))):
        label = 'CURRENT' if index == 3 else f'T-{(3-index)*interval_s:.1f}s'
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
    return encode(canvas)


def fuse_grasp(primary, verifier):
    first = primary.get('decision', 'unknown')
    second = (verifier or {}).get('decision', 'unknown')
    if first == 'yes':
        decision, reason = '抓住', 'V32主判别器确认抓住。'
    elif first == 'no' and second == 'no':
        decision, reason = '未抓住', 'V32与V40均确认未抓住。'
    elif second == 'yes':
        decision, reason = '抓住', 'V40发现V32容易漏掉的已抓住外观；按安全策略不报未抓住。'
    else:
        decision, reason = '不确定', '两套判别未形成足够一致的结论。'
    return dict(decision=decision, reason=reason, v32_primary=primary,
                v40_error_aware_verifier=verifier, policy='V32先判；仅V32与V40均为no才输出未抓住。')


def alignment_vote(results):
    if len(results) != 5:
        raise ValueError('Hole alignment majority requires five observations')
    votes = Counter(result.get('decision', '不确定') for result in results)
    decision = next((value for value in ('无明显大偏差', '明显大偏差') if votes[value] >= 3), '不确定')
    return dict(decision=decision, votes=dict(votes), observations=results,
                reason=f'5次判别多数票：{decision}')
