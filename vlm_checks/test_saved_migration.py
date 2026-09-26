"""Read-only API regression on previously saved original images."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from src.migrated_checks import alignment_roi, fixture_board
from src.vlm_three_checks import VlmClient, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--vlm-mode', choices=('online', 'local'), required=True)
    args = parser.parse_args()
    out = ROOT.parent / 'outputs/vlm_regression' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.vlm_mode)
    out.mkdir(parents=True)
    client = VlmClient(ROOT / 'config/vlm.env', None, None, 60, args.vlm_mode)
    results = {'mode': args.vlm_mode, 'input': str(args.input_dir), 'cases': [],
               'limitations': 'Historical fixture frames were sampled at 0.25s, not 1.5s; only three alignment frames exist. No fabricated five-frame vote or accuracy claim.'}
    alignment_frames = sorted(args.input_dir.glob('*_right_vlm_hole_alignment_check.jpg'))
    if not alignment_frames:
        alignment_frames = sorted(args.input_dir.glob('*_right_vlm_c_hole_check.jpg'))
    cases = [('hole-alignment', [p]) for p in alignment_frames]
    right = sorted(args.input_dir.glob('*_right_vlm_right_fixture_grasp_check_frame[1-4].jpg'))
    if len(right) == 4:
        cases.append(('right-fixture-grasp', right))
    for index, (task, paths) in enumerate(cases):
        images = [p.read_bytes() for p in paths]
        prepared = alignment_roi(images[0]) if task == 'hole-alignment' else fixture_board(images, interval_s=.25)
        (out / f'{index}_{task}_sent.jpg').write_bytes(prepared)
        record = {'task': task, 'images': [p.name for p in paths]}
        start = time.monotonic()
        try:
            # Use the actual known historic interval in the labels.
            if task == 'right-fixture-grasp':
                from src.migrated_checks import fuse_grasp
                primary = client._request('fixture-v32', [prepared])
                verifier = client._request('fixture-v40', [prepared]) if primary.get('decision') != 'yes' else None
                record['result'] = fuse_grasp(primary, verifier)
            else:
                record['result'] = client.run_bytes(task, images)
        except Exception as exc:
            record['error'] = str(exc)
        record['elapsed_s'] = round(time.monotonic() - start, 3)
        results['cases'].append(record)
        (out / 'result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print(f'Results: {out}', flush=True)


if __name__ == '__main__':
    main()
