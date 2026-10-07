"""Review screw annotations and build grouped full-view YOLO data with affine augmentation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

SITES = ((603, 605), (672, 580), (683, 616), (633, 640))
SEED_BOXES = ((657, 566, 688, 596), (668, 601, 699, 632), (617, 624, 650, 657))
WIDTH, HEIGHT = 1280, 1056


def feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4)).apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), .5)


def propose(raw: Path, seed: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    seed_image = cv2.imread(str(seed))
    templates = [feature(seed_image[y0:y1, x0:x1]) for x0, y0, x1, y1 in SEED_BOXES]
    # Visually checked extra head after the user installed the fourth site.
    extra = raw / 'frame_0010.jpg'
    if extra.is_file():
        extra_image = cv2.imread(str(extra))
        templates.append(feature(extra_image[591:624, 588:621]))
    rows = []
    for path in sorted(raw.glob('frame_*.jpg')):
        image = cv2.imread(str(path))
        boxes, scores, site_boxes = [], [], []
        for x, y in SITES:
            search_x, search_y = x - 28, y - 28
            region = image[search_y:y + 28, search_x:x + 28]
            source = feature(region)
            matches = []
            for template in templates:
                _, score, _, xy = cv2.minMaxLoc(cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED))
                matches.append((score, xy, template.shape))
            score, (dx, dy), (h, w) = max(matches, key=lambda item: item[0])
            scores.append(round(score, 4))
            site_boxes.append([search_x + dx, search_y + dy, search_x + dx + w, search_y + dy + h])
            if score >= .64:
                boxes.append(site_boxes[-1])
        rows.append(dict(file=str(path.resolve()), boxes_xyxy=boxes, site_scores=scores,
                         site_boxes=site_boxes, exclude=False))
    (out / 'proposed_annotations.json').write_text(json.dumps(rows, indent=2) + '\n')
    review(rows, out / 'proposal_review')
    print('proposed counts:', dict(Counter(len(row['boxes_xyxy']) for row in rows)))


def boxed(image, boxes):
    result = image.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(result, (round(x0), round(y0)), (round(x1), round(y1)), (0, 255, 0), 2)
        cv2.putText(result, 'screw', (round(x0), round(y0) - 3), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 0), 1)
    return result


def review(rows, root):
    root.mkdir(parents=True, exist_ok=True)
    tiles = []
    for row in rows:
        image = cv2.imread(row['file'])
        annotated = boxed(image, row['boxes_xyxy'])
        crop = annotated[540:710, 550:735]
        tile = cv2.resize(crop, (370, 340))
        margin = np.full((60, 370, 3), 245, np.uint8)
        title = Path(row['file']).stem + f" n={len(row['boxes_xyxy'])}" + (' EXCLUDE' if row.get('exclude') else '')
        cv2.putText(margin, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1)
        if 'site_scores' in row:
            cv2.putText(margin, str(row['site_scores']), (5, 47), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 1)
        tiles.append(np.concatenate((tile, margin)))
    for start in range(0, len(tiles), 12):
        canvas = np.full((1600, 1110, 3), 230, np.uint8)
        for k, tile in enumerate(tiles[start:start + 12]):
            y, x = divmod(k, 3)
            canvas[y * 400:(y + 1) * 400, x * 370:(x + 1) * 370] = tile
        cv2.imwrite(str(root / f'sheet_{start // 12:02d}.jpg'), canvas)


def affine_boxes(boxes, matrix):
    transformed = []
    for x0, y0, x1, y1 in boxes:
        points = np.array(((x0, y0, 1), (x1, y0, 1), (x1, y1, 1), (x0, y1, 1))) @ matrix.T
        lo, hi = points.min(axis=0), points.max(axis=0)
        lo, hi = np.maximum(lo, (0, 0)), np.minimum(hi, (WIDTH, HEIGHT))
        if np.all(hi - lo >= 4):
            transformed.append([*lo, *hi])
    return transformed


def build(annotation: Path, out: Path):
    rows = [row for row in json.loads(annotation.read_text()) if not row.get('exclude')]
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    # Hold out contiguous time blocks for each screw count, with adjacent
    # source frames assigned to a gap instead of training. Augment only train.
    by_count = {}
    for index, row in enumerate(rows):
        occupied = tuple(sorted(int(np.argmin([
            np.linalg.norm(np.asarray(((box[0] + box[2]) / 2,
                                       (box[1] + box[3]) / 2)) - site)
            for site in SITES])) for box in row['boxes_xyxy']))
        by_count.setdefault(occupied, []).append(index)
    validation, gap = set(), set()
    for indices in by_count.values():
        # Keep rare real count configurations in training as well.
        if len(indices) < 3:
            continue
        n = min(max(1, round(len(indices) * .2)), len(indices) // 3)
        selected = indices[-n:]
        validation.update(selected)
        gap_size = 2 if len(indices) >= 8 else 0
        gap.update(indices[len(indices) - n - gap_size:len(indices) - n])
    records = []
    for index, row in enumerate(rows):
        if index in gap:
            continue
        split = 'val' if index in validation else 'train'
        source = cv2.imread(row['file'])
        if source.shape[:2] != (HEIGHT, WIDTH):
            raise ValueError('Source size differs from the fixed second-reference crop')
        for variant in range(1 if split == 'val' else 6):
            angle = 0 if variant == 0 else float(rng.uniform(-8, 8))
            tx = 0 if variant == 0 else float(rng.uniform(-30, 30))
            ty = 0 if variant == 0 else float(rng.uniform(-25, 25))
            matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle, 1.)
            matrix[:, 2] += (tx, ty)
            image = source.copy() if variant == 0 else cv2.warpAffine(
                source, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT_101)
            boxes = affine_boxes(row['boxes_xyxy'], matrix)
            stem = Path(row['file']).stem + f'_aug{variant:02d}'
            image_path = out / 'images' / split / (stem + '.jpg')
            label_path = out / 'labels' / split / (stem + '.txt')
            preview_path = out / 'boxed_all' / split / (stem + '.jpg')
            for path in (image_path, label_path, preview_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 97])
            lines = []
            for x0, y0, x1, y1 in boxes:
                lines.append(f'0 {(x0+x1)/2/WIDTH:.8f} {(y0+y1)/2/HEIGHT:.8f} {(x1-x0)/WIDTH:.8f} {(y1-y0)/HEIGHT:.8f}')
            label_path.write_text('\n'.join(lines) + ('\n' if lines else ''))
            cv2.imwrite(str(preview_path), boxed(image, boxes), [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append(dict(image=str(image_path.relative_to(out)), source=row['file'], split=split,
                                count=len(boxes), angle_deg=angle, translation_px=[tx, ty],
                                boxes_xyxy=boxes, affine_matrix=matrix.tolist()))
    data = f'path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: plate_screw\n'
    (out / 'data.yaml').write_text(data)
    (out / 'samples.json').write_text(json.dumps(records, indent=2) + '\n')
    info = dict(camera='kHandRightColor /camera/hand_right_color', crop_xyxy=[0, 0, WIDTH, HEIGHT],
                classes={'0': 'plate_screw'}, annotation_file=str(annotation.resolve()),
                counts=dict(Counter(record['split'] for record in records)),
                raw_count_histogram=dict(Counter(len(row['boxes_xyxy']) for row in rows)),
                gap_source_files=[rows[i]['file'] for i in sorted(gap)],
                augmentation='train only: rotations +-8deg, x +-30px, y +-25px; fixed image size',
                split='last time block per occupied-site configuration held out; two-frame gap for >=8 sources; no shared sources')
    (out / 'dataset_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    review(rows, out / 'raw_annotation_review')
    print(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('propose')
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('build')
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'propose':
        propose(args.raw, args.seed, args.out)
    else:
        build(args.annotations, args.out)


if __name__ == '__main__':
    main()
