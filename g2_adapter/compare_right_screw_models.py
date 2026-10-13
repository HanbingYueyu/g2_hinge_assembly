"""Compare frozen screw weights on identical held-out full-view and ROI images."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .evaluate_right_plate_screws import compare
from .prepare_right_screw_dataset import WIDTH, HEIGHT, affine_boxes
from .right_plate_screw_detection import ROI, detect


def visible_boxes(boxes):
    x0, y0, x1, y1 = ROI
    result = []
    for b in boxes:
        clipped = [max(x0, b[0]), max(y0, b[1]), min(x1, b[2]), min(y1, b[3])]
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            result.append(clipped)
    return result


def cases(data, extra):
    rows = [r for r in json.loads((data/'samples.json').read_text()) if r['split']=='val']
    for row in rows:
        source = cv2.imread(str(data/row['image']))
        for i, (angle, tx, ty) in enumerate(((0,0,0), (12,45,-35), (-12,-45,35))):
            matrix = cv2.getRotationMatrix2D((WIDTH/2, HEIGHT/2), angle, 1)
            matrix[:, 2] += (tx, ty)
            image = source if i == 0 else cv2.warpAffine(source, matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_REFLECT_101)
            yield dict(kind='heldout' if i == 0 else 'affine_stress',
                       stem=Path(row['image']).stem+f'_test{i}', source=row['source'],
                       source_group='supplement' if row.get('supplement_batch') else 'original',
                       expected=affine_boxes(row['boxes_xyxy'], matrix), image=image)
    if extra:
        for i, row in enumerate(json.loads(extra.read_text())):
            yield dict(kind='separate_capture', stem=f'extra_{i:04d}', source=row['file'],
                       source_group=row.get('capture_group', 'separate_capture'),
                       expected=row['boxes_xyxy'], image=cv2.imread(row['file']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--old-model', required=True)
    parser.add_argument('--new-model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra-annotations', type=Path)
    args = parser.parse_args()
    from ultralytics import YOLO
    models = {name: YOLO(path) for name, path in (('old', args.old_model), ('new', args.new_model))}
    assert all(model.names == {0:'plate_screw'} for model in models.values())
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    pages = {'crop': [], 'full': []}
    for case in cases(args.data, args.extra_annotations):
        if case['image'] is None or case['image'].shape[:2] != (HEIGHT, WIDTH):
            raise ValueError(f"Invalid source image: {case['source']}")
        for mode in ('crop', 'full'):
            expected = visible_boxes(case['expected']) if mode == 'crop' else case['expected']
            pair = []
            for name, model in models.items():
                if mode == 'crop':
                    _, preview, detected = detect(model, case['image'], confidence=.5)
                    boxes = [d['bbox_xyxy'] for d in detected['detections']]
                    confidences = [d['confidence'] for d in detected['detections']]
                else:
                    prediction = model.predict(case['image'], imgsz=1280, conf=.5, iou=.5, max_det=100, verbose=False)[0]
                    boxes = prediction.boxes.xyxy.cpu().numpy().tolist()
                    confidences = prediction.boxes.conf.cpu().numpy().tolist()
                    preview = prediction.plot(labels=False, conf=False, line_width=2)
                metrics = compare(boxes, expected)
                path = Path(mode)/case['kind']/name/(case['stem']+'.jpg')
                (args.out/path).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.out/path), preview):
                    raise RuntimeError(f'Unable to save {path}')
                record = dict(model=name, mode=mode, kind=case['kind'], source_group=case['source_group'],
                              source=case['source'], stem=case['stem'], expected_count=len(expected),
                              predicted_count=len(boxes), expected_boxes_xyxy=expected,
                              predicted_boxes_xyxy=boxes, confidences=confidences, preview=str(path), **metrics)
                records.append(record)
                pair.append(record)
            pages[mode].append(pair)
    summary = {}
    for mode in ('crop', 'full'):
        summary[mode] = {}
        for name in models:
            summary[mode][name] = {}
            for kind in sorted({r['kind'] for r in records}):
                group = [r for r in records if r['mode']==mode and r['model']==name and r['kind']==kind]
                histogram = defaultdict(lambda:dict(images=0,count_correct=0,tp=0,fp=0,fn=0))
                for r in group:
                    bucket = histogram[r['expected_count']]
                    bucket['images'] += 1
                    bucket['count_correct'] += int(r['count_correct'])
                    for key in ('tp','fp','fn'):
                        bucket[key] += r[key]
                summary[mode][name][kind] = dict(images=len(group), count_correct=sum(r['count_correct'] for r in group),
                    tp=sum(r['tp'] for r in group), fp=sum(r['fp'] for r in group), fn=sum(r['fn'] for r in group),
                    by_expected_count=dict(histogram),
                    by_source_group={source:dict(images=sum(r['source_group']==source for r in group),
                        count_correct=sum(r['count_correct'] for r in group if r['source_group']==source),
                        fp=sum(r['fp'] for r in group if r['source_group']==source),
                        fn=sum(r['fn'] for r in group if r['source_group']==source))
                        for source in sorted({r['source_group'] for r in group})})
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.out/'detections.json').write_text(json.dumps(records, indent=2)+'\n')
    for mode, pairs in pages.items():
        parts = ['<!doctype html><meta charset="utf-8"><title>螺丝新旧模型对比</title>',
                 '<style>body{font-family:sans-serif}section{margin-bottom:25px}main{display:flex;gap:15px}figure{margin:0;max-width:48%}img{width:100%;max-width:510px}</style>',
                 f'<h1>{mode} 新旧模型对比</h1><p>左侧旧权重，右侧新权重。置信度≥0.5；计数之外另以 IoU≥0.5 检查逐框漏检/误检。扰动图是留出图的合成变体。</p>']
        for pair in pairs:
            parts.append(f"<section><h3>{html.escape(pair[0]['kind']+' / '+pair[0]['stem'])}，标注 {pair[0]['expected_count']} 颗</h3><main>")
            for r in pair:
                title = f"{r['model']}：检测 {r['predicted_count']}，漏检 {r['fn']}，多检 {r['fp']}"
                parts.append(f'<figure><a href="{r["preview"]}"><img loading="lazy" src="{r["preview"]}"></a><figcaption>{title}</figcaption></figure>')
            parts.append('</main></section>')
        (args.out/f'{mode}_新旧模型对比.html').write_text('\n'.join(parts), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
