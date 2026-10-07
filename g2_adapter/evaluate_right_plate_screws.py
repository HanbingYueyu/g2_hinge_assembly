"""Evaluate individual detection boxes and counts on held-out and affine stress images."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from .prepare_right_screw_dataset import WIDTH, HEIGHT, affine_boxes


def compare(predicted, expected):
    candidates = []
    for p, box in enumerate(predicted):
        for g, target in enumerate(expected):
            low = np.maximum(box[:2], target[:2])
            high = np.minimum(box[2:], target[2:])
            intersection = np.prod(np.maximum(high-low, 0))
            union = np.prod(np.asarray(box[2:])-box[:2]) + np.prod(np.asarray(target[2:])-target[:2]) - intersection
            iou = float(intersection / union)
            if iou >= .5:
                candidates.append((iou, p, g))
    used_p, used_g, ious = set(), set(), []
    for iou, p, g in sorted(candidates, reverse=True):
        if p not in used_p and g not in used_g:
            used_p.add(p); used_g.add(g); ious.append(iou)
    return dict(tp=len(ious), fp=len(predicted)-len(ious), fn=len(expected)-len(ious),
                matched_ious=ious, count_correct=len(predicted)==len(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--extra-annotations', type=Path, help='Visually reviewed later capture, not used for training')
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    assert model.names == {0: 'plate_screw'}, model.names
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [row for row in json.loads((args.data/'samples.json').read_text()) if row['split']=='val']
    results = []
    for row in rows:
        source = cv2.imread(str(args.data/row['image']))
        for index, (angle, tx, ty) in enumerate(((0,0,0), (12,45,-35), (-12,-45,35))):
            matrix = cv2.getRotationMatrix2D((WIDTH/2,HEIGHT/2),angle,1)
            matrix[:,2] += (tx,ty)
            image = source if index==0 else cv2.warpAffine(source,matrix,(WIDTH,HEIGHT),borderMode=cv2.BORDER_REFLECT_101)
            expected = affine_boxes(row['boxes_xyxy'],matrix)
            prediction = model.predict(image,imgsz=1280,conf=.5,iou=.5,max_det=100,verbose=False)[0]
            predicted = prediction.boxes.xyxy.cpu().numpy().tolist()
            metrics = compare(predicted,expected)
            kind = 'heldout' if index==0 else 'affine_stress'
            stem = Path(row['image']).stem + f'_test{index}'
            directory = args.out/kind;directory.mkdir(exist_ok=True)
            cv2.imwrite(str(directory/(stem+'.jpg')),prediction.plot(labels=False, line_width=2))
            results.append(dict(kind=kind,image=stem,source=row['source'],expected_count=len(expected),
                                predicted_count=len(predicted),confidence=prediction.boxes.conf.cpu().numpy().tolist(),
                                expected_boxes_xyxy=expected,predicted_boxes_xyxy=predicted,**metrics))
    if args.extra_annotations:
        for index, row in enumerate(json.loads(args.extra_annotations.read_text())):
            image = cv2.imread(row['file'])
            expected = row['boxes_xyxy']
            prediction = model.predict(image,imgsz=1280,conf=.5,iou=.5,max_det=100,verbose=False)[0]
            predicted = prediction.boxes.xyxy.cpu().numpy().tolist()
            directory=args.out/'fresh';directory.mkdir(exist_ok=True)
            cv2.imwrite(str(directory/f'frame_{index:04d}.jpg'),prediction.plot(labels=False, line_width=2))
            results.append(dict(kind='fresh',image=f'frame_{index:04d}',source=row['file'],
                                expected_count=len(expected),predicted_count=len(predicted),
                                expected_boxes_xyxy=expected,predicted_boxes_xyxy=predicted,
                                **compare(predicted,expected)))
    summary = {}
    for kind in sorted({row['kind'] for row in results}):
        group = [r for r in results if r['kind']==kind]
        histogram = defaultdict(lambda:dict(images=0,count_correct=0,tp=0,fp=0,fn=0))
        for r in group:
            record=histogram[r['expected_count']];record['images']+=1;record['count_correct']+=int(r['count_correct'])
            for key in ('tp','fp','fn'):record[key]+=r[key]
        summary[kind]=dict(images=len(group),count_correct=sum(r['count_correct'] for r in group),
                           tp=sum(r['tp'] for r in group),fp=sum(r['fp'] for r in group),fn=sum(r['fn'] for r in group),
                           by_expected_count=dict(histogram))
    (args.out/'detections.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
