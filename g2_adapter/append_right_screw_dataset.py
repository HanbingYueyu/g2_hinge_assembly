"""Append a visually reviewed batch, preserving old images, labels and splits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from .prepare_right_screw_dataset import build


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def append_batch(annotation, data, work, batch_id, session):
    if not batch_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in batch_id):
        raise ValueError('Batch ID must be a safe unique filename prefix')
    rows = json.loads(annotation.read_text())
    if not rows or any(row.get('review_status') != 'visually_reviewed' for row in rows):
        raise ValueError('Every annotation must be visually reviewed first')
    retained = [r for r in rows if not r.get('exclude')]
    if not retained:
        raise ValueError('No retained frames')
    state = json.loads(session.read_text())
    weight = Path(state['protected_weight'])
    if hashlib.sha256(weight.read_bytes()).hexdigest() != state['weight_sha256']:
        raise ValueError('Protected model weight changed')
    old_records = json.loads((data/'samples.json').read_text())
    old_info = json.loads((data/'dataset_info.json').read_text())
    sources = {str(Path(r['source']).resolve()) for r in old_records}
    if any(str(Path(r['file']).resolve()) in sources for r in retained):
        raise ValueError('Batch source already present in dataset')
    if batch_id in {b['batch_id'] for b in old_info.get('supplement_batches', [])}:
        raise ValueError('Batch already appended')
    build(annotation, work)
    additions = json.loads((work/'samples.json').read_text())
    batch_info = json.loads((work/'dataset_info.json').read_text())
    copies = []
    for record in additions:
        source_image = Path(record['image'])
        target_image = source_image.with_name(batch_id+'_'+source_image.name)
        for directory, suffix in (('images', '.jpg'), ('labels', '.txt'), ('boxed_all', '.jpg')):
            source = work/directory/record['split']/(source_image.stem+suffix)
            target = data/directory/record['split']/(target_image.stem+suffix)
            if not source.is_file() or target.exists():
                raise ValueError(f'Missing source or target would overwrite: {target}')
            copies.append((source, target))
        record['image'] = str(target_image)
        record['supplement_batch'] = batch_id
    review_root = data/'supplement_reviews'/batch_id
    if review_root.exists():
        raise ValueError('Review batch already exists')
    records = old_records + additions
    info = dict(old_info)
    info['counts'] = dict(Counter(r['split'] for r in records))
    histogram = Counter({str(k): v for k, v in info['raw_count_histogram'].items()})
    histogram.update(str(len(r['boxes_xyxy'])) for r in retained)
    info['raw_count_histogram'] = dict(histogram)
    info['gap_source_files'] = old_info.get('gap_source_files', []) + batch_info['gap_source_files']
    info['supplement_batches'] = old_info.get('supplement_batches', []) + [dict(
        batch_id=batch_id, annotation_file=str(annotation.resolve()), retained_raw=len(retained),
        appended_images=len(additions), split_counts=batch_info['counts'],
        appended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())]
    parts = ['<!doctype html><meta charset="utf-8"><title>右手螺丝全部标注图</title>',
             '<style>body{font-family:sans-serif}main{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}figure{margin:0}img{width:100%}</style>',
             f'<h1>全部标注图：{len(records)} 张</h1><p>每颗螺丝单独标框；0 颗为空标签。点击图片查看原尺寸。</p><main>']
    for r in records:
        name = Path(r['image']).name
        url = f"boxed_all/{r['split']}/{name}"
        title = html.escape(f"{name} | {r['split']} | {r['count']} boxes")
        parts.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}"></a><figcaption>{title}</figcaption></figure>')
    parts.append('</main>')
    # Keep all previous dataset metadata and the gallery as a recoverable snapshot.
    backup = data/'supplement_backups'/batch_id
    backup.mkdir(parents=True, exist_ok=False)
    metadata = ('samples.json', 'dataset_info.json', '全部标注图.html', 'README.md')
    for name in metadata:
        if (data/name).exists():
            shutil.copy2(data/name, backup/name)
    written = []
    try:
        for source, target in copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst:
                written.append(target)
                shutil.copyfileobj(src, dst)
        shutil.copytree(work/'raw_annotation_review', review_root)
        shutil.copy2(annotation, review_root/'reviewed_annotations.json')
        write_json(data/'samples.json', records)
        write_json(data/'dataset_info.json', info)
        (data/'全部标注图.html').write_text('\n'.join(parts), encoding='utf-8')
        with (data/'README.md').open('a') as stream:
            stream.write(f"\n## 补采 {batch_id}\n\n新增 {len(retained)} 张已核对原图，生成 {len(additions)} 张训练/验证图。当前总数：{info['counts']}；当前原始样本颗数分布：{dict(histogram)}。旧图、旧标签和旧划分保留；补采验证使用该批末尾时间块，与训练间留两帧间隔。原权重保持不变。\n")
        assert hashlib.sha256(weight.read_bytes()).hexdigest() == state['weight_sha256']
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        if review_root.exists():
            shutil.rmtree(review_root)
        for name in metadata:
            if (backup/name).exists():
                shutil.copy2(backup/name, data/name)
        raise
    print(json.dumps(dict(batch=batch_id,added=len(additions),total=len(records),counts=info['counts']),ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--batch-id', required=True)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    append_batch(args.annotations, args.data, args.work, args.batch_id, args.session)


if __name__ == '__main__':
    main()
