# Historical stereo-camera two-hole detector

This earlier model is not used by the head RGB-D localization workflow. Use
`models/two_hole_head_rgbd_yolo/best.pt` for that workflow.

This is a YOLO11n **detection** model trained from the GDK `kHeadStereoRight`
camera. It distinguishes:

| ID | Name | Target |
| --- | --- | --- |
| 0 | `irregular_table_hole` | Irregular white ring in the tabletop |
| 1 | `round_hinge_tip_hole` | Round opening at the silver hinge tip |

For a native 1920×1536 image, crop `[x0, y0, x1, y1] =
[280, 575, 1180, 1215]` before inference. The crop is 900×640. Do not apply
random crop shifts or rotations at inference. Add `(280, 575)` to detection
coordinates to recover native-image positions.

Training data: `datasets/two_hole_yolo_lit_v3`, captured at about 3-second
intervals under changing colored lights. Split by source frame, with 240
training images from 48 frames and 12 unaugmented validation images from 12
other frames. The model used `imgsz=960`, `batch=8`, `epochs=120`,
`patience=25`, AdamW with `lr0=0.001`, cosine decay, and no mosaic, mixup, or
flips. Early stopping ended training after 34 epochs; the best epoch was 9.

The validation frames come from the same fixed scene and are not an
independent geometry test. A later 12-frame capture under colored lighting
detected both classes in every frame at confidence threshold 0.25.

Reproduction:

```bash
.venv-sam3/bin/python -m g2_adapter.prepare_two_hole_training
.venv-sam3/bin/python -m g2_adapter.train_two_hole_yolo
```
