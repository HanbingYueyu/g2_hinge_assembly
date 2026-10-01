# 头部 RGB-D 双孔检测模型 v2

在线定位默认使用本目录的 `best.pt`，只读取 GDK `kHeadColor` 与 `kHeadDepth`。固定 RGB 裁剪框为原生 640×400 画面中的 `[70, 90, 450, 390]`（380×300），深度只对齐到同一头部 RGB-D 相机的彩色图像。

| 类别 | 名称 | 目标 |
| --- | --- | --- |
| 0 | `irregular_table_hole` | 台面上的不规则孔 |
| 1 | `round_hinge_tip_hole` | 银色铰链尖端的圆孔 |

数据集：`datasets/two_hole_head_rgbd_yolo_v2/`。来源共 276 张头部 RGB-D 彩色原图：原始人工参考框 60 张、历史定位结果 168 张、新采集图 48 张。新图的两个框由已核对的参考框模板匹配传播，48 张的匹配位置一致；模板分数约为类别 0 的 0.86、类别 1 的 0.92。经过最大约 ±65 px 水平平移、±42 px 垂直平移、±18° 旋转，得到 1008 张训练图和 24 张验证图。全部 1032 张标注结果都有 `boxed/train/` 或 `boxed/val/` 中对应的带框图。另有 72 张极端变换检验图在 `datasets/two_hole_head_rgbd_yolo_v2_stress/`。

训练从旧模型继续，YOLO11n，`imgsz=960`，`batch=16`，目标 50 epoch，提前停止。记录在 `runs/two_hole_head_rgbd_yolo/v2_large_roi_shift_rotate/`。在 24 张留出图上，新模型按 `conf=0.5` 检出两类 48/48，旧模型在相同大裁剪框下只检出 5/48；在 72 张平移旋转检验图上，新模型检出 144/144，旧模型检出 7/144。留出图包含与训练图相同摆位的新采集帧，因此这些数字不等于新场景的泛化保证。

新权重与新裁剪框启用后，用只读 GDK 相机连续定位 12 次：两类各检出 12/12，类别置信度最低分别为 0.952 和 0.969；12 次 `base_link` 平面定位最大偏差 0.89 mm，并通过双孔间距检查。这是当前摆位的重复性验证，不代表绝对三维精度或尚未采集的摆位。

重新构建相同数据集需要指定新的输出目录，不会覆盖旧数据：

```bash
cd /home/flyfuture/g2_hinge_assembly
.venv-sam3/bin/python -m g2_adapter.build_head_rgbd_two_hole_v2 \
  --out-dir datasets/two_hole_head_rgbd_yolo_v2_new
```

按新数据训练：

```bash
cd /home/flyfuture/g2_hinge_assembly
.venv-sam3/bin/python -m g2_adapter.train_head_rgbd_two_hole_yolo \
  --data datasets/two_hole_head_rgbd_yolo_v2_new/data.yaml \
  --model models/two_hole_head_rgbd_yolo_v2/best.pt \
  --epochs 50 --imgsz 960 --batch 16 \
  --project runs/two_hole_head_rgbd_yolo --name v2_retrain
```
