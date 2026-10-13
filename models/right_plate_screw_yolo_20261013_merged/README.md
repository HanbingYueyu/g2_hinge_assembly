# 合并数据集重新训练的螺丝模型（2026-10-13）

权重：`best.pt`。单类别 `plate_screw`，每颗螺丝单独检测，计数为保留检测框数量。

训练使用合并后的 `../../datasets/right_plate_screw_yolo_20261007_v1/data.yaml`：954 张训练图、47 张留出验证图，原始帧来源不重叠。真实标注原图共 224 张，数量分布为 0颗44张、1颗44张、2颗27张、3颗82张、4颗27张；其中含训练/验证之间的时间间隔帧。增强图不跨训练/验证。

从 2026-10-07 螺丝模型初始化，在独立目录训练，旧权重 SHA256 保持不变。YOLO11n、imgsz=1280、batch=8、AdamW、seed=42，最多120轮、patience=25；实际完成 40 轮，最佳第 15 轮。

标准整图验证：precision=0.9993、recall=1.0000、mAP50=0.9950、mAP50–95=0.9823。这是 Ultralytics 标准验证指标。

实际推理输入完整 1280×1056 原图，先裁剪 `[525,435,780,815]` 得到255×380，再使用 imgsz=384、conf=0.5、NMS IoU=0.5。输出 `bbox_xyxy` 是完整原图像素坐标，`bbox_crop_xyxy` 是裁剪图坐标。

## 同图比较（实际裁剪推理）

| 数据 | 旧模型计数正确 | 新模型计数正确 |
| --- | --- | --- |
| 47 张留出原图，0–4颗 | 32/47 | 47/47 |
| 留出原图的94张±12°旋转及±45/35px平移变体 | 69/94 | 94/94 |
| 当天另采12张四颗图，不参与训练 | 0/12 | 12/12 |
| 之前另采12张空孔图，不参与训练 | 12/12 | 12/12 |

计数之外还逐框检查 IoU≥0.5：新模型在以上165张裁剪图的336个目标上无漏检、多检。整图推理同样全部正确。原来的22张留出图、新增25张留出图分别全部正确。

另外用当前相机单帧验证，识别到4颗，置信度约0.949–0.977，输出保存在 `../../outputs/right_plate_screws/20261013_new_model_readonly_check/`。

这些样本来自目前工位，验证帧与训练帧仍来自相同采集段，94张扰动图是合成图；上述成绩不代表其他工位、明显遮挡或大幅视角变化下的表现。

## 查看结果

- [裁剪画面新旧模型并排对比](../../runs/right_plate_screw_yolo/20261013_merged_v1/comparison/crop_新旧模型对比.html)
- [整图新旧模型并排对比](../../runs/right_plate_screw_yolo/20261013_merged_v1/comparison/full_新旧模型对比.html)
- [逐组计数和框匹配统计](../../runs/right_plate_screw_yolo/20261013_merged_v1/comparison/summary.json)
- 逐帧坐标和置信度：同目录 `detections.json`。
- 训练曲线：`../../runs/right_plate_screw_yolo/20261013_merged_v1/results.png`。
- 数据来源和训练前哈希快照：`../../runs/right_plate_screw_yolo/20261013_merged_v1_manifest.json`。

## 浏览器实时预览新模型

保持已有 DDS 相机服务5001运行。命令使用独立端口8794，显式选择本次新权重：

```bash
cd /home/flyfuture/g2_hinge_assembly && \
.venv-sam3/bin/python -m g2_adapter.live_right_plate_screws_web \
  --model models/right_plate_screw_yolo_20261013_merged/best.pt \
  --camera-url 'http://127.0.0.1:5001/get_all_images?indices=18' \
  --confidence 0.5 \
  --interval 0.25 \
  --host 127.0.0.1 \
  --port 8794
```

浏览器打开 http://127.0.0.1:8794 。Ctrl+C退出。实时服务由用户启动。

## 单张识别

```bash
cd /home/flyfuture/g2_hinge_assembly && \
.venv-sam3/bin/python -m g2_adapter.predict_right_plate_screws \
  --model models/right_plate_screw_yolo_20261013_merged/best.pt \
  --camera-url 'http://127.0.0.1:5001/get_all_images?indices=18' \
  --confidence 0.5 \
  --output "outputs/right_plate_screws/new_$(date +%Y%m%d_%H%M%S_%N)"
```

本地图片调用可将 `--camera-url` 替换为 `--image /实际完整原图路径.jpg`。
输出：source.jpg（完整原图）、crop.jpg（裁剪原图）、detections.jpg（裁剪带框图）、detections.json（数量和坐标）。输入必须为1280×1056完整原图。

原权重仍在 `../right_plate_screw_yolo_20261007/best.pt`；备份在 `../right_plate_screw_preserved_20261013/best.pt`。程序默认路径保持原配置；使用新模型时显式传入本页的 `--model`。
