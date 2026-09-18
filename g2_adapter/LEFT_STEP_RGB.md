# 左手铰链 STEP + RGB 定位与抓取

归档提示（2026-09-17）：独立预览入口 `run_left_step_rgb.py` 已移到仓库外归档，本文相关历史命令须先恢复脚本。共用 STEP 模块及标定配置保留；[恢复说明](../docs/cleanup-archive-2026-09-17.md)。

STEP 模式的粗对准和精对准现使用连续 XY 伺服：每帧目标从实际末端位置计算，前瞻距离不超过配置的 2 mm，不再等待每小步停稳。进入对准容差后撤销剩余前瞻、保持实际位置，重新确认视觉对准和实际到位，再下探。`--xy-speed-mm-s 15` 单独设置 XY 速度（1–30 mm/s，默认 15），`--speed-mm-s 5` 继续控制下探和原后续运动速度。单步测试仍只发送一个目标。未做真机验证。

闭爪后的反馈确认不再经过 STEP 孔位门槛，避免夹爪遮孔阻塞上抬。下探和闭爪阶段的只读视觉监控检查整件目标是否丢失，不要求孔位持续可见；夹爪反馈、抓取验证及搬运监控仍保留。

最终下探阶段可用 `--final-descent-visual-servo` 对下探前锁定的同一 YOLO 目标持续做受限 XY 修正；估计水平误差不超过 5 mm 时保持，超过后才修正。力反馈接近接触时冻结横向修正，下探的 Z 距离仍只由力控逻辑决定。

启用 STEP 的流程先用原 YOLO 分割中心和原配置像素基准做水平粗对准，粗对准不要求孔位匹配。到达后切换 STEP 精定位；STEP 未通过时等待，不下探。两阶段使用连续有限前瞻目标，最终对准后确认实际到位。日志 `alignment_stage` 为 `coarse_yolo` 或 `fine_step`；`--stop-after xy` 等待精对准完成才结束，重抓从粗对准重新开始。

## 近似夹持中心直接 XY 对准

此模式不需要 `--step-reference`。每帧利用图像时间的 GDK TF，将 STEP 抓取点与近似夹持中心转换到 base_link，计算 XY 偏差；不改变 Z 和姿态。TF 不可用或匹配失败时不提交新修正。

`--step-tcp-end-mm` 是夹持中心相对 `arm_l_end_link` 的 XYZ 估计，默认 `0,0,145` mm 为可调初值，并非从精确闭环求得。`--step-compensation-xy-mm` 是 base_link 下附加到抓取目标的 X/Y 补偿；正 X 向前，正 Y 向左。视觉仅观察工件，TCP 固定误差需要用这些参数调整。

```bash
cd /home/flyfuture/g2_hinge_assembly && \
./scripts/run_gdk_sam3_python.sh g2_adapter/left_grasp_gdk.py \
  --step-profile g2_adapter/left_step_profile.json \
  --step-metric-xy \
  --step-tcp-end-mm=0,0,145 \
  --step-compensation-xy-mm=0,0 \
  --step-xy-tolerance-mm 2 \
  --yolo-model models/left_d405_hinge_shaft_yolo/best.pt \
  --selection leftmost --stop-after xy --speed-mm-s 5 \
  --web-port 8767 --allow-motion
```

目标相对实际位置最多前伸配置中的 2 mm，随新视觉帧连续更新。对准满意后，将 `--stop-after xy` 换成 `--full-cycle` 执行完整左手流程。未做真机验证。

## URDF 夹持中心核验

只读命令：`./scripts/run_gdk_sam3_python.sh g2_adapter/inspect_left_gripper_urdf.py --opening-rad=-0.51025`。
输出各 link 相对 `arm_l_end_link` 的变换，以及两处 `loop_joint` 的连接点距离。
按当前 URDF 的 mimic 系数计算，闭合时残差约 0.572 mm；65% 开度时外侧约 26.142 mm、内侧约 23.593 mm。
这些是模型闭环约束不一致的残差，不是实测 TCP 误差。不能由此把网格接触面的 FK 中点当成准确 TCP。
网格位于 `/home/flyfuture/agi/G2_crsB_omnipicker/genie_robot_description_210/meshes/gripper/omnipicker/`。
精确的闭环夹持中心仍未求得；上面的近似模式使用显式可调 TCP，不能把它当成已验证的 URDF 几何中心。

入口 `run_left_step_rgb.py` 只打开左彩色相机，使用 YOLO 铰链分割区域内的 RGB 孔轮廓估计 STEP 相对相机的位姿。此入口没有运动接口。`left_grasp_gdk.py` 和双手串联入口现已支持显式选择 STEP 目标，必须使用新采集的对准基准。

## 模型准备

`extract_left_step_profile.py` 从 `models/顶铰链.stp` 提取七个孔位，保留源文件 SHA256 和圆柱面实体编号。长圆孔使用两端轴线的中点，孔位高度取模型上侧孔口。原导出的轴线坐标只用于识别孔面，实际几何从新文件重新读取，不依赖旧 STEP 文件。

新文件中红色面 `#14000`、`#14073` 对应两侧夹持区域。取每个面顶点范围的中心，再取两中心的中点作为对称抓取基准：`(-264.678065, 1323.880548, 18.774347) mm`。这是模型坐标，不是机器人坐标，也不是曲面面积重心。配置已生成，可直接运行预览；重新导出模型后可运行下面的提取命令更新。

```bash
cd /home/flyfuture/g2_hinge_assembly
./scripts/run_gdk_sam3_python.sh g2_adapter/extract_left_step_profile.py \
  --step models/顶铰链.stp \
  --output g2_adapter/left_step_profile.json

./scripts/run_gdk_sam3_python.sh g2_adapter/run_left_step_rgb.py \
  --profile g2_adapter/left_step_profile.json \
  --web-host 0.0.0.0 --web-port 8777
```

网页 `http://127.0.0.1:8777`：黄色圆点为候选孔中心，绿色数字为模型孔编号，红十字为模型抓取点投影。JSON 输出 `camera_from_model`、`grasp_xyz_camera_m`、`grasp_pixel_px` 和重投影误差。`accepted=true` 仅表示本帧通过几何拟合，并不代表可以运动。

## 限制与后续接入条件

- 首次至少匹配六孔，优先七孔；锁定后可用至少四个分散孔续跟（`reason=occlusion_tracked`）。通过上一帧位姿预测和双向唯一对应匹配，拒绝拥挤、歧义和跳变，不将四孔作为新目标初始化。不再按失配时长清锁，也不要求定期六孔重新确认；缺失检测时保留历史锁，恢复后仍需通过位置关联和几何检查。没有有效匹配时不提交新运动目标。候选孔不等于已确认孔位。
- JSON 的 `matched_count` 表示实际匹配孔数，`matched_model_indices` 是模型孔编号；网页绿色编号也保持模型编号，不因缺孔重新编号。
- 使用相机内参与 OpenCV 畸变参数，平均重投影误差上限 2 px，单点上限 4 px；多解无法区分时拒绝。
- 金属反光、长圆孔轮廓不完整及透视下椭圆中心偏差会影响精度。必须先用真实相机画面核对拟合；合成测试不能替代现场精度验证。
- 自动流程使用 STEP 抓取点的图像投影做水平像素伺服，沿用实测像素到 base XY 的局部映射，不把 PnP XYZ 直接作为机器人目标。保留连续帧对准确认和力反馈下探；下探期间丢失 STEP 匹配会触发现有视觉保护。搬运监控仍跟踪整件分割目标。
- 当前程序没有 `--allow-motion`，不会发出下探或闭爪指令。

离线图像也可使用 `--image image.jpg --intrinsics intrinsics.json --output outputs/left_step/rgb_preview.jpg`；内参必须包含 `fx,fy,ppx,ppy,width,height,distortion`，图像分辨率必须一致。

## 新目标标定与实机测试

### 三高度像素标定

斜视相机会让同一个夹持中心在 10 cm、5 cm 和夹取面出现在不同像素位置。先保持左夹爪张开，视野内只保留一个可识别铰链，启动下面的只读标定命令；程序会持续更新网页。按照终端提示，手动把夹爪中心依次放到铰链抓取点上方 100 mm、50 mm、0 mm，每个位置按一次回车，程序采集 5 个通过 STEP PnP 的 RGB 帧并取中位数。整个过程不发送机器人或夹爪命令。

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/run_left_step_rgb.py \
  --profile g2_adapter/left_step_profile.json \
  --yolo-model models/left_d405_hinge_yolo/best.pt \
  --save-height-reference g2_adapter/left_step_reference.json \
  --height-capture-frames 5 \
  --web-host 0.0.0.0 --web-port 8777
```

网页：`http://127.0.0.1:8777`。保存的 `left_step_reference.json` 含三个高度、每个高度的相机距离和目标像素。运行抓取时，程序按当前 STEP PnP 的 `grasp_xyz_camera_m` 距离在相邻标定点之间线性插值；超出 0–100 mm 范围时使用最近端点。日志字段 `height_reference` 会记录当前使用的上下端点和插值比例。旧的单点 `sample_count >= 20` 文件仍可继续使用，但不会做高度补偿。

如需兼容旧流程，也可以只在最终下探前采集一个 20 帧基准（只读，不发运动）；该方式没有高度补偿：

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/run_left_step_rgb.py \
  --profile g2_adapter/left_step_profile.json \
  --save-reference g2_adapter/left_step_reference.json \
  --web-port 8777
```

首次先做 XY 对准测试，不进入下探：

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/left_grasp_gdk.py \
  --step-profile g2_adapter/left_step_profile.json \
  --step-reference g2_adapter/left_step_reference.json \
  --yolo-model models/left_d405_hinge_shaft_yolo/best.pt \
  --selection leftmost --stop-after xy --speed-mm-s 5 \
  --web-port 8767 --allow-motion
```

完整左手流程将 `--stop-after xy` 替换成 `--full-cycle`。串联命令追加 `--left-step-profile g2_adapter/left_step_profile.json --left-step-reference g2_adapter/left_step_reference.json`。省略 STEP 参数仍使用原流程。修改 STEP/抓取点或相机内参后必须重新采集基准，启动会核对它们。
