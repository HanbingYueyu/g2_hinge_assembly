# G2 纯 GDK 适配层

这里是当前 `G2_t2v2_crsB + Omnipicker + GDK 4.1.5` 的传统视觉流程控制边界。

当前已经落实：

- 双臂顺序固定为左 7 + 右 7，按关节名读取 GDK 状态；
- 使用当前 G2 URDF 的位置限位，拒绝 NaN/Inf 和越界目标；
- 100 Hz 调用官方 `move_arm_joint_servo()`，轨迹段先检查软件速度上限；
- 每段从当前实测状态开始，并周期性复核全身/运动错误和急停字段；
- Omnipicker 只接受 `[-pi/4, 0] rad`，组名为 `left_tool/right_tool`；
- 所有控制默认只读，真机运动只通过 `allow_motion` 参数启用；
- 任一异常、超时或取消都会停止继续发送新命令；不会假设存在未确认的 `stop()`。

### P4 后右手 hole/target 检测（当前只读）

`right_hole_target_detect_gdk.py` 用训练好的 `hole/target` YOLO 框检测模型读取右手彩色相机。它假设前序左手抓取、上抬和 P4 已完成；带 `--allow-motion` 时只把右臂移动到记录的 `right_post_grasp` 姿态，随后持续输出同一帧中的 `hole`（固定工装孔）和 `target`（被夹铰链销轴）框、置信度和中心误差。当前阶段不执行 XY 对孔、下探、力控或释放夹爪。

检测模型只有 2D 框，不能把像素误差直接转换为毫米运动。`paired` 仅表示同一帧同时看到两类目标；目标缺失、过期或出现多候选时都保持检测状态，不产生运动建议。后续对孔必须在 P4 姿态单独标定像素到右臂基座水平位移的 Jacobian，并另行定义力传感器成功条件。

如果需要先验证 SAM3 是否能理解这两个物体，可运行 `right_hole_target_sam3_preview.py`。它在同一右手彩色帧上分别使用 `the exposed cylindrical metal pin at the end of the hinge` 和 `the white circular opening in the fixture` 两个文本提示，只读显示掩膜，不接入动作状态机。SAM3 的英文提示应描述可见外观和用途；“axis”容易被理解成数学轴，优先使用 `pin`、`shaft`、`cylindrical metal pin`。

`annotate_hole_target_video.py` 是对应的快速框标注器：输入视频后按间隔抽帧，用当前 `hole/target` YOLO 低阈值预标，浏览器中只需补漏框、删误框或拖动边界，最终写出标准 YOLO `images/`、`labels/` 和 `classes.txt`。它不连接 GDK；默认每 6 帧取一帧，适合 30 FPS 视频快速建立初始数据集。

如果需要直接采集实时数据，使用 `capture_right_hole_target_yolo_dataset.py`。它只打开右手彩色相机，网页持续刷新实时画面；冻结时运行一次 YOLO 预标，修正后保存为 `hole/target` 框。空画面也可以保存，作为负样本。

更推荐先用 `record_right_camera_video.py` 录制右手彩色视频，再用 `annotate_hole_target_video.py` 离线抽帧和修框。这样可以重复回放、补标困难帧，并按场景分组划分训练/测试集。

多台电脑完成标注后，使用 `merge_hole_target_annotations.py` 合并各目录的 `images/` 和 `labels/`，生成可直接训练的 `data.yaml`。

`preflight.py` 和 `record_pose.py` 只读状态，不含任何运动 API 调用。
`control.py` 直接封装 GDK 的规划式关节运动、100 Hz 伺服与 Omnipicker。
`camera.py` 直接读取 GDK 的头部、左手和右手 RGB-D，不依赖 ROS 或 RealSense。
左手 D405 的彩色、深度内参不同，即便帧尺寸相同也不能同像素取深度；
`capture_left_d405_gdk.py` 使用 GDK 的 `kLeftHandDepthToLeftHandColor` 外参，把深度投影至
彩色坐标系，并保存 `kLeftHandRGBDToArmLEndLink` 外参，作为传统左手视觉定位的 GDK 输入。
`camera_web_viewer.py` 是只读网页查看器；它显式请求六路 RGB-D，在浏览器显示三张彩色
和三张深度伪彩图，不保存图像。默认只监听本机回环地址：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/camera_web_viewer.py
```

在机器人本机浏览器打开 `http://127.0.0.1:8765`，用 `Ctrl-C` 关闭。若浏览器在另一台可信
主机上，操作员可自行指定机器人网卡地址，例如 `--host 192.168.x.x`；该模式没有鉴权，勿
绑定至公网地址。`--browser-fps` 默认是 4，可按设备负载降低。
鼠标悬停在任一深度图上可实时读取该像素的 Z16 深度（米）。

默认 GDK Python binding 使用官方 Python 3.10。若已执行
`scripts/build_gdk_py312.sh`，可用 `scripts/run_gdk_sam3_python.sh` 在同一 Python 3.12
进程中访问 GDK 和 SAM3；该 binding 与官方 3.10 安装并存，不会替换它。

使用单进程实时检测网页（默认最左候选、托盘 ROI、1 FPS 推理）：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/live_left_d405_sam3_web.py
```

在机器人本机浏览器打开 `http://127.0.0.1:8766`。页面显示 RGB 叠加、候选和深度状态，
不保存图像；`Ctrl-C` 停止。它只使用左手 GDK `Camera`、TF 和本地 SAM3，不导入或调用
机器人、夹爪控制接口。当前最左候选深度不足时，页面状态为 `no_usable_target`，不会伪造
三维抓取点。SAM3 只接收彩色图像中心 70%（每边去掉 15%）；完整画面上的黄色框就是实际
推理视野。掩膜随后会恢复到完整图像坐标，所以网页坐标、深度计算、标定和视觉伺服仍是原始
`1280×1056` 像素坐标。

### 可选：YOLO 主检测 + SAM3 低频复核

`models/left_d405_hinge_yolo/best.pt` 是左 D405 的单类铰链 YOLO11n-seg 权重；该模型目录被
忽略，不提交到 Git。先安装只影响 Python 3.12 SAM3 环境的依赖：

```bash
uv pip install --python .venv-sam3/bin/python "ultralytics==8.4.80"
```

使用下面的**只读**入口验证。训练相机的完整画面是 `680×420`；为不拉伸 G2 的原始 `1280×1056`
画面，YOLO 先取与 `680:420` 同宽高比的最大中心区域（当前为全宽 `1280×791`，即
`[0,132,1280,923]`），再缩放到 `680×420` 推理。这样保留大部分视野，而不是取一个固定
`680×420` 像素窗口。YOLO 输出掩膜会按最近邻放回完整图像坐标。只有中心位于黄色框标示中心 70% 内的候选才有资格被选择；紫色框是 YOLO 的源视野，黄色框是 SAM3/作业 ROI。SAM3 保持中心 70% 输入。
YOLO 默认不作软件 FPS 限制，SAM3 每秒复核一次同一 RGB-D 帧；仅两者中心距离不超过 80 px、
深度中值差不超过 50 mm 时，网页状态才会为 `accepted` 并显示 `published_target`。`--yolo-confidence`
与 `--yolo-target-confidence` 均默认 `0.01`，以保留本模型的低分铰链；网页会显示所有满足面积与 ROI 的
YOLO 掩膜及各自的有效深度点数。只有同时具有有效深度的候选才能被选择或交给 SAM3 复核。SAM3 不再独立选择“最左”掩膜，而是只复核距离该 YOLO 候选最近的掩膜。该程序不创建
`Robot`、不发送运动或夹爪命令：

```bash
./scripts/run_gdk_sam3_python.sh \
  g2_adapter/live_left_d405_yolo_sam_review_web.py \
  --host 0.0.0.0 --port 8767
```

权重训练于 `ultralytics 8.4.80`，运行环境固定为该版本。原模型包建议 YOLO 阈值为 `0.35`，但
当前项目中保存的一张 2026-08-28 历史图在该阈值没有输出，中心 `680×420` 图中三件的分数约为
`0.177`、`0.154`、`0.016`，故默认使用 `0.01` 并依赖深度与 SAM3 复核兜底。`680×420` 是模型训练相机的完整画面分辨率；运行时使用同宽高比的最大中心区域，而不是固定像素裁切或拉伸整幅图。
在多组当前 GDK 画面都稳定 `accepted` 前，该入口不会接入抓取状态机。

### 重新采集并训练 G2 原生 YOLO 数据集

现有权重来自另一套 `680×420` 相机视角，和 G2 左 D405 的原生 `1280×1056` 视角、光照、托盘位置
不一致。不要继续用它的低置信度输出来自动生成真值；应重新采集 G2 原生彩色图，并人工勾出每个可见
铰链的轮廓。`capture_left_d405_hinge_yolo_dataset.py` 是浏览器标注器：冻结一帧后会自动按提示词运行一次 SAM，
再逐个确认、删除误检或清空后手画；SAM 建议绝不会自动保存。用左键沿铰链边缘点多边形，
点完一个铰链按“完成当前实例”，一帧中所有铰链都标完再保存。没有铰链的空托盘、机械臂遮挡、手进入画面的
负样本也要保存为空标签。

建议至少采集 120 个正样本帧（每帧可含 1--3 个铰链）和 40 个负样本帧；覆盖左/中/右位置、不同数量、
轻微姿态变化、亮暗变化、夹爪遮挡和空托盘。不要为采集脚本移动机器人，机器人位置由操作员用平板调整。

```bash
cd /home/flyfuture/traditional_auto_flow_g2_migration

./scripts/run_gdk_sam3_python.sh \
  g2_adapter/capture_left_d405_hinge_yolo_dataset.py \
  --host 0.0.0.0 --port 8770 \
  --sam-prompt "silver triangular hinge"
```

在浏览器打开 `http://<机器人IP>:8770`。原始帧和标签写入被忽略跟踪的
`datasets/left_d405_hinge_yolo_raw_g2_v1/`。结束采集后构建确定性的 train/val/test 划分：

```bash
./.venv-sam3/bin/python g2_adapter/prepare_left_d405_hinge_yolo_dataset.py \
  --min-positive 120 --min-negative 40
```

然后训练新的 G2 原生模型。YOLO 会以 `imgsz=640` 对每个原生图等比 letterbox；不再把运行时图像裁成
固定的 `680×420` 或拉伸图像。首次使用 `yolo11n-seg.pt` 时，Ultralytics 若本机缓存没有该初始化权重，
会下载它。

```bash
./.venv-sam3/bin/python g2_adapter/train_left_d405_hinge_yolo.py \
  --epochs 150 --batch 8 --name g2_native_v1
```

训练完成后先在验证集和实时只读页面检查新权重，确认三件铰链及负样本均正常，再替换运行时权重；新模型
测试时必须使用原生输入模式：

```bash
./scripts/run_gdk_sam3_python.sh \
  g2_adapter/live_left_d405_yolo_sam_review_web.py \
  --yolo-model models/left_d405_hinge_yolo/best.pt \
  --yolo-input-mode native \
  --host 0.0.0.0 --port 8767
```

`native` 会将完整 G2 图交给 YOLO，由 `imgsz=640` 的内部 letterbox 保持比例。旧模型若仍需对比，
保留默认的 `aspect-crop`。两种模式均只读，均不接入抓取状态机。

左手 P2/P3 是预移动路径点，不是检测工位；抓取确认后的运输流程会先在 `base_link +Z` 上抬 100 mm，随后运行 P4（后续工位点）。机器人由操作员完成预移动并到达最终
检测位后，再采集一组可供视觉定位复现的 GDK RGB-D 数据：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/capture_left_d405_gdk.py --scene left_final_detection
```

`left_final_detection` 是可自由命名的最终检测场景。结果写在忽略跟踪的
`outputs/left_d405_gdk/<scene>/`；含原始深度、投影到彩色坐标系的深度、实时内外参和时间戳。
对已采集样本做离线 SAM3 定位时，使用独立的 Python 3.12 环境（它不加载 GDK，也不发送
任何机器人命令）：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_sam3_python.sh \
  g2_adapter/localize_left_d405_sam3.py --scene left_final_detection
```

脚本取该最终检测场景最新一次采集，使用已对齐的深度和 GDK 标定，将结果写到该采集目录的
`sam3_localization/`：`overlay.jpg`、`masks.npy`、`localization.json`。JSON 同时包含左手
彩色相机系坐标和按 `kLeftHandRGBDToArmLEndLink` 变换后的左臂末端链接系坐标。
默认提示词为 `triangular perforated metal plate`。完整的几何描述在当前 SAM3 文本接口上会
过于严格、可能返回零掩膜；如需现场对比，可用 `--prompt "..."` 覆盖它。
若需要按托盘左右位置取件，可用像素检测区域和选择策略过滤夹爪，例如
`--roi 330,430,1000,760 --selection leftmost`。仅会改变离线定位的候选选择，不会发送运动。
`kinematics.py` 直接解析 G2 URDF，提供 FK、Jacobian 和带步长限制的阻尼最小二乘视觉修正。

`workcell.example.json` 中的左手 P2/P3/P4 故意为空。请使用 `record_pose.py`
在实际工位逐点读取；完成左手验证前不得启动完整自动流程。

只读预检命令（由操作员执行，Codex 不启动机器人）：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/preflight.py
```

运行 `run_preset.py` 时，`--allow-motion` 是唯一运动确认参数。未指定时只读取并
打印当前状态和目标。启动前仍必须满足《G2机器人控制注意事项与依赖路径.md》的
现场、网络、PTP、急停和状态检查。

当前左手视觉抓取是**纯位控**。根据 GDK 离线文档，`base` 应用模式支持机器人运动控制，`develop`
用于自定义相机开关；因此操作员应先切到 `base` 并重启机器人/GDK 服务，再以只读方式确认左 D405 RGB-D
仍可用，最后才可运行带 `--allow-motion` 的抓取。脚本不会切换 GDK 应用模式或 GDK control mode，也不会
重启机器人/GDK 服务。若重启后的 `base` 中左 D405 深度不可用，则这是需要先解决的相机配置问题，不能假设
在 `develop` 中执行位控可行。

注意：GDK 的应用模式 `base/develop` 与运动 `control_mode` 是两件事。纯位控要求运行前的
`control_mode=1`；`control_mode=3` 是关节阻抗，虽然 GDK 允许它调用末端位姿接口，但本项目不使用它。
如状态显示为 3，可由操作员执行以下命令切换到 1；它只调用 `set_control_mode(1)`，不发送任何运动目标：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/set_left_position_control_mode.py --allow-motion
```

## 单轴 XY 映射标定

完整视觉伺服之前，先实测 `base_link` 的水平轴与左 D405 像素坐标的对应关系。`probe_left_xy_mapping.py`
从 P3 开始，动作前、后各锁定同一个 SAM3 目标至少三帧；`--allow-motion` 时最多通过与抓取完全相同的
GDK mode-1 `end_effector_pose_control()`，以 50 Hz 插补发送一次 1 mm 的基座 `+X` 或 `+Y` 末端位姿运动。
它不控制夹爪、不下探、不回退、不切模式，也绝不重获取第二个目标。旧的“自算 IK + 关节位置伺服”标定
已作废，不能用于这个末端控制器。

先由操作员执行 P2→P3，再运行 `+X` 测试：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/probe_left_xy_mapping.py --axis x --direction positive \
  --selection second-leftmost --allow-motion
```

结果中的 `pixel_delta_post_minus_pre_px` 即该一步造成的 `[Δu, Δv]`；同时记录
`actual_end_link_base_delta_m`，以 GDK `motion_control_status.frame_poses` 为准。把完整 JSON 提供回来后，再重复 P2→P3 并测试
`--axis y --direction positive`。这两组实测结果确定后，才修改连续视觉伺服的 XY 映射。

当前配置中的旧 `pixel_jacobian_px_per_m` 已被作废，因此在新标定前抓取程序只可进行只读检测，不能产生
 视觉伺服动作。两组 GDK 末端位姿实测完成后，将它们的“实际 XY 位移矩阵”与“像素位移矩阵”相除得到新的
`pixel_jacobian_px_per_m`，同时写入 `mapping_controller=gdk_mode1_end_effector_pose_control`；正式抓取才会按该
矩阵的反矩阵求解 `base_link` XY 纠偏。

方向复核使用 `--single-xy-step`：它可等待多帧以跨过 SAM3 的单帧波动，但只要得到有效目标就最多发送一次
XY 末端位姿纠偏，然后立刻停止；不下探、不操作夹爪。每帧日志均包含 `candidate_diagnostics` 与
`eligible_candidate_indices`，可区分“未识别”与“掩膜无有效深度”。

完成 `+X`、`+Y` 两次试验后，可离线生成并写入映射（该命令不初始化 GDK、不接触机器人，也不会把
`calibrated` 改为 `true`）：

```bash
python3 g2_adapter/derive_left_xy_mapping.py \
  --x-log logs/left_xy_mapping_end_pose_<x 时间戳>.jsonl \
  --y-log logs/left_xy_mapping_end_pose_<y 时间戳>.jsonl \
  --write-config
```

## 左手最终位视觉抓取（待现场标定）

`left_grasp_gdk.py` 迁移了旧 OpenArm 的抓取状态机，但不再依赖 ROS、MoveIt 或额外重力补偿：

1. 默认仅在操作员已到达**最终检测位**后运行；带 `--full-cycle` 时，同一进程会在 YOLO/SAM3 预加载完成后执行 P2→P3；
2. 默认使用新训练的 G2 原生 YOLO 选择具有足够对齐深度的视觉最左目标；同一目标必须通过 SAM3 低频复核才会交给 XY 状态机。复核未通过、过期或目标缺少深度时不发送运动；
3. 连续对心确认后，锁定 `arm_l_end_link` 法兰姿态；视觉深度仅用于确认目标可用，不能请求额外下探；
4. 以 50 Hz、固定法兰姿态进行最多 250 mm 的力优先接触搜索。追加 `--final-descent-visual-servo` 后，会仅对下探前锁定的同一实例做实时 XY 复位：每次最多 1 mm，默认不设累计 XY 上限（可用 `--final-descent-visual-max-total-m` 设正值重新启用上限）；相对力达到接触阈值的 50% 即冻结视觉修正。视觉绝不改变 Z 向下探距离。相对力首次达到 2 N 都会停止向下；目标真正连续丢失仍会停止并进入安全恢复。接触力 ≤3 N 时保持当前位置直接闭爪；仅接触力 >3 N 才最多上提 15 mm，并在相对力连续 3 帧 ≤3 N 后闭爪。未触发接触或未能卸力均拒绝闭爪；
5. Omnipicker 先向 `0 rad` 收拢以测得本次铰链的卡住角度；角度稳定 0.5 秒（变化 ≤0.005 rad）后，仅从该角度再闭合 0.03 rad。随后先以 `base_link +Z` 上抬 20 mm，并连续比较同一 YOLO 实例的像素中心和掩膜面积：若它未随夹爪移动，则张爪、回到视觉对准并重新抓取；验证通过后，`--full-cycle` 会持续锁定并监控该同一实例完成 100 mm 上抬，且必须取得一帧新的健康观测才会请求 P4。P4 请求前掉件会张爪并重新抓取；观测暂时陈旧时保持闭爪并等待，只有新鲜观测确认实例丢失或形态异常才进入恢复；P4 请求已发出后若确认异常，程序停止发送后续命令并报错，**不会**在运动中的机械臂上自动重抓（当前 GDK 没有经验证的规划运动抢占接口）。

完整抓取先以 GDK `left_end_state` 确认左 Omnipicker 已张开至配置的 `-0.58875 rad`（机械最大张开量的 75%）附近，才允许开始视觉伺服；单次 GDK 接口调用成功不再被视为张开成功。若 6 秒内未收到符合容差的反馈，流程在 P3 停止，不会下探或闭爪。

受看护的 `--stop-after xy` 与 `--stop-after descend` 不会改变夹爪状态；只有完整且已标定的抓取流程才会先张开夹爪。

左手相机外参的终点是 GDK `arm_l_end_link`，末端命令的目标位置也以此法兰在 `base_link` 下的位姿表示；
不得改用带固定额外旋转的 `gripper_l_base_link`。

其中所有物理量必须为 G2 实测值，不能复制 OpenArm 的值。左 Omnipicker 已实测为
**机械最大张开 `-0.785 rad`、机械全闭合 `0.0 rad`**；当前运行时张开目标为 **75% (`-0.58875 rad`)**，以避开周围铰链。配置不再包含固定 `close_rad`。实际夹紧角度由每次闭合的稳定卡住角度加 `0.03 rad` 得出。配置中的 `target_pixel_px`
必须在“夹爪将要最终下探”的实际姿态读取，它是目标在左 D405 图像中的中心（视觉伺服的抓取参考点）。
`pre_descent_*` 为兼容旧配置保留，不参与竖直运动决策；G2 基座坐标为 `+X` 前、`+Y` 左、`+Z` 上。完整抓取仅当这些值已实测且
`calibrated=true` 后，`--allow-motion` 才会被接受。
默认不带该参数时可检查实时相机、YOLO/SAM3 复核、最左目标锁定和深度，且日志只写入 `logs/`：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/left_grasp_gdk.py \
  --config g2_adapter/left_grasp.example.json \
  --yolo-model models/left_d405_hinge_yolo/best.pt \
  --max-frames 20
```

该命令是只读预览，不打开/关闭夹爪、不移动机械臂、也不切换任何模式。真机运行仅在操作员完成现场检查后
使用同一命令追加唯一确认参数 `--allow-motion`；Codex 不执行该命令。程序读取当前
`base_link→arm_l_end_link` 位姿，并以 GDK `end_effector_pose_control()` 在 mode 1 直接发布目标；发送器固定为
50 Hz，并将位置变化限为配置速度（当前默认 `10 mm/s`，可用 `--speed-mm-s` 在 `1～18 mm/s` 内调整），不发送阶跃目标。目标丢失、重新获取、深度、GDK 返回值或运行异常都会停止
后续末端目标。XY 对准后，程序以 50 Hz 进行最多 250 mm 的力接触搜索，首次相对力达到 2 N 即停止向下；GDK 状态实测位移不足或横向
偏差超限时不闭爪。末端起点和最终位移均从该状态的 `frame_poses` 读取，不使用 TF 查询作为末端控制反馈。该 GDK 接口没有路径规划或碰撞检测，仅用于已验证的局部小步；抓取完成后保持当前位置和夹爪状态，不会自动回 P3。

最后一个 XY 小步后，程序会先最多等待 5 秒，确认 50 Hz 末端位置流已经到达最后的水平设定点，才允许构造向下目标；等待期间仅保持现有 XY 目标。若仍未到位则停止且不下探。可用 `--pre-final-settle-timeout-s` 在 `1--10` 秒内调整该保护等待。

视觉候选按 YOLO 置信度、面积和 ROI 的最左顺序选择；对齐深度仅记录诊断，不再筛掉最左铰链或参与下探决策。真机动作确认仍只有 `--allow-motion`。

已记录 P2/P3/P4 后，完整连续作业只运行一次以下脚本。它只初始化一次 GDK，并在 P2 请求前加载一次 YOLO 和一次 SAM3；任一预设、视觉抓取、接触卸力或夹爪确认失败都会阻止后续阶段：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/left_grasp_gdk.py \
  --full-cycle \
  --selection leftmost \
  --speed-mm-s 10 \
  --web-host 0.0.0.0 \
  --web-port 8766 \
  --allow-motion
```

不要把 `--full-cycle` 与 `--stop-after` 或 `--single-xy-step` 合用；这三者是互斥的运行模式。
P2/P3 的规划接口可将 GDK `control_mode` 留在 `3`；全流程会在 P3 到位后显式切回并确认 `control_mode=1`，随后才张开夹爪或发布视觉伺服末端目标。这是 GDK 控制状态切换，不是 `base/develop` 应用模式切换，也不重启 GDK 或 SAM3。

需要在抓取时实时监看同一份 SAM3 推理结果时，**不要**另起 `live_left_d405_sam3_web.py`；在抓取命令
追加 `--web-port 8766` 即可。浏览器打开 `http://127.0.0.1:8766`。内嵌网页不创建第二个相机或模型，
仅显示抓取循环正在使用的掩膜、最左目标、状态和深度；黄色框显示 SAM3 的中心 70% 实际输入范围。从最终力接触下探开始，会持续复用同一个已加载的 SAM3 模型更新叠加图，直至自适应闭爪、上抬和 P4 完成。网页仅供观察，不参与下探、上抬或夹爪决策，也没有控制接口。

首次真机验证应使用受限阶段测试，且 `--allow-motion` 仍是唯一的动作确认参数：

```bash
# 仅视觉水平对准；对准完成即停止，绝不下探或闭爪。
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/left_grasp_gdk.py --web-port 8766 --stop-after xy --allow-motion

# 重新回到 P3 后：XY 对准、完成最多 250 mm 的力优先接触搜索，绝不闭爪。
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/left_grasp_gdk.py --web-port 8766 --stop-after descend \
  --allow-motion
```

阶段测试日志会给出 `test_stop`；达到限制即停止发送后续命令。`--stop-after xy` 和
`--stop-after descend` 均可在 `calibrated=false` 时运行，且都不会发送闭爪命令。确认水平对准和
下探位移均正确后，才将配置设为 `calibrated=true` 并省略 `--stop-after` 执行完整抓取。

`probe_left_cartesian_contact.py` 和 `inspect_force_control.py` 仅为后续力控排障保留，当前位控抓取
流程不会调用它们；在 GDK 的 mode 2 问题解决前，请不要把接触力作为闭爪条件。
