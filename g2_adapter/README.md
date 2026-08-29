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

1. 默认仅在操作员已到达**最终检测位**后运行；带 `--full-cycle` 时，同一进程会在 SAM3 预加载完成后执行 P2→P3；
2. SAM3 只在具有足够对齐深度的候选中锁定视觉最左目标，先在 `base_link` 水平面完成 XY 对心；
3. 连续对心确认后，锁定 `arm_l_end_link` 法兰姿态；视觉深度仅用于确认目标可用，不能请求额外下探；
4. 以 50 Hz、固定法兰姿态进行最多 250 mm 的力优先接触搜索；左末端相对力首次达到 2 N 时立即停止下探；若接触力 ≤3 N，保持当前位置直接闭合 Omnipicker；仅接触力 >3 N 时才最多上提 15 mm，并在相对力连续 3 帧 ≤3 N 后闭爪。未触发接触或未能卸力均拒绝闭爪；
5. Omnipicker 先向 `0 rad` 收拢以测得本次铰链的卡住角度；角度稳定 0.5 秒（变化 ≤0.005 rad）后，仅从该角度再闭合 0.03 rad。它不再要求某个固定夹紧角度；`--full-cycle` 随后以 `base_link +Z` 上抬 100 mm，并执行 P4，期间不再发送夹爪命令。

完整抓取先以 GDK `left_end_state` 确认左 Omnipicker 已张开至 `-0.785 rad` 附近，才允许开始视觉伺服；单次 GDK 接口调用成功不再被视为张开成功。若 6 秒内未收到符合容差的反馈，流程在 P3 停止，不会下探或闭爪。

受看护的 `--stop-after xy` 与 `--stop-after descend` 不会改变夹爪状态；只有完整且已标定的抓取流程才会先张开夹爪。

左手相机外参的终点是 GDK `arm_l_end_link`，末端命令的目标位置也以此法兰在 `base_link` 下的位姿表示；
不得改用带固定额外旋转的 `gripper_l_base_link`。

其中所有物理量必须为 G2 实测值，不能复制 OpenArm 的值。左 Omnipicker 已实测为
**张开 `-0.785 rad`、机械全闭合 `0.0 rad`**；配置不再包含固定 `close_rad`。实际夹紧角度由每次闭合的稳定卡住角度加 `0.03 rad` 得出。配置中的 `target_pixel_px`
必须在“夹爪将要最终下探”的实际姿态读取，它是目标在左 D405 图像中的中心（视觉伺服的抓取参考点）。
`pre_descent_*` 为兼容旧配置保留，不参与竖直运动决策；G2 基座坐标为 `+X` 前、`+Y` 左、`+Z` 上。完整抓取仅当这些值已实测且
`calibrated=true` 后，`--allow-motion` 才会被接受。
默认不带该参数时可检查实时相机、SAM3、最左目标锁定和深度，且日志只写入 `logs/`：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/left_grasp_gdk.py \
  --config g2_adapter/left_grasp.example.json --max-frames 20
```

该命令是只读预览，不打开/关闭夹爪、不移动机械臂、也不切换任何模式。真机运行仅在操作员完成现场检查后
使用同一命令追加唯一确认参数 `--allow-motion`；Codex 不执行该命令。程序读取当前
`base_link→arm_l_end_link` 位姿，并以 GDK `end_effector_pose_control()` 在 mode 1 直接发布目标；发送器固定为
50 Hz，并将位置变化限为 `0.01 m/s`，不发送阶跃目标。目标丢失、重新获取、深度、GDK 返回值或运行异常都会停止
后续末端目标。XY 对准后，程序以 50 Hz 进行最多 250 mm 的力接触搜索，首次相对力达到 2 N 即停止向下；GDK 状态实测位移不足或横向
偏差超限时不闭爪。末端起点和最终位移均从该状态的 `frame_poses` 读取，不使用 TF 查询作为末端控制反馈。该 GDK 接口没有路径规划或碰撞检测，仅用于已验证的局部小步；抓取完成后保持当前位置和夹爪状态，不会自动回 P3。

已记录 P2/P3/P4 后，完整连续作业只运行一次以下脚本。它只初始化一次 GDK、只加载一次 SAM3，SAM3 在 P2 请求前已经加载完成；任一预设、视觉抓取、接触卸力或夹爪确认失败都会阻止后续阶段：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/left_grasp_gdk.py \
  --full-cycle \
  --selection leftmost \
  --web-host 0.0.0.0 \
  --web-port 8766 \
  --allow-motion
```

不要把 `--full-cycle` 与 `--stop-after` 或 `--single-xy-step` 合用；这三者是互斥的运行模式。
P2/P3 的规划接口可将 GDK `control_mode` 留在 `3`；全流程会在 P3 到位后显式切回并确认 `control_mode=1`，随后才张开夹爪或发布视觉伺服末端目标。这是 GDK 控制状态切换，不是 `base/develop` 应用模式切换，也不重启 GDK 或 SAM3。

需要在抓取时实时监看同一份 SAM3 推理结果时，**不要**另起 `live_left_d405_sam3_web.py`；在抓取命令
追加 `--web-port 8766` 即可。浏览器打开 `http://127.0.0.1:8766`。内嵌网页不创建第二个相机或模型，
仅显示抓取循环正在使用的掩膜、最左目标、状态和深度；黄色框显示 SAM3 的中心 70% 实际输入范围。从最终力接触下探开始，会持续复用同一个已加载的 SAM3 模型更新叠加图，直至自适应闭爪、上抬和 P4 完成。这些结果仅供显示，绝不参与下探、卸力或夹爪决策。它没有控制接口。

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
