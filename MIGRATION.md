# G2 traditional 自动流程

本项目运行架构为纯 `Python + GDK 4.1.5`，目标机器人是
`G2_t2v2_crsB + Omnipicker`。不依赖 ROS 2、MoveIt、ros2_control 或额外重力补偿
控制器。

## 数据流

```text
GDK 头部/双手 RGB-D
→ 传统 SAM/目标检测与三维定位
→ 实测像素 Jacobian 与 `base_link→arm_l_end_link` 位姿
→ GDK end_effector_pose_control()（mode 1，50 Hz）
→ GDK move_end_effector_joint()
```

预设点使用 `joint_control_request()`，由 GDK 执行规划和碰撞检查。连续视觉闭环使用
`end_effector_pose_control()`，在 mode 1 下由 GDK 全身逆解，并以 50 Hz 的连续、限速目标发布。
该接口不规划且官方标注无碰撞检测，因此只允许已验证的局部小步；运行时仍检查状态、急停、NaN/Inf、
目标有效性和超时。代码不会修改或关闭 GDK 碰撞配置。

运动学直接读取项目内的：
`robot_description/G2_t2v2_crsB_omnipicker.urdf`。

项目不依赖旧的 AGI 工作目录。Python 运行环境定义在根目录的 `pyproject.toml`，
`scripts/run_gdk_python.sh` 会从官方 GDK 安装位置加载运行时并使用 Python 3.10；可用
`G2_GDK_RUNTIME` 覆盖官方 GDK 的 `app` 目录。该启动器用于 GDK 核心脚本，不解析
可选视觉依赖；`numpy` 与 `opencv-python-headless` 定义在可选 `vision` 依赖组中。

### 可选：GDK + SAM3 的 Python 3.12 单进程环境

官方 GDK 默认 binding 是 Python 3.10。项目提供安全的 CPython 3.12 构建配方，把同一套
官方 GDK C++ 库链接到 `.venv-sam3`，以便 GDK 相机与 SAM3 同进程运行。它不会删除或替换
官方 3.10 binding：

```bash
sudo apt update
sudo apt install -y libprotobuf-dev protobuf-compiler

/home/flyfuture/traditional_auto_flow_g2_migration/scripts/build_gdk_py312.sh
```

之后以 `scripts/run_gdk_sam3_python.sh` 启动 Python 3.12 的 GDK + SAM3 程序。该启动器会
保留官方动态库环境，但刻意不继承官方 Python 3.10 的 `PYTHONPATH`。

构建验证后，可用单进程实时左手检测网页：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/live_left_d405_sam3_web.py
```

默认监听 `127.0.0.1:8766`，在机器人本机浏览器查看。它只读取左手 RGB-D 和 GDK TF，
执行本地 SAM3；不创建 `Robot` 对象，不发送运动或夹爪命令。默认每秒推理一帧，可用
`--inference-fps` 按显卡负载调整。SAM3 输入固定为彩色图像中心 70%（左右、上下各排除
15%）；网页以黄色框显示该区域。掩膜会映射回完整彩色图坐标，因此深度、目标像素和视觉伺服
仍使用完整 `1280×1056` 坐标系。

## 主要文件

- `g2_adapter/control.py`：唯一 GDK 运动调用边界；
- `g2_adapter/camera.py`：GDK RGB-D 同步读取与解码；
- `g2_adapter/safety.py`：关节、状态、速度和夹爪安全规则；
- `g2_adapter/record_pose.py`：只读记录左手 P2/P3/P4；
- `g2_adapter/run_preset.py`：只读预检或执行一个已记录预设；
- `g2_adapter/run_left_premove_sequence.py`：按到位确认顺序执行左 P2→P3，P3 为最终检测位；
- `g2_adapter/run_left_post_grasp_sequence.py`：抓取确认后先在 `base_link +Z` 实测上抬 100 mm，再按到位确认执行左 P4，并保持夹爪状态；
- `g2_adapter/probe_left_xy_mapping.py`：从 P3 在 `base_link` 的单一 X 或 Y 轴发送一次 1 mm 位置伺服，并测量同一 SAM3 目标的像素变化；
- `g2_adapter/inspect_cameras.py`：只读检查实际相机流和 binding；
- `g2_adapter/workcell.example.json`：待填充工位配置。
- `g2_adapter/left_grasp_gdk.py`：最终检测位的左手 D405/SAM3 视觉抓取状态机；默认只读，
  通过唯一 `--allow-motion` 参数且完成 G2 实测标定后才会调用 GDK 运动/夹爪接口。加 `--full-cycle`
  后同一进程只初始化一次 GDK、只加载一次 SAM3，依次完成 P2→P3、抓取、`base_link +Z` 上抬 100 mm、P4。图像对心仍为
  GDK mode 1 的直接末端位姿控制；接近和闭爪全程不切换 GDK control mode。
- `g2_adapter/inspect_force_control.py`、`g2_adapter/probe_left_cartesian_contact.py`：仅为后续力控排障
  保留，当前正式抓取流程不调用。

当前左手视觉抓取使用纯位控。根据 GDK 离线文档，机器人运动控制应在 `base` 应用模式运行，`develop`
用于自定义相机开关。项目不会自动切换 GDK 应用模式或重启机器人/GDK 服务；应用模式变更后必须由
操作员重启才会生效。操作员切到 `base` 并重启后，必须先只读检查左 D405 RGB-D 仍可用，再运行任何
带 `--allow-motion` 的抓取脚本；若深度不可用，应先解决相机配置，不能假设 `develop` 下可位控。

## 记录预设姿态

由操作员手动将机器人移动到目标位置，然后运行只读记录：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/record_pose.py --name left_p2 --save
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/record_pose.py --name left_p3 --save
```

结果写入 `g2_adapter/recorded_poses.json`。左 P2→P3 是到最终检测位的顺序预移动路径，
P3 即实际执行左手视觉检测和抓取前的最终位置。脚本记录指定侧 7 关节、双臂
14 关节、全身 motor position 和两侧 Omnipicker，不发送任何控制命令。

## 预设点预检与执行

不加参数时只读：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/run_preset.py --preset left_p2
```

只有操作员完成现场检查后才可增加唯一运动确认参数：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/run_preset.py \
  --preset left_p2 \
  --allow-motion
```

### 单轴 XY 映射标定

完整视觉伺服之前，先从 P3 对基座水平坐标和左 D405 图像坐标做实测。每次运行只会发送**一次**最多
1 mm 的关节位置伺服；不控制夹爪、不下探、不自动回退、不切换 GDK 模式，且动作后只允许锁定初始目标
附近的同一 SAM3 掩膜。目标丢失时不会以其他铰链替代。
规划验收保留请求的水平主方向和至少 25% 的有效位移；仅允许不超过 0.05 mm 的 IK 竖直数值残差。

先用 P2→P3 到最终检测位，再测试 `base_link +X`：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/run_left_premove_sequence.py --allow-motion

/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/probe_left_xy_mapping.py --axis x --direction positive --allow-motion
```

输出中的 `pixel_delta_post_minus_pre_px = [Δu, Δv]` 是实测的图像变化，
`actual_tip_base_delta_m` 是动作后按实际关节反馈重新计算的末端基座位移。请保留并提供该 JSON。
然后由操作员重新执行 P2→P3，再运行相同命令但把 `--axis x` 改为 `--axis y`。这两组结果是后续校正
视觉 XY 方向的依据；没有结果前不应再运行连续 XY 对心。

当前 `left_grasp.example.json` 已包含上述两次实测得到的
`alignment.pixel_jacobian_px_per_m`。抓取的水平阶段直接求解这个实测矩阵的反矩阵，
不会再采用理论相机/TF 的水平映射；每个水平指令日志均会记录矩阵、当前像素误差和
`command_source=base_horizontal_empirical_visual_xy`。

### 最终检测位左手抓取

旧 OpenArm 的实际抓取逻辑现已按 G2 结构迁移：有有效深度的最左目标的图像 XY 对心、锁定姿态、在
`base_link` 水平面保持高度后，以已标定时长竖直向下接近、闭合 Omnipicker、读取闭合反馈确认。G2 版本读取
当前 `base_link→arm_l_end_link` 位姿，直接以 GDK `end_effector_pose_control()`（mode 1）发布小范围末端
目标；不使用 ROS、MoveIt、自写 IK 或自建重力/碰撞控制器。

GDK 左手 RGB-D 外参以 `arm_l_end_link` 为终点；视觉 FK、Jacobian、IK 和姿态锁定均使用该链接，
避免与 `gripper_l_base_link` 的固定额外旋转混用。

默认入口不控制 P2/P3，也不猜测最终检测位：操作员先完成预移动并确认已经处于
最终检测位，再启动抓取状态机。若全部 P2/P3/P4 已记录，可在同一入口追加 `--full-cycle`，由单一 GDK/SAM3 进程完成整条流程。左 Omnipicker 的机械行程已实测为张开 `-0.785 rad`、全闭合 `0.0 rad`；夹紧不再依赖固定绝对角度：它先测出当前铰链位置的稳定卡住角度，再从该角度额外闭合 `0.03 rad`；
`g2_adapter/left_grasp.example.json` 的 `target_pixel_px` 必须在夹爪将要下探的位置实测，表示视觉伺服的
抓取参考像素。视觉深度仅用于筛除无效目标，**不再用于决定垂直下探距离**；不同铰链的绝对相机深度不能作为公共高度参考。XY 连续对准后，
`approach.final_descent_speed_mps=0.01` 与 `approach.final_descent_duration_s=25.0` 定义一个最多 250 mm 的、受相对力优先保护的接触搜索行程。每次搜索前读取 20 帧新鲜基线；相对力首次达到 2 N 即停止向下。接触时相对力不超过 3 N 则保持当前位置直接闭爪；仅超过 3 N 才最多上提 15 mm，且相对力连续 3 帧降至或等于 3 N 后才闭爪。250 mm 和 15 mm 均为操作员确认的故障兜底上限。G2 基座方向经操作员确认：`+X` 前、`+Y` 左、`+Z` 上。完整抓取在文件未完成标定时会拒绝执行。
完整抓取在 P3 会先以 `left_end_state` 确认左 Omnipicker 的实际角度已到张开目标 `-0.785 rad` 附近；若 6 秒内没有这一反馈，流程停止且绝不开始视觉伺服或下探。
只读观察可由操作员运行：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_sam3_python.sh \
  g2_adapter/left_grasp_gdk.py \
  --config g2_adapter/left_grasp.example.json --max-frames 20
```

输出仅写到 `logs/left_grasp_*.jsonl`。每帧都会由实测像素 Jacobian 求出受限小步 `base_link` 位移，
并在 mode 1 直接以 GDK `end_effector_pose_control()` 控制 `arm_l_end_link`；发送器以 50 Hz 保持目标并将
平移速率限制为 `0.01 m/s`，并从 `motion_control_status.frame_poses` 读取末端起点与最终位移，不依赖 SAM3 的固定帧率，也不使用笛卡尔阻抗或 mode 2。任何无效深度、
目标丢失、目标重新获取、GDK 返回值或运行异常都会停止继续发送末端目标。XY 连续确认后，直接以 50 Hz 进行最多 250 mm 的力接触搜索；相对力达到 2 N 时立即停止向下。接触触发后，GDK 状态实测末端位移或横向偏差
超限时不会闭爪。该 GDK 接口不做路径规划或碰撞检测，仅限已验证局部小步。脚本不改 payload、TCP、扭矩零点、
重力补偿，也不会自动回 P3 或切换任何模式。

若抓取时需要浏览器监看，请在同一个 `left_grasp_gdk.py` 进程加 `--web-port 8766`，并在本机打开
`http://127.0.0.1:8766`。不要同时启动独立 `live_left_d405_sam3_web.py`：内嵌网页复用抓取进程的
GDK 相机和同一个已加载的 SAM3 模型；从力接触下探开始到自适应闭爪、上抬及 P4 完成，持续显示 SAM3 叠加结果。该结果只用于观察，不提供控制接口，也不参与下探或夹爪决策。

首次真机运行使用 `--stop-after xy --allow-motion` 验证仅水平对准；它达到对准后立即停止，不会下探或
闭爪。确认后从 P3 再运行 `--stop-after descend --allow-motion`，验证力优先的最多 250 mm 接触搜索且不闭爪。
两个受看护阶段均允许在 `calibrated=false` 时执行，且不发送闭爪命令。唯一动作确认参数仍是
`--allow-motion`。

## GDK 相机

G2 相机不按 USB 序列号打开。`camera.py` 使用 `agibot_gdk.Camera` 的头部、左手、
右手彩色和深度流，并按彩色时间戳读取最近深度帧。创建 Camera 时必须显式传入所需
相机类型，不能使用会默认启动全部相机的无参构造。实机探测（2026-08-28）确认左右手
深度已可用（1280×1056、Z16），胸口 `head_depth` 虽可读取内参但尚未收到图像；不能
在它恢复前把胸口深度接入三维定位。

项目内 [thor_camera_conf_hand_depth.json](config/thor_camera_conf_hand_depth.json) 只保留胸口和
双手 RGB-D 六路：`head_depth`、`hand_left_depth` 与 `hand_right_depth` 均为 `true`；三路鱼眼和
头部双目左右相机均关闭。先预览，不会初始化 GDK：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/configure_hand_depth.py
```

现场确认后，操作员可用唯一确认参数 `--apply` 让 GDK 接受该配置；随后按官方文档切换到
`develop` 模式并重启使配置生效，再重新运行相机探针。该过程改变相机发布配置，不属于运动控制，但会增加
传感器负载；在验证成功前不得对左右手执行依赖深度的三维定位或自动抓取。

若要隔离胸口深度问题，可改用
[thor_camera_conf_head_depth_diagnostic.json](config/thor_camera_conf_head_depth_diagnostic.json)：
它只发布胸口 RGB-D 和双手彩色，关闭双手深度、鱼眼和头部双目。该验证会暂时令双手深度
不可用，完成后须重新应用上面的常规配置。预览和应用命令如下：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/configure_hand_depth.py \
  --config config/thor_camera_conf_head_depth_diagnostic.json

/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/configure_hand_depth.py \
  --config config/thor_camera_conf_head_depth_diagnostic.json --apply
```

现有 GDK Python 示例明确提供图像、尺寸、FPS 和时间戳；安装包还包含
`GetCameraParametersService` 的内参、外参和分辨率消息。为确认当前 4.1.5 Python
binding 实际公开的参数方法，请由操作员运行：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/inspect_cameras.py
```

该脚本只读取每路一帧，不保存图像、不发送控制命令。得到输出后即可确定深度单位、
彩色/深度对齐关系及标定参数入口，再把三维定位接入 GDK 帧。

实时查看胸口和双手六路图像时，使用项目自带的只读网页查看器：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/camera_web_viewer.py
```

在机器人本机浏览器打开 `http://127.0.0.1:8765`。默认仅本机监听、不会保存图像；以
`Ctrl-C` 关闭。深度以 Z16 毫米值在浏览器端显示为 0.2–2.0 m 的伪彩图，可用
`--browser-fps` 降低浏览器刷新负载；鼠标悬停在深度图上可读取当前像素的米值。

排查胸口深度时，应只请求胸口两路，避免探针自身打开手部相机：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/inspect_cameras.py --streams head_color,head_depth
```

### 左手 D405 视觉定位输入

传统左手 SAM 定位的 RealSense 输入已由 GDK 左手 RGB-D 输入替代。彩色和深度不能以相同
像素坐标直接配对：使用 `capture_left_d405_gdk.py` 会读取 GDK 的两组内参，并通过
`SensorExtrinsicType.kLeftHandDepthToLeftHandColor` 把 Z16 深度投影到彩色坐标系；同时保存
`kLeftHandRGBDToArmLEndLink`，供后续将相机系三维点变换到左臂末端系。Z16 单位固定为毫米。

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh \
  g2_adapter/capture_left_d405_gdk.py --scene left_final_detection
```

P2/P3 仅用于预移动，不能作为检测数据集。操作员完成预移动、机器人到达最终检测位后，
再执行采集（`left_final_detection` 可替换为任意场景名）。脚本只读取相机和 GDK TF，数据
写到 `outputs/left_d405_gdk/<scene>/`；不保存到 Git，也不发送运动或夹爪命令。

采集后，传统左手视觉定位改由 `localize_left_d405_sam3.py` 消费 GDK 数据，而不再打开
RealSense。它只消费最终检测位的采集，不消费 P2/P3 的预移动数据：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_sam3_python.sh \
  g2_adapter/localize_left_d405_sam3.py --scene left_final_detection
```

该脚本在隔离的 `.venv-sam3`（Python 3.12）中运行，以匹配 SAM3；GDK 的 Python 3.10
采集过程与其分离。它只读取最新的 `left_final_detection` 采集，使用已投影到彩色坐标系的 Z16 深度、
内参和 `kLeftHandRGBDToArmLEndLink` 外参，输出相机系及左臂末端链接系三维点至
`sam3_localization/localization.json`，并生成 `overlay.jpg`。没有机器人运动、夹爪或 GDK
控制调用。默认 SAM3 提示词为较稳健的 `triangular perforated metal plate`；可通过
`--prompt` 进行现场对比。
