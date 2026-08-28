# G2 traditional 自动流程

本项目运行架构为纯 `Python + GDK 4.1.5`，目标机器人是
`G2_t2v2_crsB + Omnipicker`。不依赖 ROS 2、MoveIt、ros2_control 或额外重力补偿
控制器。

## 数据流

```text
GDK 头部/双手 RGB-D
→ 传统 SAM/目标检测与三维定位
→ G2 URDF 运动学、关节限位和视觉误差转换
→ GDK joint_control_request() / move_arm_joint_servo()
→ GDK move_end_effector_joint()
```

预设点使用 `joint_control_request()`，由 GDK 执行规划和碰撞检查。连续视觉闭环使用
`move_arm_joint_servo()`，仍保留软件关节限位、速度限制、100 Hz 插值、状态错误、
急停、NaN/Inf 和超时检查。代码不会修改或关闭 GDK 碰撞配置。

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
`--inference-fps` 按显卡负载调整。

## 主要文件

- `g2_adapter/control.py`：唯一 GDK 运动调用边界；
- `g2_adapter/camera.py`：GDK RGB-D 同步读取与解码；
- `g2_adapter/safety.py`：关节、状态、速度和夹爪安全规则；
- `g2_adapter/record_pose.py`：只读记录左手 P1/P2/P3；
- `g2_adapter/run_preset.py`：只读预检或执行一个已记录预设；
- `g2_adapter/inspect_cameras.py`：只读检查实际相机流和 binding；
- `g2_adapter/workcell.example.json`：待填充工位配置。

## 记录预设姿态

由操作员手动将机器人移动到目标位置，然后运行只读记录：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/record_pose.py --name left_p1 --save
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/record_pose.py --name left_p2 --save
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/record_pose.py --name left_p3 --save
```

结果写入 `g2_adapter/recorded_poses.json`。脚本记录指定侧 7 关节、双臂
14 关节、全身 motor position 和两侧 Omnipicker，不发送任何控制命令。

## 预设点预检与执行

不加参数时只读：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/run_preset.py --preset left_p1
```

只有操作员完成现场检查后才可增加唯一运动确认参数：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/run_preset.py \
  --preset left_p1 \
  --allow-motion
```

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
`develop` 模式，并重新运行相机探针。该过程改变相机发布配置，不属于运动控制，但会增加
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

P1/P2/P3 仅用于预移动，不能作为检测数据集。操作员完成预移动、机器人到达最终检测位后，
再执行采集（`left_final_detection` 可替换为任意场景名）。脚本只读取相机和 GDK TF，数据
写到 `outputs/left_d405_gdk/<scene>/`；不保存到 Git，也不发送运动或夹爪命令。

采集后，传统左手视觉定位改由 `localize_left_d405_sam3.py` 消费 GDK 数据，而不再打开
RealSense。它只消费最终检测位的采集，不消费 P1/P2/P3 的预移动数据：

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
