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

运动学直接读取：
`/home/flyfuture/agi/G2_crsB_omnipicker/genie_robot_description_210/urdf/G2_t2v2_crsB/G2_t2v2_crsB_omnipicker.urdf`。

## 主要文件

- `g2_adapter/control.py`：唯一 GDK 运动调用边界；
- `g2_adapter/camera.py`：GDK RGB-D 同步读取与解码；
- `g2_adapter/safety.py`：关节、状态、速度和夹爪安全规则；
- `g2_adapter/record_pose.py`：只读记录 P1–P4；
- `g2_adapter/run_preset.py`：只读预检或执行一个已记录预设；
- `g2_adapter/inspect_cameras.py`：只读检查实际相机流和 binding；
- `g2_adapter/workcell.example.json`：待填充工位配置。

## 记录预设姿态

由操作员手动将机器人移动到目标位置，然后运行只读记录：

```bash
cd /home/flyfuture/agi
source "$HOME/.cache/agibot/app/env.sh" "$HOME/.cache/agibot/app"

uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/record_pose.py --name left_p1 --save
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/record_pose.py --name left_p2 --save
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/record_pose.py --name left_p3 --save
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/record_pose.py --name left_p4 --save
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/record_pose.py --name right_post_grasp --save
```

结果写入 `g2_adapter/recorded_poses.json`。脚本记录指定侧 7 关节、双臂
14 关节、全身 motor position 和两侧 Omnipicker，不发送任何控制命令。

## 预设点预检与执行

不加参数时只读：

```bash
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/run_preset.py --preset left_p1
```

只有操作员完成现场检查后才可增加唯一运动确认参数：

```bash
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/run_preset.py \
  --preset left_p1 \
  --allow-motion
```

## GDK 相机

G2 相机不按 USB 序列号打开。`camera.py` 使用 `agibot_gdk.Camera` 的头部、左手、
右手彩色和深度流，并按彩色时间戳读取最近深度帧。

现有 GDK Python 示例明确提供图像、尺寸、FPS 和时间戳；安装包还包含
`GetCameraParametersService` 的内参、外参和分辨率消息。为确认当前 4.1.5 Python
binding 实际公开的参数方法，请由操作员运行：

```bash
cd /home/flyfuture/agi
source "$HOME/.cache/agibot/app/env.sh" "$HOME/.cache/agibot/app"
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/inspect_cameras.py
```

该脚本只读取每路一帧，不保存图像、不发送控制命令。得到输出后即可确定深度单位、
彩色/深度对齐关系及标定参数入口，再把三维定位接入 GDK 帧。
