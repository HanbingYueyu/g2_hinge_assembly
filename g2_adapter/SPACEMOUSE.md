# SpaceMouse Compact 控制 G2 左右末端

单臂入口：`g2_adapter/run_spacemouse_gdk.py`；左右视觉流程串联入口：`g2_adapter/run_bilateral_flow_gdk.py`。两者仅支持 Linux USB SpaceMouse Compact（`256f:c635`），通过只读、非阻塞 `hidraw` 读取，不需要安装 PySpaceMouse 或启动 spacenavd。Python 环境需要已有的 NumPy；真机使用仓库 GDK Python 3.10 启动脚本。

## 启动

列出设备（不连接机器人）：

```bash
python3 g2_adapter/run_spacemouse_gdk.py --list-devices
```

输入与模拟目标预览（不导入/初始化 GDK，不读取真实末端）：

```bash
./scripts/run_gdk_python.sh g2_adapter/run_spacemouse_gdk.py --side right
```

多设备时指定 `--device /dev/hidrawN`；也可以使用下方 udev 创建的固定别名 `/dev/spacemouse-compact`。终端显示 `preview_no_robot`、当前手、原始六轴、按钮和模拟目标；模拟初始位置为零，姿态为单位四元数。可加 `--duration-s 10` 定时退出。

真机运行：

```bash
./scripts/run_gdk_python.sh g2_adapter/run_spacemouse_gdk.py \
  --side right \
  --speed-mm-s 10 \
  --rotation-deg-s 10 \
  --allow-motion
```

真机启动时会先检查 `mode=0/1` 和故障码；程序会自动准备双臂关节阻抗 `control_mode=3` 和标准 `kp=400 400 400 500 300 200 200`，等待状态确认成功后才发送第一条末端位姿命令。手动接管期间所选手臂改用位控 `control_mode=1`（刚性保持，不会被随手推动），接管结束时恢复该臂的标准关节阻抗 `control_mode=3` 再恢复自动流程；运行中 `control_mode` 变成 1/3 以外或出现故障会停止发送。程序不会自动启动机器人或改变碰撞配置。不要与抓取程序或其他运动发送程序同时运行；设备文件锁只防止本工具重复打开同一鼠标，不能阻止其他机器人控制程序。

## 操作

1. 启动显示 `waiting_for_neutral`。轻推并扭动摇杆后完全松开，再按下并松开任一按钮。程序必须先收到两组轴数据和按钮状态，且六轴回中、按钮释放，才能显示 `ready`。这次初始按键不切手、不动夹爪；不存在按住才可运动的使能键。
2. `ready` 后，推拉控制当前手末端 XYZ，扭转控制三个旋转轴，偏转越大速度越大。位置和旋转都采用 `base_link` 坐标系；控制 `arm_l_end_link` / `arm_r_end_link`，不添加夹爪尖端 TCP 偏移。
3. **左按钮**：松开时只开放 XYZ 平移；按住时开放完整 6D（XYZ + RX/RY/RZ）。单击不会切手；在回中保持至少 0.15 秒的前提下，两次短按并释放（双击窗口 0.55 秒）才切换左右手。长按达到 0.40 秒会取消双击候选，不会切手。未回中时显示 `switch_rejected_not_centered`，本次切换不会排队。切换先向旧手发送实测保持目标，等待至少 0.12 秒，再从新手实测位姿开始；等待期间的按键不排队执行。
4. **右按钮**：单击切换当前手夹爪打开/闭合。全开 `-0.785 rad`，全闭 `0 rad`；两手分别记录切换状态，每次选中时按该手实测开度中点重新同步。夹爪目标与位姿通过同一个请求发送。若切手中止了尚未完成的夹爪动作，切回来不会自动续做；再次点击从实测开合状态切换目标。
5. 右按钮单击消抖间隔 0.25 秒，长按不连发。`Ctrl+C` / `SIGTERM` 退出。

启动电批：先按住左按钮达到约 0.40 秒，保持左按钮时按下右按钮，触发一次电批周期；两键释放前不会再次触发。两键重叠后屏蔽切手、夹爪和阶段推进手势，右按钮按得过早时本次不触发，需释放后重试。电批周期在独立线程执行，不暂停 G2 视觉或 50 Hz 位姿控制；集成流程同时后台轮询 RM1010 的 P10 下降沿，物理按钮也触发同一个周期；RM1010 错误记录为 `dianpi_error`，不会让视觉流程退出。

键盘操作：按空格触发一次相同的电批周期；按 Enter 仅用于等待门的阶段确认。空格不会推进阶段，也不会改变 SpaceMouse 的原有手势。

## 左右视觉流程串联与临时接管

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/run_bilateral_flow_gdk.py \
  --device /dev/spacemouse-compact \
  --left-web-port 8767 \
  --right-web-port 8767 \
  --spacemouse-speed-mm-s 10 \
  --spacemouse-rotation-deg-s 10 \
  --auto-speed-mm-s 10 \
  --auto-rotation-deg-s 10 \
  --auto-preset-velocity-rad-s 0.08 \
  --deadzone 0.12 \
  --allow-motion
```

统一入口先运行左手视觉 demo。左手正常完成后不会自动启动右手，而是保持左手的 GDK、SpaceMouse 接管和 Web 服务运行，等待右按钮长按 2 秒作为手动“下一步”；等待期间左按钮双击或右按钮双击仍可临时接管对应手。如果在左手流程执行中长按右按钮 2 秒，则停止左手视觉/接管，在释放左手 GDK 前先将左夹爪完全张开并确认反馈，再使用同一 GDK 连接依次移动到已记录的 `left_handoff_1 → left_handoff_2`，两点到位并释放左手 GDK 后才启动右手。左手正常完成后按下一步也执行相同交接路径；交接移动期间暂停 Web 和 SpaceMouse 服务，张爪失败不执行交接点。缺少交接点时启动前拦截。`--spacemouse-speed-mm-s` / `--spacemouse-rotation-deg-s` 只控制 SpaceMouse 接管速度（兼容旧名 `--speed-mm-s` / `--rotation-deg-s`）；`--auto-speed-mm-s` / `--auto-rotation-deg-s` 控制视觉自动末端速度；`--auto-preset-velocity-rad-s` 控制关节预设速度。未指定自动参数时使用左右 demo 原有配置。接管时视觉状态机暂停发送，但保留当前阶段和模型状态；接管不会因无操作自动退出，当前接管手再次双击才会发送实测保持目标并恢复自动流程，双击另一手则切换接管对象。左手完成后的等待状态没有自动视觉动作，释放接管后继续等待右手“下一步”。左右阶段串行复用 `http://127.0.0.1:8767`；右手阶段的右按钮长按不会重启右手流程。接管期间左按钮松开只平移，按住才开放完整 6D。

## 方向与速度

默认轴映射 `--axis-map=-2,-1,-3,-5,-4,-6` 对应 HID 到机器人常用的前/左/上约定。操作者摆放方向会影响手感，先在预览模式逐轴确认方向。

映射参数依次指定输出 `X,Y,Z,RX,RY,RZ` 从哪个原始通道取值。`1,2,3` 是平移报告三个通道，`4,5,6` 是旋转报告三个通道，负号反向。例如保持原始通道顺序：

```bash
./scripts/run_gdk_python.sh g2_adapter/run_spacemouse_gdk.py \
  --axis-map=1,2,3,4,5,6
```

`raw_axes` 显示映射前的值；位姿的四元数顺序为 XYZW。旋转通过角速度增量更新四元数，不累加欧拉角。

默认平移合速度 10 mm/s，旋转合速度 10°/s，六轴死区 0.12。`--speed-mm-s 30` 是满量程时的上限，不是任何偏转下的恒定速度；死区后的轴幅度会按比例缩放，且对角线输入仍受合速度限制。例如死区 0.10 时，原始轴值 0.5 约对应 13.3 mm/s，满量程才对应 30 mm/s。可用 `--speed-mm-s`（最高 30）、`--rotation-deg-s`（最高约 28.6）、`--deadzone` 调整。

## 停止与跟随保护

- 已移除相对于首次激活位姿的累计平移半径和累计旋转角度限制；持续推动会持续产生速度积分目标，直到回中或退出。控制循环以 50 Hz 为目标，并按实际循环间隔积分（最长按 80 ms 计一步），从最新实测位姿开始；通信/执行延迟不会把旧目标不断累积成远距离目标，也不会把设定速度按固定 20 ms 周期隐式缩小。
- 已移除目标相对实测的跟随误差保护（原 10 mm / 约 5.7°）：机器人暂时跟不上或被人为推动不再中止接管，命令仍从最新实测位姿积分，不会累积成远距离目标。GDK 状态异常、模式丢失、运动轴数据超过 150 ms 未更新或控制循环严重超时时仍会退出。
- 摇杆回中时锁存一个固定保持目标（进入回中那一刻的实测位姿），不再每拍把目标重锁到实测；空闲时关节阻抗因此对外力有回复力，不会被随手推走。再次偏转摇杆时从当前实测位姿重新积分。停止或切手时也尝试取消尚未完成的夹爪目标。
- 输入断连或异常不自动重连恢复运动。退出时若反馈健康，发送一次实测保持目标；若通信已失败，只能等待最后命令 0.10 秒的有效期结束。此机制不保证机械瞬间停止，也不是硬件急停。
- GDK 位姿接口不做轨迹碰撞规划，也没有工作空间/碰撞自动限制；必须由操作者控制移动方向和距离。单臂切换及未选中手在命令到期后的行为、六轴方向、USB 报告频率与真机停止响应仍需现场低速验证。

## Linux 权限

如果列得出设备但报无权读取，桌面本地登录会话可按设备限定设置 udev（仅需人工配置一次）：

```bash
echo 'KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0660", TAG+="uaccess", SYMLINK+="spacemouse-compact"' | sudo tee /etc/udev/rules.d/70-g2-spacemouse.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hidraw
```

然后拔插 SpaceMouse，或检查固定别名：

```bash
readlink -f /dev/spacemouse-compact
```

之后命令固定使用 `--device /dev/spacemouse-compact`，不再依赖 `hidrawN` 编号。SSH/无桌面会话不一定获得 `uaccess` 权限，应由管理员设置专用设备组并将运行用户加入该组；不要用 root 运行整个遥操作程序。

报告格式依据 [PySpaceMouse Compact 设备定义](https://github.com/JakubAndrysek/PySpaceMouse/blob/master/pyspacemouse/devices.toml)，默认方向依据其 [HID/ROS 坐标约定](https://github.com/JakubAndrysek/PySpaceMouse/blob/master/pyspacemouse/config_helpers.py)。实现只解码对应报告，没有依赖上游库的缓存时间戳。

## 离线验证

```bash
python3 -m unittest discover -s g2_adapter -p 'test_spacemouse*.py' -v
```

测试使用原始 HID 字节、非阻塞读取替身和模拟 GDK 反馈，不发送真机命令。
# 右按钮单击与长按

双手串联流程会在右手抓取成功后记录实测末端位置和姿态（日志事件 `right_grasp_return_pose_saved`）。右手自动控制因 SpaceMouse 接管暂停时以及右手流程完成后的等待状态，右手网页仍持续读取相机、检测 Tag 并刷新画面。右手完成后长按右按钮 2 秒：停止接管，按自动速度先到记录点的 base_link +Z 100 mm 上方，恢复记录姿态，再回到记录点；两段均确认到位后张开右夹爪并退出。放回期间不再接管，任一段到位失败都不松爪；Ctrl+C 直接停止，不触发放回。

SpaceMouse 发生 `EIO`、USB 短暂断开或固定设备链接暂时消失时，串联流程不会退出。程序停止当前手动位姿流并保持实测位置，暂停自动流程，每 0.5 秒尝试重新打开设备。重连后会清空旧按钮和轴状态；重新双击选择要接管的手，再次双击退出接管后才恢复自动流程。使用 `/dev/spacemouse-compact` 固定链接时，即使内核重新分配 `hidrawN` 也能恢复。

如果接管瞬间一次 GDK 状态读取超过 60 ms，程序丢弃该次已经过期的 SpaceMouse 输入和接管目标，不把它升级为整个视觉流程故障。当前自动阶段从反馈状态继续执行，操作者可在输入回中后重新双击接管。

右按钮短按后松开才切换夹爪；按住达到 0.4 秒后取消本次单击，松开也不触发夹爪。双手流程中的右按钮长按 2 秒推进阶段沿用原规则。左按钮长按期间按右按钮属于电批组合，不会切换夹爪。

独立运行 `dianpi/dp/Control.py` 时仍使用 P10 低电平下降沿触发；集成流程不要同时启动另一个占用同一 RM1010 的电批进程。
