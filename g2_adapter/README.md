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
三维抓取点。

左手 P1/P2/P3 是预移动路径点，不是检测工位。机器人由操作员完成预移动并到达最终
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

`workcell.example.json` 中的左手 P1/P2/P3 故意为空。请使用 `record_pose.py`
在实际工位逐点读取；完成左手验证前不得启动完整自动流程。

只读预检命令（由操作员执行，Codex 不启动机器人）：

```bash
/home/flyfuture/traditional_auto_flow_g2_migration/scripts/run_gdk_python.sh g2_adapter/preflight.py
```

运行 `run_preset.py` 时，`--allow-motion` 是唯一运动确认参数。未指定时只读取并
打印当前状态和目标。启动前仍必须满足《G2机器人控制注意事项与依赖路径.md》的
现场、网络、PTP、急停和状态检查。
