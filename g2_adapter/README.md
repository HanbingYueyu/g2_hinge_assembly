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
`kinematics.py` 直接解析 G2 URDF，提供 FK、Jacobian 和带步长限制的阻尼最小二乘视觉修正。

`workcell.example.json` 中的 P1/P2/P3/P4 故意为空。请使用 `record_pose.py`
在实际工位逐点读取；未完成前不得启动完整自动流程。

只读预检命令（由操作员执行，Codex 不启动机器人）：

```bash
cd /home/flyfuture/agi
source "$HOME/.cache/agibot/app/env.sh" "$HOME/.cache/agibot/app"
uv run python /home/flyfuture/traditional_auto_flow_g2_migration/g2_adapter/preflight.py
```

运行 `run_preset.py` 时，`--allow-motion` 是唯一运动确认参数。未指定时只读取并
打印当前状态和目标。启动前仍必须满足《G2机器人控制注意事项与依赖路径.md》的
现场、网络、PTP、急停和状态检查。
