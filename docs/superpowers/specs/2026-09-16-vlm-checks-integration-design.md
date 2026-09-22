# G2 全流程 VLM 三项判别集成设计

## 目标

将迁移包中的三项 VLM 语义判别集成进 G2 分阶段双臂流程：

- `gripper`：判断左夹爪中的铰链是否仍然夹持；
- `pin`：判断凸轴 B 是否完全插入；
- `c-hole`：判断是否存在明显的 C 孔外漏。

第一版只输出判别结果并写入全流程 JSONL 日志。任何判别结果、`unknown`、网络错误或超时均不控制、停止或改变机械臂动作。

## 迁移范围

在项目根目录新增 `vlm_checks/`，只迁移运行和离线验证必需内容：

- `src/vlm_three_checks.py`；
- `prompts/` 下三个生产提示词；
- `references/pin/` 下两张生产参考图；
- `references/c_hole/` 下两张生产参考图；
- `config/vlm.env.example`；
- `docs/verified_results.json`；
- `tests/test_package.py`；
- 精简后的 `README.md`；
- `requirements.txt`。

不迁移以下内容：

- `.pytest_cache`、`__pycache__`；
- `references/review_all/` 人工总览图；
- `original_gripper_ros/` 旧 ROS 实现与历史说明；
- 原迁移包的壳脚本和其他不被全流程调用的文件。

## 密钥管理

真实密钥复制到 `vlm_checks/config/vlm.env`，文件权限保持 `600`。项目 `.gitignore` 明确加入：

```gitignore
/vlm_checks/config/vlm.env
```

迁移后必须执行 `git check-ignore vlm_checks/config/vlm.env`，确认密钥文件不会进入 Git。代码和日志不得打印密钥内容。

## 画面来源

VLM 不直接读取 GDK 相机，也不暂停 Web 原画面线程。`BilateralWeb` 已在线程安全缓存中保存 `head`、`left`、`right` 三路最新 JPEG；VLM 阶段通过 `snapshot_pane()` 复制当前 JPEG 字节。

因此：

- Web 继续按原帧率刷新；
- VLM 不与 YOLO、AprilTag 或原画面线程竞争相机；
- 不创建临时图片文件；
- 输入可能是当前原画面，也可能是阶段最近发布的叠加画面；迁移包允许使用新鲜叠加图。

若 Web 关闭、某路缓存为空或 JPEG 无法解码，则该项判别输出警告记录并跳过，不影响后续动作。

## 阶段安排

新增两个明确、可由 `--from-stage` / `--through-stage` 选择的阶段：

1. `vlm_gripper_check`
   - 位于 `left_grasp` 之后、`right_to_lr1` 之前；
   - 获取左手最新 JPEG；
   - 使用当前全图，并按 `(80, 120, 560, 480)` 裁剪夹爪近景；
   - 调用 `gripper` 判别；
   - 不要求抓取成功参考近景。

2. `vlm_assembly_check`
   - 位于 `left_release_lift_to_ll2` 之后、`right_grasp` 之前；
   - `pin` 输入顺序为右手、左手、头部；
   - `c-hole` 输入顺序为左手、右手；
   - 两个远程请求并行执行，以总耗时接近较慢的一项，而不是两项耗时相加。

从晚于头部定位的阶段启动时，现有前置 `head_localize` 行为保持不变。VLM 阶段只依赖 Web JPEG 缓存，不重新执行机械臂阶段。

## 运行组件

新增轻量适配模块，职责分离如下：

- `vlm_checks/src/vlm_three_checks.py`
  - 保留迁移包的 payload、API 请求、环境加载和 JSON 解析能力；
  - 增加接受内存 JPEG 字节的调用入口，避免临时文件。
- `g2_adapter/vlm_flow_checks.py`
  - 从 `BilateralWeb` 获取快照；
  - 裁剪左夹爪近景；
  - 校验各任务返回字段；
  - 并行执行 `pin` 和 `c-hole`；
  - 将异常转换成非阻断结果记录。
- `run_bilateral_flow_gdk.py`
  - 增加两个阶段处理函数；
  - 将简短结果输出到终端；
  - 将完整结果交给 `BilateralRuntime.emit_event()` 写入 JSONL。

## 配置

全流程增加参数：

- `--vlm-enable`：显式启用全部 VLM 判别；未指定时跳过 VLM；
- `--vlm-env-file`：默认 `vlm_checks/config/vlm.env`；
- `--vlm-model`：默认 `qwen3.8-flash`；
- `--vlm-api-url`：默认北京 DashScope OpenAI-compatible 地址；
- `--vlm-timeout-s`：默认 60 秒；
- `--vlm-gripper-crop`：默认 `80,120,560,480`。

VLM 只有在显式指定 `--vlm-enable` 时启用。密钥缺失、配置错误或 `requests` 不可用时，阶段输出警告并继续，不把依赖问题升级为机械臂流程失败。

## 输出与日志

终端只打印简要结果，例如：

```text
[VLM 夹持] state=attached confidence=0.94 reason=铰链仍在两指间
[VLM 凸轴] state=inserted confidence=0.91 reason=B轴已落入安装孔
[VLM C孔] obvious_c_leak=false confidence=0.96 reason=板外未见小孔
```

完整 JSONL 事件分别使用：

- `vlm_gripper_result`；
- `vlm_pin_result`；
- `vlm_c_hole_result`；
- `vlm_check_warning`。

记录任务名、模型、耗时、输入画面来源状态和原始结构化结果；不记录 Base64 图片或 API Key。

## 并发与失败处理

- 每个阶段最多一个同任务请求在途；
- `pin`、`c-hole` 使用最多两个工作线程并行；
- 请求完成前该 VLM 阶段等待，但 Web 和相机线程继续运行；
- HTTP、解析、超时、缓存缺帧全部转换为 `vlm_check_warning`；
- 无论结果如何，阶段自身以成功结束并进入下一机械臂阶段；
- `Ctrl+C` 仍应终止整个程序，不得被当成普通 VLM 错误吞掉。

## 测试与验收

离线测试不得调用真实 API：

- 验证迁移必需文件存在、密钥被 Git 忽略且权限为 `600`；
- 验证内存 JPEG 正确生成 Data URL；
- 验证三个任务的图片顺序和参考图顺序；
- 验证左手裁剪框；
- 验证 `pin` 与 `c-hole` 并行调度；
- 验证缺帧、超时、HTTP 错误和无效 JSON 均非阻断；
- 验证两个新阶段的顺序和 `--from-stage` / `--through-stage` 行为；
- 验证终端摘要不包含 Base64 和密钥。

实机验收时只观察相机、终端和日志：VLM 阶段运行期间 Web 画面必须持续刷新，机械臂动作顺序与未启用 VLM 时一致。
