# VLM checks

## 文件与迁移来源

必要提示词、参考图和人工核对资料来自历史迁移包，运行时只使用本目录，不调用下载目录或 ROS 工作区。视觉模型不在机器人本机部署；全流程可切换局域网 OpenAI-compatible API 或在线 API。

- `references/pin/`：凸轴 B 目标说明与插入/未插入状态参考，`pin` 请求自动发送这两张参考图，再发送待测原图。
- `references/c_hole/`：C/D 孔定义与状态参考，`c-hole` 请求自动发送这两张参考图，再发送待测原图。
- `references/convex_inserted/`：凸轴完全插入方案的三张严格参考图，`convex-inserted` 请求自动发送后再发送一张右相机原图。
- `references/right_fixture_v47/`：20260923 V47 包的正/负参考板；`right-fixture-grasp` 先调用 V32，仅在非 yes 时调用 V40，同为 no 才输出“未抓住”，复核 yes 输出“抓住”，其他冲突输出“不确定”。保留两次原始结果。提示词取原程序 PROMPT 原文；V32 文本副本末尾的迁移提醒不追加到实际请求。
- `references/hole_alignment/`：保留 20260923 孔对齐稳定版的正/负双图。当前按现场要求改用右相机 1280×1056 原图，裁剪 x=400,y=200,w=500,h=400，通过 FFmpeg Lanczos 缩放到 900×713。发送顺序为正例、负例、当前 ROI；temperature=0，max_tokens=160。该 ROI 已用保存的右相机图核对，但 LR2 动作后的取景仍需实机核验；原包左视角参考图在右视角下的精度尚未验证。
- 右手 V47 按原程序 `fixture_roi → contact_roi → grid` 生成 2040×1710 四宫格，V32/V40 的 max_tokens 分别为 240/260。原包提示词写“6秒”，但程序实际是每 1.5 秒一帧、四帧跨度 4.5 秒；本项目遵循程序间隔。运行环境需 OpenCV、NumPy、requests 和 ffmpeg。
- `prompts/four_screws.txt`：四颗螺丝 V4 保守遮挡判别提示词；该任务不使用静态参考图，只发送四张连续右相机原图。
- `references/review_all/`：完整人工核对卡，供人回看；不把所有卡片塞进每次请求，也不把它们当作夹爪成功参考。
- `docs/verified_results.json`：原包八段现场标签；不代表当前相机和实时任务的准确率。
- [原迁移包运行说明](docs/原迁移包运行说明.md)：来源说明保留版；当前夹持输入已调整为两张当前图，不依赖 ROS。

源包没有独立的夹爪成功参考图片。全流程夹持监视现在仅发送当前原图和同一帧的中央近景，不发送历史参考，避免历史成功画面干扰。近景居中保留高度 60%、宽度 40%，源图来自原图缓存而非标注图。重抓清空旧投票。

请求串行，完成后立即取最新画面，不再额外等待 2 秒。空闲或取图/请求失败时短暂退避，避免忙循环。保留 5 次结果，连续 3 次一致可输出稳定标签；连续两次失败（`dropped` 和 `not_attached` 可混合）输出 `stable=failed` 并请求重抓。网络错误计为 unknown，阶段切换时丢弃旧请求结果。终端显示图片时间（按缓存年龄估算的本机时间）和请求耗时。

## API 调用与离线图片测试

全流程可用 `--vlm-mode local|online` 选择 API 配置档，默认 `local`。配置档从 `config/vlm.env` 读取：local 使用 `VLM_LOCAL_API_URL`、`VLM_LOCAL_MODEL`、`VLM_LOCAL_API_KEY`；online 使用 `VLM_ONLINE_API_URL`、`VLM_ONLINE_MODEL`、`VLM_ONLINE_API_KEY`。在线档默认指向 DashScope-compatible API 和 `qwen3.8-flash`；局域网档默认指向 `http://192.168.30.13:8000/v1` 和 `qwen3-vl-8b-local`。旧变量 `VLM_API_URL`、`VLM_MODEL`、`VLM_LAN_API_KEY`、`DASHSCOPE_API_KEY` 继续兼容。CLI 的 `--vlm-api-url` 和 `--vlm-model` 若显式传入，会覆盖所选配置档。图片以 Base64 Data URL 上传，解析 `choices[0].message.content` 的 JSON；随机种子不保证绝对可复现，远程请求也不是实时安全控制。

全流程增加 `--tts-enable true|false`（默认 `false`），用 GDK `Interaction.play_tts()` 以自然短句播报 VLM 结论，不播报理由或证据。例如“铰链已经夹稳了”“凸轴还没有插到位”。同一次运行中，同一检查的同一结果只排队播报一次；结论变化时会播报新结果。播报在后台队列执行，TTS 不可用时只输出警告，不改变机械臂流程。启用前需按 G2 语音服务说明启动 `agibot_voice` 和远程 GDK 桥接；API 密钥保存在被 Git 忽略的 `config/vlm.env`，不要提交或写入日志。

```bash
cd /home/flyfuture/g2_hinge_assembly
bash vlm_checks/run.sh pin \
  --image /path/to/right_final.jpg \
  --image /path/to/left_final.jpg \
  --image /path/to/head_final.jpg
bash vlm_checks/run.sh c-hole \
  --image /path/to/left_final.jpg --image /path/to/right_final.jpg
bash vlm_checks/run.sh gripper \
  --image /path/to/current_left_raw.jpg \
  --image /path/to/current_gripper_crop.jpg \
  --image /path/to/post_grasp_reference.jpg
bash vlm_checks/run.sh convex-inserted --image /path/to/right_raw.jpg
bash vlm_checks/run.sh four-screws \
  --image /path/to/right_frame_1.jpg --image /path/to/right_frame_2.jpg \
  --image /path/to/right_frame_3.jpg --image /path/to/right_frame_4.jpg
bash vlm_checks/run.sh right-fixture-grasp \
  --image /path/to/right_frame_1.jpg --image /path/to/right_frame_2.jpg \
  --image /path/to/right_frame_3.jpg --image /path/to/right_frame_4.jpg
```

`--image` 可重复，顺序应与提示词一致；夹爪历史参考可省略，没有本次历史参考不能仅凭空夹爪判断抓后掉落。脚本优先使用项目 `.venv-sam3`，可通过 `VLM_PYTHON` 指定解释器；无需初始化 GDK。可覆盖 `--vlm-mode`、`--model`、`--api-url`、`--env-file`、`--timeout`，例如超时使用 `--timeout 120`。

`pin_inserted=true` 表示模型判断 B 轴插入；`obvious_c_leak=false` 仅表示没有明显板外 C 孔，不等于螺丝孔精确同轴，也不是完整安装合格证明。遮挡/证据不足必须保留 unknown/null，不转为成功。

## 密钥与故障排查

真实密钥只用本地 `config/vlm.env`，维持权限 `600` 和 Git 忽略；不要上传、打印或写日志。服务端和客户端的 local key 必须一致；缺少在线密钥时设置 `VLM_ONLINE_API_KEY`（或旧名 `DASHSCOPE_API_KEY`），缺少局域网 key 时设置 `VLM_LOCAL_API_KEY`（或旧名 `VLM_LAN_API_KEY`）。401 检查对应服务地址和密钥；429 检查服务限流；超时可延长等待但不要判为掉件；JSON 解析失败检查模型输出。原始 HTTP 错误也应审查后再分享，避免泄露敏感信息。

无需请求 API 的检查：

```bash
.venv-sam3/bin/python -m unittest vlm_checks.tests.test_package
```

本目录提供六个只读语义判别：左夹爪是否仍夹持铰链、凸轴 B 是否插入、是否存在明显 C 孔外漏、凸轴严格完全插入、四颗螺丝是否完成、右手是否抓住白色三二维码夹具。

默认使用 `http://192.168.30.13:8000/v1` 局域网 OpenAI-compatible 接口和 `qwen3-vl-8b-local`。真实密钥仅保存在被 Git 忽略的 `config/vlm.env`，不得打印、提交或写入流程日志。

主流程用 `--vlm-enable` 显式打开 VLM；不带此参数时不创建 VLM 客户端，也不会发送判别请求。主流程示教统一用 `--teach-enable` 打开，并同时提供左臂轨迹和右臂轨迹参数；不带此参数时使用原来的 LL1/下探及右手原流程。

夹持判别在 `post_grasp` 阶段连续两次为 `not_attached` 或 `dropped`（允许混合）时，向主控制线程请求重新夹取。后台 VLM 线程不下发运动。主线程结束当前左手动作，保持实测关节目标，然后重新执行 `left_grasp`，使用本次缓存的头部铰链定位及原预接近姿态/高度回盘，不在运输位置原地张爪重抓。重新进入抓取前阶段清空投票和请求；松爪前检查未处理的失败请求，主动释放后停用夹持监视。unknown、网络错误、抓取前未夹住都不会触发重抓。定位或回盘失败停止报告，不继续盲夹。

凸轴 B、凸轴严格完全插入、C 孔、四颗螺丝和右手夹具判别均仅输出终端结果并写日志，不影响动作；右手夹具检查在右手抓取动作返回后执行。示教模式下，最后一段结束并确认 gate_right 后，右手先沿 Z 轴上抬 30 mm，再到当前 `--right-hole-*` 指定的位置与姿态（不重复抓取后的 100 mm 上抬）；到位后等待新的右相机原图，采集四帧并判别，最后归位电批。移动未到位时不取图、不继续归位。

全流程阶段：

- `left_grasp` 开始进入抓取前监视；夹爪闭合保持后、上抬前进入 `post_grasp`，仅使用新鲜当前图。后续在运输、放置及门1等待期间持续检查。`vlm_gripper_check` 保留阶段入口，但不阻塞等待远程响应；松爪阶段开始停止夹持检查。
- `vlm_assembly_check`：左手释放并到达 LL2 后仅执行严格 `convex-inserted` 3 轮，每轮取右相机新原图，轮间等待 1 秒。旧 `pin` 不再在主流程调用：其 2 张参考图加 3 张当前图超过本地服务每请求最多 4 张的限制。旧接口仍保留独立调用。
- `vlm_c_hole_check`：保留原阶段名方便旧命令使用；右手完成 LR2 夹爪闭合 1 秒并张开后、抓电批前，使用右相机孔对齐判别。连续取 5 次新图，轮间等待 1 秒；同一明确结论至少 3 票才通过多数票，否则“不确定”。终端显示 `[VLM 孔对齐]`，语音只播最终投票；只输出结果，不作为动作开关。旧 `c-hole` 仅保留独立调用。
- 收到 `vlm_regrasp_requested` 时清空夹持结果的语音去重记录；新一轮可再次播报相同夹持结果，其他任务的去重不变。

两种 API 模式共用完全相同的提示词、参考图、裁剪、推理参数、V47 复核及投票逻辑，只切换服务地址、模型 ID 和密钥。模型输出可能不同。

历史图片测试（结果和实际送检图写入 `outputs/vlm_regression/时间_模式/`）：

```bash
.venv-sam3/bin/python vlm_checks/test_saved_migration.py --input-dir outputs/vlm/20260923_211148_385924 --vlm-mode local
# 改为 --vlm-mode online 测试在线 API
```

该历史目录只有 3 张孔位图，不伪造 5 次投票；右手历史 4 帧是 0.25 秒间隔，测试标签按实际间隔显示，不能作为 4.5 秒时序的完整回归。

VLM 只调用独立的线程安全原图缓存，不读取页面叠加图、不暂停页面、不直接读取 GDK 相机。缓存无新鲜原图时短暂等待，仍不可用则输出警告，不以叠加图替代。主流程通过 `--vlm-enable` 显式启用，未指定时跳过全部 VLM 检查。

## 独立实时测试

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/test_vlm_live_gdk.py gripper
```

任务可替换为 `pin` 或 `c-hole`。每次运行取图、请求一次，不发送机械臂运动指令。
默认优先读取 `http://127.0.0.1:8767/raw/<摄像头>.jpg` 当前原图；Web 不可连接时独立打开所需 GDK 彩色相机，取图后关闭。
Web 正在运行但尚无画面时会报错，不另开相机。`--web-url ''` 可强制独立相机模式（不要与已有相机采集进程同时使用）。
图片读取不保证三路硬件时间同步。同一次运行的送检图片统一保存到 `outputs/vlm/<运行开始时间>/`，目录格式为 `YYYYMMDD_HHMMSS_ffffff`；全流程的后台夹爪检查与安装检查共用该目录，独立实时测试每次新建运行目录。已有历史图片不移动。图片文件名仍含微秒时间戳、摄像头和阶段，变体后缀为 `_crop`、`_reference`；全图保留缓存原始 JPEG 字节。独立实时脚本每次只有当前全图和近景，不伪造历史参考，也不做跨进程投票。日志 `input_status.saved_images` 记录具体路径。
