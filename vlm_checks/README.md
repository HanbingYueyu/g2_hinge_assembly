# VLM checks

## 文件与迁移来源

必要提示词、参考图和人工核对资料来自历史迁移包，运行时只使用本目录，不调用下载目录或 ROS 工作区。当前所有任务都通过配置的 DashScope/OpenAI-compatible HTTP API 调用，暂不部署本地视觉模型。

- `references/pin/`：凸轴 B 目标说明与插入/未插入状态参考，`pin` 请求自动发送这两张参考图，再发送待测原图。
- `references/c_hole/`：C/D 孔定义与状态参考，`c-hole` 请求自动发送这两张参考图，再发送待测原图。
- `references/convex_inserted/`：凸轴完全插入方案的三张严格参考图，`convex-inserted` 请求自动发送后再发送一张右相机原图。
- `references/right_fixture/`：右手夹具 V32 正/负参考板，`right-fixture-grasp` 请求自动发送后再发送四张连续右相机原图。
- `prompts/four_screws.txt`：四颗螺丝 V4 保守遮挡判别提示词；该任务不使用静态参考图，只发送四张连续右相机原图。
- `references/review_all/`：完整人工核对卡，供人回看；不把所有卡片塞进每次请求，也不把它们当作夹爪成功参考。
- `docs/verified_results.json`：原包八段现场标签；不代表当前相机和实时任务的准确率。
- [原迁移包运行说明](docs/原迁移包运行说明.md)：来源说明保留版；当前夹持输入已调整为两张当前图，不依赖 ROS。

源包没有独立的夹爪成功参考图片。全流程夹持监视现在仅发送当前原图和同一帧的中央近景，不发送历史参考，避免历史成功画面干扰。近景居中保留高度 60%、宽度 40%，源图来自原图缓存而非标注图。重抓清空旧投票。

请求串行，完成后立即取最新画面，不再额外等待 2 秒。空闲或取图/请求失败时短暂退避，避免忙循环。保留 5 次结果，连续 3 次一致可输出稳定标签；连续两次失败（`dropped` 和 `not_attached` 可混合）输出 `stable=failed` 并请求重抓。网络错误计为 unknown，阶段切换时丢弃旧请求结果。终端显示图片时间（按缓存年龄估算的本机时间）和请求耗时。

## API 调用与离线图片测试

北京 DashScope Chat Completions，默认 `qwen3.8-flash`；`temperature=0`、`seed=42`、`enable_thinking=false`、`max_tokens=256`。图片以 Base64 Data URL 上传，解析 `choices[0].message.content` 的 JSON。随机种子不保证绝对可复现，远程请求也不是实时安全控制。

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

`--image` 可重复，顺序应与提示词一致；夹爪历史参考可省略，没有本次历史参考不能仅凭空夹爪判断抓后掉落。脚本优先使用项目 `.venv-sam3`，可通过 `VLM_PYTHON` 指定解释器；无需初始化 GDK。可覆盖 `--model`、`--api-url`、`--env-file`、`--timeout`，例如超时使用 `--timeout 120`。

`pin_inserted=true` 表示模型判断 B 轴插入；`obvious_c_leak=false` 仅表示没有明显板外 C 孔，不等于螺丝孔精确同轴，也不是完整安装合格证明。遮挡/证据不足必须保留 unknown/null，不转为成功。

## 密钥与故障排查

真实密钥只用本地 `config/vlm.env`，维持权限 `600` 和 Git 忽略；不要上传、打印或写日志。北京密钥与北京接口配套，不混用国际站地域。缺少 Key 检查文件配置；401 检查密钥和地域；429 检查限流/额度；超时可延长等待但不要判为掉件；JSON 解析失败检查模型输出。原始 HTTP 错误也应审查后再分享，避免泄露敏感信息。

无需请求 API 的检查：

```bash
.venv-sam3/bin/python -m unittest vlm_checks.tests.test_package
```

本目录提供六个只读语义判别：左夹爪是否仍夹持铰链、凸轴 B 是否插入、是否存在明显 C 孔外漏、凸轴严格完全插入、四颗螺丝是否完成、右手是否抓住白色三二维码夹具。

默认使用北京 DashScope OpenAI-compatible 接口和 `qwen3.8-flash`。真实密钥仅保存在被 Git 忽略的 `config/vlm.env`，不得打印、提交或写入流程日志。

夹持判别在 `post_grasp` 阶段连续两次为 `not_attached` 或 `dropped`（允许混合）时，向主控制线程请求重新夹取。后台 VLM 线程不下发运动。主线程结束当前左手动作，保持实测关节目标，然后重新执行 `left_grasp`，使用本次缓存的头部铰链定位及原预接近姿态/高度回盘，不在运输位置原地张爪重抓。重新进入抓取前阶段清空投票和请求；松爪前检查未处理的失败请求，主动释放后停用夹持监视。unknown、网络错误、抓取前未夹住都不会触发重抓。定位或回盘失败停止报告，不继续盲夹。

凸轴 B、凸轴严格完全插入、C 孔、四颗螺丝和右手夹具判别均仅输出终端结果并写日志，不影响动作；右手夹具检查在右手抓取动作返回后执行，四颗螺丝检查在收尾阶段执行。

全流程阶段：

- `left_grasp` 开始进入抓取前监视；夹爪闭合保持后、上抬前进入 `post_grasp`，仅使用新鲜当前图。后续在运输、放置及门1等待期间持续检查。`vlm_gripper_check` 保留阶段入口，但不阻塞等待远程响应；松爪阶段开始停止夹持检查。
- `vlm_assembly_check`：左手释放并到达 LL2 后检查 3 轮，每轮从原图缓存重新取右、左、头图，轮间等待 1 秒；`pin`、`c-hole` 和严格 `convex-inserted` 每轮输出结果。既有静态判别参考图不变，不把检查变成动作开关。

VLM 只调用独立的线程安全原图缓存，不读取页面叠加图、不暂停页面、不直接读取 GDK 相机。缓存无新鲜原图时短暂等待，仍不可用则输出警告，不以叠加图替代。可用 `--disable-vlm-checks` 临时关闭。

## 独立实时测试

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/test_vlm_live_gdk.py gripper
```

任务可替换为 `pin` 或 `c-hole`。每次运行取图、请求一次，不发送机械臂运动指令。
默认优先读取 `http://127.0.0.1:8767/raw/<摄像头>.jpg` 当前原图；Web 不可连接时独立打开所需 GDK 彩色相机，取图后关闭。
Web 正在运行但尚无画面时会报错，不另开相机。`--web-url ''` 可强制独立相机模式（不要与已有相机采集进程同时使用）。
图片读取不保证三路硬件时间同步。同一次运行的送检图片统一保存到 `outputs/vlm/<运行开始时间>/`，目录格式为 `YYYYMMDD_HHMMSS_ffffff`；全流程的后台夹爪检查与安装检查共用该目录，独立实时测试每次新建运行目录。已有历史图片不移动。图片文件名仍含微秒时间戳、摄像头和阶段，变体后缀为 `_crop`、`_reference`；全图保留缓存原始 JPEG 字节。独立实时脚本每次只有当前全图和近景，不伪造历史参考，也不做跨进程投票。日志 `input_status.saved_images` 记录具体路径。
