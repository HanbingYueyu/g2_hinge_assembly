# VLM checks

本目录提供三个只读语义判别：左夹爪是否仍夹持铰链、凸轴 B 是否插入、是否存在明显 C 孔外漏。

默认使用北京 DashScope OpenAI-compatible 接口和 `qwen3.8-flash`。真实密钥仅保存在被 Git 忽略的 `config/vlm.env`，不得打印、提交或写入流程日志。

这些判别只输出终端结果并写日志，不直接或间接控制机械臂。

全流程阶段：

- `vlm_gripper_check`：左手运输到 hole 上方后，从 Web 的左手最新 JPEG 生成全图和夹爪裁剪图。
- `vlm_assembly_check`：左手释放并到达 LL2 后，从 Web 缓存取右、左、头图；`pin` 与 `c-hole` 并行请求。

VLM 只调用线程安全的 Web 快照接口，不暂停页面、不直接读取 GDK 相机。可用 `--disable-vlm-checks` 临时关闭。
