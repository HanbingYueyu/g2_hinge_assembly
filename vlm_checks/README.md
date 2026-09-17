# VLM checks

本目录提供三个只读语义判别：左夹爪是否仍夹持铰链、凸轴 B 是否插入、是否存在明显 C 孔外漏。

默认使用北京 DashScope OpenAI-compatible 接口和 `qwen3.8-flash`。真实密钥仅保存在被 Git 忽略的 `config/vlm.env`，不得打印、提交或写入流程日志。

这些判别只输出终端结果并写日志，不直接或间接控制机械臂。

全流程阶段：

- `vlm_gripper_check`：左手运输到 hole 上方后，仅发送左手相机完整原图，不带检测标注、不裁剪。
- `vlm_assembly_check`：左手释放并到达 LL2 后，从原图缓存取右、左、头图；`pin` 与 `c-hole` 并行请求。既有静态判别参考图不变。

VLM 只调用独立的线程安全原图缓存，不读取页面叠加图、不暂停页面、不直接读取 GDK 相机。缓存无新鲜原图时短暂等待，仍不可用则输出警告，不以叠加图替代。可用 `--disable-vlm-checks` 临时关闭。

## 独立实时测试

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/test_vlm_live_gdk.py gripper
```

任务可替换为 `pin` 或 `c-hole`。每次运行取图、请求一次，不发送机械臂运动指令。
默认优先读取 `http://127.0.0.1:8767/raw/<摄像头>.jpg` 当前原图；Web 不可连接时独立打开所需 GDK 彩色相机，取图后关闭。
Web 正在运行但尚无画面时会报错，不另开相机。`--web-url ''` 可强制独立相机模式（不要与已有相机采集进程同时使用）。
图片读取不保证三路硬件时间同步。发送的实时原图按原始 JPEG 字节保存到项目根目录的 `outputs/vlm/`，命名为微秒时间戳、摄像头和对应 VLM 阶段。不再生成 `_crop` 图片；已有历史图片不删除。
