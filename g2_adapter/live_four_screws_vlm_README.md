# 四颗螺丝独立实时判别

只读取图像并调用 VLM，不连接 Robot、不发送运动或电批指令。现有全流程代码不变。

## 单独启动右手彩色相机

```bash
cd /home/flyfuture/g2_hinge_assembly && \
./scripts/run_gdk_sam3_python.sh g2_adapter/live_four_screws_vlm.py \
  --crop 430,400,870,830 \
  --web-host 0.0.0.0 --web-port 8770 \
  --vlm-mode local \
  --vlm-env-file vlm_checks/config/vlm.env \
  --vlm-timeout-s 60
```

打开 `http://127.0.0.1:8770`；其他电脑访问 `http://机器人IP:8770`。Ctrl+C 停止。

`--crop` 为原图像素坐标 `x0,y0,x1,y1`，示例得到 440×430 的 JPEG。根据浏览器预览调整区域，必须保留完整目标铰链及白色定位标记；裁掉标记或螺丝位置可能让模型无法判别。默认区域按用户提供的完整图与裁剪示例估算，对应 1280×1056 原图中的三枚二维码架与下方铰链；可根据实时预览微调。

## 全流程已占用相机时

追加以下参数，读取全流程网页现有右手原图接口，避免重复打开相机：

```bash
--source-url http://127.0.0.1:8767/raw/right
```

此模式不初始化 GDK。原图接口暂不可用时，网页显示相机错误，不把旧图当作新的 VLM 输入。

## 预览与判别

- 上方：实时裁剪预览。
- 下方三张：最近一次实际提交给 VLM 的 JPEG，顺序固定为零颗参考图、四颗参考图、待测当前裁剪图；保持冻结直到下一次提交。
- 结果区：显示新方案的 model_count、complete、raw；计数 4 才是完成，0–3 为未完成，看不清或解析失败为 unknown。latest_decision 是从 complete 映射的预览兼容字段，model_count 仅供诊断。
- 每隔至少 0.25 秒采样，积累四张后请求；请求串行执行，结束后等待 1 秒，取最新一张裁剪图，连同两张固定参考图提交。可用 `--sample-interval-s`、`--request-interval-s` 调整。
- 图像预览独立于 VLM 请求持续刷新，判别刷新速度由模型响应时间决定。
- 采用下载目录中《四颗螺丝齐全判别_20261006》主程序 predict.py 的原始提示词及零颗、四颗参考图，已逐字节复制到 `g2_adapter/vlm_four_screws_20261006/`。三颗参考图和未采用试验不发送。追加待测指令原文，模型计数仅用于诊断；程序映射 complete：4 为 yes，0–3 为 no，解析失败或不确定为 unknown。请求 temperature=0、max_tokens=100。服务地址、模型、密钥、超时仍使用原脚本配置。参考图保持原样，只有待测相机图按 --crop 裁剪。网络错误清空旧结果并显示失败；现有全流程提示词、参考图和判别逻辑均未修改。
