# Left Placement Teach Switch Implementation Plan

**Goal:** 当前主分支完整流程支持 original/teach 左手放置切换。

**Architecture:** 默认 original 不变。按用户最新要求，teach 保留抓取、150mm上抬、同目标确认及原 hole 偏移点运输；之后在 left_to_ll1_and_seat 回放左臂轨迹，只替代LL1及力保护下压。录制及回放模块迁入当前项目，不依赖外部工作区；VLM及后续释放、右臂流程不变。

**Tech Stack:** Python、GDK、unittest。

**Spec:** 本次会话用户确认的设计。

## Constraints

- 当前主分支修改，不合并旧工作区。
- teach 轨迹为绝对关节角，不自动根据 hole 重定位；从洞偏移点之后开始，至放置完成，不包含松爪。
- 保留开头静止样本；不添加空间范围限制。

## Tasks

- [x] 写CLI默认/teach参数、保留洞运输、示教替代原动作的失败测试。
- [x] 通过apply_patch迁入teach_trajectory和teach_trajectory_gdk及已有测试。
- [x] 保持左手抓取与洞运输代码完全不变。
- [x] 校验左臂轨迹及倍率，接入原阶段与现有运动暂停锁。
- [x] 运行相关unittest及diff检查，补精简运行说明。

## Verification

110项相关unittest通过，git diff --check通过。只读审查发现到位等待暂停计时问题，已增加45秒暂停回归测试并修正。未进行整机运动联调。
