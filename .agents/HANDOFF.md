# 项目交接簿（HANDOFF）

> **交接定位与说明（强制遵循 [AGENTS.md](../AGENTS.md) 第 8 条防流水账规则）**：
> 本文件为项目单帧即时交接簿，旨在帮助下一任接手开发者（人类开发者、评审导师、小组成员或后续开发 Agent）在 1 分钟内清晰掌握：**项目刚刚完成的最新成果（上一步是什么）**与**即将开展的未开始事项（下一步做什么）**。
> **严禁作为流水账追加**：正文必须严格控制在 100~150 行（绝对上限 200 行）。早期历史记录已归档至 [归档/.agents/HANDOFF_历史累积归档_20261108.md](../归档/.agents/HANDOFF_历史累积归档_20261108.md)。

---

## 当前交接元信息

- **交接时间**：2026-11-11
- **当前移交人**：ZCode Agent（会话：`git-release-v5-20261111`，承接 `PLATFORM-042` 完成 `GIT-007` 发布）
- **接收角色**：项目负责人（陈德昕）、课题组成员（李世显、陈龙辉）、指导教师（王笛）及下一任接手 Agent
- **当前活动任务状态**：`PLATFORM-042` 与 `GIT-007` 均已完成并释放写入范围，[TASKS.md](TASKS.md) 当前无 `in_progress` 阻塞任务
- **当前后台进程状态**：**后端服务与测试进程数为 0**（未启动常驻后台任务，可随时双击启动 `8769` 或 `8768` 平台）

---

## 一、刚完成的最新成果（上一步：`PLATFORM-042` → `GIT-007`）

本轮围绕 **GeoNav V5.0 全栈体验升级、防白屏冷启动加固、双便携 Demo 重建、中期答辩材料同步、微信汇报图文指南优化及 GitHub 正式发布（`5b81597`）**完成闭环交付，所有历史 4.18 原稿与 10.25 旧版均保留未动或镜像归档：

### 1. GeoNav V5.0 全功能一体化平台全面升级（对齐 GeoNav Lite 交互并清扫冗余）
- **前端矢量重构与说明气泡化**（[geonav_v5.html](../project/src/webapp/geonav_v5.html)、[geonav_v5.css](../project/src/webapp/geonav_v5.css)、[geonav_v5.js](../project/src/webapp/geonav_v5.js)）：
  - 彻底移除全站卡通 Emoji，统一改为单色线性 SVG 图标与精炼短标签 + 旁侧 `!` 感叹号点击气泡说明（Popover）；
  - 重构平台总览首屏为紧凑零滚动指标条（`.v5-ov-strip` + 四大黄金预设 + 六协议消融总表），左侧保留 6 大业务工作台导航；
  - 工作台 2（`#ws-navigator`）注入 `/api/v5/interactive/*` 精选交互探索：涵盖 **192 张独立航拍底图（960 题）+ 192 道地面样本**、一键随机换图、动态多尺度网格点击设起终点、小眼睛人类预览全图（底层策略严守盲视隔离）、四方向倾向度进度条及 M0 双分支决策解释。
- **全局纯内存切片与云端 API 自检融断**（[service_v17.py](../project/src/platform_core/service_v17.py)、[workbench_server_v28.py](../project/src/webapp/workbench_server_v28.py)）：
  - 顶部全局【自定义切片】支持 `3×3~10×10` 本地图片纯内存实时切分并由 M0 寻路，关闭服务自动销毁不污染样本库；
  - 顶部全局【云端 API】弹窗支持在线配置 OpenAI/DeepSeek 接口、一键连通性自检与 `thinking={'type':'disabled'}` 关闭长思考防截断适配；
  - `BENCHMARK_OVERVIEW` 同步精确指标（S2 `0.320格`、S3 `86.80%/0.300格`、S4 `77.13%/1.201格`、SwissView `86.80%/0.381格`）。
- **双端启动防白屏与端到端回归全绿**：
  - [启动平台.ps1](../平台/启动平台.ps1)（`8769`）、[启动极简演示平台.ps1](../平台/启动极简演示平台.ps1)（`8768`）与 [build_pure_lite_demo.py](../temp/build_pure_lite_demo.py) 均改为 `Start-Process -PassThru` + 绕过系统代理（`$req.Proxy = $null`）HTTP 200 就绪轮询后才拉起浏览器；[geonav_lite.js](../project/src/webapp/geonav_lite.js) 同步增加前端自动重连兜底。
  - [test_workbench_v5.py](../project/src/tests/test_workbench_v5.py)（6 项端到端测试：`Ran 6 tests in 269.604s · OK`）与 [test_lite_workbench.py](../project/src/tests/test_lite_workbench.py)（6 项回归测试：`Ran 6 tests in 159.578s · OK`）均在源码终改后实测通过。

### 2. 双独立便携交付包重建（完全解耦 + 脱敏去 `.env` + 冒烟验证）
- **[demo/小型演示demo/](../demo/小型演示demo/) 与 [demo/小型演示demo.zip](../demo/小型演示demo.zip)**（端口 `8768`）：由 [build_pure_lite_demo.py](../temp/build_pure_lite_demo.py) 重建，含 `4,993` 个文件/条目，解压 **`304.14 MB`** / 压缩 **`292.01 MB`**；已含防白屏启动轮询，彻底剥离中期大工作台。
- **[demo/中期demo/](../demo/中期demo/) 与 [demo/中期demo.zip](../demo/中期demo.zip)**（端口 `8769`）：由 [build_v5_midterm_demo.py](../temp/build_v5_midterm_demo.py) 同步重建为 GeoNav V5.0，含 `5,923` 个文件/条目，解压 **`449.39 MB`** / 压缩 **`360.86 MB`**；已去除失效 `.env` 改附 `.env.example`，剔除无用 `MassGIS grid15`（`128.57 MB`），补齐 `OpenCVL/输入` 与 CV031/CV041 等全套证据；已在隔离目录实测启动及 `/`, `/api/meta`, `/api/v5/overview`, `/api/v5/interactive/meta`（`tasks= 1152`）, `/api/perception/catalog`, `/api/spatial-evidence/catalog` 全部返回 HTTP `200 OK`。

### 3. 架构图重绘、中期报告/PPT 全套同步与 4.18 原稿镜像归档
- **原稿安全归档**：4.18 原交付稿完整备份至 [交付_20261109_418原稿备份/](../归档/中期报告相关/中期成果更新_20261025/交付_20261109_418原稿备份/)；旧版图 5 备份至 [05_GeoNav418原版备份_20261109.{png,pdf}](../归档/绘图/架构与方法图/)。
- **架构图与报告/PPT 同步审校**：
  - 由 [redraw_fig05.py](../temp/redraw_fig05.py) 重绘 [05_GeoNav418前后端分离工作台与云API熔断架构图.{png,pdf}](../绘图/架构与方法图/05_GeoNav418前后端分离工作台与云API熔断架构图.png)（300 DPI 白底矢量风），并由 [capture_v5_screenshots.py](../temp/capture_v5_screenshots.py) 采集 3 张 `1440×960` V5.0 实机界面图；
  - 由 [update_midterm_docx.py](../temp/update_midterm_docx.py) 外科更新 13 处段落与等比替换图 5 并导出 **22 页 [中期报告.pdf](../中期报告相关/中期成果更新_20261025/交付/中期报告.pdf)**，逐页预览零溢出；
  - 由 [update_midterm_pptx.py](../temp/update_midterm_pptx.py) 更新 [中期成果汇报.pptx](../中期报告相关/中期成果更新_20261025/交付/中期成果汇报.pptx)（**25 页**）并经 PowerPoint COM 导出渲染校验修正标题与表格间距，视觉复核零重叠；
  - 同步更新 [汇报讲稿.md](../中期报告相关/中期成果更新_20261025/交付/汇报讲稿.md)、[中期汇报准备.md](../中期报告相关/中期汇报准备.md)（含优化后的微信汇报图文组合文案与 4 张高清图选单，修正 2 处相对归档路径）、[中期报告相关/README.md](../中期报告相关/README.md)、根 [README.md](../README.md)、[demo/README.md](../demo/README.md)、[docs/PROJECT_STATE.md](../docs/PROJECT_STATE.md) 与 [版本生命周期.json](../docs/维护/版本生命周期.json)。

### 4. GitHub 精简合规发布（`GIT-007`）
- 经 5 份新 Python AST、2 份 JSON、全量本地导入闭包、链接有效性及密钥扫描核验，主发布提交 `5b81597`（41 个文件）已通过 SSH 直连推送至 `origin/main`；严格排除 `视频/`、`归档/`、`temp/`、`demo/`、`.codex/`、`.env*` 及本地大数据与模型。

---

## 二、未开始的下一步工作（下一步）

1. **中期检查答辩现场展示**：
   - 会前直接双击 **[平台/启动平台.cmd](../平台/启动平台.cmd)**（端口 `8769`，V5.0 全功能一体工作台）或 **[平台/启动极简演示平台.cmd](../平台/启动极简演示平台.cmd)**（端口 `8768`，Lite 单屏极简台）；
   - 向导师或评委发送材料默认选用 **[中期报告.pdf](../中期报告相关/中期成果更新_20261025/交付/中期报告.pdf)**、**[中期成果汇报.pptx](../中期报告相关/中期成果更新_20261025/交付/中期成果汇报.pptx)**，便携包按需单独发送 **[小型演示demo.zip](../demo/小型演示demo.zip)**（`292.01 MB`）或 **[中期demo.zip](../demo/中期demo.zip)**（`360.86 MB`）；群发话术与 3 分钟实机操作详见 [中期汇报准备.md](../中期报告相关/中期汇报准备.md)。
2. **如需在现场演示云端 API 实测（可选）**：
   - 两套便携包均已脱敏移除 `.env`；演示云端大模型时可在网页右上角【云端 API】弹窗在线填入有效 Key（如 DeepSeek 并保持“关闭长思考”勾选），点击【测试连通性】自检后运行。
3. **下半学期科研攻关主线（中期后新阶段）**：
   - 保持已封存 4.18 文件与研究约束不变，围绕“跨视角地面手持照片 → 高空航拍”引入几何位姿先验与区域/方向级弱监督小闭环探索方案（需先另行冻结协议与验收门槛）。

---

## 三、当前核心入口与关键资产速查

| 资产 / 入口类别 | 路径链接 | 状态与关键指标 |
| :--- | :--- | :--- |
| **V5.0 主平台启动入口** | [平台/启动平台.cmd](../平台/启动平台.cmd) · [workbench_server_v28.py](../project/src/webapp/workbench_server_v28.py) | 端口 `8769`；[test_workbench_v5.py](../project/src/tests/test_workbench_v5.py) 6/6 通过（`269.604s`） |
| **Lite 极简台启动入口** | [平台/启动极简演示平台.cmd](../平台/启动极简演示平台.cmd) · [lite_server.py](../project/src/webapp/lite_server.py) | 端口 `8768`；[test_lite_workbench.py](../project/src/tests/test_lite_workbench.py) 6/6 通过（`159.578s`） |
| **便携交付包（独立解压即用）** | [demo/小型演示demo.zip](../demo/小型演示demo.zip) · [demo/中期demo.zip](../demo/中期demo.zip) | Lite `292.01 MB`（`4993` 项） / V5 `360.86 MB`（`5923` 项，冒烟 `200 OK`） |
| **中期汇报定稿材料** | [中期报告.pdf](../中期报告相关/中期成果更新_20261025/交付/中期报告.pdf) · [中期成果汇报.pptx](../中期报告相关/中期成果更新_20261025/交付/中期成果汇报.pptx) · [汇报讲稿.md](../中期报告相关/中期成果更新_20261025/交付/汇报讲稿.md) | 22 页报告 + 25 页可编辑 PPT + 25 页讲稿 + 重绘架构图 5 |
| **历史 4.18 原稿备份** | [归档/中期报告相关/中期成果更新_20261025/交付_20261109_418原稿备份/](../归档/中期报告相关/中期成果更新_20261025/交付_20261109_418原稿备份/) | 已按 `归档/` 相对路径完整镜像留存 |
