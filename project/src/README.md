# 当前源码入口与历史边界

当前工作台为 **4.13.1**：webapp/workbench_server_v21.py，内核platform_core/service_v11.py，搬迁器tools/package_workbench_v25.py。唯一操作入口见[平台](../../平台/README.md)，研究现状见[PROJECT_STATE](../../docs/PROJECT_STATE.md)。

- 实验创建/执行/持久化由platform_core负责；Web为可替换视图。航拍CLI仍为cli_v2，地面登记/旧分数诊断为cli_v4；CLI不是所有GUI功能的总入口。
- 场景保存/只读隔离用scene_workspace_v2；地面单对证据与人工连续观测分别由ground_live/ground_walk组件组合。
- CV031/CV041证据读取内核不加载模型；空间小头页面采用spatial_evidence_v2.js，原v1保留封存。
- 当前服务仍明确依赖server12及其父模块、service2至11等。旧编号不等于可删；详见[生命周期](../../docs/维护/版本生命周期.json)。本轮不重构导航内核或改变研究模型。
- 研究执行源码、失败记录和复核绑定原址保留。不能运行旧prepare/execute/publish覆盖封存。新任务另认领、冻结和授权。
- 旧源码说明原字节见[整理前快照](../../平台/实验工作台_v4/发布修复与存储收束_v1/入口快照/02.before)，其中“当前CV003/005/006”等仅反映旧时点，不是接手指令。模块中较早README亦须按其版本理解。
