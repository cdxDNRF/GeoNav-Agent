# GeoNav实验内核4.18

当前Python组合入口为`service_v16.Workbench`，Web入口为`workbench_server_v27`。内核管理任务、环境、状态与记录；Web负责表单和显示。旧编号模块仍有明确继承/适配用途，不是并列的当前启动入口。

| 模块 | 当前职责与边界 |
|---|---|
| `engine_v2` | 航拍任务与实验、单步/单题/队列、临时租期和持久记录；恢复未完成内存不承诺续跑 |
| `service_v13` | 在既有service_v11链上组合拍摄点人工任务，航拍M0/区域方案/云接口保持原协议 |
| [capturepoint_catalog_v1](capturepoint_catalog_v1.py) | CV042固定192题及原图/封存核验、显示裁切；真值留在环境端 |
| [capturepoint_workspace_v1](capturepoint_workspace_v1.py) | 任务创建、人工四动作、保存配置/终局、只读重开和同题新建；没有策略模型工厂 |
| `scene_workspace_v2` | 航拍自由场景、人工/Agent接管、终局快照与异常隔离；不同控制类型分账 |
| `ground_walk_v1` | 原MM-GAG配对人工连续观察和按需CPU编码证据；不提供可靠自主方向 |
| `ground_tasks_v2` / `ground_live_v3` | 地面任务登记/已存诊断和实时单对编码；未知照片不借用旧缓存分数 |
| `storage_v1` | 单一服务独占存储锁；不能让第二服务接管活动目录 |

## 拍摄点任务的Python接口

`workbench.capturepoints`提供`catalog/create/save_task/step/finish/save_run/state/reopen/image/heartbeat/release`。create只接受已登记任务编号与名称，不接目标覆盖、预算增改或M0/云模型；固定5×5/B10。底层目录与环境规则不依赖HTTP，可供后续GUI复用。

显式保存位于store/capturepoint_tasks/<id>/task.json与record.json，照片只在数据根保存一份，重放依赖原资产。未访问图块默认拒绝；观察者全图不改变动作/访问历史，但使用留痕。人工结果的Agent SR恒为null，不纳入科研验收。标签、路径、来源身份和全图不是Agent输入。

## 已有CLI与新资料接收

航拍执行继续使用`python -m project.src.platform_core.cli_v2`，地面登记/旧分数诊断使用`cli_v4`；仅覆盖各自已实现功能，不宣称新拍摄点人工动作已有CLI命令。完整开发仓库中的资料接收可独立使用`python -m project.src.tools.inspect_capturepoint_intake_v1 --pool`，只读哈希/编码头/标签，不运行模型，也不代替训练或确认准入。

## 保存和恢复

输入、任务身份和实际执行源码在运行前绑定；保存配置/终局不覆盖不同记录。未完成写盘不进入有效目录，损坏原件保留，其它有效记录可继续使用。临时过程离页或超时释放，不自动落盘；保存的任务可另开同题，保存的终局可只读重放，不把已退出进程当作可续训/续跑状态。

原航拍及资料复核的通过范围继续按原协议；地面照片自主导航仍未完成。新协议需适配数据、信息/行动权限和指标，不能因为有注册机制就视为科学验收通过。


## 原拍摄点控制模块（仍为继承和旧记录依赖）

新增[navigation_controller_v1](navigation_controller_v1.py)公开状态契约和四动作转移、[capturepoint_sessions_v1](capturepoint_sessions_v1.py)人工/Frontier接管及同题比较。新会话保存于store/capturepoint_sessions，旧manual namespace只读兼容；同一环境分类人工/规则/混合，不把Frontier计成视觉目标模型。

无需HTTP的[controller_cli_v1](controller_cli_v1.py)可只读catalog，或在明确选定工作目录显式run-rule。当前统一入口改用下述CLIv2；旧CLI只作原交付参考，不对活动store执行。[操作与恢复](../../../平台/实验工作台_v4/统一控制器与规则对照_v1/使用与恢复.md)。

## 原4.16统一控制模块与验收（复用依赖）

[controller_hub_v1](controller_hub_v1.py)在原SceneWorkspace/MasaRuntime上适配Masa S2航拍M0、Frontier和人工；地面目标只接原人工/Frontier。不重写冻结策略、环境或缓存；外部模型调用及所有历史评测器尚未迁入该统一接口。[service_v14](service_v14.py)与server24为原4.16组合；当前正式见下方4.17。

[controller_batch_v1](controller_batch_v1.py)为最多10题×M0/Frontier保存队列。每次step最多执行一个动作；已完成轨迹保存到原scenes模板并释放对应内存。重启后只读保留已完成题，未完成题的模型内存不能恢复，须显式选择从起点另开；旧尝试动作和策略调用数保留。SR按计划分母，未跑全时平均SG/覆盖/移动为null，Masa尺度未知不造米数。人工/混合/模型/规则分别记录，均属已消费开发记录。

[controller_cli_v2](controller_cli_v2.py)支持catalog/run/queue/resume/export，非catalog操作先取得StoreLock再构造Workbench。队列需要显式保存，不能和服务共用活动store；真实运行仍须满足任务授权。新UI分开当前一条轨迹与整个队列自动运行，保存任务/终局和旧记录兼容由同一Hub提供。新旧导出接口均限制终局，暂停不冒充可恢复快照。

[27项合成检查](../tests/test_controller_hub_v1.py)已通过：公开输入、M0模态隔离、因果历史同步、末步成功、算法锁定、计划分母、完成题不重跑、中断显式重开、跨页权限、写盘失败计数与CLI锁/HTTP边界。测试只在内存中使用合成环境和策略替身，无新目录、真实模型或真实导航。运行：`D:/PYTHON/python.exe -X utf8 -B -m unittest project.src.tests.test_controller_hub_v1`。

真实执行与GUI已通过并切换4.16：20主轨迹172动作，M0 8/10、SG0.50格，Frontier4/10、SG2.70格；另2GUI过程12动作，总22条184动作。10条M0与历史轨迹一致，原权重/均值/阈值保持。实际只验seed0和4已消费来源，不是新科研确认。[操作与恢复](../../../平台/实验工作台_v4/航拍统一接入与小队列_v1/使用与恢复.md)。

## 队列修复（4.17）

原4.17采用[controller_batch_v2](controller_batch_v2.py)、[service_v15](service_v15.py)与[CLI v3](controller_cli_v3.py)，继续作为当前继承/本地命令依赖；API组合见4.18。batch_v1和service14仍为冻结的继承依赖，原字节保留；当前控制页改用controller_console_v2.js，原HTML/CSS复用。

v2在判断live前先清理租期；严格检查定义/进度结构，坏队列逐个隔离。所有完成行都通过Hub核对提交记录及题目/权重/预算/控制来源/输入/结果；证据异常时原JSON不改，evidence_complete=false、evidence_issues说明原因、cached_metrics仅供诊断，SR/SG置空并拒绝继续执行与导出。恢复原件后重新核验即可；计划分母不变，不自动重算。空/仅地面目录均正确初始化。

[新增12项内存回归](../tests/test_controller_batch_v2.py)与原27项通过，17GUI及旧20行只读核验通过，无新模型或导航。[当前操作与恢复](../../../平台/实验工作台_v4/队列恢复与证据修复_v1/使用与恢复.md)。旧CLIv2仅作原交付参考，当前操作优先CLI v3，不对活动store并行执行。

## API统一接口（4.18）

当前[service_v16](service_v16.py)组合[controller_hub_v2](controller_hub_v2.py)、[scene_workspace_v3](scene_workspace_v3.py)、[API执行与账本](api_controller_v1.py)和[固定十题对照](api_comparison_v1.py)。本地队列继续复用controller_batch_v2，原M0/Frontier环境与权重不改；API CLI是[controller_cli_v4](controller_cli_v4.py)，只提供API组操作，本地命令仍用CLI v3，不宣称整个历史评测器已迁入。

方法标为单API＋M0失败回退：API只收两图与公开状态，M0每步备用而不发送其提议；历史同步只热身本地，不补发API。每轨迹10/固定组100请求，发送前create-only记录，结果另写create-only，不重置消耗；未决/连续3错误/上限会停。配置、任务、原基线/源码哈希绑定，保存/导出仅重读。凭据/图片正文不入账本；未核单价时货币成本未知。

52唯一测试、10离线与10真实结果只读GUI通过；真实96请求0回退/重试，API1/10/SG3.20低于原M0与Frontier，因此可选不升级默认。本轮没有新训练/编码/确认。发布server27只修两位数计数窄屏换行，真正执行绑定server26/console3和98依赖仍保留。[使用与恢复](../../../平台/实验工作台_v4/单模型API统一接入与同题对照_v1/使用与恢复.md)。
