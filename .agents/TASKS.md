# 协作任务表

更新时间：2026-10-05。这里只登记已知认领情况，不推定未登记Agent是否在执行。`in_progress`表示实际已认领写入；`ready`可认领，`planned`等待前置结果，`done`已经交接并释放。

| ID | 任务 | 状态 | 执行者/会话 | 独占写入范围 | 依赖 | 下一动作 |
|---|---|---|---|---|---|---|
| DOC-001 | 拆分README并建立协作入口 | done | Codex / docs-20261003 | 已释放；原范围为AGENTS.md、README.md、.agents/、docs/、项目导航/README.md、项目导航/当前进度.md、项目导航/目录结构与维护.md | — | 链接、规则迁移及849文件哈希核验通过；已交接 |
| GIT-001 | 提交截至2026-10-03的项目更新 | done | Codex / git-20261003-current-chat | 已释放；原范围为Git暂存区与本地main引用、.gitignore大型文件规则、README测试目录登记及本任务协调记录；测试夹具已安全清理 | 用户明确授权本地提交；保留既有更新和冻结原件 | 2037文件本地提交完成；119项单测、语法/凭据/链接检查通过，849保护哈希一致；没有启动实验或推送；详见HANDOFF |
| GIT-002 | 通过SSH推送当前main到GitHub | done | Codex / ssh-push-20261003-current-chat | 已释放；原范围为origin/main推送、本地Git引用及本任务协调记录；代码和实验原件只读 | 用户明确授权SSH推送；GIT-001完成 | 项目提交6a9af59已通过SSH推送，远程SHA一致；协调记录单独提交并同步；详见HANDOFF |
| DATA-001 | 补全官方分区目录及3×3连续区域候选 | done | Codex / catalog-20261003 | 已释放；原范围为新area_catalog_v1/area_catalog_links_v2相关6源文件、冻结方案/测试、新调查批次和共享入口 | DOC-001完成 | 15单测、263标签及最大12区独立复核通过；像素质量待验；已交接 |
| DATA-002 | 真实15×15影像坐标、像素与质量工程 | done | Codex / real-area15-20261003 | 已释放；新real_area15三模块、方案/测试、新原生数据及共享入口 | DATA-001目录/标签通过 | 执行及可信复核结束；质量最大7区，10区门槛未过；2已知工程图完成，无新SR |
| DATA-003 | 同源同尺度窗口静态可行性 | done | Codex / native-window-20261004 | 已释放；新area_windows_v1三源码、方案/测试、新静态批次及共享入口 | DATA-002只读冻结原件 | 577合格窗口，双优化器精确最大7；10区未过，因素收束；12单测/独立复核及交付封存通过，无新SR |
| DOC-002 | 明确报告适用范围与未来工作 | done | Codex / native-window-20261004 | 已释放；中期报告相关/报告适用范围与未来工作_v1.md及README、状态和协调入口 | 用户要求写清边界、暂不扩大研究 | 任务定义/未来工作引用文字及链接已核验，原Word/PDF/PPT保留 |
| COORD-001 | Codex与ZCode最小CLI往返验证 | done | Codex / external-smoke-20261004 | 已释放；external_agent_smoke_v1/v2、工具联通验证目录、调用说明与README/协调入口 | 用户明确要求试验双向调用；不启动研究实验 | 试验结束，联通未过：ZCode buddy渠道拒绝；Codex独立120秒超时，无随机回执，未执行ZCode→Codex；1918保护项一致，未接管桌面会话 |
| COORD-002 | Pi及重新授权8790模型接口验证 | done | Codex / pi-gateway-20261004 | 已释放；pi_gateway_smoke_v1及Guard、Pi独立试验目录/说明、README/AGENTS一条授权规则及协调记录 | 用户本轮重新授权截图5模型和无key接口，优先于旧弃用规则 | 四模型文本/工具通过，Pi+buddy/deepseek读取回执/Guard通过；qoder20秒超时；17请求，1918保护项一致，无新SR |
| DATA-004 | 更大连续同模态影像池调查 | done | Codex / massgis-pool-20261004 | 已释放；原范围为新MassGIS四调查模块、独立复核/两测试、冻结方案、新调查批次及共享入口 | DATA-003裁切因素收束；用户授权继续 | 官方1540索引/20几何见证/三代表原生头通过；像素质量pending；18测试/12请求/独立重算/1964保护一致，无新SR |
| COORD-003 | Pi首项真实只读规则摘录 | done | Codex调度 / Pi buddy-deepseek / pi-read-20261004 | 已释放；原范围pi_read_task_v1.py、pi_allowlist_guard_v2.ts及.agents/Pi只读协作_20261004_v1/；Pi无仓库写入 | 用户授权简单重复活外包；COORD-002通过 | 六成功read/两回合/14538报告token，Guard5检查及封存通过；旧像素条件/行号由Codex勘误后采用，不替代主线验收 |
| DATA-005 | MassGIS原生解码与小区域质量工程 | done | Codex / massgis-native-20261004 | 已释放；原范围为native_v1/v2/v3、600PNG环境、目标输入桥接、复核/测试、四源原件/新工程批次和共享入口 | 用户按计划授权；DATA-004通过 | 1已见工程区/225原生600 PNG/29混合格通过；37测试/225reset/独立像素复核/2039保护通过；v1/v2拒绝保留，v3码流读取，无新SR |
| DATA-006 | MassGIS正式池获取与兼容协议准备 | done | Codex / massgis-confirm-pool-20261004 | 已释放；原范围为正式池获取_v1新代码/测试/方案/raw/processed及共享入口 | DATA-005工程通过；所有旧原件只读 | 20候选/80源取满，13区合格/10正式保留/3备用；16测试/独立像素来源复核/2355保护通过；0导航/新SR，已交接 |
| DATA-007 | MassGIS同产品开发导航兼容 | done | Codex / massgis-navigation-dev-20261004 | 已释放；原范围为native_area_v2、navigator_v1、navigation_compat_v1、audit_v1/v2、两个测试/两个交付生成器、新开发批次/方案/测试及共享入口 | DATA-006数据通过；DATA-005已消费开发区，正式10区/3备用未消费 | 工程兼容/目标开发门槛通过，10格Coverage SR68.00%/SG646.67m；34+4测试/2300轨迹独立重放/6040保护、交付QA/554文件封存通过；15格无导航，EVAL-001可认领协议准备 |
| EVAL-001 | 冻结MassGIS正式10×10/15×15协议与工程接口 | done | Codex / massgis-confirm-protocol-20261004 | 已释放；原5源码/方案/测试、新正式导航协议冻结_v1批次及共享入口 | DATA-006数据及DATA-007开发兼容通过；正式10区未消费 | 10区750/1000题/门槛冻结；30测试/450旧开发题7217动作/独立协议复核/6603保护/104文件封存通过；无正式模型消费或新SR，EVAL-002待认领 |
| EVAL-002 | MassGIS正式双网格执行与导航确认 | done | Codex / massgis-formal-run-20261004 | 已释放；原范围为四新执行/审计/报告/测试源码、新正式十乘十十五乘十五确认_v1批次、执行方案/测试和共享入口 | EVAL-001原冻结协议；全部旧原件只读 | 42测试/14000轨迹255125动作/6715保护/753封存通过；10/15格SR54.62%/20.27%，新域必要门槛未过，目标控制未启动、无默认升级；已交接 |
| DOC-003 | 状态同步规则与ZCode接手提示词 | done | Codex / handoff-prompt-20261004 | 已释放；原范围为AGENTS一条协调规则、.agents/ZCode接手EVAL-003提示词_v1.md、TASKS/HANDOFF及README提示词入口 | 用户要求额度切换接手；EVAL-002完成，EVAL-003未启动 | 规则/提示词范围、113链接/白空及753封存SHA通过；已交接，无实验或外部调用 |
| EVAL-003 | MassGIS误触发与覆盖只读失败诊断 | done | ZCode / zcode-eval003-20261004 | 已释放；原范围为诊断源码 project/src/eval/{diagnose_massgis_mistrigger_v1,build_eval003_input_binding_v1}.py、project/src/tests/test_diagnose_massgis_mistrigger_v1.py、新批次 新域误触发与覆盖诊断_v1/ 及共享入口 | EVAL-002已完整封存，两网格方法验收未过 | 10500轨迹/282480探针只读诊断完成：机会=成功3563、失败0机会、误触发83.4%在距离≥2且重定向劣于探索器提议、15格全程性塌陷；9项单测与另实现复核通过；唯一候选=cue门槛校准（冻结方案已存，不启动）；详见HANDOFF |
| DEV-001 | 误触发治理（cue接受门槛校准）开发验证 | done | ZCode / zcode-dev001-20261005 | 已释放；原范围为新源码 project/src/agents/massgis_threshold_navigator_v1.py、project/src/eval/{massgis_threshold_calibration_v1,massgis_threshold_dev_run_v1}.py、project/src/tests/test_massgis_threshold_navigator_v1.py、新批次 误触发治理开发验证_v1/ 及共享入口 | 用户明确授权启动；EVAL-003冻结方案 | 校准τ\*=0.95（规范先于曲线冻结）；1950条双臂导航，grid5/10 T050回归525条全等；门槛①④过、②③未过→**总判定未过，按停止条件收束因素**，不调门槛不升级默认；详见HANDOFF |
| DOC-004 | 整理交接并编写Codex接手提示词 | done | ZCode / zcode-dev001-20261005 | 已释放；原范围为.agents/Codex接手EVAL-004提示词_v1.md、DEV-001批次内下一项草案、TASKS/HANDOFF/README/PROJECT_STATE协调入口 | 用户要求转回Codex并给下一步提示词；DEV-001已完成 | 草案与提示词已写（EVAL-004=15×15到达能力只读诊断，首步配对logits实验为核心）；锚点几何已预核实并要求接手者独立重推；无实验无提交 |
| EVAL-004 | 15×15到达能力与输入表征漂移只读诊断 | done | Codex / massgis-arrival-drift-audit-v11-20261005 | 已释放；v1-v6主分析及审计v2-v6完整保留；v6补充诊断报告/输入绑定/源码快照/配对与路径明细；v11审计源码/测试及审计v6最终复核；README、PROJECT_STATE、TASKS、HANDOFF、实验索引 | DEV-001已按停止条件收束；只读EVAL-002正式封存/EVAL-003/DEV-001原件 | 复核753封存、160主/规则轨迹seal及分母、188246输入SHA、750共同题/4500首步配对、SR分层/10区域bootstrap/路线分叉/491520训练上下文通过；8项测试通过；补充报告与可追溯交付绑定见十五乘十五到达能力漂移诊断_v6；坐标格点对齐是唯一未执行候选，另立DEV再评估；0模型forward/训练/导航/网络/备用消费/默认改动；无提交/推送 |
| DEMO-001 | 项目进度复查、轻量轨迹回放与平台V0.1 | done | Codex / replay-platform-20261005 | 已释放；原范围为webapp六源码/verify与静态界面、test_replay_platform_v1.py、平台/主动探索演示_v1/、进度计划及共享入口 | EVAL-004完成；封存日志/任务/图块只读 | 已交接：15000轨迹/261622动作/3250图块、原汇总/753封存/32交付/3默认和10单测通过；所列浏览器检查通过，下载事件回执未验证；8766回放服务保留，无新模型/训练/导航/备用消费 |
| PLATFORM-002 | 界面驱动实时Agent导航与演示日志 | ready | 未认领 | 认领时声明新live服务/界面版本、适配与测试、开发演示批次及共享入口；保留V0.1与历史原件 | DEMO-001完成；只用已消费开发区域，原冻结策略/环境 | 先冻结演示协议、源码/模型/均值/任务身份与新输出路径；单步/自动/终止/状态隔离、公开Observation权限、冻结轨迹回归；补JSON按钮原生浏览器下载兼容检查，不默认启动15格研究或备用确认 |
| GIT-003 | 分层整理本地工具、精简Git收录并推送GitHub | done | Codex / repository-release-20261005 | 已释放；本轮目录/忽略规则、核验源码与公开维护文档均已交付，数据原址保留 | 用户明确授权整理并上传GitHub；main现场保留 | 87迁移/950退出索引SHA、753/32历史、平台25+7及3默认通过；19测试、无本地目录样本9项、429语法/230链接/只读发布审查通过。主提交2b31b644已上传main且远程SHA一致；补同步交接状态，无新实验 |

ZCode及其它Agent当前未登记，不自动指派给它们。认领前读取AGENTS.md及本表；只有`in_progress`行的写入范围处于占用，DOC-001已交接并释放。`ready/planned`行只是拆分建议，不代表任务已经开始或某Agent已获独占范围。

## 认领与交接

1. 先核对状态和写入范围，再把自己的任务改为`in_progress`；写清执行者/会话、具体路径、实际更新时间和验收要求。未知输出路径先登记再创建，不能用“整个project/src”一类范围阻塞不相交任务。
2. 更新表前读取最新版；若保存前文件已变，重新合并自己的行，保留其它Agent的记录。共享文档合并也遵守这条。
3. EVAL-004及DEMO-001均已交接并释放；当前下一任务PLATFORM-002待认领，优先实时平台与课程交付。MassGIS坐标格点因素未启动，不作为平台交付前置。默认顺序接手、共享入口合并；外部助手旧授权不自动替代本批认领或科研验收。
4. 完成/移交先写HANDOFF，再标`done`或说明剩余任务及接手状态；保留记录但释放写入范围。暂停不能被其它Agent默认为完成。

## 新任务记录模板

```text
ID：
目标、状态、执行者/会话、更新时间：
独占写入路径（具体文件/新版本目录）：
只读输入和前置依赖：
验证方式与通过要求：
实际剩余事项、交接入口：
```
