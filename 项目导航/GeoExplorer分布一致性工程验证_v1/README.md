# 分布一致性真实工程与执行准备

BENCH-024 / 2026-10-12。**本批真实工程已通过；工程交付时无新SR。后继BENCH-025成对开发现已完成并按必要门槛收束，见[结果报告](../../DATA/processed_data/Masa/训练结果/GeoExplorer分布一致性成对开发_v1/训练完成与开发对照报告_v1.md)。**

- [当前工程结果](隔离环境恢复_v2/工程运行/result.json) / [另实现复核](隔离环境恢复_v2/另实现工程复核.json)：三份冻结history的246张量SHA相同，12次真实history调用与原接口输出一致，原采样及Legacy更新保持，部分rollout恢复后参数/optimizer/scheduler/RNG完全相同。仅5头工程更新，骨干未更新；峰值1022MiB、35.83秒，原上限24调用/8工程更新/300秒/2560MiB未放宽。
- [工程执行协议](隔离环境恢复_v2/冻结工程执行协议.json)绑定34项输入与源码；[CPU测试](CPU测试记录.json)为8项边界/恢复，另3项开发接口合成测试通过（3轨迹/24动作，无模型）。两次CPU optimizer夹具更新与5次真实工程更新合计7，不算科研训练。
- 首轮错误解释器记录：[原执行协议](冻结工程执行协议.json) / [失败回执](执行回执.json) / [stderr](worker.stderr.log)。误用D盘全局Transformers5.9，模型构造时缺rope_theta，0history调用/优化更新；旧原件不改。v2直接复用既有隔离解释器4.46.3，与原训练六包版本完全相同，没有安装、下载或改全局依赖，也没有改公开模型代码。
- [下一成对开发](../../DATA/processed_data/Masa/训练结果/GeoExplorer分布一致性成对开发_v1/README.md)：774源码/资产绑定、109fit/28已知开发、140固定同题、50错目标距离例外提前登记；骨干/编码器冻结，仅165k参数头训练。每臂每种子40960环境步，同新初始化，不冒充旧optimizer精确续训。

实际原训练环境为 `project/.runtime/geoexplorer_probe_v1/venv/Scripts/python.exe`。当前工作台与全局D盘Python保持自己的环境。新工程由run_geoexplorer_mask_v2调度、真实执行仍复用绑定的geoexplorer_mask_ppo_v1；最终开发派发run_geoexplorer_mask_v3增加完整源码/队列/目标对照绑定。v1首失败、v2成功原件保留，不能从旧全局协议再次运行。

只读核验：用上述隔离Python调用 `audit_geoexplorer_mask_v1.engineering(协议路径,工程运行路径)`；不得重复prepare/engineering覆盖已有记录。恢复文件是本轮小头工程夹具，不能拿它继续科研训练。原科研权重、默认、S2/S3/S4及失败结论均不改，3备用区域未消费。
