# CV-023A：官方单权重取得与真实工程

**已完成取得、严格CPU加载和普通照片输入工程；尚未验证跨视角定位效果。** 不训练、不生成导航SR，不改当前4.7网页或航拍默认。旧8192参数小头路线保持收束。

- [工程报告与限制](工程报告.md)
- [首取得/输入前冻结协议](执行协议.json)、[历史保护绑定](历史保护绑定.json)
- [权重取得/CRC/SHA](权重取得结果.json)、[逐次Range回执](取得请求事件.jsonl)
- [模型/预处理/依赖输入绑定](工程源码输入绑定.json)、[严格加载343张量](严格加载与逐张量.json)
- [6次编码与资源](工程结果.json)、[工程事件](工程事件.jsonl)、[冻结核验](编码器冻结核验.json)
- [照片处理回执](输入适配回执.json)、[地面输入预览](预览/ground_input.png)、[当前航拍输入预览](预览/aerial_input.png)
- [10边界测试](测试结果.json)、[另实现复核](另实现核验.json)、[下一批条件草案](下一项单次开发对照草案.md)
- [封存](阶段封存.json)、[交付源码](交付源码绑定.json)、[最终收尾](收尾核验.json)

权重唯一保留在`models/sample4geo_cvusa_public_v1/weights_e40_98.6830.pth`；SHA739cf00d303836fba1195f1bb8d54c4b13c53926ce3212fc9ceced335fc59a23，350402687字节。官方整包未保存、未下载完整内容，不能声称取得整包SHA或核过所有包内许可；没有出版社签名/上游公开SHA认证。团队内部验证，不将未知权重许可写成允许再分发，不进Git/搬迁包。

恢复先读回执与封存；`sample4geo_acquire_v1.py`和`sample4geo_engineering_v1.py`均已消费，禁止覆盖重跑。只读复核命令（不加`--save`不写出）：

```powershell
D:\PYTHON\python.exe -B project/src/eval/audit_sample4geo_engineering_v1.py
```

依赖仅`project/.runtime/sample4geo_engineering_v1/`中的timm1.0.20（8683815字节），原torch2.6.0+cu124/torchvision0.21.0+cu124不升级。其它机器按安装回执重建同版本依赖，核模型/代码/输入身份后在新批次运行；本批恢复不重新读原图或编码。不得运行旧collect/prepare/train/导航来“补验收”。
