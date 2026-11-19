# OpenCVL `in_the_wild_test` 野测洁净簇冻结清单 v1（`CV-049`）

本目录保存 `CV-049` 对 `DATA/raw_data/OpenCVL/OpenCVL/mapillary/in_the_wild_test/{SE,PL}` 全部 `1,361` 对原始档案执行零像素解码核验与宏区域地理隔离冻结后的轻量化元信息（总计约 `2.57 MiB`，零原图复制、零像素解码、零模型前向与确认消耗）。上游 [inspect_capturepoint_intake_v1.py](../../../project/src/tools/inspect_capturepoint_intake_v1.py) 面向早期 sample-pack 字段保持原址不改，本批通过独立适配模块 [opencvl_wild_intake_v1.py](../../../project/src/data/opencvl_wild_intake_v1.py) 实现。

## 核心角色划分与基准先验

| 分区角色 | 对数 | 说明与隔离口径 | `AlwaysCell12` 先验率 | `opensfm == mast3rpnp` 一致率 |
|---|---:|---|---:|---:|
| `blocked_known_overlap` | `351` | `SE` 中与 62 个已知/封存源（含 `CV-045` 3 候选）保守间隔 `< 500 m` 或 ID 重合；**硬禁入模** | `29.06%` (`102/351`) | `77.78%` (`273/351`) |
| `quarantined_cv045_vicinity` | `12` | `SE` 中虽过 `500 m` 间隔但位于 `CV-045` 同城近邻（`< 2 km`，Uppsala/Skellefteå/Kiruna）；**隔离防城风格泄漏** | `33.33%` (`4/12`) | `33.33%` (`4/12`) |
| `development_fit_train` | `435` | `M01` Malmö (`288`) + `M12` Lund (`1`) + 已消费城市外围 `M03/M04/M05` (`146`)；允许 ≤24 对限额抽检 | `32.41%` (`141/435`) | `74.94%` (`326/435`) |
| `development_validation` | `116` | 未见城市 `M02` Linköping（距 `fit_train` `≥ 95.203 km`、距已知源 `≥ 97.014 km`）；禁止人工调图窥视 | `42.24%` (`49/116`) | `56.90%` (`66/116`) |
| `isolated_confirmation_blind` | `28` | 9 个独立未见瑞典城市/城镇（距已知源间隔 `≥ 62.135 km`，距 `fit_train` `≥ 84.807 km`，距 `dev_val` `≥ 109.465 km`）；**单次盲测封存** | `39.29%` (`11/28`) | `71.43%` (`20/28`) |
| `pl_pending_visual_dedup` | `419` | `PL` 全部缺失 `aerial_latlon`（含 `CV-044` 3 个元数据已知 ID）；暂不列入地理隔离结论 | `29.83%` (`125/419`) | `68.26%` (`286/419`) |

## 文件清单与防泄漏边界

- [协议冻结_v1.json](协议冻结_v1.json)：机器可读 `CV-049` 方案协议、几何约定、`AlwaysCell12` 与 `CenterCross` 基线约束。
- [公开洁净划分清单_v1.json](公开洁净划分清单_v1.json)：供开发脚本读取的 ID 与相对路径索引，**不包含任何样本的 `dx/dy` 偏移、格子编号或盲测坐标**。
- [核验/评估器专用冻结台账_禁止入模.json](核验/评估器专用冻结台账_禁止入模.json)：评估器专用完整坐标、目标格子及 `IHDR` 校验台账（明确标注 `policy_input: false`）。
- [核验/源码与输入绑定_v1.json](核验/源码与输入绑定_v1.json) · [核验/另实现复核_v1.json](核验/另实现复核_v1.json)：上游保护源码哈希、生成源码哈希与 3D 球面弦距/`bisect_right` 独立复核（最大距离方程差 `1.004e-9 m`，`2,722` 张 PNG `IHDR` CRC-32 全过）。

## 只读复核命令

- `D:/PYTHON/python.exe -X utf8 -B -m project.src.eval.audit_opencvl_wild_intake_v1`（不加 `--save`，重验全部 1,361 对与 2,722 个 PNG 头）。
- `D:/PYTHON/python.exe -X utf8 -B -m unittest project.src.tests.test_opencvl_wild_intake_v1 project.src.tests.test_capturepoint_intake_v1`（运行 23 项边界单测）。
