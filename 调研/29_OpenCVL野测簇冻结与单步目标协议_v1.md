# OpenCVL 野测簇冻结与单步跨视角定位协议（v1 / `CV-049`）

2026-11-14。针对 `CV-048` 零像素解码审计暴露出的中心偏置、真值分歧与城市邻近风险，本协议落实**三项改进修正**：先完成字段适配、宏区域地理隔离冻结与评估口径修订，**本轮不启动训练、不调用模型、不读取或消耗保留确认集**。

- **机器协议与冻结清单**：[协议冻结_v1.json](../DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/协议冻结_v1.json) · [公开洁净划分清单_v1.json](../DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/公开洁净划分清单_v1.json) · [批次说明](../DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/README.md)
- **前置证据与约束**：[CV-045 三候选预检](../DATA/processed_data/OpenCVL/公开瑞典三候选校正标签预检_v1/README.md) · [CV-046 三源确认协议](27_三源确认协议_v1.md) · [CV-047 真实上下文准备](28_真实上下文与同尺度取窗准备_v1.md)

---

## 一、落实的三项核心修正

### 修正 1：承认原生 `100m×100m` 图幅约束，第一阶段改立“单步目标定位”协议
- **为何放弃在原生小图上强行做“随机偏心裁剪”**：实测 `SE` 航拍图为 `625×625 @ 0.16 m/px`、`PL` 为 `1000×1000 @ 0.10 m/px`，原生覆盖范围均为 **`100m × 100m`**（四周无实拍外延上下文）；而 `max_offset_m = 20.0` 使全部 `1,361` 对目标落入 5×5 中心 9 格。若无外延图幅而强行平移窗口，只能边缘补零（违反 `CV-047` 禁补零约定）或缩格（破坏 `20m/格`）；同时 `CV-046` 已证明在原生 5×5 / `B10` 导航下无目标 `CenterCross` 至多 9 步可遍历全部中心 9 格，纯导航 SR 无法有效区分目标表征收益。
- **阶段一冻结评估口径**：改为**单步跨视角目标格/方位预测（Single-Step Cross-View Target Localization）**——模型仅见普通地面 RGB 与完整 `100m×100m` 参考图（按 5×5、每格 `20m×20m` 划分），输出 25 格分布，评测 **25 格 Top-1 / Top-3 / Top-5 准确率、Center-9 条件准确率、交叉熵与期望平面误差（米）**。
- **强制基线门槛**：
  - `AlwaysCell12Prior`（恒猜中心格 `12`）：`development_validation` 地板为 **`42.24%`（`49/116`）**，`isolated_confirmation_blind` 地板为 **`39.29%`（`11/28`）**；候选模型必须严格胜过 `AlwaysCell12Prior` 才具备准入意义。
  - 同步披露 `UniformCenter9Prior`（均猜中间 9 格，`11.11%`）与 `CenterCross ≤9 步` 无目标覆盖对照边界（不把单步格点 Top-1 准确率宣称为多步主动导航 SR）。
- **真值口径与双切面报告**：沿用 [capturepoint_corrected_v1.py](../project/src/data/capturepoint_corrected_v1.py) 冻结 `mast3rpnp`（`x = W/2 + dx_mast3rpnp`, `y = H/2 - dy_mast3rpnp`）作为主键，同时在评估台账中标记 `opensfm_cell_agree`（两套位姿 5×5 落格一致率：开发 `74.94%`、验证 `56.90%`、确认 `71.43%`），强制按 **`all_pairs` 全集** 与 **`opensfm_cell_agree` 双真值共识子集** 分开报数。

### 修正 2：`SE` 宏簇隔离划分、盲测保护与仅开发簇限额质检
对 `SE` 全部 `942` 对按已知/封存 `62` 源 `500 m` 保守间隙与 `15 km` 宏区域连通图冻结 5 个互斥角色（`CV-045` 3 候选与 `MM-GAG` 6 确认组继续原封不动）：

| 角色 | 对数 | 地理与成员构成 | 使用权限与隔离距离 |
|---|---:|---|---|
| `blocked_known_overlap` | `351` | `< 500 m` 重叠或精确命中（`257` 近 ZOD + `94` 近 `CV-045` 含 3 个原候选） | **永久禁入**，禁止读取、训练或当新确认 |
| `quarantined_cv045_vicinity` | `12` | `M07` Uppsala (`10`)、`M10` Skellefteå (`1`)、`M16` Kiruna (`1`)，距 `CV-045` 中心 `0.58–1.53 km` | **同城隔离**，防城市建筑/植被先验泄漏至 `CV-045` |
| `development_fit_train` | `435` | `M01` Malmö (`288`) + `M12` Lund (`1`) + 已消费城市外围 `M03` Gothenburg (`100`)、`M04` Stockholm (`29`)、`M05` Örebro (`17`) | 仅此分区开放训练/特征适配；后续**质检目视上限 `≤ 24` 对**（优先抽检 `opensfm != mast3rpnp` 分歧样例） |
| `development_validation` | `116` | 未见城市 `M02` Linköping（4 子簇、5 creators） | 仅用于模型/超参本地开发验证；距已知源 `≥ 97.014 km`、距 `fit_train` `≥ 95.203 km`；**禁止人工调图窥视** |
| `isolated_confirmation_blind` | `28` | `9` 个独立未见瑞典城镇（`M06` Umeå `12`、`M08` Luleå `8`、`M09` Sundsvall `2` 及 6 个偏远单点） | 距已知源 `≥ 62.135 km`、距 `fit_train` `≥ 84.807 km`、距 `dev_val` `≥ 109.465 km`；**严格单次盲测**，冻结前 0 窥视、0 解码 |

- **公开清单防泄漏设计**：[公开洁净划分清单_v1.json](../DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/公开洁净划分清单_v1.json) 仅保留 ID、相对路径与宏簇标签，剥离一切目标 `dx/dy`、格号及盲测区经纬度；目标真值仅存于标注 `policy_input: false` 的评估器专用台账。

### 修正 3：`PL`（`419` 对）单列非隔离参考账户，不混入地理隔离结论
- `PL` 全量缺失 `aerial_latlon`（`0/419`），且包含 `CV-044` 已登记的 3 个元数据参考 ID（`1004803046722240`、`1016275108779955`、`1020327673038045`）。
- 全部 `419` 对冻入 `pl_pending_visual_dedup`：已核验 `IHDR` CRC-32 与双位姿一致性（`68.26%`），**不计入 500 m 地理隔离确认结论**，保留待未来通过感知哈希/编码器特征与 `CV-042` Krakow (`3` 对)及 `PL` 内部完成视觉去重后，再单独开启“非 GPS 去重跨国参考评估”。

---

## 二、源码实现、测试与独立复核证据

- **核心模块**：[opencvl_wild_intake_v1.py](../project/src/data/opencvl_wild_intake_v1.py) · [freeze_opencvl_wild_v1.py](../project/src/tools/freeze_opencvl_wild_v1.py)（不修改受保护的 `capturepoint_intake_v1.py` / `capturepoint_corrected_v1.py` / `inspect_capturepoint_intake_v1.py`）
- **独立复核与单元测试**：[audit_opencvl_wild_intake_v1.py](../project/src/eval/audit_opencvl_wild_intake_v1.py) · [test_opencvl_wild_intake_v1.py](../project/src/tests/test_opencvl_wild_intake_v1.py)
  - 另实现采用三维单位球弦距公式与 `bisect_right` 格边映射，重查 `2,722` 张 PNG 的签名/`IHDR`/CRC-32 并验证公开清单零标签字段泄漏：与 Haversine 最大距离差 `1.004e-9 m`，`auditor_image_decodes = 0`、`model_calls = 0`、`training = 0`、`formal_confirmation_consumed = 0`、`independent_personnel_review = false`。
- **下一步（待另行立项 `CV-050` 授权后执行）**：① 在 `development_fit_train` 内按 `≤ 24` 对限额抽检画质与 `opensfm != mast3rpnp` 分歧主因；② 冻结基线与单一轻量单步目标定位实验（如冻结视觉特征探针 / 线索头评估 `development_validation` 是否稳定高于 `42.24%` 的 `AlwaysCell12Prior` 地板）。
