# OpenCVL 野测开发集分歧抽检与单步定位基线协议（v1 / `CV-050`）

2026-11-14。在 [CV-049 野测簇冻结协议](29_OpenCVL野测簇冻结与单步目标协议_v1.md) 完成 `SE` 宏簇隔离（`435 fit_train / 116 dev_val / 28 isolated_confirmation_blind / 12 quarantined / 351 blocked`）基础上，本报告落实 **`CV-050`**：① 仅对 `development_fit_train`（`435` 对）执行严格受限的 **`24` 对（`48` 张图像）分层画质与分歧抽检**，全程 `development_validation`（`116` 对）与 `isolated_confirmation_blind`（`28` 对）**零像素解码、零模型前向、零盲测统计**；② 揭示全量开发集 `opensfm != mast3rpnp` 分歧的几何主因与跨城地面画幅纵横比偏移；③ 在 `fit_train`（`435` 对）与 `dev_val`（`116` 对）上冻结 4 个零图像统计基线的分层对照地板，并预登记面向 `CV-051` 的单步跨视角 25 格视觉定位实验协议。

- **产出批次与协议资产**：[开发集分歧抽检与单步基线_v1 README](../DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/README.md) · [抽检 24 对核验 JSON](../DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/抽检24对画质与分歧核验_v1.json) · [统计基线对照 JSON](../DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/单步定位统计基线对照_v1.json) · [CV-051 视觉协议冻结 JSON](../DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/单步视觉实验冻结协议_v1.json)

---

## 一、`opensfm != mast3rpnp` 分歧几何机制与开发集双纵横比发现

对 [评估器专用冻结台账](../DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/评估器专用冻结台账_禁止入模.json) 中获批分析的两个开发分片（`development_fit_train` `435` 对、`development_validation` `116` 对；严禁触碰 `isolated_confirmation_blind` `28` 对）执行纯元数据分析（0 像素解码）：

### 1. 分歧几乎全部发生在 `20m×20m` 内部网格边界 1 格邻域内
| 开发分片 | 总对数 | 5×5 格一致 (`opensfm == mast3rpnp`) | 格分歧对数 | 分歧 `Chebyshev <= 1` (8邻域) | 分歧 `L1 == 1` (4邻域) | 分歧对距 20m 网格线中位边距 | 一致对距 20m 网格线中位边距 | 分歧平面差中位数 / 最大值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `development_fit_train` | `435` | `326` (`74.94%`) | `109` (`25.06%`) | **`108/109` (`99.08%`)** | `94/109` (`86.24%`) | **`1.20 m`**（`<1m` 占 `44.95%`） | **`3.12 m`**（`<1m` 占 `12.88%`） | `5.20 m` / `32.35 m` |
| `development_validation` (`M02` Linköping) | `116` | `66` (`56.90%`) | `50` (`43.10%`) | **`49/50` (`98.00%`)** | `30/50` (`60.00%`) | **`2.24 m`**（`<1m` 占 `22.00%`） | **`2.96 m`**（`<1m` 占 `10.61%`） | `9.95 m` / `29.23 m` |

- **关键科学结论**：`development_fit_train` 的 `109` 对分歧中，**`99.08%` (`108/109`) 仅跨相邻 8 邻域格线**（仅 `1` 对因 `32.35 m` 偏移跨达 `Chebyshev=2`），且 `47.71%`（`52/109`）平面差 $\le 5.0\text{ m}$，直接体现 **近边界点微米/亚米级位移引起的 5×5 硬离散化量化敏感性**（分歧对中位边距 `1.20 m` 显著低于一致对 `3.12 m`）；而 `dev_val` 中存在 `24/50` 对分歧平面差 $> 10.0\text{ m}$（全集共 `33/116` 对平面差 $> 10.0\text{ m}$）。因此，Stage-1 评测不能只看硬 1-hot Top-1，必须配套 **高斯软目标分布（$\sigma = 5.0\text{ m}$）KL/交叉熵、期望平面误差（米）与 `Chebyshev-1` / `L1-1` 邻域容忍度**，并按四分层（`all_pairs`、`opensfm_cell_agree`、`interior_margin_ge_2m`、`large_discrepancy_gt_10m`）同步报数。

### 2. 地面图像双原生纵横比（`952×714` vs `952×532`）跨城显著偏移
- 元数据与 `IHDR` 核验发现，`SE` 地面图像存在 **两种合法原生画幅**（均按宽 `952 px` 归一化）：`952×714`（4:3）与 `952×532`（16:9），而航拍参考图统一为 `625×625`。
- 跨分片比例发生明显偏移：
  - `development_fit_train`（`435` 对）：`952×714` 占 **`259` 对（`59.54%`）**，`952×532` 占 **`176` 对（`40.46%`）**；
  - `development_validation`（`116` 对）：`952×714` 仅 **`14` 对（`12.07%`）**，`952×532` 高达 **`102` 对（`87.93%`）**。
- 该现象若不加约束，模型极易在非等比直接 `Resize(224, 224)` 时受相机画幅比例域迁移影响。因此 `CV-051` 协议强制规定等比几何预处理与 `952×714` / `952×532` 分切面审计，严禁向模型传入图像原始尺寸字段。

---

## 二、`development_fit_train` 受限 24 对（48 图）分层画质与分歧抽检

通过 [opencvl_wild_qa_v1.py](../project/src/data/opencvl_wild_qa_v1.py) 与 [run_opencvl_wild_qa_v1.py](../project/src/tools/run_opencvl_wild_qa_v1.py)，**严格限定在 `development_fit_train`（435 对）内** 按确定性规则分层抽取 **24 对（48 张 PNG）** 解码验质：

1. **大偏移分歧层（`large_discrepancy_disagree`，10 对）**：按双管线平面差由大到小选取前 10（`15.45 m ~ 32.35 m`，覆盖 `M01/M03/M04/M05`）；
2. **近边界跨格层（`boundary_straddling_disagree`，10 对）**：在平面差 $\le 5.0\text{ m}$ 的分歧样本中按宏簇（`M01/M03/M05/M12`）轮转取距 20m 网格线最近的前 10 对（边距 `0.08 m ~ 1.68 m`）；
3. **跨城一致对照层（`consensus_control`，4 对）**：从 `M01/M03/M04/M05` 核心宏簇各取 1 对边界边距 $\ge 3.0\text{ m}$ 且双管线一致样本作为正常对照。

**抽检核验结果（100% 通过）**：
- **解码会计**：`image_decodes = 48`（`fit_train_decodes = 48`、`development_validation_decodes = 0`、`isolated_confirmation_blind_decodes = 0`、`blocked_or_quarantined_decodes = 0`、`pl_decodes = 0`）；
- **尺寸与画幅覆盖**：`16` 对地面图为 `952×714`、`8` 对为 `952×532`，全部 `24` 张航拍图为 `625×625`，逐图精确匹配冻结台账 `IHDR`；
- **画质门限（对齐 `CV-045`）**：全部 48 张图像 `nonblank = true`，地面图 `gray_std` 分布 `[41.78, 83.59]`（远高于门限 `2.0`），航拍图 `gray_std` 分布 `[33.38, 52.42]`；地面近黑/近白占比上限分别仅 `0.1433` / `0.1993`（航拍 `0.0` / `0.0183`），远低于异常阈值 `0.95`；主实现与独立整点二阶矩复核的均值/标准差最大绝对差仅 `1.274e-10 <= 1e-6`。

---

## 三、单步 25 格定位统计基线对照总表（零模型、零盲测评测）

完全依据 `development_fit_train`（`435` 对）历史标签统计构造 4 个零图像先验分布，并在 `development_fit_train`（435 对）与 `development_validation`（116 对）四个分层上评测（`TrainEmpiricalPrior` 在 `fit_train` 的 25 格经验排序前 9 为 `[12, 7, 17, 11, 13, 16, 18, 8, 6]`，Dirichlet 平滑 $\alpha = 0.5$；`AlwaysCell12Prior` 与 `UniformCenter9Prior` 采用 $\epsilon = 10^{-3}$ 均匀底噪平滑以计算有限 KL 散度）：

### 1. `development_validation`（116 对，`M02` Linköping — `CV-051` 核心选型基准）
| 统计基线（零图像输入） | 子集 | 样本数 | Top-1 | Top-3 | Top-5 | Cheb-1 ($3\times3$) | L1-1 (4邻域) | 期望平面误差 (m) | Soft-KL ($\sigma=5\text{m}$) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Uniform25Prior` (`4%` 随机均值) | `all_pairs` | `116` | `0.00%` (`4.00%`*) | `0.00%` (`12.00%`*) | `0.00%` (`20.00%`*) | `3.45%` | `0.00%` | `39.26` | `2.761` |
| `UniformCenter9Prior` (`11.10%` 均值) | `all_pairs` | `116` | `42.24%` (`11.10%`*) | `68.97%` (`33.31%`*) | `89.66%` (`55.52%`*) | `100.00%` | `89.66%` | `24.64` | `1.740` |
| **`AlwaysCell12Prior`** | **`all_pairs`** | `116` | **`42.24%`** (`49/116`) | **`68.97%`** (`80/116`) | `89.66%` | `100.00%` | `89.66%` | **`12.51`** | `5.604` |
| `AlwaysCell12Prior` | `opensfm_cell_agree` | `66` | `40.91%` (`27/66`) | `66.67%` | `87.88%` | `100.00%` | `87.88%` | `12.62` | `5.652` |
| `AlwaysCell12Prior` | `interior_margin_ge_2m` | `70` | `40.00%` (`28/70`) | `67.14%` | `88.57%` | `100.00%` | `88.57%` | `12.71` | `5.882` |
| `AlwaysCell12Prior` | `large_discrepancy_gt_10m` | `33` | `45.45%` (`15/33`) | `75.76%` | `100.00%` | `100.00%` | `100.00%` | `11.81` | `5.342` |
| **`TrainEmpiricalPrior`** | **`all_pairs`** | `116` | **`42.24%`** (`49/116`) | `67.24%` (`78/116`) | `89.66%` | `100.00%` | `89.66%` | `21.42` | **`1.420`** |
| `TrainEmpiricalPrior` | `opensfm_cell_agree` | `66` | `40.91%` (`27/66`) | `66.67%` | `87.88%` | `100.00%` | `87.88%` | `21.51` | `1.481` |
| `TrainEmpiricalPrior` | `interior_margin_ge_2m` | `70` | `40.00%` (`28/70`) | `62.86%` | `88.57%` | `100.00%` | `88.57%` | `21.60` | `1.655` |
| `TrainEmpiricalPrior` | `large_discrepancy_gt_10m` | `33` | `45.45%` (`15/33`) | `63.64%` | `100.00%` | `100.00%` | `100.00%` | `20.93` | `1.346` |

### 2. `development_fit_train`（435 对参考表）
- `AlwaysCell12Prior`（`all_pairs`）：`Top-1 = 32.41%` (`141/435`)、`Top-3 = 57.24%`、`Top-5 = 77.01%`、期望平面误差 `14.20 m`、`Soft-KL = 6.509 nats`；
- `TrainEmpiricalPrior`（`all_pairs`）：`Top-1 = 32.41%`、`Top-3 = 58.39%`、`Top-5 = 77.01%`、期望平面误差 `22.37 m`、`Soft-KL = 1.563 nats`。

**为何必须同时对照 `AlwaysCell12Prior` 与 `TrainEmpiricalPrior`**：`AlwaysCell12Prior` 通过重注中心格 `12` 将期望平面误差压到 **`12.51 m`**（且在 `dev_val` Top-3 达 **`68.97%`**），但过度自信导致软标签 KL 高达 `5.604`；而 `TrainEmpiricalPrior` 虽然把软标签 KL 降到 **`1.420 nats`**，期望误差却升高至 `21.42 m`。**真正依赖视觉线索的定位模型，必须在保持更低软标签 KL（`< 1.420`）的同时，稳定超越 `AlwaysCell12Prior` 的 `Top-1 > 42.24%`、`Top-3 > 68.97%` 与 `期望平面误差 < 12.51 m`**。

---

## 四、面向 `CV-051` 的单步视觉基线冻结协议与独立复核

[单步视觉实验冻结协议_v1.json](../DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/单步视觉实验冻结协议_v1.json) 已锁定后续 `CV-051` 实验边界：
1. **本地已核实骨干与执行复用**：复用 `models/clip_vit_b32_public_v1`（地面）与 `models/Sat2Cap`（航拍）及 [frozen_crossview_encoder_v2.py](../project/src/agents/frozen_crossview_encoder_v2.py) 单活体 CPU 槽位，对比 ① 零训练局部 token 余弦匹配基线 与 ② 仅在 `development_fit_train`（`435` 对）拟合的轻量 25 格空间读出头（高斯软目标 $\sigma = 5.0\text{ m}$ KL 损失）。
2. **双画幅等比标准化与禁捷径约束**：地面图严禁传入原始宽高，统一采用中心正方形裁切后双三次缩放（或等比 letterbox 一致预处理）并强制分列 `952×714` 与 `952×532` 诊断切面。
3. **四重硬门槛（`G1–G4`）与停止条件**：在 `development_validation` 上须同时满足 `G1`（Top-1 在 `all_pairs > 42.24%`、`opensfm_cell_agree > 40.91%`、`interior_margin_ge_2m > 40.00%` 均严胜）、`G2`（`Soft-KL < 1.420 nats` 且 `期望误差 < 12.51 m` 且 `Top-3 > 68.97%`）、`G3`（双画幅无崩塌）、`G4`（相对去地面图/打乱航拍/均值图像消融对照严格提升 $\ge 3$ 个百分点或更低期望误差）。若 `dev_val` 未过线，按失败结论诚实收束，**绝不消耗 `isolated_confirmation_blind`（28 对）**。
4. **独立复核与单元测试结论**：[audit_opencvl_wild_qa_v1.py](../project/src/eval/audit_opencvl_wild_qa_v1.py) 以无解码 PNG `IHDR` CRC-32、`bisect_right` 分箱及独立高斯软 KL 方程复核通过全部 12 项门禁（`max_moment_diff = 1.274e-10 <= 1e-6`、`max_discrepancy_diff = 0.0`、`max_baseline_metric_diff = 8.88e-16 <= 1e-9`、`auditor_image_decodes = 0`、`model_calls = 0`、`training = 0`、`formal_confirmation_consumed = 0`、`independent_personnel_review = false`）；联合单元测试 [test_opencvl_wild_qa_v1.py](../project/src/tests/test_opencvl_wild_qa_v1.py) 共 `29` 项全绿（`0.249 s`）。
