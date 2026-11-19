# `CV-051` OpenCVL 阶段二单步跨视角视觉定位基线与消融报告（`v1`）

- **实验编号 / 阶段**：`CV-051`（对应四阶段 OpenCVL 科研路线 **Stage 2：单步视觉基线 + Multi-Scale `CV-051b` 增强与消融**）
- **生成日期**：2026-11-14
- **核心结论**：`STOP_AT_STAGE2_NEGATIVE_AND_PROCEED_TO_STAGE4_CLOSURE`（`development_validation` `116` 对上 **G1–G4 四项门禁全未通过**；按预设停止条件诚实收束，**未触碰、未解码、未评测 `28` 对盲测确认集 `isolated_confirmation_blind`**，直接转入 Stage 4 平台与报告收官）
- **产物入口**：[单步视觉定位基线与消融_v1/README.md](../DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/README.md) · [四阶段科研路线与阶段二判定_v1.json](../DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/四阶段科研路线与阶段二判定_v1.json) · [另实现复核_v1.json](../DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/核验/另实现复核_v1.json)

---

## 一、四阶段 OpenCVL 科研路线与当前阶段状态

| 阶段 | 任务 | 当前状态 | 阶段通过门槛 / 停止条件 | 本轮实际判定 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | `CV-048 ~ CV-050` | `done`（已完成） | SE 分域隔离（`435 fit_train / 116 dev_val / 28 blind / 12 quar / 351 block`）+ `PL 419`、24 对 `fit_train` QA 全非空白、无视觉统计地板冻结 | 已通过并封存 |
| **Stage 2** | `CV-051`（含 `CV-051b`） | `completed_in_this_run`（本轮完成，负向） | 在 `development_validation`（`116` 对，Linköping `M02`）同时通过 **G1–G4** 四项门禁；任一未过即触发停止条件 | `result_passed = False`；`STOP_AT_STAGE2_NEGATIVE_AND_PROCEED_TO_STAGE4_CLOSURE` |
| **Stage 3** | `CV-052`（条件触发） | `not_triggered_by_stopping_rule`（**未触发**） | 仅当 Stage 2 通过全部 G1–G4 时，才允许对 `28` 对盲测确认集执行单次盲评及 `B=3/5` 外环导航对照 | **严格未触发**（`0` 解码、`0` 评测、`28` 对保持封存） |
| **Stage 4** | `CV-053 / FINAL-001` | `next_recommended`（建议下一步） | 将 GeoNav V5.0 已验证正向航拍探索结果与 Stage 2 跨视角诚实负向边界结论写入最终报告 | 下一步重点收官 |

---

## 二、样本边界、冻结双塔核算与等比多尺度特征设计

### 2.1 数据边界与 CPU 单槽核算

依据 [CV-049 评估器冻结台账](../DATA/processed_data/OpenCVL/野测洁净簇冻结清单_v1/核验/评估器专用冻结台账_禁止入模.json) 与 [CV-051 单步视觉冻结协议](../DATA/processed_data/OpenCVL/开发集分歧抽检与单步基线_v1/单步视觉实验冻结协议_v1.json)，本轮实验严格锁定以下运行边界（见 [opencvl_single_step_visual_v1.py](../project/src/agents/opencvl_single_step_visual_v1.py)）：

1. **严格白名单分域**：仅解码 `development_fit_train`（`435` 对）与 `development_validation`（`116` 对，与训练集地理间隔 `≥ 95.203 km`），合计 `551` 对；`isolated_confirmation_blind`（`28` 对）、`quarantined_cv045_vicinity`（`12` 对）、`blocked_known_overlap`（`351` 对）及 `PL`（`419` 对）均被 `_assert_stage2_entry` 拒绝。
2. **单槽位冻结前向与权重要素校验**：在 CPU 上强制 `max_live_encoders = 1` 顺序挂载 `models/clip_vit_b32_public_v1`（`ground` 分支）与 `models/Sat2Cap`（`aerial` 分支），运行前后权重张量 SHA-256 完全不变：
   - 地面编码器权重前/后张量摘要：`3c3635af58b60b6d6a1832459be180f8c10314c3c5680836c51aa92f39023c24`
   - 航拍编码器权重前/后张量摘要：`a83f15403728bed8c971ec591a7d088120ccc820e6d0fe248b4773b3d1206556`
   - 累计冻结前向共 `1,653` 次（`551` 地面等比图+三高度条带 + `551` 全幅 `625×625` 航拍 + `551` Center-9 `375×375` 航拍），首次全量提取耗时 `137.648 s`，复用 `.npz` 缓存重算耗时 `0.25 s` + 5 折交叉验证拟合耗时 `47.443 s`，全过程 `0` 主干微调训练、`0` 盲测解码。

### 2.2 预处理与多尺度 25 格特征构造

1. **双宽高比等比预处理与多尺度裁剪**：
   - 地面分支对 `952×714`（`fit_train 259 / dev_val 14`）与 `952×532`（`fit_train 176 / dev_val 102`）统一做中心等边方形裁剪（`min(W, H)`）+ 双三次重采样至 `224×224`，再额外切取上/中/下 3 条水平高度带；**禁止将原始图像宽高作为模型输入特征**。
   - 航拍分支保持 `625×625` 原生居中不偏移：全幅 `R1` 将 ViT patch token 按面积精确重叠矩阵映射至 `5×5` 网格（`aerial_full25_512`）；`CV-051b` 多尺度 `R2` 额外截取中心 `375×375`（恰好覆盖 `3×3` 中心 9 格 `CENTER9_CELLS`）按面积精确映射为 `aerial_center9_512`（`9×512`）。
2. **严格训练集拟合的子空间与特征维度**：
   - `PCA/CCA` 子空间（`CCA_RANK = 16`）、特征均值/标准差、`TrainEmpiricalPrior` 及空间线性读出头权重均严格仅在 `development_fit_train`（`435` 对）上以 `gaussian_soft_target_distribution(..., sigma=5.0m)` 拟合，优化器为 `scipy.optimize.minimize(..., method="L-BFGS-B")`。
   - 单元格特征维度：`R1` 每格 `D = 27`（10 维基础余弦/对比/局部差分统计 + 16 维全局中心化 CCA 逐元素乘积 + 1 维 CCA 内积）；`R2` 在 `R1` 基础上对 `CENTER9_CELLS` 拼接 22 维 Center-9 局部多尺度块，形成每格 `D = 49` 维特征。
   - **5 折 `fit_train` 交叉验证选定超参**：
     - `M1_ZeroShot_FullTile_Cosine`（`R1`）与 `M1b_ZeroShot_Center9Crop_Cosine`（`R2`）：`temperature = 10.0`（零参数余弦打分）；
     - `M2a_SpatialHead_PureVisual_NoPrior`（`R1`，均匀先验 `1/25`）：`lam = 0.05, beta = 1.0, cv_objective = 33.657`；
     - `M2b_SpatialHead_PriorCalibrated_Full`（`R1`，叠加 `TrainEmpiricalPrior` 对数先验与 `TOP_LOCAL_INDICES_R1` 中心 9 格局部头）：`lam = 5.0, beta = 0.5, cv_objective = 21.405`；
     - `M2c_MultiScale_PriorCalibrated_CV051b`（`R2`，叠加 `TrainEmpiricalPrior` 对数先验与 `TOP_LOCAL_INDICES_R2` 多尺度中心 9 格局部头）：`lam = 20.0, beta = 0.5, cv_objective = 21.150`（**5 折交叉验证最优，选为主评估候选模型**）。

---

## 三、`development_validation`（116 对）与 `fit_train`（435 对）全量结果对照

### 3.1 `all_pairs` 全量主表（`dev_val=116` vs `fit_train=435`）

| 模型 / 基线 / 对照组 | `dev_val` Top-1 | `dev_val` Top-3 | `dev_val` Top-5 | `dev_val` Cheb-1 | `dev_val` 期望误差 (m) | `dev_val` Soft-KL (nats) | `fit_train` Top-1 | `fit_train` 期望误差 (m) | `fit_train` Soft-KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Uniform25Prior` | `0.00%` | `0.00%` | `0.00%` | `3.45%` | `39.26` | `2.761` | `0.00%` | `39.71` | `2.774` |
| `UniformCenter9Prior` | `42.24%` | `68.97%` | `89.66%` | `100.00%` | `24.64` | `1.740` | `32.41%` | `25.38` | `1.753` |
| `AlwaysCell12Prior`（Top-1/误差地板） | **`42.24%`** | **`68.97%`** | **`89.66%`** | **`100.00%`** | **`12.51`** | `5.604` | `32.41%` | **`14.20`** | `6.509` |
| `TrainEmpiricalPrior`（KL 地板） | **`42.24%`** | `67.24%` | `89.66%` | `100.00%` | `21.42` | **`1.420`** | `32.41%` | `22.37` | `1.563` |
| `M1_ZeroShot_FullTile_Cosine` | `2.59%` | `8.62%` | `12.07%` | `23.28%` | `40.52` | `3.105` | `1.15%` | `41.26` | `3.130` |
| `M1b_ZeroShot_Center9Crop_Cosine` | `5.17%` | `19.83%` | `36.21%` | `62.93%` | `25.56` | `1.990` | `5.29%` | `26.27` | `1.980` |
| `M2a_SpatialHead_PureVisual_NoPrior` | `4.31%` | `17.24%` | `25.86%` | `41.38%` | `37.45` | `3.292` | `42.99%` | `26.02` | `1.510` |
| `M2b_SpatialHead_PriorCalibrated_Full` | `24.14%` | `56.90%` | `79.31%` | `87.93%` | `22.27` | `1.705` | `60.00%` | `17.03` | `0.922` |
| **`M2c_MultiScale_PriorCalibrated_CV051b`** | **`24.14%`** | **`51.72%`** | **`73.28%`** | **`89.66%`** | **`22.18`** | **`1.730`** | **`68.74%`** | **`16.07`** | **`0.810`** |
| `Control_GroundZeroed`（基于 `M2c`） | `14.66%` | `39.66%` | `62.07%` | `81.90%` | `23.94` | `2.275` | `16.78%` | `24.17` | `2.507` |
| `Control_AerialShuffled`（基于 `M2c`） | `22.41%` | `55.17%` | `77.59%` | `89.66%` | `22.81` | `1.791` | `19.77%` | `22.96` | `1.966` |
| `Control_MeanVisual`（基于 `M2c`） | **`42.24%`** | `62.93%` | `81.03%` | **`100.00%`** | **`21.78`** | **`1.524`** | `32.41%` | `22.56` | `1.644` |

### 3.2 `development_validation`（`116` 对）六分层细粒度明细

各分层样本数：`all_pairs` (`116`)、`opensfm_cell_agree` (`66`)、`interior_margin_ge_2m` (`70`)、`large_discrepancy_gt_10m` (`33`)、`ground_size_952x714` (`14`)、`ground_size_952x532` (`102`)。

| 模型 / 基线 / 对照组 | `all_pairs` (116)<br>Top-1 / 误差 / KL | `opensfm_agree` (66)<br>Top-1 / 误差 / KL | `interior_≥2m` (70)<br>Top-1 / 误差 / KL | `large_disc_>10m` (33)<br>Top-1 / 误差 / KL | `952×714` (14)<br>Top-1 / 误差 / KL | `952×532` (102)<br>Top-1 / 误差 / KL |
|---|---:|---:|---:|---:|---:|---:|
| `AlwaysCell12Prior` | `42.24% / 12.51 / 5.604` | `40.91% / 12.62 / 5.652` | `40.00% / 12.71 / 5.882` | `45.45% / 11.81 / 5.342` | `50.00% / 12.20 / 5.222` | `41.18% / 12.55 / 5.657` |
| `TrainEmpiricalPrior` | `42.24% / 21.42 / 1.420` | `40.91% / 21.51 / 1.481` | `40.00% / 21.60 / 1.655` | `45.45% / 20.93 / 1.346` | `50.00% / 21.35 / 1.585` | `41.18% / 21.43 / 1.397` |
| `M1_ZeroShot_FullTile_Cosine` | `2.59% / 40.52 / 3.105` | `3.03% / 40.60 / 3.139` | `0.00% / 40.58 / 3.376` | `3.03% / 39.92 / 3.061` | `0.00% / 40.66 / 3.219` | `2.94% / 40.50 / 3.089` |
| `M1b_ZeroShot_Center9Crop_Cosine` | `5.17% / 25.56 / 1.990` | `4.55% / 25.57 / 2.014` | `2.86% / 25.84 / 2.257` | `6.06% / 25.04 / 1.957` | `7.14% / 25.58 / 2.155` | `4.90% / 25.56 / 1.967` |
| `M2a_SpatialHead_PureVisual_NoPrior` | `4.31% / 37.45 / 3.292` | `4.55% / 37.10 / 3.279` | `7.14% / 37.55 / 3.609` | `0.00% / 38.11 / 3.293` | `7.14% / 35.13 / 3.227` | `3.92% / 37.77 / 3.301` |
| `M2b_SpatialHead_PriorCalibrated_Full` | `24.14% / 22.27 / 1.705` | `28.79% / 22.02 / 1.728` | `25.71% / 22.56 / 1.979` | `27.27% / 21.96 / 1.565` | `42.86% / 20.89 / 1.715` | `21.57% / 22.46 / 1.703` |
| `M2c_MultiScale_PriorCalibrated_CV051b` | `24.14% / 22.18 / 1.730` | `21.21% / 21.90 / 1.764` | `22.86% / 22.48 / 2.004` | `30.30% / 21.89 / 1.598` | `42.86% / 21.11 / 1.835` | `21.57% / 22.33 / 1.715` |
| `Control_GroundZeroed` | `14.66% / 23.94 / 2.275` | `15.15% / 23.79 / 2.434` | `15.71% / 24.02 / 2.576` | `18.18% / 22.87 / 1.773` | `14.29% / 23.01 / 2.243` | `14.71% / 24.06 / 2.279` |
| `Control_AerialShuffled` | `22.41% / 22.81 / 1.791` | `22.73% / 22.63 / 1.886` | `24.29% / 22.82 / 2.057` | `27.27% / 22.61 / 1.767` | `21.43% / 21.74 / 1.797` | `22.55% / 22.96 / 1.791` |
| `Control_MeanVisual` | `42.24% / 21.78 / 1.524` | `40.91% / 21.84 / 1.583` | `40.00% / 21.95 / 1.761` | `45.45% / 21.47 / 1.497` | `50.00% / 21.74 / 1.708` | `41.18% / 21.79 / 1.498` |

---

## 四、`G1–G4` 门禁逐条判定与机理诊断

以 5 折交叉验证选定的 `M2c_MultiScale_PriorCalibrated_CV051b` 为主候选模型（其余 4 个视觉模型 `M1 / M1b / M2a / M2b` 同样全部 `stage2_passed = False`）：

1. **`G1_top1_beats_AlwaysCell12 = False`**：
   要求候选模型在 `dev_val` 的 `all_pairs`、`opensfm_cell_agree`、`interior_margin_ge_2m` 三个分层上 Top-1 均严格高于 `AlwaysCell12Prior`。实测 `M2c` 为 `24.14% / 21.21% / 22.86%`，全面低于 `AlwaysCell12Prior`（`42.24% / 40.91% / 40.00%`）。
2. **`G2_distributional_and_error_advantage = False`**：
   要求同时满足 `soft_target_kl_nats < TrainEmpiricalPrior (1.420)`、`expected_planar_error_m < AlwaysCell12Prior (12.51 m)` 且 `top3_acc > AlwaysCell12Prior (68.97%)`。实测 `M2c` 为 `Soft-KL 1.730 nats`、`期望误差 22.18 m`、`Top-3 51.72%`，三项指标均未过线。
3. **`G3_aspect_ratio_no_collapse = False`**：
   要求 `952×714` 与 `952×532` 两项纵横比分层同时满足 `top1_acc >= AlwaysCell12Prior - 0.05` 且 `soft_target_kl_nats <= TrainEmpiricalPrior + 0.05`。实测 `M2c` 在 `952×714` 上 `Top-1 = 42.86%`（低 `AlwaysCell12` `7.14 个百分点`）、`KL = 1.835`（超 `TrainEmpirical + 0.05` `0.200`）；在占比 `87.93%`（`102/116`）的 `952×532` 上更降至 `21.57%`（低 `AlwaysCell12` `19.61 个百分点`、`KL 1.715`）。
4. **`G4_visual_ablation_gain = False`（最关键机理证据）**：
   虽然 `M2c` 相对退化破坏对照组 `Control_GroundZeroed`（Top-1 `+9.48%`、误差降 `1.76 m`）和 `Control_AerialShuffled`（Top-1 `+1.72%`、误差降 `0.63 m`）呈小幅正增益，但在关键对照 `Control_MeanVisual`（将验证集视觉特征冻结替换为训练集视觉均值向量）上，`Control_MeanVisual` 取得 `Top-1 = 42.24%` / `期望误差 = 21.78 m` / `Soft-KL = 1.524`，反而领先实时视觉推理的 `M2c` **高出 `+18.10 个百分点` Top-1 且误差更低 `0.40 m`**（`top1_gain = -0.1810`、`planar_error_reduction_m = -0.3964 m`、`passed = False`）。

**核心科学机理诊断（为什么在 `fit_train` 有 `68.74%`，跨城即坍塌？）**：
1. **零样本跨视角表征缺位**：`M1`（`2.59%`）与 `M1b`（`5.17%`）接近随机均匀猜测，说明通用图文冻结 `CLIP-ViT-B/32` 地面特征与 `Sat2Cap` 航拍 patch 特征在 `20m` 细粒度子格尺度上并不具备开箱即用的跨视角几何对齐度。
2. **城内拟合与跨城泛化断层**：在 `fit_train`（`435` 对）内部，无先验的 `M2a` 可拟合至 `42.99%`、`M2b` 达 `60.00%`、`M2c` 达 `68.74%`（`Soft-KL 0.810`），说明 L-BFGS-B 优化充分收敛；但一旦迁移到距训练集至少 `95.203 km` 的未见城市 Linköping `M02`（`dev_val=116`），纯视觉 `M2a` 暴跌至 `4.31%`（`37.45 m`），`M2b/M2c` 跌至 `24.14%`。结合 `Control_MeanVisual` 回弹至 `42.24%` 的对照结果，可严格证伪“冻结一般双塔在跨城 20m 子格任务上的可辨识性”——训练集上学到的视觉判别主要是城内场景分布与统计关联记忆，在未见城市的 `dev_val` 上视觉扰动反而劣化了中心先验。

---

## 五、独立复核、全套单测与存储合规核验

1. **零 `torch` 独立复核通过**：运行 [audit_opencvl_single_step_visual_v1.py](../project/src/eval/audit_opencvl_single_step_visual_v1.py) 生成 [另实现复核_v1.json](../DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/核验/另实现复核_v1.json)，`10/10` 项核验全部通过（`zero_torch_import = True`、`upstream_cv049_cv050_hashes_match = True`、`encoder_asset_hashes_match = True`、`cv051_artifact_and_cache_hashes_match = True`、`max_live_encoders = 1`、区域边积分重算 `max_overlap_diff = 3.331e-16`、`max_feature_norm_err = 2.525e-08`、6 分层指标最大差 `8.882e-16`、`auditor_image_decodes = 0`、`blind_decodes = 0`、`blind_set_evaluated = False`）。注：由同一 Antigravity 会话编写与运行，`independent_personnel_review = False`。
2. **全套 `35` 项单元测试通过**：覆盖 `test_opencvl_single_step_visual_v1`（`6` 项）、`test_opencvl_wild_qa_v1`（`6` 项）、`test_opencvl_wild_intake_v1`（`7` 项）与 `test_capturepoint_intake_v1`（`16` 项），全部 `35` 项通过。
3. **产物保留级别与空间声明（实测总计 `45.774 MiB` / `47,997,597 B`，声明上限调准为 `≤ 48 MiB`）**：
   - `开发集冻结双塔特征缓存_v1.npz`（`41,896,026 B` / `39.955 MiB`）：本地可再生特征缓存，已由 `.gitignore` 的 `*.npz` 规则自动排除出 Git 仓库。
   - `单步视觉定位模型与消融结果_v1.json`（`6,075,615 B`）、`四阶段科研路线与阶段二判定_v1.json`（`16,751 B`）、`核验/源码与输入绑定_v1.json`（`5,950 B`）、`README.md`（`2,130 B`）、`核验/另实现复核_v1.json`（`1,125 B`）：研究复核必需原件（共 `6,101,571 B`）。

---

## 六、结论与下一步决策（Stage 4 收尾）

1. **科研边界定论**：严格遵守预设停止条件，不在看到负向成绩后改动软标签 `sigma`、网格切分、门禁阈值或调参试探 `28` 对 `isolated_confirmation_blind`（保留完整零泄漏科研证据）。
2. **Stage 4 汇入方向（`CV-053 / FINAL-001`）**：将 GeoNav V5.0 已通过协议的积极航拍视觉导航成果，与 OpenCVL 野测线（`CV-048 ~ CV-051`）形成的跨视角适用边界实证完整写入总结报告与答辩口径：未经过跨视角几何微调或空间监督学习的冻结通用图文/航拍表征，难以承担未知城市 20m 子格跨视角定位，后续需针对性跨视角空间对齐训练而非盲目堆叠后验头。
