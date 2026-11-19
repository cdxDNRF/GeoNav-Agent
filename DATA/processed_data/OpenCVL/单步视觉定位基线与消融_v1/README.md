# OpenCVL 阶段二单步视觉定位基线与消融结果（`CV-051`）

- **阶段二总判定**：`STOP_AT_STAGE2_NEGATIVE_AND_PROCEED_TO_STAGE4_CLOSURE`（未全部通过 G1–G4，按停止条件定性收束（严禁触碰 28 对盲测集））
- **主评估候选模型（5 折 `fit_train` CV 选定）**：`M2c_MultiScale_PriorCalibrated_CV051b`
- **盲测保护**：`blind_set_evaluated = False`（`28` 对盲测确认集 0 解码/0 评测）

## `development_validation`（116 对）核心对照总表（`all_pairs`）

| 模型 / 对照组 | Top-1 | Top-3 | Top-5 | Cheb-1 | 期望误差 (m) | Soft-KL ($\sigma=5\text{m}$) | `952×714` Top-1 | `952×532` Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `AlwaysCell12Prior` | `42.24%` | `68.97%` | `89.66%` | `100.00%` | `12.51` | `5.604` | `50.00%` | `41.18%` |
| `TrainEmpiricalPrior` | `42.24%` | `67.24%` | `89.66%` | `100.00%` | `21.42` | `1.420` | `50.00%` | `41.18%` |
| `M1_ZeroShot_FullTile_Cosine` | `2.59%` | `8.62%` | `12.07%` | `23.28%` | `40.52` | `3.105` | `0.00%` | `2.94%` |
| `M1b_ZeroShot_Center9Crop_Cosine` | `5.17%` | `19.83%` | `36.21%` | `62.93%` | `25.56` | `1.990` | `7.14%` | `4.90%` |
| `M2a_SpatialHead_PureVisual_NoPrior` | `4.31%` | `17.24%` | `25.86%` | `41.38%` | `37.45` | `3.292` | `7.14%` | `3.92%` |
| `M2b_SpatialHead_PriorCalibrated_Full` | `24.14%` | `56.90%` | `79.31%` | `87.93%` | `22.27` | `1.705` | `42.86%` | `21.57%` |
| `M2c_MultiScale_PriorCalibrated_CV051b` | `24.14%` | `51.72%` | `73.28%` | `89.66%` | `22.18` | `1.730` | `42.86%` | `21.57%` |
| `Control_GroundZeroed` | `14.66%` | `39.66%` | `62.07%` | `81.90%` | `23.94` | `2.275` | `14.29%` | `14.71%` |
| `Control_AerialShuffled` | `22.41%` | `55.17%` | `77.59%` | `89.66%` | `22.81` | `1.791` | `21.43%` | `22.55%` |
| `Control_MeanVisual` | `42.24%` | `62.93%` | `81.03%` | `100.00%` | `21.78` | `1.524` | `50.00%` | `41.18%` |

## 门禁明细（`G1–G4`）
- `G1_top1_beats_AlwaysCell12`: `False`
- `G2_distributional_and_error_advantage`: `False`
- `G3_aspect_ratio_no_collapse`: `False`
- `G4_visual_ablation_gain`: `False`
