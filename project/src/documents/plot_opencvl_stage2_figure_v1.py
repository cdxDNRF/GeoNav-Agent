"""Plot CV-051 OpenCVL Stage-2 single-step cross-view localization and ablation results directly from verified JSON."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SUMMARY_JSON = ROOT / "DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/四阶段科研路线与阶段二判定_v1.json"
DETAIL_JSON = ROOT / "DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/单步视觉定位模型与消融结果_v1.json"
OUT_PNG = ROOT / "绘图/数据结果图/08_OpenCVL阶段二单步视觉定位与消融实证对比.png"
OUT_PDF = ROOT / "绘图/数据结果图/08_OpenCVL阶段二单步视觉定位与消融实证对比.pdf"


def configure_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


def main() -> None:
    configure_fonts()
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8-sig"))
    detail = json.loads(DETAIL_JSON.read_text(encoding="utf-8-sig"))
    dev_all = summary["summary_comparison_all_pairs"]["development_validation"]
    fit_all = summary["summary_comparison_all_pairs"]["development_fit_train"]
    dv_strata = detail["evaluations"]["development_validation"]["M2c_MultiScale_PriorCalibrated_CV051b"]

    ordered = [
        ("AlwaysCell12Prior", "AlwaysCell12\n(中心常数先验)", "#64748B"),
        ("TrainEmpiricalPrior", "TrainEmpirical\n(拟合经验先验)", "#475569"),
        ("M1_ZeroShot_FullTile_Cosine", "M1 零样本\n全图余弦", "#94A3B8"),
        ("M1b_ZeroShot_Center9Crop_Cosine", "M1b 零样本\nCenter9余弦", "#38BDF8"),
        ("M2a_SpatialHead_PureVisual_NoPrior", "M2a 纯视觉\n无先验校准", "#F59E0B"),
        ("M2b_SpatialHead_PriorCalibrated_Full", "M2b 全图\n先验校准", "#3B82F6"),
        ("M2c_MultiScale_PriorCalibrated_CV051b", "M2c 多尺度\n先验校准(主选)", "#1D4ED8"),
        ("Control_GroundZeroed", "消融: 地面置零\nGroundZeroed", "#14B8A6"),
        ("Control_AerialShuffled", "消融: 航拍打乱\nAerialShuffled", "#0D9488"),
        ("Control_MeanVisual", "消融: 均值视觉\nMeanVisual", "#EF4444"),
    ]

    labels = [x[1] for x in ordered]
    colors = [x[2] for x in ordered]
    top1_dev = [100.0 * dev_all[k]["top1_acc"] for k, _, _ in ordered]
    top1_fit = [100.0 * fit_all[k]["top1_acc"] for k, _, _ in ordered]
    err_dev = [dev_all[k]["expected_planar_error_m"] for k, _, _ in ordered]
    kl_dev = [dev_all[k]["soft_target_kl_nats"] for k, _, _ in ordered]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.5), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.patch.set_facecolor("white")

    x = np.arange(len(ordered))
    w = 0.38
    bars_dev = ax1.bar(x - w / 2, top1_dev, width=w, color=colors, edgecolor="#1E293B", linewidth=0.5, label="跨城验证集 dev_val (N=116, ≥95.2 km)")
    bars_fit = ax1.bar(x + w / 2, top1_fit, width=w, color="#CBD5E1", edgecolor="#475569", linewidth=0.5, hatch="//", alpha=0.88, label="城内拟合集 fit_train (N=435)")

    g1_floor = 100.0 * dev_all["AlwaysCell12Prior"]["top1_acc"]
    ax1.axhline(g1_floor, color="#DC2626", linestyle="--", linewidth=1.3, label=f"G1 门槛地板 AlwaysCell12 ({g1_floor:.2f}%)")

    for i, (bd, bf) in enumerate(zip(bars_dev, bars_fit)):
        hd = bd.get_height()
        hf = bf.get_height()
        ax1.text(bd.get_x() + bd.get_width() / 2, hd + 1.1, f"{hd:.1f}%", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color="#0F172A")
        if hf >= 10.0:
            ax1.text(bf.get_x() + bf.get_width() / 2, hf + 1.1, f"{hf:.1f}%", ha="center", va="bottom", fontsize=6.2, color="#475569")

    # Inset text box summarizing G1-G4 & aspect-ratio slice for M2c
    top1_714 = 100.0 * dv_strata["ground_size_952x714"]["top1_acc"]
    top1_532 = 100.0 * dv_strata["ground_size_952x532"]["top1_acc"]
    note_txt = (
        f"M2c 画幅分层 (dev_val):\n"
        f"• 952×714 (n=14): {top1_714:.2f}%\n"
        f"• 952×532 (n=102): {top1_532:.2f}%\n"
        f"结论: G1–G4 全否，止损转入阶段四"
    )
    ax1.text(
        0.02,
        0.96,
        note_txt,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=7.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#94A3B8", "linewidth": 0.8},
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.8, rotation=24, ha="right")
    ax1.set_ylim(0, 92)
    ax1.set_ylabel("25 格 Top-1 准确率 (% ↑)", fontsize=9, fontweight="bold")
    ax1.set_title("(a) 跨城验证 vs 城内拟合 Top-1 准确率与消融对比", fontsize=10, fontweight="bold", pad=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=6.6, frameon=True, facecolor="white", framealpha=0.95)

    # Panel (b): Expected planar error & Soft-target KL divergence on dev_val
    ax2_kl = ax2.twinx()
    b_err = ax2.bar(x - w / 2, err_dev, width=w, color="#2563EB", alpha=0.88, edgecolor="#1E293B", linewidth=0.5, label="期望平面误差 (m ↓, 左轴)")
    b_kl = ax2_kl.bar(x + w / 2, kl_dev, width=w, color="#F59E0B", alpha=0.88, edgecolor="#78350F", linewidth=0.5, label="软标签 KL 散度 (nats ↓, 右轴)")

    err_emp = dev_all["TrainEmpiricalPrior"]["expected_planar_error_m"]
    kl_emp = dev_all["TrainEmpiricalPrior"]["soft_target_kl_nats"]
    ax2.axhline(err_emp, color="#1D4ED8", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2_kl.axhline(kl_emp, color="#B45309", linestyle=":", linewidth=1.1, alpha=0.8)

    for be, bk in zip(b_err, b_kl):
        he = be.get_height()
        hk = bk.get_height()
        ax2.text(be.get_x() + be.get_width() / 2, he + 0.6, f"{he:.1f}", ha="center", va="bottom", fontsize=6.2, color="#1E3A8A")
        ax2_kl.text(bk.get_x() + bk.get_width() / 2, hk + 0.08, f"{hk:.2f}", ha="center", va="bottom", fontsize=6.2, color="#92400E")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.8, rotation=24, ha="right")
    ax2.set_ylim(0, 48)
    ax2_kl.set_ylim(0, 6.6)
    ax2.set_ylabel("期望平面误差 (m ↓)", fontsize=9, fontweight="bold", color="#1D4ED8")
    ax2_kl.set_ylabel("软标签分布 KL 散度 (nats ↓)", fontsize=9, fontweight="bold", color="#B45309")
    ax2.set_title("(b) dev_val (N=116) 误差与分布校准分解 (虚线为经验先验基线)", fontsize=10, fontweight="bold", pad=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_kl.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.8, frameon=True, facecolor="white", framealpha=0.92)

    plt.tight_layout(pad=1.0)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"png": str(OUT_PNG), "pdf": str(OUT_PDF)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
