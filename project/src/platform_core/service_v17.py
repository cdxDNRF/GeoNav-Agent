"""5.0 unified all-in-one SPA workbench service aggregating all navigation, sandbox, arena, ground, and evidence engines."""
from __future__ import annotations
import json
from pathlib import Path
from .service_v16 import Workbench as Previous
from project.src.webapp.binding_v3 import source_closure


BENCHMARK_OVERVIEW = [
    {
        "protocol_id": "masa-s2",
        "stage": "S2 航拍确认",
        "domain": "Masa 5×5 (B=10)",
        "tasks": "4图 / 100题×3",
        "method": "M0 (Target-Conditioned Edge)",
        "sr": "89.33%",
        "baseline_sr": "73.00% (禁用目标)",
        "delta": "+16.33pp",
        "sg": "0.320 格",
        "note": "较 Frontier (53.00%) 净增 +36.33pp；三权重 (Seed 0/1/2) 全过门槛",
        "name": "Masa S2 标准航拍确认",
        "grid": "5×5",
        "budget": 10,
        "episodes": 300,
        "m0_sr": 0.8933,
        "m0_sg_cells": 0.320,
        "explorer_only_sr": 0.7300,
        "cue_mean_sr": 0.6900,
        "cue_wrong_sr": 0.6967,
        "frontier_sr": 0.5300,
        "highlight": "较禁用目标 +16.33pp，较 Frontier +36.33pp",
        "detail": "在 Masa S2 验证集（4源 100 题 × 3 随机种子）上验证 M0 双分支架构：真实目标 SR 达 89.33%（SG 0.320 格），显著优于禁用目标、均值目标与错误目标对照，证实定位增益来自真实目标拼缝线索而非位置记忆。",
    },
    {
        "protocol_id": "masa-s3",
        "stage": "S3 独立源泛化",
        "domain": "Masa 5×5 (10新源)",
        "tasks": "10图 / 250题×3",
        "method": "M0 (0.93M 轻量网络)",
        "sr": "86.80%",
        "baseline_sr": "35.60% (GeoExplorer)",
        "delta": "+51.20pp",
        "sg": "0.300 格",
        "note": "同题显著优于有限预算 GeoExplorer 复现 (35.60%)",
        "name": "Masa S3 独立源图确认",
        "grid": "5×5",
        "budget": 10,
        "episodes": 750,
        "m0_sr": 0.8680,
        "m0_sg_cells": 0.300,
        "explorer_only_sr": 0.6420,
        "cue_mean_sr": 0.6140,
        "cue_wrong_sr": 0.6120,
        "frontier_sr": 0.5420,
        "highlight": "10 新源图确认通过；同题领先 GeoExplorer +51.20pp",
        "detail": "在 10 张未见独立测试源图（250 题 × 3 种子）及 6,000 条共同评测轨迹上，0.93M 参数的 M0 取得 86.80% SR（SG 0.300 格），而有限预算复现的 GeoExplorer 好奇心基线仅 35.60%（目标控制未过）。",
    },
    {
        "protocol_id": "masa-s4",
        "stage": "S4 高密度大网格",
        "domain": "Masa 10×10 (B=20)",
        "tasks": "20图 / 500题×3",
        "method": "M0 (Target-Conditioned Edge)",
        "sr": "77.13%",
        "baseline_sr": "53.80% (禁用目标)",
        "delta": "+23.33pp",
        "sg": "1.201 格",
        "note": "候选空间扩大 4 倍 (100格)，仍保持 +23.33pp 真实目标增益",
        "name": "Masa S4 高密 10×10 网格",
        "grid": "10×10",
        "budget": 20,
        "episodes": 1500,
        "m0_sr": 0.7713,
        "m0_sg_cells": 1.201,
        "explorer_only_sr": 0.5380,
        "cue_mean_sr": 0.5120,
        "cue_wrong_sr": 0.5087,
        "frontier_sr": 0.3460,
        "highlight": "100 格高密切图，相对纯探索净增 +23.33pp",
        "detail": "切图密度提升至 10×10（100 候选格，预算 B=20 步），M0 依靠 20 维拼缝边缘门控将成功率从纯探索的 53.80% 提升至 77.13%（+23.33 个百分点），SG 降至 1.201 格。",
    },
    {
        "protocol_id": "swissview-5x5",
        "stage": "SwissView 跨地域",
        "domain": "瑞士航拍 5×5 (B=10)",
        "tasks": "20图 / 500题×3",
        "method": "M0 冻结零样本迁移",
        "sr": "86.80%",
        "baseline_sr": "70.40% (禁用目标)",
        "delta": "+16.40pp",
        "sg": "0.381 格",
        "note": "欧洲高分航拍零样本测试，验证 20 维边缘线索跨地域稳健性",
        "name": "SwissView 跨地域迁移",
        "grid": "5×5",
        "budget": 10,
        "episodes": 1500,
        "m0_sr": 0.8680,
        "m0_sg_cells": 0.381,
        "explorer_only_sr": 0.7040,
        "cue_mean_sr": 0.6880,
        "cue_wrong_sr": 0.6820,
        "frontier_sr": 0.5600,
        "highlight": "冻结权重零样本迁移欧洲航拍，SR 86.80%",
        "detail": "不重训任何参数，直接将 Masa 训练的冻结 M0 权重迁移至瑞士高分航拍（20 新源 500 题 × 3 种子），取得 86.80% SR 与 0.381 格 SG，验证了跨地域同模态泛化能力。",
    },
    {
        "protocol_id": "massgis-9km2",
        "stage": "9km² 连续大区域",
        "domain": "MasaRoads 10×10 (300m/格)",
        "tasks": "10区 / 750题×3",
        "method": "Coverage3Radial (H=3+径向保护)",
        "sr": "68.04%",
        "baseline_sr": "58.84% (原 M0)",
        "delta": "+9.20pp",
        "sg": "648.53 m",
        "note": "无需重训网络，恢复 295 题 / 损伤 88 题 (净增 207 题)，10/10 区域全正增益",
        "name": "9km² 连续大区域径向保护",
        "grid": "10×10 (300m/格)",
        "budget": 20,
        "episodes": 2250,
        "m0_sr": 0.6804,
        "m0_sg_cells": 2.162,
        "explorer_only_sr": 0.5884,
        "cue_mean_sr": None,
        "cue_wrong_sr": None,
        "frontier_sr": 0.4120,
        "highlight": "H=3 前瞻+径向保护较原 M0 净增 +9.20pp (10/10区全正)",
        "detail": "在 10 个空间隔离的 9km² 连续大区域（300m/格，2,250 条轨迹）上，无需重训的 Coverage3Radial 将同题 SR 从原 M0 的 58.84% 提升至 68.04%（净恢复 207 题），平均终局距离降至 648.53 米；另在异源 MassGIS 10×10 (54.62%) 如实保留泛化边界。",
    },
    {
        "protocol_id": "openvl-capturepoint-192",
        "stage": "地面采集点探索",
        "domain": "OpenCVL 三城 5×5",
        "tasks": "192题 (开发队列)",
        "method": "Frontier / 人工同台对照",
        "sr": "71.88%",
        "baseline_sr": "50.00% (无目标随机)",
        "delta": "探索基线",
        "sg": "0.92 格",
        "note": "192 道跨视角地面拍摄点队列已接入统一探索台与显微镜",
        "name": "OpenCVL 地面跨视角探索",
        "grid": "5×5",
        "budget": 10,
        "episodes": 192,
        "m0_sr": 0.7188,
        "m0_sg_cells": 0.920,
        "explorer_only_sr": 0.5000,
        "cue_mean_sr": None,
        "cue_wrong_sr": None,
        "frontier_sr": 0.7188,
        "highlight": "192 道三城实景配对队列，Frontier 基线 71.88%",
        "detail": "完成 CV001–CV047 六阶段跨视角排障，查明‘可见地标格 ≠ 相机拍摄点格’与连续回归高方差瓶颈；建立 192 道合法三城市地面拍摄点队列，无目标 Frontier 达 71.88%，作为下一阶段跨视角弱监督门控的严格基准。",
    },
]


class Workbench(Previous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sources.update(source_closure(self.root, ['project/src/platform_core/service_v17.py']))

    def capabilities(self):
        v = super().capabilities()
        v.update(
            kernel='geonav-core-v17',
            unified_v5_spa=True,
            interactive_curated_aerial=True,
            volatile_multi_grid_upload=True,
            online_cloud_api_config=True,
            workspaces=['overview', 'navigator', 'sandbox', 'arena', 'ground', 'evidence'],
        )
        return v

    def _curated_pool_counts(self):
        curated_path = self.root / 'project/src/webapp/lite_curated_tasks.json'
        if not curated_path.is_file():
            return 4, 100
        try:
            datasets = json.loads(curated_path.read_text(encoding='utf-8')).get('datasets', {})
            img_set = set()
            total_tasks = 0
            for bank_id, rows in datasets.items():
                for r in rows:
                    img_set.add((bank_id, r.get('area')))
                    total_tasks += 1
            return len(img_set) or 4, total_tasks or 100
        except Exception:
            return 4, 100

    def v5_overview(self, owner, perception_reader=None, spatial_reader=None, interactive_stats=None):
        ctrl_cat = self.controllers.catalog(owner)
        batch_cat = self.controller_batches.catalog(owner)
        api_cat = self.api_comparisons.catalog(owner)
        scene_cat = self.scenes.catalog(owner)
        gw_cat = self.ground_walk.catalog(owner)
        gl_cat = self.ground_live.catalog(owner)

        perception_ok = False
        perception_cases = 0
        if perception_reader is not None:
            try:
                p_cat = perception_reader.catalog()
                perception_ok = True
                perception_cases = len(p_cat.get('cases', []))
            except Exception:
                perception_ok = False

        spatial_ok = False
        spatial_cases = 0
        if spatial_reader is not None:
            try:
                s_cat = spatial_reader.catalog()
                spatial_ok = True
                spatial_cases = len(s_cat.get('cases', []))
            except Exception:
                spatial_ok = False

        tasks = ctrl_cat.get("tasks", [])
        aerial_count = sum(1 for t in tasks if t.get("dataset_id") == "masa-s2")
        cp_count = sum(1 for t in tasks if t.get("dataset_id") != "masa-s2")

        if isinstance(interactive_stats, dict) and interactive_stats.get("aerial_tasks"):
            curated_imgs = int(interactive_stats.get("aerial_images", 192))
            curated_aerial = int(interactive_stats.get("aerial_tasks", 960))
        else:
            curated_imgs, curated_aerial = self._curated_pool_counts()

        api_ready = self.config is not None
        api_model = getattr(self.config, "model", None) if api_ready else "离线/本地模式"

        assets = [
            {
                "label": "统一探索控制台 (多源精选库 + 标准审计库)",
                "detail": f"{curated_imgs} 张独立航拍底图 ({curated_aerial} 题) + {cp_count} 地面题 | 封存审计 {aerial_count} 题 · 已存终局 {len(ctrl_cat.get('saved_runs', []))} 条",
                "ready": len(tasks) > 0,
            },
            {
                "label": "自由场景沙箱 & 多尺度内存切片 (3×3 ~ 10×10)",
                "detail": f"已存自定义场景 {len(scene_cat.get('scenes', []))} 个 · 运行 {len(scene_cat.get('saved_runs', []))} 条 | 支持本地图片纯内存动态切片",
                "ready": True,
            },
            {
                "label": "固定10题云 API 打擂 & 同题小队列批测",
                "detail": f"十题三方对照 {len(api_cat.get('comparisons', []))} 组 | 成对批测队列 {len(batch_cat.get('batches', []))} 组 (含请求预写审计)",
                "ready": True,
            },
            {
                "label": "地面连续漫游 (Walk) & 实时特征编码 (Live)",
                "detail": f"漫游记录 {len(gw_cat.get('saved', []))} 条 | 单对 512 维编码记录 {len(gl_cat.get('saved', []))} 条",
                "ready": bool(gw_cat.get("available", False) or gl_cat.get("real_encoding_available", False) or len(gw_cat.get("saved", [])) > 0),
            },
            {
                "label": "空间感知显微镜 (CV031 20例 + CV041 三城150例)",
                "detail": f"CCVPE 局部筛查 {perception_cases} 例 | 7×7 空间小头多方法对照 {spatial_cases} 例",
                "ready": bool(perception_ok and spatial_ok),
            },
        ]

        return {
            "version": "geonav-workbench-v5.0",
            "kernel": "geonav-core-v17",
            "capabilities": self.capabilities(),
            "api_status": {
                "ready": api_ready,
                "model": api_model,
            },
            "counts": {
                "aerial_images": curated_imgs,
                "curated_aerial_tasks": curated_aerial,
                "total_interactive_tasks": curated_aerial + cp_count,
                "aerial_tasks": aerial_count,
                "capturepoint_tasks": cp_count,
                "controller_datasets": len(ctrl_cat.get("datasets", [])),
                "controller_tasks": len(tasks),
                "saved_tasks": len(ctrl_cat.get("saved_tasks", [])),
                "saved_runs": len(ctrl_cat.get("saved_runs", [])),
                "saved_batches": len(batch_cat.get("batches", [])),
                "saved_api_comparisons": len(api_cat.get("comparisons", [])),
                "saved_scenes": len(scene_cat.get("scenes", [])),
                "saved_scene_runs": len(scene_cat.get("saved_runs", [])),
                "ground_walk_saved": len(gw_cat.get("saved", [])),
                "ground_walk_available": bool(gw_cat.get("available", False)),
                "ground_live_saved": len(gl_cat.get("saved", [])),
                "ground_live_available": bool(gl_cat.get("real_encoding_available", False)),
                "perception_available": perception_ok,
                "perception_cases": perception_cases,
                "spatial_evidence_available": spatial_ok,
                "spatial_evidence_cases": spatial_cases,
            },
            "benchmarks": BENCHMARK_OVERVIEW,
            "benchmark_overview": BENCHMARK_OVERVIEW,
            "assets": assets,
        }
