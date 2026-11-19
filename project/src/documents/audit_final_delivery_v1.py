"""Independent verification and final closure seal for FINAL-005 & FINAL-006.

Usage:
    D:\\PYTHON\\python.exe -X utf8 -B -m project.src.documents.audit_final_delivery_v1
    D:\\PYTHON\\python.exe -X utf8 -B -m project.src.documents.audit_final_delivery_v1 --verify-only

Full mode runs all structural checks, automated V5/Lite/OpenCVL/FinalReportV5
audits, zip hygiene scans, blind-set zero-decode checks, and seals SHA-256
manifests under 总结报告相关/总结报告_v1/核验/{答辩PPT_v1,提交封存_v1}.
`--verify-only` re-checks sealed digests, PPTX/PDF/zip invariants, and JSON
validity read-only after final documentation updates without re-running suites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import fitz
from PIL import Image
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "总结报告相关" / "总结报告_v1"
MANUSCRIPT_DIR = REPORT_DIR / "稿件"
PPT_REVIEW_DIR = REPORT_DIR / "审阅" / "答辩PPT_v1"
PPT_AUDIT_DIR = REPORT_DIR / "核验" / "答辩PPT_v1"
SEAL_AUDIT_DIR = REPORT_DIR / "核验" / "提交封存_v1"

PPTX_PATH = MANUSCRIPT_DIR / "总结报告_基于主动探索的视觉地理定位方法设计与实现_结题答辩PPT.pptx"
PPT_PDF_PATH = MANUSCRIPT_DIR / "总结报告_基于主动探索的视觉地理定位方法设计与实现_结题答辩PPT.pdf"
SCRIPT_PATH = MANUSCRIPT_DIR / "结题答辩讲稿与现场操作卡.md"
REHEARSAL_PATH = PPT_AUDIT_DIR / "平台实机预演回执_v1.json"
PPT_AUDIT_PATH = PPT_AUDIT_DIR / "PPT构建与结构核验_v1.json"
SEAL_MANIFEST_PATH = SEAL_AUDIT_DIR / "最终提交物清单与封存_v1.json"
UNIFIED_SUMMARY_PATH = SEAL_AUDIT_DIR / "最终一键验收汇总_v1.json"

DELIVERY_V5_HASHES_PATH = REPORT_DIR / "核验" / "终稿_v5" / "交付文件哈希_v5.json"
OPENCV_STAGE2_JSON = ROOT / "DATA" / "processed_data" / "OpenCVL" / "单步视觉定位基线与消融_v1" / "四阶段科研路线与阶段二判定_v1.json"
LIFECYCLE_JSON = ROOT / "docs" / "维护" / "版本生命周期.json"

MIDTERM_DIR = ROOT / "中期报告相关" / "中期成果更新_20261025" / "交付"
DEMO_V5_ZIP = ROOT / "demo" / "中期demo.zip"
DEMO_LITE_ZIP = ROOT / "demo" / "小型演示demo.zip"

FORBIDDEN_TEXT_PATTERNS = [
    "中期检查汇报",
    "390.8",
    "61.40%",
    "61.20%",
    "66.80%",
    "eval_opencvl_v1.py",
    "coverage_radial_guard.py",
]
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{16,}",
    r"AIza[0-9A-Za-z_-]{20,}",
    r"DEEPSEEK_API_KEY\s*=\s*['\"]?[^'\"\s]+",
    r"OPENAI_API_KEY\s*=\s*['\"]?[^'\"\s]+",
]

SUITE_COMMANDS = [
    ("test_workbench_v5", [sys.executable, "-X", "utf8", "-B", "project/src/tests/test_workbench_v5.py"]),
    ("test_lite_workbench", [sys.executable, "-X", "utf8", "-B", "project/src/tests/test_lite_workbench.py"]),
    ("audit_opencvl_stage2", [sys.executable, "-X", "utf8", "-B", "project/src/eval/audit_opencvl_single_step_visual_v1.py"]),
    ("audit_final_report_v5", [sys.executable, "-X", "utf8", "-B", "-m", "project.src.documents.audit_final_report_v5", "audit"]),
]

SEALED_DELIVERABLE_PATHS = [
    "总结报告相关/总结报告_v1/稿件/总结报告_基于主动探索的视觉地理定位方法设计与实现_终稿_v5.docx",
    "总结报告相关/总结报告_v1/稿件/总结报告_基于主动探索的视觉地理定位方法设计与实现_终稿_v5.pdf",
    "总结报告相关/总结报告_v1/稿件/正文文字稿_v5.md",
    "总结报告相关/总结报告_v1/稿件/总结报告_基于主动探索的视觉地理定位方法设计与实现_结题答辩PPT.pptx",
    "总结报告相关/总结报告_v1/稿件/总结报告_基于主动探索的视觉地理定位方法设计与实现_结题答辩PPT.pdf",
    "总结报告相关/总结报告_v1/稿件/结题答辩讲稿与现场操作卡.md",
    "总结报告相关/总结报告_v1/准备/增量证据绑定_v2.json",
    "总结报告相关/总结报告_v1/核验/终稿_v5/最终正文与原件核验_v5.json",
    "总结报告相关/总结报告_v1/核验/答辩PPT_v1/平台实机预演回执_v1.json",
    "总结报告相关/总结报告_v1/核验/答辩PPT_v1/PPT构建与结构核验_v1.json",
    "绘图/数据结果图/08_OpenCVL阶段二单步视觉定位与消融实证对比.png",
    "绘图/数据结果图/08_OpenCVL阶段二单步视觉定位与消融实证对比.pdf",
    "demo/中期demo.zip",
    "demo/小型演示demo.zip",
    "中期报告相关/中期成果更新_20261025/交付/中期成果汇报.pptx",
    "中期报告相关/中期成果更新_20261025/交付/中期报告.docx",
    "中期报告相关/中期成果更新_20261025/交付/中期报告.pdf",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit_pptx_and_pdf() -> dict[str, Any]:
    failures: list[str] = []
    if not PPTX_PATH.is_file():
        return {"passed": False, "failures": [f"Missing PPTX: {PPTX_PATH}"]}
    if not PPT_PDF_PATH.is_file():
        return {"passed": False, "failures": [f"Missing PPT PDF: {PPT_PDF_PATH}"]}

    prs = Presentation(str(PPTX_PATH))
    slide_count = len(prs.slides)
    if slide_count != 22:
        failures.append(f"Expected 22 PPTX slides, got {slide_count}")

    fonts_used: set[str] = set()
    all_pptx_text_parts: list[str] = []
    footer_mismatches: list[str] = []
    picture_count = 0

    for idx, slide in enumerate(prs.slides, start=1):
        slide_texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    for r in p.runs:
                        if r.font.name:
                            fonts_used.add(r.font.name)
                    slide_texts.append(p.text)
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        slide_texts.append(cell.text)
                        for p in cell.text_frame.paragraphs:
                            for r in p.runs:
                                if r.font.name:
                                    fonts_used.add(r.font.name)
            if shape.shape_type == 13:  # PICTURE
                picture_count += 1
                _ = len(shape.image.blob)
        joined = "\n".join(slide_texts)
        all_pptx_text_parts.append(joined)
        if idx >= 2:
            expected_num = f"{idx:02d} / 22"
            if expected_num not in joined:
                footer_mismatches.append(f"Slide {idx} missing pagination '{expected_num}'")

    if "微软雅黑" not in fonts_used:
        failures.append(f"Expected primary font '微软雅黑', got {sorted(fonts_used)}")
    if footer_mismatches:
        failures.extend(footer_mismatches)

    pptx_full_text = "\n".join(all_pptx_text_parts)
    for pat in FORBIDDEN_TEXT_PATTERNS:
        if pat in pptx_full_text:
            failures.append(f"Forbidden stale string '{pat}' found in PPTX text")
    for regex in SECRET_PATTERNS:
        if re.search(regex, pptx_full_text):
            failures.append(f"Secret pattern matched in PPTX: {regex}")

    # PDF audit
    doc = fitz.open(str(PPT_PDF_PATH))
    pdf_pages = len(doc)
    pdf_texts: list[str] = []
    empty_pdf_pages: list[int] = []
    for i in range(pdf_pages):
        txt = doc[i].get_text("text").strip()
        if not txt:
            empty_pdf_pages.append(i + 1)
        pdf_texts.append(txt)
    doc.close()

    if pdf_pages != 22:
        failures.append(f"Expected 22 PDF pages, got {pdf_pages}")
    if empty_pdf_pages:
        failures.append(f"Empty PDF pages: {empty_pdf_pages}")

    pdf_full_text = "\n".join(pdf_texts)
    for pat in FORBIDDEN_TEXT_PATTERNS:
        if pat in pdf_full_text:
            failures.append(f"Forbidden stale string '{pat}' found in PPT PDF")

    # Review PNG & contact sheet verification
    expected_pngs = [PPT_REVIEW_DIR / f"slide_{i:02d}.png" for i in range(1, 23)]
    expected_pngs += [
        PPT_REVIEW_DIR / "contact_sheet_01_08.png",
        PPT_REVIEW_DIR / "contact_sheet_09_16.png",
        PPT_REVIEW_DIR / "contact_sheet_17_22.png",
        PPT_REVIEW_DIR / "lite_screen_demo.png",
        PPT_REVIEW_DIR / "v5_rehearsal_overview.png",
        PPT_REVIEW_DIR / "v5_rehearsal_navigator.png",
        PPT_REVIEW_DIR / "v5_rehearsal_evidence.png",
    ]
    missing_pngs: list[str] = []
    corrupt_pngs: list[str] = []
    for p in expected_pngs:
        if not p.is_file() or p.stat().st_size == 0:
            missing_pngs.append(str(p.relative_to(ROOT)).replace("\\", "/"))
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:
            corrupt_pngs.append(f"{p.name}: {exc}")

    if missing_pngs:
        failures.append(f"Missing review PNGs: {missing_pngs}")
    if corrupt_pngs:
        failures.append(f"Corrupt review PNGs: {corrupt_pngs}")

    # Speaker script & rehearsal receipt audit
    if not SCRIPT_PATH.is_file():
        failures.append(f"Missing speaker script: {SCRIPT_PATH}")
    else:
        script_txt = SCRIPT_PATH.read_text(encoding="utf-8")
        missing_slide_cues = [f"Slide {i:02d}" for i in range(1, 23) if f"Slide {i:02d}" not in script_txt]
        if missing_slide_cues:
            failures.append(f"Speaker script missing slides: {missing_slide_cues}")
        for pat in FORBIDDEN_TEXT_PATTERNS:
            if pat in script_txt:
                failures.append(f"Forbidden stale string '{pat}' found in speaker script")
        for regex in SECRET_PATTERNS:
            if re.search(regex, script_txt):
                failures.append(f"Secret pattern matched in speaker script: {regex}")

    if not REHEARSAL_PATH.is_file():
        failures.append(f"Missing rehearsal receipt: {REHEARSAL_PATH}")
        rehearsal_ok = False
    else:
        rh = read_json(REHEARSAL_PATH)
        rehearsal_ok = (
            rh.get("passed") is True
            and rh.get("external_api_calls") == 0
            and rh.get("blind_set_touched") is False
            and rh.get("v5_rehearsal_8769", {}).get("passed") is True
            and rh.get("lite_rehearsal_8768", {}).get("passed") is True
            and len(rh.get("v5_rehearsal_8769", {}).get("errors", [])) == 0
            and len(rh.get("lite_rehearsal_8768", {}).get("errors", [])) == 0
        )
        if not rehearsal_ok:
            failures.append("Live platform rehearsal receipt failed validation")

    return {
        "schema": "final-ppt-audit/v1",
        "date": "2026-11-14",
        "passed": len(failures) == 0,
        "failures": failures,
        "pptx_slide_count": slide_count,
        "pdf_page_count": pdf_pages,
        "embedded_picture_count": picture_count,
        "fonts_used": sorted(fonts_used),
        "review_images_verified": len(expected_pngs) - len(missing_pngs) - len(corrupt_pngs),
        "rehearsal_passed": rehearsal_ok,
        "pptx_sha256": sha256_file(PPTX_PATH),
        "pdf_sha256": sha256_file(PPT_PDF_PATH),
        "script_sha256": sha256_file(SCRIPT_PATH) if SCRIPT_PATH.is_file() else None,
    }


def scan_demo_zip_hygiene(zip_path: Path) -> dict[str, Any]:
    if not zip_path.is_file():
        return {"path": str(zip_path.relative_to(ROOT)).replace("\\", "/"), "passed": False, "error": "missing"}
    forbidden_hits: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for name in names:
            norm = name.replace("\\", "/").lower()
            parts = [p for p in norm.split("/") if p]
            if any((p == ".env" or p.startswith(".env.")) and p != ".env.example" for p in parts):
                forbidden_hits.append(name)
            elif any(p == ".env.example" for p in parts):
                txt = zf.read(name).decode("utf-8", errors="ignore")
                if any(re.search(rgx, txt) for rgx in SECRET_PATTERNS):
                    forbidden_hits.append(name)
            elif "data/raw_data" in norm or "/raw_data/" in norm:
                forbidden_hits.append(name)
    return {
        "path": str(zip_path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": zip_path.stat().st_size,
        "mib": round(zip_path.stat().st_size / (1024 * 1024), 2),
        "sha256": sha256_file(zip_path),
        "entry_count": len(names),
        "forbidden_hits": forbidden_hits,
        "passed": len(forbidden_hits) == 0,
    }


def check_core_invariants() -> dict[str, Any]:
    failures: list[str] = []

    # 1. OpenCVL 28 blind pairs
    cv051 = read_json(OPENCV_STAGE2_JSON)
    blind_ok = (
        cv051.get("blind_set_evaluated") is False
        and cv051.get("blind_pairs_sealed") == 28
        and cv051.get("extraction_accounting", {}).get("blind_decodes") == 0
    )
    if not blind_ok:
        failures.append("OpenCVL 28-pair blind isolation gate failed")

    # 2. FINAL-003/FINAL-004 intact vs 交付文件哈希_v5.json (note: 最终正文与原件核验_v5.json may be refreshed deterministically by audit_final_report_v5.py audit)
    v5_drift: list[str] = []
    if not DELIVERY_V5_HASHES_PATH.is_file():
        failures.append(f"Missing v5 delivery manifest: {DELIVERY_V5_HASHES_PATH}")
    else:
        for item in read_json(DELIVERY_V5_HASHES_PATH).get("files", []):
            p = ROOT / item["path"]
            if not p.is_file():
                v5_drift.append(f"missing:{item['path']}")
            elif sha256_file(p) != item["sha256"]:
                v5_drift.append(item["path"])
    if v5_drift:
        failures.append(f"FINAL-004 v5 deliverable SHA drift: {v5_drift}")

    # 3. Demo zip hygiene
    zips = [scan_demo_zip_hygiene(DEMO_V5_ZIP), scan_demo_zip_hygiene(DEMO_LITE_ZIP)]
    for z in zips:
        if not z.get("passed"):
            failures.append(f"Zip hygiene failed for {z['path']}: {z.get('forbidden_hits') or z.get('error')}")

    # 4. Disposable temp cleanup check
    leftover_temp = sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in (ROOT / "temp").glob("rehearse_*.py")
        if p.is_file()
    )
    if leftover_temp:
        failures.append(f"Uncleaned temporary scripts in temp/: {leftover_temp}")

    # 5. Storage budget for 总结报告相关/总结报告_v1 (note v5 audit checks <= 60 MiB)
    total_report_bytes = sum(p.stat().st_size for p in REPORT_DIR.rglob("*") if p.is_file())
    report_mib = round(total_report_bytes / (1024 * 1024), 2)
    if report_mib > 60.0:
        failures.append(f"总结报告_v1 exceeds 60.0 MiB limit: {report_mib} MiB")

    # 6. Lifecycle JSON parse check
    try:
        _ = read_json(LIFECYCLE_JSON)
        lifecycle_ok = True
    except Exception as exc:
        lifecycle_ok = False
        failures.append(f"Invalid JSON in 版本生命周期.json: {exc}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "opencvl_blind_28_untouched": blind_ok,
        "final004_v5_intact": len(v5_drift) == 0,
        "demo_zips": zips,
        "leftover_rehearsal_scripts": leftover_temp,
        "report_v1_total_bytes": total_report_bytes,
        "report_v1_total_mib": report_mib,
        "lifecycle_json_valid": lifecycle_ok,
    }


def run_subprocess_suites() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{prev_pp}" if prev_pp else str(ROOT)
    for name, cmd in SUITE_COMMANDS:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = round(time.perf_counter() - t0, 3)
        results.append(
            {
                "suite": name,
                "cmd": " ".join(cmd),
                "returncode": proc.returncode,
                "elapsed_sec": dt,
                "passed": proc.returncode == 0,
                "stdout_tail": proc.stdout.strip().splitlines()[-8:] if proc.stdout else [],
                "stderr_tail": proc.stderr.strip().splitlines()[-8:] if proc.stderr else [],
            }
        )
    return results


def build_seal_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for rel in SEALED_DELIVERABLE_PATHS:
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        entries.append(
            {
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    return {
        "schema": "final-delivery-seal/v1",
        "date": "2026-11-14",
        "task": "FINAL-005/FINAL-006",
        "opencvl_blind_28_untouched": True,
        "missing": missing,
        "files": entries,
    }


def verify_sealed_manifest() -> dict[str, Any]:
    if not SEAL_MANIFEST_PATH.is_file():
        return {"passed": False, "failures": [f"Missing seal manifest: {SEAL_MANIFEST_PATH}"]}
    manifest = read_json(SEAL_MANIFEST_PATH)
    mismatches: list[str] = []
    for item in manifest.get("files", []):
        p = ROOT / item["path"]
        if not p.is_file():
            mismatches.append(f"missing:{item['path']}")
        elif p.stat().st_size != item["bytes"] or sha256_file(p) != item["sha256"]:
            mismatches.append(f"drift:{item['path']}")
    return {
        "passed": len(mismatches) == 0 and len(manifest.get("missing", [])) == 0,
        "checked_count": len(manifest.get("files", [])),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="Re-verify sealed artifacts read-only without re-running suites")
    args = parser.parse_args()

    PPT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    SEAL_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    ppt_audit = audit_pptx_and_pdf()

    if args.verify_only:
        inv = check_core_invariants()
        seal_check = verify_sealed_manifest()
        existing_summary = read_json(UNIFIED_SUMMARY_PATH) if UNIFIED_SUMMARY_PATH.is_file() else {}
        passed = ppt_audit["passed"] and inv["passed"] and seal_check["passed"] and existing_summary.get("passed") is True
        print(
            json.dumps(
                {
                    "mode": "verify-only",
                    "passed": passed,
                    "ppt_passed": ppt_audit["passed"],
                    "invariants_passed": inv["passed"],
                    "seal_verified": seal_check["passed"],
                    "seal_checked_count": seal_check.get("checked_count", 0),
                    "report_v1_total_mib": inv["report_v1_total_mib"],
                    "failures": ppt_audit["failures"] + inv["failures"] + seal_check.get("mismatches", []),
                },
                ensure_ascii=False,
            )
        )
        if not passed:
            raise SystemExit(1)
        return

    # Full run: first run audit_final_report_v5 & test suites cleanly BEFORE hashing PPT_AUDIT_PATH into SEAL_MANIFEST_PATH
    suites = run_subprocess_suites()
    suites_passed = all(s["passed"] for s in suites)

    # Re-check ppt_audit + core invariants AFTER suites (audit_final_report_v5.py audit updated 交付文件哈希_v5.json)
    ppt_audit = audit_pptx_and_pdf()
    PPT_AUDIT_PATH.write_text(json.dumps(ppt_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    inv = check_core_invariants()
    manifest = build_seal_manifest()
    SEAL_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_passed = suites_passed and ppt_audit["passed"] and inv["passed"] and not manifest["missing"]
    summary = {
        "schema": "final-delivery-unified-summary/v1",
        "date": "2026-11-14",
        "task": "FINAL-005/FINAL-006",
        "passed": all_passed,
        "ppt_audit_passed": ppt_audit["passed"],
        "invariants_passed": inv["passed"],
        "suites_passed": suites_passed,
        "suites": suites,
        "invariants": inv,
        "sealed_file_count": len(manifest["files"]),
        "seal_manifest_path": str(SEAL_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
        "seal_manifest_sha256": sha256_file(SEAL_MANIFEST_PATH),
    }
    UNIFIED_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "mode": "full",
                "passed": all_passed,
                "suites": [{s["suite"]: s["passed"], "sec": s["elapsed_sec"]} for s in suites],
                "ppt_passed": ppt_audit["passed"],
                "invariants_passed": inv["passed"],
                "sealed_files": len(manifest["files"]),
                "report_v1_total_mib": inv["report_v1_total_mib"],
                "failures": ppt_audit["failures"] + inv["failures"] + manifest["missing"],
            },
            ensure_ascii=False,
        )
    )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
