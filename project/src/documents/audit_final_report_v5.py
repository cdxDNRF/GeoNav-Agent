"""Final report v5 archival, Word COM PDF export, OOXML normalization, rendering, and independent audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import fitz
from lxml import etree
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = ROOT / "总结报告相关" / "总结报告_v1"
MANUSCRIPT_DIR = REPORT_DIR / "稿件"
PREP_DIR = REPORT_DIR / "准备"
AUDIT_DIR = REPORT_DIR / "核验" / "终稿_v5"
REVIEW_DIR = REPORT_DIR / "审阅" / "终稿_v5"
ARCHIVE_MANUSCRIPT_DIR = ROOT / "归档" / "总结报告相关" / "总结报告_v1" / "稿件"

DOCX_PATH = MANUSCRIPT_DIR / "总结报告_基于主动探索的视觉地理定位方法设计与实现_终稿_v5.docx"
PDF_PATH = MANUSCRIPT_DIR / "总结报告_基于主动探索的视觉地理定位方法设计与实现_终稿_v5.pdf"
MD_PATH = MANUSCRIPT_DIR / "正文文字稿_v5.md"
CONTENT_JSON_PATH = ROOT / "project" / "src" / "documents" / "final_report_content_v5.json"

TEMPLATE_PATH = ROOT / "课程相关报告模版" / "综合工程设计-3-总结报告.docx"
MIDTERM_HASHES_PATH = ROOT / "总结报告相关" / "总结准备_v1" / "中期原件角色与哈希.json"
FINAL001_INPUTS_PATH = REPORT_DIR / "核验" / "输入与源码绑定.json"
FINAL001_HASHES_PATH = REPORT_DIR / "核验" / "交付文件哈希.json"
CLAIM_V1_PATH = PREP_DIR / "论断与证据.json"
CLAIM_V2_PATH = PREP_DIR / "增量证据绑定_v2.json"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STYLE_ORDER = [
    "name",
    "aliases",
    "basedOn",
    "next",
    "link",
    "autoRedefine",
    "hidden",
    "uiPriority",
    "semiHidden",
    "unhideWhenUsed",
    "qFormat",
    "locked",
    "personal",
    "personalCompose",
    "personalReply",
    "rsid",
    "pPr",
    "rPr",
    "tblPr",
    "trPr",
    "tcPr",
    "tblStylePr",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_with_archive(rel: str) -> Path | None:
    p1 = ROOT / rel
    if p1.is_file():
        return p1
    p2 = ROOT / "归档" / rel
    if p2.is_file():
        return p2
    return None


def pointer(doc: Any, ptr: str) -> Any:
    cur = doc
    if not ptr:
        return cur
    for tok in ptr.lstrip("/").split("/"):
        tok = tok.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            cur = cur[int(tok)]
        else:
            cur = cur[tok]
    return cur


def archive_old_drafts() -> dict[str, Any]:
    """Move superseded v4 drafts byte-identical to 归档/ and emit mapping JSON."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        "总结报告_基于主动探索的视觉地理定位方法设计与实现_初稿_v4.docx",
        "总结报告_基于主动探索的视觉地理定位方法设计与实现_初稿_v4.pdf",
        "正文文字稿_v3.md",
    ]
    records = []
    for name in targets:
        src = MANUSCRIPT_DIR / name
        dst = ARCHIVE_MANUSCRIPT_DIR / name
        if src.is_file():
            old_sha = sha256_file(src)
            old_bytes = src.stat().st_size
            shutil.copy2(src, dst)
            new_sha = sha256_file(dst)
            if old_sha != new_sha:
                raise RuntimeError(f"Archive hash mismatch on {name}")
            src.unlink()
            status = "moved_verified"
        elif dst.is_file():
            old_sha = sha256_file(dst)
            new_sha = old_sha
            old_bytes = dst.stat().st_size
            status = "already_archived"
        else:
            raise FileNotFoundError(f"Missing both active and archived draft: {name}")
        records.append({
            "filename": name,
            "old_path": f"总结报告相关/总结报告_v1/稿件/{name}",
            "new_path": f"归档/总结报告相关/总结报告_v1/稿件/{name}",
            "bytes": old_bytes,
            "sha256_before": old_sha,
            "sha256_after": new_sha,
            "byte_identical": old_sha == new_sha,
            "status": status,
        })
    mapping = {
        "schema": "superseded-draft-archive-map/v5",
        "date": "2026-11-14",
        "superseded_by": "总结报告相关/总结报告_v1/稿件/总结报告_基于主动探索的视觉地理定位方法设计与实现_终稿_v5.docx",
        "files": records,
    }
    out = AUDIT_DIR / "被替代稿迁移映射_v5.json"
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[archive] {len(records)} files verified -> {out.relative_to(ROOT)}")
    return mapping


def export_word_pdf() -> None:
    """Update Word TOC & fields and export PDF via Word COM automation."""
    import pythoncom
    import win32com.client

    if not DOCX_PATH.is_file():
        raise FileNotFoundError(DOCX_PATH)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pythoncom.CoInitialize()
    word = None
    doc = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(DOCX_PATH), ReadOnly=False)
        for idx in range(1, doc.TablesOfContents.Count + 1):
            toc = doc.TablesOfContents(idx)
            toc.Update()
        doc.Fields.Update()
        doc.Repaginate()
        doc.Save()
        # 17 == wdExportFormatPDF
        doc.ExportAsFixedFormat(
            OutputFileName=str(PDF_PATH),
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0,
            Range=0,
            Item=0,
            IncludeDocProps=True,
            KeepIRM=True,
            CreateBookmarks=1,
            DocStructureTags=True,
            BitmapMissingFonts=True,
            UseISO19005_1=False,
        )
        print(f"[export] saved updated DOCX & exported PDF -> {PDF_PATH.relative_to(ROOT)}")
    finally:
        if doc is not None:
            try:
                doc.Close(SaveChanges=False)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit(SaveChanges=False)
            except Exception:
                pass
        pythoncom.CoUninitialize()


def finalize_ooxml() -> None:
    """Normalize word/styles.xml child ordering after Word COM save."""
    order_idx = {name: i for i, name in enumerate(STYLE_ORDER)}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_zip = Path(tmp) / "repacked.docx"
        with zipfile.ZipFile(DOCX_PATH, "r") as zin, zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/styles.xml":
                    root = etree.fromstring(data)
                    for style_el in root.findall(f"{{{W_NS}}}style"):
                        children = list(style_el)
                        children.sort(key=lambda el: order_idx.get(etree.QName(el).localname, 999))
                        for c in children:
                            style_el.remove(c)
                        for c in children:
                            style_el.append(c)
                    data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
                zout.writestr(item, data)
        shutil.copy2(tmp_zip, DOCX_PATH)
    print(f"[finalize] normalized word/styles.xml -> {DOCX_PATH.relative_to(ROOT)}")


def render_review_pages() -> dict[str, Any]:
    """Render all PDF pages and generate multi-page contact sheets in 审阅/终稿_v5/."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(PDF_PATH)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in REVIEW_DIR.glob("*.png"):
        old.unlink()

    doc = fitz.open(PDF_PATH)
    page_paths: list[Path] = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=140, alpha=False)
        out_p = REVIEW_DIR / f"page-{i + 1:02d}.png"
        pix.save(str(out_p))
        page_paths.append(out_p)
    doc.close()

    # Build 4x3 grid contact sheets (1200x1732 px matching v1 reviewer pattern)
    sheet_paths: list[str] = []
    cols, rows = 4, 3
    per_sheet = cols * rows
    cell_w, cell_h = 280, 405
    pad = 16
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad + 32

    for s_idx in range(0, len(page_paths), per_sheet):
        chunk = page_paths[s_idx : s_idx + per_sheet]
        canvas = Image.new("RGB", (sheet_w, sheet_h), "white")
        draw = ImageDraw.Draw(canvas)
        s_num = s_idx // per_sheet + 1
        draw.text(
            (pad, 8),
            f"Final Report v5 Review Contact Sheet {s_num} (Pages {s_idx + 1}-{s_idx + len(chunk)} / {len(page_paths)})",
            fill="#111111",
        )
        for j, p_path in enumerate(chunk):
            r, c = divmod(j, cols)
            x0 = pad + c * (cell_w + pad)
            y0 = 32 + pad + r * (cell_h + pad)
            with Image.open(p_path) as im:
                thumb = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x0, y0))
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline="#CBD5E1", width=1)
        s_out = REVIEW_DIR / f"contact-{s_num:02d}.png"
        canvas.save(str(s_out))
        sheet_paths.append(str(s_out.relative_to(ROOT)).replace("\\", "/"))
    print(f"[render] {len(page_paths)} pages + {len(sheet_paths)} contact sheets -> {REVIEW_DIR.relative_to(ROOT)}")
    return {"page_count": len(page_paths), "contact_sheets": sheet_paths}


def check_pointer_bundle(bundle_path: Path, label: str, inputs_override_path: Path | None = None) -> dict[str, Any]:
    bundle = read_json(bundle_path)
    inputs_meta = read_json(inputs_override_path)["inputs"] if inputs_override_path else bundle["inputs"]
    docs: dict[str, Any] = {}
    missing_sources = []
    sha_mismatches = []
    for k, meta in inputs_meta.items():
        rel = meta["path"]
        resolved = resolve_with_archive(rel)
        if resolved is None:
            missing_sources.append(rel)
            continue
        actual_sha = sha256_file(resolved)
        if k != "launcher" and actual_sha != meta["sha256"]:
            sha_mismatches.append({"key": k, "path": rel, "expected": meta["sha256"], "actual": actual_sha})
        docs[k] = read_json(resolved) if resolved.suffix.lower() == ".json" else resolved.read_text(encoding="utf-8", errors="ignore")

    checked = 0
    value_mismatches = []
    for claim in bundle["claims"]:
        for fact in claim["facts"]:
            src = fact["source"]
            if src not in docs or isinstance(docs[src], str):
                continue
            checked += 1
            actual = pointer(docs[src], fact["pointer"])
            if actual != fact["value"]:
                value_mismatches.append({"claim": claim["id"], "source": src, "pointer": fact["pointer"]})
    return {
        "bundle": label,
        "claims": len(bundle["claims"]),
        "json_facts_checked": checked,
        "missing_sources": missing_sources,
        "sha_mismatches": sha_mismatches,
        "value_mismatches": value_mismatches,
        "passed": not missing_sources and not sha_mismatches and not value_mismatches,
    }


def run_audit() -> dict[str, Any]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # 1. OOXML geometry & element counts
    with zipfile.ZipFile(DOCX_PATH, "r") as z:
        doc_xml = etree.fromstring(z.read("word/document.xml"))
        styles_xml = etree.fromstring(z.read("word/styles.xml"))
        media_pngs = [n for n in z.namelist() if n.startswith("word/media/") and n.lower().endswith(".png")]

    sects = doc_xml.findall(f".//{{{W_NS}}}sectPr")
    pg_ok = len(sects) == 2
    geom_details = []
    for s in sects:
        sz = s.find(f"{{{W_NS}}}pgSz")
        mar = s.find(f"{{{W_NS}}}pgMar")
        d = {
            "w": sz.attrib.get(f"{{{W_NS}}}w") if sz is not None else None,
            "h": sz.attrib.get(f"{{{W_NS}}}h") if sz is not None else None,
            "top": mar.attrib.get(f"{{{W_NS}}}top") if mar is not None else None,
            "right": mar.attrib.get(f"{{{W_NS}}}right") if mar is not None else None,
            "bottom": mar.attrib.get(f"{{{W_NS}}}bottom") if mar is not None else None,
            "left": mar.attrib.get(f"{{{W_NS}}}left") if mar is not None else None,
            "header": mar.attrib.get(f"{{{W_NS}}}header") if mar is not None else None,
            "footer": mar.attrib.get(f"{{{W_NS}}}footer") if mar is not None else None,
            "gutter": mar.attrib.get(f"{{{W_NS}}}gutter") if mar is not None else None,
        }
        expected = {
            "w": "11906",
            "h": "16838",
            "top": "1440",
            "right": "1134",
            "bottom": "1440",
            "left": "1985",
            "header": "851",
            "footer": "992",
            "gutter": "170",
        }
        if d != expected:
            pg_ok = False
        geom_details.append(d)
    if not pg_ok:
        failures.append("Page geometry mismatch against template")

    tables = doc_xml.findall(f".//{{{W_NS}}}tbl")
    table_widths_ok = True
    for idx, tbl in enumerate(tables):
        w_el = tbl.find(f"./{{{W_NS}}}tblPr/{{{W_NS}}}tblW")
        if w_el is not None and int(w_el.attrib.get(f"{{{W_NS}}}w", "0")) > 8620:
            table_widths_ok = False
            failures.append(f"Table {idx + 1} width exceeds 8617 dxa")
    if len(tables) != 7:
        failures.append(f"Expected 7 tables, got {len(tables)}")
    if len(media_pngs) != 9:
        failures.append(f"Expected 9 PNG media files (logo + F01..F08), got {len(media_pngs)}")

    # Verify styles ordering
    order_idx = {name: i for i, name in enumerate(STYLE_ORDER)}
    styles_ordered = True
    for st in styles_xml.findall(f"{{{W_NS}}}style"):
        ranks = [order_idx.get(etree.QName(c).localname, 999) for c in list(st)]
        if ranks != sorted(ranks):
            styles_ordered = False
            failures.append("word/styles.xml child order non-canonical")
            break

    # 2. Content CJK count & sequential citations
    content = read_json(CONTENT_JSON_PATH)
    texts = list(content["abstract_cn"]["paragraphs"])
    for ch in content["chapters"]:
        texts.append(ch["title"])
        for sec in ch["sections"]:
            texts.append(sec["title"])
            texts.extend(sec["paragraphs"])
            texts.extend(sec.get("extra_paragraphs", []))
    for sec in content["appendix"]["sections"]:
        texts.append(sec["title"])
        texts.extend(sec["paragraphs"])
    joined = "\n".join(texts)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", joined))
    if cjk_count < 12000:
        failures.append(f"CJK character count too low: {cjk_count}")

    cites = [int(x) for x in re.findall(r"\[(\d+)\]", joined)]
    seen_cites = []
    for c in cites:
        if c not in seen_cites:
            seen_cites.append(c)
    if seen_cites != list(range(1, len(content["references"]) + 1)):
        failures.append(f"Non-sequential citations: {seen_cites}")

    # 3. PDF page margin & TOC check
    pdf = fitz.open(PDF_PATH)
    overflow_pages = []
    toc_errors = []
    toc_pages_text = ""
    for i, page in enumerate(pdf):
        pw, ph = page.rect.width, page.rect.height
        # left margin ~1.38 in + 0.12 in gutter ~= 107.7 pt; right margin ~0.79 in ~= 56.7 pt
        for blk in page.get_text("blocks"):
            x0, y0, x1, y1, txt, *_ = blk
            if not txt.strip():
                continue
            if x0 < 70 or x1 > pw - 35:
                overflow_pages.append({"page": i + 1, "x0": round(x0, 1), "x1": round(x1, 1), "pw": round(pw, 1), "snippet": txt.strip()[:40]})
        if i <= 6:
            page_txt = page.get_text()
            toc_pages_text += page_txt
            if "Error!" in page_txt or "未定义书签" in page_txt or "错误!" in page_txt or "未找到目录项" in page_txt:
                toc_errors.append(i + 1)
    pdf_page_count = len(pdf)
    pdf.close()
    toc_has_chapter_numbers = ("第一章" in toc_pages_text and "第五章" in toc_pages_text and re.search(r"(?:第一章|绪论)[^\n]*\d+", toc_pages_text) is not None)
    if overflow_pages:
        failures.append(f"PDF horizontal margin overflow on {len(overflow_pages)} blocks")
    if toc_errors:
        failures.append(f"TOC errors on pages {toc_errors}")
    if not toc_has_chapter_numbers:
        failures.append("TOC did not populate chapter headings and page numbers")

    # 4. Claim & source verification (FINAL-001 & FINAL-003)
    c1_check = check_pointer_bundle(CLAIM_V1_PATH, "论断与证据.json", FINAL001_INPUTS_PATH)
    c2_check = check_pointer_bundle(CLAIM_V2_PATH, "增量证据绑定_v2.json")
    if not c1_check["passed"]:
        failures.append(f"FINAL-001 claim check failed: {c1_check}")
    if not c2_check["passed"]:
        failures.append(f"FINAL-003 claim check failed: {c2_check}")

    # 5. Historical integrity: 203 original files + template + FINAL-001 files + 28 blind OpenCVL pairs
    mid_manifest = read_json(MIDTERM_HASHES_PATH)
    mid_missing, mid_drift = [], []
    for item in mid_manifest["files"]:
        res = resolve_with_archive(item["path"])
        if res is None:
            mid_missing.append(item["path"])
        elif sha256_file(res) != item["sha256"]:
            mid_drift.append(item["path"])
    if mid_missing or mid_drift:
        failures.append(f"Midterm original integrity drift: missing={len(mid_missing)}, drift={len(mid_drift)}")

    f1_inputs = read_json(FINAL001_INPUTS_PATH)["inputs"]
    non_launcher_drift = []
    for k, meta in f1_inputs.items():
        if k == "launcher":
            continue
        res = resolve_with_archive(meta["path"])
        if res is None or sha256_file(res) != meta["sha256"]:
            non_launcher_drift.append(k)
    if non_launcher_drift:
        failures.append(f"FINAL-001 input drift on keys: {non_launcher_drift}")

    f1_hashes = read_json(FINAL001_HASHES_PATH)["files"]
    f1_drift = []
    for item in f1_hashes:
        p = REPORT_DIR / item["path"]
        if not p.is_file() or sha256_file(p) != item["sha256"]:
            f1_drift.append(item["path"])
    if f1_drift:
        failures.append(f"FINAL-001 prepared artifact drift: {f1_drift}")

    # Verify OpenCVL blind set untouched
    cv051 = read_json(ROOT / "DATA/processed_data/OpenCVL/单步视觉定位基线与消融_v1/四阶段科研路线与阶段二判定_v1.json")
    blind_untouched = (
        cv051.get("blind_set_evaluated") is False
        and cv051.get("blind_pairs_sealed") == 28
        and cv051.get("extraction_accounting", {}).get("blind_decodes") == 0
    )
    if not blind_untouched:
        failures.append("OpenCVL 28-pair blind protection gate failed")

    # 6. Directory size budget <= 60 MiB
    total_dir_bytes = sum(p.stat().st_size for p in REPORT_DIR.rglob("*") if p.is_file())
    size_mib = total_dir_bytes / (1024 * 1024)
    if size_mib > 60.0:
        failures.append(f"总结报告_v1 directory exceeds 60 MiB: {size_mib:.2f} MiB")

    report = {
        "schema": "final-report-audit/v5",
        "date": "2026-11-14",
        "passed": len(failures) == 0,
        "failures": failures,
        "ooxml_geometry": {
            "sections_matched_template": pg_ok,
            "sections": geom_details,
            "tables_count": len(tables),
            "table_widths_within_limit": table_widths_ok,
            "media_png_count": len(media_pngs),
            "styles_xml_canonical_order": styles_ordered,
        },
        "content_checks": {
            "cjk_character_count": cjk_count,
            "citation_sequence": seen_cites,
            "sequential_1_to_13": seen_cites == list(range(1, 14)),
        },
        "pdf_checks": {
            "page_count": pdf_page_count,
            "margin_overflow_blocks": overflow_pages,
            "toc_errors": toc_errors,
            "toc_populated": toc_has_chapter_numbers,
        },
        "source_claim_verification": {
            "final001_v1": c1_check,
            "final003_v2": c2_check,
        },
        "historical_protection": {
            "template_sha256": sha256_file(TEMPLATE_PATH),
            "midterm_203_files_checked": len(mid_manifest["files"]),
            "midterm_missing": len(mid_missing),
            "midterm_drift": len(mid_drift),
            "final001_non_launcher_49_intact": len(non_launcher_drift) == 0,
            "final001_prepared_10_intact": len(f1_drift) == 0,
            "opencvl_blind_28_untouched": blind_untouched,
        },
        "storage_budget": {
            "report_dir": "总结报告相关/总结报告_v1",
            "total_bytes": total_dir_bytes,
            "total_mib": round(size_mib, 2),
            "limit_mib": 60.0,
        },
    }
    out_audit = AUDIT_DIR / "最终正文与原件核验_v5.json"
    out_audit.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write FINAL-004 delivery hashes manifest without modifying FINAL-001 manifest
    delivery_files = [
        DOCX_PATH,
        PDF_PATH,
        MD_PATH,
        PREP_DIR / "增量证据绑定_v2.json",
        PREP_DIR / "增量证据与修订清单_v2.md",
        PREP_DIR / "图表与表格清单_v2.md",
        AUDIT_DIR / "生成回执_v5.json",
        AUDIT_DIR / "被替代稿迁移映射_v5.json",
        out_audit,
    ]
    manifest = {
        "schema": "final-report-delivery-v5",
        "task": "FINAL-003/FINAL-004",
        "date": "2026-11-14",
        "files": [
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in delivery_files
            if p.is_file()
        ],
    }
    (AUDIT_DIR / "交付文件哈希_v5.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "pages": pdf_page_count, "cjk": cjk_count, "mib": round(size_mib, 2), "failures": failures}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["archive", "export", "finalize", "render", "audit", "all"])
    args = parser.parse_args()
    if args.mode in ("archive", "all"):
        archive_old_drafts()
    if args.mode in ("export", "all"):
        export_word_pdf()
    if args.mode in ("finalize", "all"):
        finalize_ooxml()
    if args.mode in ("render", "all"):
        render_review_pages()
    if args.mode in ("audit", "all"):
        run_audit()


if __name__ == "__main__":
    main()
