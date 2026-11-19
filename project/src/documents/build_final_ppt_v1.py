# -*- coding: utf-8 -*-
"""Build the 22-slide final defense PPTX, export PDF and 22 PNGs via PowerPoint COM,
and assemble 3 review contact sheets under 总结报告相关/总结报告_v1/审阅/答辩PPT_v1/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PPTX = ROOT / "中期报告相关" / "中期成果更新_20261025" / "交付" / "中期成果汇报.pptx"
MANUSCRIPT_DIR = ROOT / "总结报告相关" / "总结报告_v1" / "稿件"
REVIEW_DIR = ROOT / "总结报告相关" / "总结报告_v1" / "审阅" / "答辩PPT_v1"

OUT_PPTX = MANUSCRIPT_DIR / "总结报告_基于主动探索的视觉地理定位方法设计与实现_结题答辩PPT.pptx"
OUT_PDF = MANUSCRIPT_DIR / "总结报告_基于主动探索的视觉地理定位方法设计与实现_结题答辩PPT.pdf"

FONT_CN = "微软雅黑"
INK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
SLATE = RGBColor(0x64, 0x74, 0x8B)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
TEAL = RGBColor(0x0D, 0x94, 0x88)
AMBER = RGBColor(0xD9, 0x77, 0x06)
PURPLE = RGBColor(0x7C, 0x3A, 0xED)
RED = RGBColor(0xDC, 0x26, 0x26)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
SOFT_RED = RGBColor(0xFE, 0xF2, 0xF2)
ROW_ALT = RGBColor(0xF8, 0xFA, 0xFC)

FOOTER_TEXT = (
    "综合工程设计结题答辩  |  基于主动探索的视觉地理定位方法设计与实现  |  "
    "陈德昕 · 李世显 · 陈龙辉（指导教师：王笛）"
)


def drop_slide(prs: Presentation, idx_0: int) -> None:
    sld_id_lst = prs.slides._sldIdLst
    sld_id = sld_id_lst[idx_0]
    prs.part.drop_rel(sld_id.rId)
    sld_id_lst.remove(sld_id)


def move_slide(prs: Presentation, from_idx: int, to_idx: int) -> None:
    sld_id_lst = prs.slides._sldIdLst
    el = sld_id_lst[from_idx]
    sld_id_lst.remove(el)
    sld_id_lst.insert(to_idx, el)


def remove_shapes_by_name(slide, names: Iterable[str]) -> None:
    target = set(names)
    for sh in list(slide.shapes):
        if sh.name in target:
            sh._element.getparent().remove(sh._element)


def shape_by_name(slide, name: str):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    raise KeyError(f"Shape {name!r} not found on slide")


def apply_run_font(
    run,
    size_pt: float,
    bold: bool = False,
    color: RGBColor = INK,
    font_name: str = FONT_CN,
) -> None:
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = color


def set_para_segments(
    para,
    segments: list[tuple[str, bool, RGBColor | None]],
    size_pt: float,
    default_color: RGBColor = INK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    space_after_pt: float = 2.5,
    line_spacing: float = 1.15,
) -> None:
    para.text = ""
    para.alignment = align
    para.space_after = Pt(space_after_pt)
    para.line_spacing = line_spacing
    for text, is_bold, color in segments:
        r = para.add_run()
        r.text = text
        apply_run_font(r, size_pt=size_pt, bold=is_bold, color=color or default_color)


def set_text_lines(
    shape,
    lines: list[list[tuple[str, bool, RGBColor | None]] | str],
    size_pt: float | None = None,
    default_color: RGBColor | None = None,
    align: PP_ALIGN | None = None,
    space_after_pt: float = 2.5,
    line_spacing: float = 1.14,
) -> None:
    tf = shape.text_frame
    tf.word_wrap = True
    orig_size = size_pt
    orig_color = default_color
    orig_align = align
    if len(tf.paragraphs) > 0:
        p0 = tf.paragraphs[0]
        if orig_align is None and p0.alignment is not None:
            orig_align = p0.alignment
        for r in p0.runs:
            if orig_size is None and r.font.size is not None:
                orig_size = r.font.size.pt
            if orig_color is None and r.font.color and r.font.color.type is not None:
                try:
                    orig_color = r.font.color.rgb
                except Exception:
                    pass
            break
    orig_size = orig_size or 11.0
    orig_color = orig_color or INK
    orig_align = orig_align or PP_ALIGN.LEFT

    while len(tf.paragraphs) > 1:
        p_el = tf.paragraphs[-1]._p
        p_el.getparent().remove(p_el)

    for idx, item in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        segs = [(item, False, orig_color)] if isinstance(item, str) else item
        set_para_segments(
            p,
            segs,
            size_pt=orig_size,
            default_color=orig_color,
            align=orig_align,
            space_after_pt=space_after_pt,
            line_spacing=line_spacing,
        )


def replace_plain_text(
    shape,
    text: str,
    size_pt: float | None = None,
    bold: bool | None = None,
    color: RGBColor | None = None,
) -> None:
    tf = shape.text_frame
    tf.word_wrap = True
    while len(tf.paragraphs) > 1:
        p_el = tf.paragraphs[-1]._p
        p_el.getparent().remove(p_el)
    p = tf.paragraphs[0]
    keep_size = size_pt
    keep_bold = bold
    keep_color = color
    if p.runs:
        r0 = p.runs[0]
        if keep_size is None and r0.font.size is not None:
            keep_size = r0.font.size.pt
        if keep_bold is None and r0.font.bold is not None:
            keep_bold = r0.font.bold
        if keep_color is None:
            try:
                keep_color = r0.font.color.rgb
            except Exception:
                pass
    p.text = ""
    r = p.add_run()
    r.text = text
    apply_run_font(r, size_pt=keep_size or 11.0, bold=bool(keep_bold), color=keep_color or INK)


def add_picture_fit(
    slide,
    img_path: Path,
    box_left_pt: float,
    box_top_pt: float,
    box_w_pt: float,
    box_h_pt: float,
):
    with Image.open(img_path) as im:
        iw, ih = im.size
    aspect = iw / ih
    target_aspect = box_w_pt / box_h_pt
    if aspect >= target_aspect:
        w = box_w_pt
        h = box_w_pt / aspect
    else:
        h = box_h_pt
        w = box_h_pt * aspect
    left = box_left_pt + (box_w_pt - w) / 2.0
    top = box_top_pt + (box_h_pt - h) / 2.0
    return slide.shapes.add_picture(str(img_path), Pt(left), Pt(top), Pt(w), Pt(h))


def add_white_card(
    slide,
    left_pt: float,
    top_pt: float,
    width_pt: float,
    height_pt: float,
    fill_color: RGBColor = WHITE,
):
    sh = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Pt(left_pt), Pt(top_pt), Pt(width_pt), Pt(height_pt)
    )
    sh.adjustments[0] = 0.03
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill_color
    sh.line.color.rgb = BORDER
    sh.line.width = Pt(1.0)
    return sh


def add_card_with_header(
    slide,
    left_pt: float,
    top_pt: float,
    width_pt: float,
    height_pt: float,
    header_text: str,
    header_color: RGBColor,
    body_lines: list[list[tuple[str, bool, RGBColor | None]] | str],
    body_size_pt: float = 9.5,
    header_h_pt: float = 26.0,
    fill_color: RGBColor = WHITE,
) -> None:
    add_white_card(slide, left_pt, top_pt, width_pt, height_pt, fill_color=fill_color)

    hdr = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Pt(left_pt), Pt(top_pt), Pt(width_pt), Pt(header_h_pt)
    )
    hdr.adjustments[0] = 0.08
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = header_color
    hdr.line.fill.background()
    tf_h = hdr.text_frame
    tf_h.word_wrap = True
    tf_h.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_h.margin_left = Pt(8)
    tf_h.margin_right = Pt(8)
    tf_h.margin_top = Pt(2)
    tf_h.margin_bottom = Pt(2)
    p = tf_h.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = header_text
    apply_run_font(r, size_pt=10.5, bold=True, color=WHITE)

    tb = slide.shapes.add_textbox(
        Pt(left_pt + 8),
        Pt(top_pt + header_h_pt + 4),
        Pt(width_pt - 16),
        Pt(height_pt - header_h_pt - 8),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.margin_top = Pt(1)
    tf.margin_bottom = Pt(1)
    set_text_lines(tb, body_lines, size_pt=body_size_pt, default_color=INK, space_after_pt=2.5, line_spacing=1.15)


def add_caption_box(
    slide, left_pt: float, top_pt: float, width_pt: float, height_pt: float, text: str, size_pt: float = 8.4
) -> None:
    tb = slide.shapes.add_textbox(Pt(left_pt), Pt(top_pt), Pt(width_pt), Pt(height_pt))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.margin_top = Pt(1)
    tf.margin_bottom = Pt(1)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    apply_run_font(r, size_pt=size_pt, bold=False, color=MUTED)


def style_table_cell(
    cell,
    text: str,
    size_pt: float,
    bold: bool = False,
    color: RGBColor = INK,
    bg: RGBColor | None = None,
    align: PP_ALIGN = PP_ALIGN.LEFT,
) -> None:
    if bg is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Pt(4)
    cell.margin_right = Pt(4)
    cell.margin_top = Pt(2)
    cell.margin_bottom = Pt(2)
    tf = cell.text_frame
    tf.word_wrap = True
    while len(tf.paragraphs) > 1:
        el = tf.paragraphs[-1]._p
        el.getparent().remove(el)
    p = tf.paragraphs[0]
    p.text = ""
    p.alignment = align
    r = p.add_run()
    r.text = text
    apply_run_font(r, size_pt=size_pt, bold=bold, color=color)


def set_cell_border_thin(cell, color_hex: str = "CBD5E1") -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for old in tcPr.findall(qn(tag)):
            tcPr.remove(old)
        ln = tcPr.makeelement(qn(tag), {"w": "6350", "cap": "flat", "cmpd": "sng", "algn": "ctr"})
        sf = ln.makeelement(qn("a:solidFill"), {})
        sf.append(sf.makeelement(qn("a:srgbClr"), {"val": color_hex}))
        ln.append(sf)
        tcPr.append(ln)


def add_styled_table(
    slide,
    left_pt: float,
    top_pt: float,
    width_pt: float,
    height_pt: float,
    headers: list[str],
    rows: list[list[str]],
    col_widths_pt: list[float],
    header_color: RGBColor = BLUE,
    font_size_pt: float = 8.6,
    highlight_col_fn=None,
) -> None:
    nrows = 1 + len(rows)
    ncols = len(headers)
    gf = slide.shapes.add_table(nrows, ncols, Pt(left_pt), Pt(top_pt), Pt(width_pt), Pt(height_pt))
    tbl = gf.table
    for ci, cw in enumerate(col_widths_pt):
        tbl.columns[ci].width = Pt(cw)
    hdr_h = max(22.0, min(26.0, height_pt * 0.22))
    row_h = (height_pt - hdr_h) / max(1, len(rows))
    tbl.rows[0].height = Pt(hdr_h)
    for ri in range(len(rows)):
        tbl.rows[ri + 1].height = Pt(row_h)

    for ci, htxt in enumerate(headers):
        c = tbl.cell(0, ci)
        style_table_cell(c, htxt, size_pt=font_size_pt, bold=True, color=WHITE, bg=header_color, align=PP_ALIGN.CENTER)
        set_cell_border_thin(c)

    for ri, row in enumerate(rows):
        bg = WHITE if ri % 2 == 0 else ROW_ALT
        for ci, val in enumerate(row):
            c = tbl.cell(ri + 1, ci)
            fg = INK
            b = ci == 0
            if highlight_col_fn is not None:
                fg, b = highlight_col_fn(ri, ci, val, fg, b)
            al = PP_ALIGN.LEFT if ci in (0, ncols - 1) else PP_ALIGN.CENTER
            style_table_cell(c, val, size_pt=font_size_pt, bold=b, color=fg, bg=bg, align=al)
            set_cell_border_thin(c)


def update_existing_table(
    table,
    rows_data: list[list[str]],
    font_size_pt: float = 9.0,
    highlight_last_col: bool = True,
) -> None:
    for ri, row in enumerate(rows_data):
        is_hdr = ri == 0
        bg = BLUE if is_hdr else (WHITE if ri % 2 == 1 else ROW_ALT)
        for ci, val in enumerate(row):
            c = table.cell(ri, ci)
            fg = WHITE if is_hdr else INK
            bold = is_hdr or ci == 0
            if not is_hdr and highlight_last_col and ci == len(row) - 1:
                if "止损" in val or "G1–G4" in val or "[未过线]" in val:
                    fg = RED
                    bold = True
                elif "0 解码" in val or "[已达标]" in val or "[过线]" in val:
                    fg = BLUE
                    bold = True
            al = PP_ALIGN.CENTER if is_hdr else (PP_ALIGN.LEFT if ci in (0, len(row) - 1) else PP_ALIGN.CENTER)
            style_table_cell(c, val, size_pt=font_size_pt, bold=bold, color=fg, bg=bg, align=al)


def set_slide_title(slide, title_text: str, subtitle_text: str | None = None) -> None:
    sh3 = shape_by_name(slide, "TextBox 3")
    sh3.left = Pt(194)
    sh3.top = Pt(10)
    sh3.width = Pt(734)
    sh3.height = Pt(31)
    sh3.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    t_size = 18.0 if len(title_text) > 29 else 19.2
    replace_plain_text(sh3, title_text, size_pt=t_size, bold=True, color=INK)
    if subtitle_text is not None:
        sh4 = shape_by_name(slide, "TextBox 4")
        replace_plain_text(sh4, subtitle_text, size_pt=10.2, bold=False, color=MUTED)


def update_footer(slide, page_no: int, total: int = 22) -> None:
    for sh in slide.shapes:
        if sh.name == "TextBox 7" and sh.has_text_frame:
            replace_plain_text(sh, FOOTER_TEXT, size_pt=8.5, bold=False, color=SLATE)
        elif sh.name == "TextBox 8" and sh.has_text_frame:
            replace_plain_text(sh, f"{page_no:02d} / {total:02d}", size_pt=9.0, bold=True, color=SLATE)


def edit_f01_cover(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "西安电子科技大学 · 综合工程设计结题答辩")
    replace_plain_text(
        shape_by_name(slide, "Rounded Rectangle 5"),
        "项目组信息与结题总交付概览（2026年9月–12月 · 证据截止 2026-11-14）",
    )
    set_text_lines(
        shape_by_name(slide, "TextBox 6"),
        [
            [
                ("汇报团队：", True, INK),
                ("陈德昕 (23009200711，负责人/算法总装)、李世显 (23009201269，空间规划/数据工程)、陈龙辉 (23009200692，平台/云端工程)", False, INK),
            ],
            [
                ("指导教师：", True, INK),
                ("王笛 老师   |   课程：综合工程设计 / 大数据工程   |   交付：", False, MUTED),
                ("终稿 v5 (40页) + 双便携台全绿封存", True, BLUE),
            ],
            [
                ("科研成果：", True, BLUE),
                ("① 0.93M 航拍 M0 解耦架构 (Masa S3 86.80%)；② 9km² 连续区域三步径向规划 (68.04%，十区全正增益)。", False, INK),
            ],
            [
                ("证伪闭环：", True, TEAL),
                ("③ OpenCVL CV-048~051 跨视角严密证伪 (dev_val 24.14%，28对盲测 0 解码)；④ V5.0+Lite 双台封存。", False, INK),
            ],
        ],
        size_pt=9.5,
        space_after_pt=3.2,
        line_spacing=1.18,
    )
    set_text_lines(
        shape_by_name(slide, "TextBox 18"),
        [
            [("V5.0+Lite", True, PURPLE)],
            [("双形态演示平台", True, INK)],
            [("192+192题·双封存", False, MUTED)],
        ],
        size_pt=10.0,
        align=PP_ALIGN.LEFT,
        space_after_pt=1.5,
    )
    for name, num_pt, color in [
        ("TextBox 9", 21.0, BLUE),
        ("TextBox 12", 21.0, TEAL),
        ("TextBox 15", 21.0, AMBER),
        ("TextBox 18", 18.5, PURPLE),
    ]:
        sh = shape_by_name(slide, name)
        for r in sh.text_frame.paragraphs[0].runs:
            r.font.size = Pt(num_pt)
            r.font.bold = True
            r.font.color.rgb = color
        if len(sh.text_frame.paragraphs) > 1:
            for r in sh.text_frame.paragraphs[1].runs:
                r.font.size = Pt(9.5)
                r.font.bold = True
        if len(sh.text_frame.paragraphs) > 2:
            for r in sh.text_frame.paragraphs[2].runs:
                r.font.size = Pt(8.5)
                r.font.color.rgb = MUTED


def edit_f02_outline(slide) -> None:
    set_slide_title(
        slide,
        "结题汇报总体脉络与科研/工程双线证据全景",
        "围绕“问题建模 → 航拍 M0 解耦 → 9km² 扩展 → OpenCVL 跨视角证伪与 GeoNav V5.0/Lite 交付”组织 22 页答辩",
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 10"), "一、问题定义、文献对比与设计边界（Slides 03–06）")
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 13"), "二、航拍 M0“探索—识别解耦”（Slides 07–11）")
    set_text_lines(
        shape_by_name(slide, "TextBox 14"),
        [
            "• 研发四阶段演进：从耦合过拟合 (74.4%) 到显式解耦 (89.33%) 的科研决策链",
            "• 0.93M 双分支深度拆解：Small256 (797,701) + 20 维 Edge 线索头 (136,837)",
            "• 真实任务实例解剖：Masa S2 val_000_b10_g13_s04 跨拼缝动态决策实录",
            "• 四条件消融与对标：16,200 轨迹目标消融、t=1..10 曲线及领先 GeoExplorer +51.20%",
        ],
        size_pt=10.5,
        space_after_pt=5.0,
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 16"), "三、9km² 连续区域三步覆盖规划（Slides 12–13）")
    set_text_lines(
        shape_by_name(slide, "TextBox 17"),
        [
            "• 大尺度打转剖析：10×10 (300m/格，9km²) 稀疏视野困境与 21.29% 首步同向翻转",
            "• 免重训创新机制：Coverage3Radial（H=3 前瞻规划 + 公开起点径向非递减保护）",
            "• 十区独立确认：2,250 条严格同题配对轨迹，SR 58.84%→68.04%（10/10 全正增益）",
            "• 异源扩展守门边界：MassGIS 5×5 达 85.47% 过线，10×10 (54.62%) / 15×15 如实封存",
        ],
        size_pt=10.5,
        space_after_pt=5.0,
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 19"), "四、OpenCVL 证伪、GeoNav 与收尾（Slides 14–22）")
    set_text_lines(
        shape_by_name(slide, "TextBox 20"),
        [
            "• 跨视角六阶段排障与 192 基线（S14）：CV001–047 负结果闭环与 Frontier 71.88% 警示",
            "• OpenCVL 阶段一与二实证（S15–16）：SE 942→435/116/28 隔离、两大陷阱与 G1–G4 全否止损",
            "• GeoNav V5.0 + Lite（S17–19）：三层熔断架构、31B API 537.08s 对比与双便携预演全绿",
            "• 团队协同、客观边界与结题封存（S20–22）：终稿 v5 + 双包 SHA 封存与现场 3 分钟速查",
        ],
        size_pt=10.5,
        space_after_pt=5.0,
    )


def edit_f03_problem(slide) -> None:
    set_slide_title(
        slide,
        "复杂工程问题：从“静态全库检索”到“主动探索定位”",
        "聚焦局部观测稀疏、跨块纹理歧义与地面—航拍视角鸿沟三大关键工程难题",
    )


def edit_f04_pomdp(slide) -> None:
    set_slide_title(
        slide,
        "POMDP 序列决策数学建模与“零真值泄漏”沙箱契约",
        "建立严格的离散序列 POMDP 数学契约与“零真值泄漏”硬件级沙箱隔离机制（覆盖航拍 5×5–15×15 与 OpenCVL 25 宫格）",
    )
    set_text_lines(
        shape_by_name(slide, "TextBox 11"),
        [
            [("状态空间 S：", True, INK), ("N×N 离散网格 (5×5 / 10×10 / 15×15)，格心 p_t=(r_t,c_t)，未知目标 p*；", False, INK)],
            [("动作空间 A：", True, INK), ("四方向步进 {上, 右, 下, 左}，越界停原地耗一步（航拍无私设 stop）；", False, INK)],
            [("预算与目标指标：", True, INK), ("SR（计划题分母）与终局曼哈顿距离 SG（格或米）；首次/末步达即判成功；", False, INK)],
            [("跨视角软核拓展：", True, BLUE), ("OpenCVL 单步回归采用 25 宫格离散分布与 σ=5.0 m 高斯软标签及 KL 散度。", False, INK)],
        ],
        size_pt=9.6,
        space_after_pt=3.0,
    )
    set_text_lines(
        shape_by_name(slide, "TextBox 14"),
        [
            [("[合法] 策略公开输入：", True, TEAL), ("当前已访问图块 RGB / 冻结特征、给定目标图特征、相对起点公开位移、历史已访掩码、剩余步数。", False, INK)],
            [("[禁入] 私有真值隔离：", True, RED), ("目标真实坐标 p*、真实曼哈顿距离 d_t、目标网格索引 goal_index、未访问图块特征与源身份标签。", False, INK)],
            [("[审计] 首动作前 SHA-256 绑定：", True, BLUE), ("所有确认实验在首步执行前锁定任务清单、模型权重、训练集均值与源使用账本。", False, INK)],
        ],
        size_pt=9.5,
        space_after_pt=4.0,
    )


def edit_f05_literature(slide) -> None:
    set_slide_title(
        slide,
        "前沿文献机制剖析 [1]–[13] 与 6,000 条本地受控对标",
        "不盲从顶会宣称：面向 [1] GOMAA-Geo / [2] GeoExplorer / [3] DynCur-Geo 与跨视角 [12] OpenCVL / [13] CCVPE 建立统一基线评测",
    )
    tbl = shape_by_name(slide, "Table 18").table
    update_existing_table(
        tbl,
        [
            ["对比方案 / 文献基线", "可训练参数量", "真实目标 SR", "禁用目标 SR", "终局距离 SG", "本地受控实验结论与取舍判定"],
            ["GeoExplorer 有限预算 (S3)", "~2.10M (含ICM头)", "35.60%", "35.33%", "2.589 格", "Wrong(36.80%) 反超 Full，全项干预门槛未过，淘汰"],
            ["耦合目标通道 (Coupled, S2)", "0.85M (单分支)", "74.40%", "73.00%", "0.690 格", "相对 S2 禁用仅 +1.40%，显著弱于解耦 89.33%，淘汰"],
            ["本项目 M0 (Small256+Edge)", "0.93M (双分支解耦)", "86.80% (S3)", "73.60% (S3)", "0.300 格", "真实目标净增 +13.20%，领先 GeoExplorer +51.20pp"],
        ],
        font_size_pt=8.7,
        highlight_last_col=False,
    )


def edit_f06_design(slide) -> None:
    set_slide_title(
        slide,
        "阶段设计目标矩阵与非技术工程因素（算力/安全/伦理）",
        "将环境可持续性、系统安全熔断与科研诚实伦理转化为贯穿全周期的可核验工程约束",
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 10"), "[算力] 环境与可持续发展（绿色低碳算力）")
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 13"), "[安全] 健康与系统安全（API 硬熔断与脱敏）")
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 16"), "[合规] 法律合规与学术伦理（防篡改账本）")
    set_text_lines(
        shape_by_name(slide, "TextBox 14"),
        [
            [("请求账本预写（RequestLedger）：", True, INK), ("GeoNav V5.0 调用云端 API 前必先落盘请求账本，全程可追溯；", False, INK)],
            [("0 自动重试 + 3 错硬熔断：", True, INK), ("超时/配额错误实时计数，连续 3 错硬切断并回退本地 M0；.env 与正文脱敏。", False, INK)],
        ],
        size_pt=9.5,
        space_after_pt=3.0,
    )
    tbl = shape_by_name(slide, "Table 18").table
    update_existing_table(
        tbl,
        [
            ["核心研究与工程主线", "课程既定结题目标", "结题实证最终成果（2026-11-14）", "终验判定与适用边界"],
            ["主线 1：航拍同模态主动导航", "建立超轻量解耦模型，通过多批次独立源确认", "M0 (0.93M) 通过 Masa S2(89.33%)、S3(86.80%)、S4(77.13%) 及 SwissView(86.80%)", "[已达标] 平台默认主策略"],
            ["主线 2：9km² 连续大区域扩展", "在 10×10 (300m/格) 连续区超越单步 M0", "Coverage3Radial 在 10 独立新区 (2,250轨迹) 将 SR 从 58.84% 升至 68.04% (+9.20%)", "[已达标] 10/10 全正(含边界)"],
            ["主线 3：地面照片跨视角定位", "严密评估地面→航拍关联并严防统计泄露", "完成 CV001–CV051 六体系排查：OpenCVL dev_val 24.14% 低于先验 42.24%，G1–G4 全否", "[诚实止损] 28对盲测 0 解码"],
            ["主线 4：GeoNav 双形态交互平台", "可配置探索、人工接管、API 对照与便携打包", "V5.0 (8769) + Lite (8768) 12/12 端到端通过，11.14 实机全绿，输出双纯净便携包", "[已达标] 终验全绿(537.08s)"],
        ],
        font_size_pt=8.8,
        highlight_last_col=True,
    )


def edit_f08_m0_arch(slide) -> None:
    set_slide_title(
        slide,
        "M0“探索—识别显式解耦”与 20 维拼缝门控架构",
        "双分支参数完全显式解耦（934,538 参数）：冻结骨干 + Small256 循环探索 + 20 维拼缝门控",
    )


def edit_f09_m0_case(slide) -> None:
    set_slide_title(
        slide,
        "真实任务 (val_000_b10_g13_s04) 探索与门控决策解剖",
        "不仅看宏观统计指标，更深入单步真实航拍任务解剖 M0 如何在第 2 步跨拼缝锁定目标",
    )


def edit_f10_s3_ablation(slide) -> None:
    tbl = shape_by_name(slide, "Table 12").table
    for ci, w_pt in enumerate([74.0, 50.5, 50.5, 50.5, 50.5, 59.7]):
        tbl.columns[ci].width = Pt(w_pt)
    for ri, row in enumerate(tbl.rows):
        for ci, cell in enumerate(row.cells):
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if (ri == 0 or ci > 0) else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.name = FONT_CN
                    if ri > 0 and ci == 5:
                        r.font.size = Pt(8.6)
                    elif ri > 0 and ci == 0:
                        r.font.size = Pt(8.4)


def edit_f11_merged_s11_s12(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "02 | 步数效率与参数消融")
    set_slide_title(
        slide,
        "航拍步数效率、6,000 轨迹对标与 0.93M 参数拆解",
        "合并呈现 Masa S2 七策略到达曲线与 d0 分层（上），以及 Masa S3 独立源确认与 0.93M 解耦参数消融（下）",
    )
    remove_shapes_by_name(
        slide,
        ["Rounded Rectangle 9", "Picture 10", "TextBox 11", "Rounded Rectangle 12", "Rounded Rectangle 13", "TextBox 14"],
    )

    add_white_card(slide, 32, 78, 598, 206)
    add_picture_fit(slide, ROOT / "绘图" / "数据结果图" / "02_MasaS2七策略逐步到达动态曲线与初始距离分层.png", 38, 82, 586, 196)
    add_card_with_header(
        slide,
        640,
        78,
        287,
        206,
        "Masa S2 逐步效率与 Edge 净增益",
        BLUE,
        [
            [("1. Edge 净贡献 +8.33%：", True, BLUE), ("M0 (89.33%, SG 0.320 格) 显著超越移除手工边缘线索的 ZeroEdge (81.00%, SG 0.567)。", False, INK)],
            [("2. 半程预算 (t=5) 过半：", True, INK), ("第 5 步即完成 54.00% 任务，前 6 步斜率陡峭收敛，领先耦合对照 14+ 个百分点。", False, INK)],
            [("3. 远距离抗衰减 (d0≥5)：", True, TEAL), ("即便初始距离 d0=6..8 格，M0 仍保持 80%+ 到达率，避免局部打转。", False, INK)],
        ],
        body_size_pt=9.2,
    )

    add_white_card(slide, 32, 294, 598, 208)
    add_picture_fit(slide, ROOT / "绘图" / "数据结果图" / "05_M0双分支超轻量化参数解耦与跨图块边界线索消融.png", 38, 304, 586, 187)
    add_card_with_header(
        slide,
        640,
        294,
        287,
        208,
        "Masa S3 独立源与 0.93M 参数拆解",
        TEAL,
        [
            [("1. 超轻量精确配比：", True, TEAL), ("Small256 (797,701) + EdgeHead (136,837) = 934,538 参数，较重型基线缩减 >98.5%。", False, INK)],
            [("2. S3 独立源与跨区迁移：", True, INK), ("Masa S3 达 86.80% / SG 0.300 格；SwissView 跨大洲迁移同样取得 86.80% / SG 0.381 格。", False, INK)],
            [("3. 6,000 轨迹同题领先：", True, BLUE), ("同题超越复现学习基线 GeoExplorer (35.60% / 2.589格) +51.20 个百分点，无负向损伤。", False, INK)],
        ],
        body_size_pt=9.2,
    )


def edit_f13_merged_s14_s15(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "03 | 9km² 确认与异源边界")
    set_slide_title(
        slide,
        "MasaRoads 9km² 十区全正增益与 MassGIS 异源边界",
        "2,250 条同题配对轨迹确认（上）与 MassGIS 5×5/10×10/15×15 实证守门及 21.29% 首步翻转诊断（下）",
    )
    remove_shapes_by_name(
        slide,
        ["Rounded Rectangle 9", "Picture 10", "TextBox 11", "Table 12", "Rounded Rectangle 13", "Rounded Rectangle 14", "TextBox 15"],
    )

    add_white_card(slide, 32, 78, 560, 200)
    add_picture_fit(slide, ROOT / "绘图" / "数据结果图" / "04_9平方公里连续区域13级距离衰减与十区配对增益.png", 38, 83, 548, 188)
    add_styled_table(
        slide,
        602,
        78,
        325,
        200,
        ["MasaRoads 分层", "M0 基线", "Cov3Radial", "配对净升 (95% CI)"],
        [
            ["总计 (2,250轨迹)", "58.84% (755m)", "68.04% (649m)", "+9.20% [+7.1,+11.0]"],
            ["跨块拼缝 Seam", "56.18% (812m)", "67.91% (661m)", "+11.73% (最高增益)"],
            ["网格中心 Center", "61.51% (698m)", "68.18% (636m)", "+6.67% [+4.2,+9.2]"],
            ["远距离 Long≥7", "54.40% (846m)", "62.67% (782m)", "+8.27% (恢复+207题)"],
        ],
        col_widths_pt=[90, 68, 82, 85],
        header_color=BLUE,
        font_size_pt=8.2,
    )

    add_styled_table(
        slide,
        32,
        288,
        560,
        214,
        ["MassGIS 异源协议", "网格/预算", "M0 → Cov3Radial SR", "禁用 / 错误对照", "预登记门槛判定"],
        [
            ["MassGIS 5×5 新域\n(750轨/10新源)", "5×5, B=10", "85.47% / 0.341格\n(约 109.2 米)", "禁用 73.73%\n错误 70.13%", "[过线] >60% 门槛\n跨源同尺度有效"],
            ["MassGIS 10×10 连续区\n(2,250轨/9km²)", "10×10, B=20", "46.84% → 54.62%\n(+7.78%, 847.2m)", "禁用 43.11%\n错误 41.82%", "[边界] <60% 硬门槛\n保留为负向边界"],
            ["MassGIS 15×15 超大网\n(225格极限扩展)", "15×15, B=30", "16.40% → 20.27%\n(+3.87%, 1,542m)", "禁用 14.80%\n错误 14.13%", "[边界] 225格稀释\n不外推超纲声称"],
        ],
        col_widths_pt=[118, 72, 126, 98, 146],
        header_color=TEAL,
        font_size_pt=8.3,
        highlight_col_fn=lambda r, c, v, fg, b: (
            (BLUE, True) if c == 4 and "[过线]" in v else ((AMBER, True) if c == 4 and "[边界]" in v else (fg, b))
        ),
    )

    add_card_with_header(
        slide,
        602,
        288,
        325,
        214,
        "异源衰减归因与科学守门原则",
        AMBER,
        [
            [("1. 区域全正与净恢复：", True, INK), ("MasaRoads 10/10 独立区全正增益，恢复 295 题 vs 损伤 88 题（净 +207 题，-106.67 米）。", False, INK)],
            [("2. 异源漂移与 21.29% 翻转：", True, AMBER), ("MassGIS 存在色差/GSD 漂移且同向双模态首步翻转率达 21.29%；15×15 局部视野触及目标四邻域仅 1.78%。", False, INK)],
            [("3. 科学守门纪律：", True, BLUE), ("严守预登记 60% 门槛，不事后放宽阈值或挑分，未达标大尺度异源批次原封保留。", False, INK)],
        ],
        body_size_pt=8.9,
    )


def edit_f14_merged_s16_s17(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "04 | 跨视角六阶段排障")
    set_slide_title(
        slide,
        "地面—航拍几何鸿沟、六阶段排查与 192 题基线",
        "系统厘清“可见地标格 ≠ 拍摄点格”与连续回归高方差根因，并在 192 道合法任务确立 Frontier 71.88% (SG 1.099) 对照门槛",
    )
    remove_shapes_by_name(slide, ["Rounded Rectangle 9", "Picture 10", "TextBox 11"])

    add_white_card(slide, 32, 78, 444, 212)
    add_picture_fit(slide, ROOT / "绘图" / "架构与方法图" / "04_跨视角地面照片与俯视航拍几何鸿沟诊断图.png", 38, 82, 432, 186)
    add_caption_box(slide, 38, 268, 432, 19, "图 4 地面水平街景 vs 俯视正射网格视角鸿沟与 CV001–047 诊断示意", 8.2)

    add_white_card(slide, 32, 298, 444, 202)
    add_picture_fit(slide, ROOT / "绘图" / "数据结果图" / "07_跨视角六阶段演进与192题多控制器实测覆盖对比.png", 38, 304, 432, 144)
    add_caption_box(
        slide,
        38,
        452,
        432,
        44,
        "CV001–047 精度演进与 192 题 4 规则对比：Frontier 71.88% (SG 1.099 格)、CenterCross 65.63%、Lawnmower 52.60%、Random 22.92%",
        8.2,
    )

    add_styled_table(
        slide,
        486,
        78,
        441,
        268,
        ["阶段批次", "模型/参数", "核心实测结果", "关键排障结论"],
        [
            ["CV002–005 CLIP 零样本", "ViT-B/32 (16对)", "Top-1 = 0/16 (0.0%)", "前景地标格 ≠ 拍摄格，方向反指"],
            ["CV008–019 弱代理对齐", "8K 参数 (125排序)", "Top-1 = 0/125 (0.0%)", "9–12 对小样本对未见源零泛化"],
            ["CV022–023 Sample4Geo", "87.57M (1,720次)", "Top-1 = 1/5 (scene 2/5)", "街景中心先验在任意窗中失效"],
            ["CV030–036 CCVPE 迁移", "60.42M (448次前向)", "≤5m 0/4 (均差26.77m)", "全参考仍差 26.77m，表征断层"],
            ["CV037–041 RGB 空间头", "65.5K (6,750条)", "≤5m 0/9 (上界 3/9<6/9)", "留出源固定输出停在 3/9，止损"],
            ["CV042–047 F1 192题", "4 种规则 (5×5/B10)", "Frontier 71.88% (1.099)", "接通合法队列并揭示中心偏置"],
        ],
        col_widths_pt=[106, 86, 108, 141],
        header_color=BLUE,
        font_size_pt=8.1,
    )

    add_card_with_header(
        slide,
        486,
        356,
        441,
        144,
        "192 题“中心偏置”警示与立项新规（引出 CV-048~051）",
        RED,
        [
            [("• 为什么无目标 Frontier 即达 71.88%？", True, RED), ("F1 拍摄点在 5×5 窗内天然靠中心聚集，盲目优先覆盖中心邻域规则便可得高分（SG 1.099 格）。", False, INK)],
            [("• 促成严格隔离新协议：", True, BLUE), ("为此我们在 OpenCVL 上立项 CV-048~051，强制地理隔离 ≥95.2 km 并设置无图像先验及视觉均值四条件红线。", False, INK)],
        ],
        body_size_pt=9.1,
        fill_color=SOFT_RED,
    )


def edit_f15_opencvl_stage1(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "04 | OpenCVL 隔离协议")
    set_slide_title(
        slide,
        "OpenCVL CV-048~050：地理隔离协议与阶段一陷阱",
        "从源池 SE (N=942) 剔除空间重合簇，构建严格地理隔离三切分，并在零训练阶段锁定两道统计防线",
    )
    remove_shapes_by_name(slide, ["Table 9"])
    add_styled_table(
        slide,
        32,
        80,
        895,
        198,
        ["隔离切分 / 过滤环节", "样本规模与图块量", "空间最小地理间距", "关键分布统计特征", "契约状态与访问红线"],
        [
            ["SE 原始审计与重叠剔除", "942 → 591 (禁用 351 重叠 / 12 cluster)", "仅保留互不相交地理簇", "排除共享街区/跨源重叠建筑影像", "[封存] 永久隔离，禁跨源混用"],
            ["fit_train 城内训练集", "435 配对样本（3,795 图像块）", "同城/城内开发簇", "16:9 画幅占比仅 40.5%（偏 4:3）", "仅作阶段二训练与先验核拟合"],
            ["dev_val 跨城验证集", "116 配对（25 宫格软目标）", "≥ 95.203 km（相对训练集）", "16:9 画幅飙升至 87.9%（102/116）", "唯一用于阶段二门槛 G1–G4 裁决"],
            ["isolated_confirmation_blind", "28 配对（独立封装盲测区）", "≥ 109.465 km（全隔离）", "阶段二未过线即永久封存", "[锁定] 0 解码 / 0 前向 / 0 评测"],
        ],
        col_widths_pt=[155, 190, 165, 195, 190],
        header_color=BLUE,
        font_size_pt=8.7,
        highlight_col_fn=lambda r, c, v, fg, b: (
            (RED, True) if r == 3 and c == 4 else ((BLUE, True) if c == 4 else (fg, b))
        ),
    )

    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 11"), "陷阱一：画幅比例混杂统计假象（952×714 vs 952×532）")
    set_text_lines(
        shape_by_name(slide, "TextBox 12"),
        [
            [("1. 跨切分画幅协变量偏移：", True, INK), ("fit_train 中仅 40.5% 为 16:9 (952×532)，而跨城 dev_val 中高达 87.9% 为 16:9（其余为 4:3 952×714）。", False, INK)],
            [("2. 分层掉点揭示伪提升：", True, RED), ("M2c 在 4:3 子集 (n=14) 得 42.86%，但在主流 16:9 子集 (n=102) 骤跌至 21.57%！单看混合均值会被分布结构掩盖。", False, INK)],
        ],
        size_pt=9.6,
        space_after_pt=4.0,
    )

    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 14"), "陷阱二：零图像先验地板与高斯软标签核（σ=5.0 m）")
    set_text_lines(
        shape_by_name(slide, "TextBox 15"),
        [
            [("1. 不看图像已达 42.24%：", True, BLUE), ("AlwaysCell12Prior（恒猜中心格 12）在 dev_val 取得 42.24% / 12.51 m；TrainEmpiricalPrior 亦达 42.24% 且 KL=1.420。", False, INK)],
            [("2. 冻结四道准入门槛 G1–G4：", True, INK), ("确立 σ=5.0 m 高斯核与全套评测防线，要求模型必须同时在 Top-1、MeanDist、KL 上超越先验及视觉均值消融。", False, INK)],
        ],
        size_pt=9.6,
        space_after_pt=4.0,
    )


def edit_f16_opencvl_stage2(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "04 | OpenCVL 阶段二实证")
    set_slide_title(
        slide,
        "OpenCVL CV-051：阶段二实证与 G1–G4 全否止损",
        "1,653 次单编码器受控前向完整证伪：城内拟合 68.74% 跨城跌至 24.14%，弱于无图像先验（42.24%）与均值视觉消融",
    )
    remove_shapes_by_name(
        slide,
        ["Rounded Rectangle 9", "Picture 10", "TextBox 11", "Table 12", "Rounded Rectangle 13", "Rounded Rectangle 14", "TextBox 15"],
    )

    add_white_card(slide, 32, 76, 895, 276)
    add_picture_fit(slide, ROOT / "绘图" / "数据结果图" / "08_OpenCVL阶段二单步视觉定位与消融实证对比.png", 46, 79, 867, 255)
    add_caption_box(
        slide,
        46,
        334,
        867,
        16,
        "图 8  OpenCVL CV-051 阶段二 (a) 跨城 dev_val vs 城内 fit_train Top-1 准确率与消融对比；(b) dev_val (N=116) 期望平面误差与软标签 KL 散度分解",
        8.2,
    )

    add_styled_table(
        slide,
        32,
        359,
        544,
        145,
        ["模型 / 消融条件", "fit_train Top-1", "dev_val Top-1", "MeanDist (m)", "软标签 KL", "门槛判定"],
        [
            ["AlwaysCell12Prior", "32.41%", "42.24%", "12.51", "5.60", "恒中心先验地板"],
            ["TrainEmpiricalPrior", "32.41%", "42.24%", "21.40", "1.420", "经验分布 KL 地板"],
            ["M1b 零样本 Center9", "—", "5.17%", "25.56", "1.99", "零样本跨城失效"],
            ["M2c 多尺度校准(主选)", "68.74%", "24.14%", "22.18", "1.730", "[未过线] 跌 44.6pp"],
            ["Control_MeanVisual", "—", "42.24%", "21.78", "1.524", "[未过线] 反超真视觉"],
        ],
        col_widths_pt=[124, 72, 72, 74, 68, 134],
        header_color=BLUE,
        font_size_pt=8.1,
        highlight_col_fn=lambda r, c, v, fg, b: ((RED, True) if c == 5 and "[未过线]" in v else (fg, b)),
    )

    add_card_with_header(
        slide,
        586,
        359,
        341,
        145,
        "阶段二停损裁决：G1–G4 全否，28 对盲测 0 解码",
        RED,
        [
            [("1. 受控算力账本：", True, INK), ("共 1,653 次编码器前向（max_live_encoders=1），全程零真值泄露、零刷榜。", False, INK)],
            [("2. 双重证伪铁证：", True, RED), ("M2c 跨城 Top-1 (24.14%) 显著低于先验地板 18.10pp，且弱于自身视觉均值消融 MeanVisual (42.24% / KL 1.524)，GroundZeroed (14.7%) 证明背景先验主导。", False, INK)],
            [("3. 严格执行停损：", True, BLUE), ("触发 STOP_AT_STAGE2_NEGATIVE，直接转入阶段四收尾，28 对盲测永久保持 0 解码 / 0 评测！", False, INK)],
        ],
        body_size_pt=8.4,
        header_h_pt=24.0,
        fill_color=SOFT_RED,
    )


def edit_f17_platform_arch(slide) -> None:
    set_slide_title(
        slide,
        "现代工具研发：GeoNav V5.0 + Lite 三层解耦与熔断架构",
        "双端统一核服务体系：workbench_server_v28 (8769) + service_v17 + lite_server (8768)，全链路支持人工接管与 API 硬熔断",
    )


def edit_f18_api_vs_m0(slide) -> None:
    set_slide_title(
        slide,
        "工具局限性检验：0.93M 本地 M0 vs 31B 云端大模型对照",
        "用 96 次真实 API 请求与 537.08 秒耗时审计揭示通用 VLM 在离散网格中的“空间幻觉”局限",
    )
    tbl = shape_by_name(slide, "Table 12").table
    style_table_cell(tbl.cell(7, 3), "537.08秒 / 71,005", size_pt=9.0, bold=False, color=RED, bg=ROW_ALT, align=PP_ALIGN.CENTER)


def edit_f19_live_platforms(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "05 | V5.0 & Lite 实机演示")
    set_slide_title(
        slide,
        "GeoNav V5.0 (8769) + Lite (8768) 实机预演与便携封存",
        "2026-11-14 全离线无外网实机预演零报错通过：统一科研台、极简演示版与 CV041 证据显微镜同台闭环",
    )
    remove_shapes_by_name(
        slide,
        ["Rounded Rectangle 9", "Picture 10", "TextBox 11", "Rounded Rectangle 12", "Rounded Rectangle 13", "TextBox 14"],
    )

    cards = [
        (
            32,
            REVIEW_DIR / "v5_rehearsal_navigator.png",
            "GeoNav V5.0 探索台 (端口 8769)：六大工作台一体化、192 底图(960题) + 192 地面点、小眼睛预览与四向倾向度透视",
        ),
        (
            335,
            REVIEW_DIR / "lite_screen_demo.png",
            "GeoNav Lite 极简台 (端口 8768)：单屏高清演示优化、自定义 3×3~10×10 纯内存上传切片与本地 M0 秒级寻路",
        ),
        (
            638,
            REVIEW_DIR / "v5_rehearsal_evidence.png",
            "CV041 空间证据显微镜：支持三城 150 案例逐一检视、7×7 热力图与 seal_sha256 追溯核验",
        ),
    ]
    for left_x, img_p, cap in cards:
        add_white_card(slide, left_x, 78, 290, 298)
        add_picture_fit(slide, img_p, left_x + 7, 84, 276, 184)
        tb = slide.shapes.add_textbox(Pt(left_x + 8), Pt(273), Pt(274), Pt(96))
        tb.text_frame.word_wrap = True
        set_text_lines(tb, [cap], size_pt=9.0, default_color=MUTED, align=PP_ALIGN.LEFT, space_after_pt=2.0)

    set_text_lines(
        shape_by_name(slide, "TextBox 17"),
        [[("12 / 12 全绿", True, BLUE)], [("V5 + Lite 端到端测试", True, INK)], [("延续 68 项基础单测护栏", False, MUTED)]],
        size_pt=9.8,
        space_after_pt=2.0,
    )
    set_text_lines(
        shape_by_name(slide, "TextBox 20"),
        [[("0 报错预演", True, TEAL)], [("11-14 实机双端口核验", True, INK)], [("8769/8768 离线零控制台异常", False, MUTED)]],
        size_pt=9.8,
        space_after_pt=2.0,
    )
    set_text_lines(
        shape_by_name(slide, "TextBox 23"),
        [[("3 错硬熔断", True, AMBER)], [("DeepSeek/VLM 网关", True, INK)], [("先账本后请求 + thinking禁用", False, MUTED)]],
        size_pt=9.8,
        space_after_pt=2.0,
    )
    set_text_lines(
        shape_by_name(slide, "TextBox 26"),
        [[("2 净便携包", True, PURPLE)], [("中期 + 小型演示 Zip", True, INK)], [("360.86 / 292.01 MiB · 0 .env", False, MUTED)]],
        size_pt=9.8,
        space_after_pt=2.0,
    )
    for nm, c in [("TextBox 17", BLUE), ("TextBox 20", TEAL), ("TextBox 23", AMBER), ("TextBox 26", PURPLE)]:
        sh = shape_by_name(slide, nm)
        for r in sh.text_frame.paragraphs[0].runs:
            r.font.size = Pt(16.0)
            r.font.bold = True
            r.font.color.rgb = c
        for r in sh.text_frame.paragraphs[1].runs:
            r.font.size = Pt(9.5)
            r.font.bold = True
        for r in sh.text_frame.paragraphs[2].runs:
            r.font.size = Pt(8.5)
            r.font.color.rgb = MUTED


def edit_f20_team(slide) -> None:
    set_slide_title(
        slide,
        "团队成员角色定位、个人量化贡献与双实现独立复核",
        "权责清晰的角色矩阵、学号确认贡献、契约式交接簿（TASKS/HANDOFF）与双实现独立复核",
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 10"), "陈德昕 (23009200711) · 负责人/算法总装")
    set_text_lines(
        shape_by_name(slide, "TextBox 11"),
        [
            [("角色定位：", True, INK), ("总体路线设计、M0 解耦架构、OpenCVL 协议冻结与关键止损。", False, INK)],
            [("代表源码：", True, BLUE), ("edge_cue.py, decoupled_cue.py, opencvl_single_step_visual_v1.py", False, MUTED)],
            [("个人量化：", True, TEAL), ("• 主导 0.93M M0 实现 S2 89.33% / S3 86.80%；\n• 完成 GeoExplorer 6,000 轨迹同题对标；\n• 主导 CV-048~051 全否止损与终稿 v5 统稿。", False, INK)],
        ],
        size_pt=9.5,
        space_after_pt=3.0,
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 13"), "李世显 (23009201269) · 空间规划/数据工程")
    set_text_lines(
        shape_by_name(slide, "TextBox 14"),
        [
            [("角色定位：", True, INK), ("大区域覆盖规划、多源遥感/街景切分管线与双实现独立复核。", False, INK)],
            [("代表源码：", True, TEAL), ("spatial_area_confirmation.py, opencvl_wild_intake_v1.py, audit_opencvl_single_step_visual_v1.py", False, MUTED)],
            [("个人量化：", True, TEAL), ("• 提出 Coverage3Radial，9km² 升至 68.04% (+207题)；\n• 建立 SE 942→435/116/28 地理隔离切分；\n• 执行 OpenCVL & 终稿双实现独立复核 (<1e-15)。", False, INK)],
        ],
        size_pt=9.5,
        space_after_pt=3.0,
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 16"), "陈龙辉 (23009200692) · 平台与云端工程")
    set_text_lines(
        shape_by_name(slide, "TextBox 17"),
        [
            [("角色定位：", True, INK), ("GeoNav V5.0 + Lite 全栈研发、云端熔断网关与科研出图交付。", False, INK)],
            [("代表源码：", True, PURPLE), ("workbench_server_v28.py, service_v17.py, lite_server.py", False, MUTED)],
            [("个人量化：", True, PURPLE), ("• 交付 V5.0 (8769) + Lite (8768) 及 12/12 端到端全绿；\n• 完成 537.08s 31B API 审计与 DeepSeek 熔断；\n• 输出双纯净便携包及全套科研图表/PPT 封存。", False, INK)],
        ],
        size_pt=9.5,
        space_after_pt=3.0,
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 19"), "团队契约化协作与零误差核验三大工程护栏")
    set_text_lines(
        shape_by_name(slide, "TextBox 20"),
        [
            [("1. 单帧契约交接：", True, INK), ("写入前在 TASKS.md 认领独占路径，交接写入单帧 HANDOFF.md 登记 SHA-256、进程归零与恢复入口。", False, INK)],
            [("2. 双实现独立核验：", True, BLUE), ("所有指标均由独立 audit_*.py 从原始 JSONL 轨迹重算核验，OpenCVL 双实现偏差 < 10^-15、缓存零漂移。", False, INK)],
            [("3. 基于实证止损：", True, RED), ("对 GeoExplorer、CCVPE 及 OpenCVL Stage 2 等未达预设门槛分支果断冻结并如实报告，拒绝事后挑分。", False, INK)],
        ],
        size_pt=9.3,
        space_after_pt=2.5,
    )


def edit_f21_boundaries_future(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "06 | 能力边界与展望")
    set_slide_title(
        slide,
        "客观能力边界、跨视角负结果归因与后续研究方向",
        "坚持科学诚实原则：明确界定本系统已验证生效域、不可靠边界及面向端到端空间定位的下一步演进路径",
    )

    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 10"), "边界 1：航拍主方案生效域与尺度上界")
    set_text_lines(
        shape_by_name(slide, "TextBox 11"),
        [
            [("[生效域] 已验证适用范围：", True, BLUE), ("Masa S2 (89.33%)、S3 (86.80%)、S4 (77.13%)、SwissView (86.80%)、MassGIS 5×5 (85.47%) 与 9km² 连续区域 (68.04%，10/10 全正)。", False, INK)],
            [("[边界线] 严禁无据外推：", True, AMBER), ("• 不可直接外推为 MassGIS 10×10 (54.62%) / 15×15 (20.27%) 超大异源网格；\n• 不可宣称具备真实无人机倾斜/强透视形变、跨季节时相或实时多传感器融合能力；\n• 冻结编码器存在 21.29% 首步同向翻转与色差漂移敏感度。", False, INK)],
        ],
        size_pt=9.7,
        space_after_pt=4.0,
    )

    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 13"), "边界 2：跨视角定位与云端 VLM 局限")
    set_text_lines(
        shape_by_name(slide, "TextBox 14"),
        [
            [("[未破线] 跨视角地面→航拍：", True, RED), ("CV001–CV051 六阶段（含 OpenCVL M2c dev_val 24.14%，KL 1.730）在地理隔离 ≥95.2 km 条件下均未超越无图中心先验（42.24%，12.51 m）与均值视觉消融。", False, INK)],
            [("深层归因：", True, INK), ("地面水平透视遮挡导致共享局部像素先验失效，小样本下模型易退化为依赖画幅占比 (4:3→16:9) 或数据集中心偏置。", False, INK)],
            [("[未破线] 通用云端 VLM 局限：", True, RED), ("31B 云大模型十题耗时 537.08s 仅 10% SR（54 次重访/24 次越界），证实缺乏亚格级连续性归纳偏置。", False, INK)],
        ],
        size_pt=9.6,
        space_after_pt=4.0,
    )

    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 16"), "后续两大科研攻关方向（Future Work）")
    set_text_lines(
        shape_by_name(slide, "TextBox 17"),
        [
            [("方向一：显式几何视角统一与 BEV 投影对齐", True, TEAL)],
            [("放弃静态先验重加权与启发式微调，引入极坐标/逆透视鸟瞰图（BEV）几何映射模块，在统一俯视空间完成跨视角特征对齐，并辅以画幅归一化去偏。", False, INK)],
            [("方向二：尺度不变拓扑坐标与粗精分级引导", True, BLUE)],
            [("针对 15×15 及更大地理范围，在保持绝对空间基准（≥100 km）零泄露前提下，构建粗粒度区域检索与精细边缘门控的分级耦合框架，缓解稀疏触达瓶颈。", False, INK)],
        ],
        size_pt=9.7,
        space_after_pt=4.5,
    )


def edit_f22_closure_qa(slide) -> None:
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 2"), "07 | 结题总结与现场演示")
    set_slide_title(
        slide,
        "结题成果全景总览、提交封存与 3 分钟演示卡",
        "航拍导航、大区域规划、跨视角严谨证伪与 V5.0/Lite 双便携平台圆满收官，支持全离线秒级现场演示",
    )
    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 10"), "本项目结题四大核心成果全景总结")
    set_text_lines(
        shape_by_name(slide, "TextBox 11"),
        [
            [("1. 0.93M 超轻量航拍 M0 解耦架构：", True, BLUE), ("通过 Small256 + 20 维 Edge 门控在 Masa S2 (89.33%)、S3 (86.80%)、S4 (77.13%) 及 SwissView (86.80%) 全面过线，6,000 轨迹同题领先 GeoExplorer +51.20pp。", False, INK)],
            [("2. 9km² 大区域 Coverage3Radial：", True, TEAL), ("免重训 H=3 前瞻与径向保护，2,250 轨迹确认将 SR 从 58.84% 升至 68.04%（10/10 区全正增益，净恢复 +207 题），同时严谨标注 MassGIS 10×10/15×15 边界。", False, INK)],
            [("3. OpenCVL 跨视角严密证伪：", True, RED), ("完成 CV001–CV051 排查（dev_val ≥95.203km 达 24.14% < 42.24% 零图地板），G1–G4 全否果断止损，28 对盲测保持 0 解码/0 评测。", False, INK)],
            [("4. GeoNav V5.0 + Lite 双形态台：", True, PURPLE), ("192+192 题库、内存多尺度切片、DeepSeek 防截断与 3 错熔断；12/12 端到端全绿，输出 360.86 / 292.01 MiB 双纯净便携包。", False, INK)],
        ],
        size_pt=9.6,
        space_after_pt=3.8,
    )

    replace_plain_text(shape_by_name(slide, "Rounded Rectangle 13"), "交付物清单与现场 3 分钟演示速查卡")
    set_text_lines(
        shape_by_name(slide, "TextBox 14"),
        [
            [("[1] 报告与答辩：", True, INK), ("总结报告_终稿_v5.docx/.pdf (40页·8图7表·17,717字) + 本答辩 PPT (22页) + 逐页讲稿。", False, INK)],
            [("[2] 一键演示入口：", True, BLUE), ("平台/启动平台.cmd (V5.0 :8769) & 启动极简演示平台.cmd (Lite :8768)，全离线零外网秒启动。", False, INK)],
            [("[3] 3 分钟现场动线：", True, TEAL), ("0–1分 V5 总览→预设1点“小眼睛”+单步看概率；1–2分 切 Lite 上传大图 5×5 实时切分；2–3分 V5 打擂(8/10 vs 1/10)+CV041热力图。", False, INK)],
            [("[4] 高频 Q&A 要点：", True, AMBER), ("① 为何弃用 31B API？10% vs 80%（537.08s 空间幻觉）；② 为何终止 OpenCVL？24.14% 不敌零图先验，坚守盲测零泄漏。", False, INK)],
        ],
        size_pt=9.2,
        space_after_pt=3.6,
    )

    set_text_lines(
        shape_by_name(slide, "Rounded Rectangle 15"),
        [
            [("感谢各位评委老师聆听！敬请批评指正，欢迎现场提问与实机点选演示！", True, INK)],
            [("汇报团队：陈德昕 (23009200711) · 李世显 (23009201269) · 陈龙辉 (23009200692)   |   指导教师：王笛 老师", False, MUTED)],
        ],
        size_pt=12.0,
        align=PP_ALIGN.CENTER,
        space_after_pt=4.0,
    )
    sh15 = shape_by_name(slide, "Rounded Rectangle 15")
    for r in sh15.text_frame.paragraphs[0].runs:
        r.font.size = Pt(15.0)
        r.font.bold = True
        r.font.color.rgb = INK
    for r in sh15.text_frame.paragraphs[1].runs:
        r.font.size = Pt(10.5)
        r.font.color.rgb = MUTED


def export_pdf_and_pngs(pptx_path: Path, pdf_path: Path, png_dir: Path) -> list[Path]:
    import win32com.client  # type: ignore

    png_dir.mkdir(parents=True, exist_ok=True)
    for old_png in png_dir.glob("slide_*.png"):
        old_png.unlink()

    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.Visible = 1
    pres = None
    try:
        pres = app.Presentations.Open(str(pptx_path.resolve()), WithWindow=False)
        pres.SaveAs(str(pdf_path.resolve()), 32)
        for idx, sld in enumerate(pres.Slides, start=1):
            out_png = png_dir / f"slide_{idx:02d}.png"
            sld.Export(str(out_png.resolve()), "PNG", 1280, 720)
    finally:
        if pres is not None:
            pres.Close()
        app.Quit()

    return sorted(png_dir.glob("slide_*.png"))


def build_contact_sheets(slide_pngs: list[Path], out_dir: Path) -> list[Path]:
    groups = [
        ("contact_sheet_01_08.png", slide_pngs[0:8]),
        ("contact_sheet_09_16.png", slide_pngs[8:16]),
        ("contact_sheet_17_22.png", slide_pngs[16:22]),
    ]
    outputs: list[Path] = []
    thumb_w, thumb_h = 640, 360
    pad = 16
    cols = 2
    for fname, items in groups:
        rows = (len(items) + 1) // 2
        sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, rows * thumb_h + (rows + 1) * pad), (241, 245, 249))
        draw = ImageDraw.Draw(sheet)
        for idx, p in enumerate(items):
            r = idx // cols
            c = idx % cols
            x = pad + c * (thumb_w + pad)
            y = pad + r * (thumb_h + pad)
            with Image.open(p) as im:
                sheet.paste(im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x, y))
            draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(148, 163, 184), width=2)
        out_p = out_dir / fname
        sheet.save(out_p)
        outputs.append(out_p)
    return outputs


def build_all() -> dict:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    prs = Presentation(str(TEMPLATE_PPTX))
    assert len(prs.slides) == 25, f"Expected 25 template slides, got {len(prs.slides)}"

    for drop_i in (23, 19, 11):
        drop_slide(prs, drop_i)
    assert len(prs.slides) == 22

    move_slide(prs, 13, 14)

    slides = prs.slides
    edit_f01_cover(slides[0])
    edit_f02_outline(slides[1])
    edit_f03_problem(slides[2])
    edit_f04_pomdp(slides[3])
    edit_f05_literature(slides[4])
    edit_f06_design(slides[5])
    edit_f08_m0_arch(slides[7])
    edit_f09_m0_case(slides[8])
    edit_f10_s3_ablation(slides[9])
    edit_f11_merged_s11_s12(slides[10])
    edit_f13_merged_s14_s15(slides[12])
    edit_f14_merged_s16_s17(slides[13])
    edit_f15_opencvl_stage1(slides[14])
    edit_f16_opencvl_stage2(slides[15])
    edit_f17_platform_arch(slides[16])
    edit_f18_api_vs_m0(slides[17])
    edit_f19_live_platforms(slides[18])
    edit_f20_team(slides[19])
    edit_f21_boundaries_future(slides[20])
    edit_f22_closure_qa(slides[21])

    for idx in range(2, 23):
        update_footer(slides[idx - 1], idx, total=22)

    prs.save(str(OUT_PPTX))
    slide_pngs = export_pdf_and_pngs(OUT_PPTX, OUT_PDF, REVIEW_DIR)
    sheets = build_contact_sheets(slide_pngs, REVIEW_DIR)
    return {
        "pptx": str(OUT_PPTX),
        "pdf": str(OUT_PDF),
        "slides": len(slide_pngs),
        "contact_sheets": [str(s) for s in sheets],
    }


if __name__ == "__main__":
    print(json.dumps(build_all(), ensure_ascii=False))
