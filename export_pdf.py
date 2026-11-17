"""Renders TECHNICAL_DOCUMENT.md to Let'sC.pdf, so the PDF and the Markdown can never disagree.

    python export_pdf.py

Handles the Markdown the document uses: headings, paragraphs, bullet and numbered lists, fenced
code blocks, pipe tables, horizontal rules, **bold**, *italic* and `code`. Inline $math$ is printed
as plain text without the dollar signs.
"""

import html
import os
import re
import shutil

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "TECHNICAL_DOCUMENT.md")
PDF_PATH = os.path.join(HERE, "Let'sC.pdf")
PDF_PATH_SAFE = os.path.join(HERE, "Let_sC.pdf")

INK = colors.HexColor("#0f172a")
BODY = colors.HexColor("#334155")
ACCENT = colors.HexColor("#0f766e")


def make_styles():
    base = getSampleStyleSheet()["Normal"]
    return {
        "h1": ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21,
                             textColor=INK, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=12, leading=16,
                             textColor=INK, spaceBefore=10, spaceAfter=4, keepWithNext=True),
        "h3": ParagraphStyle("h3", parent=base, fontName="Helvetica-Bold", fontSize=10, leading=13,
                             textColor=BODY, spaceBefore=7, spaceAfter=3, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base, fontName="Helvetica", fontSize=8.5,
                               leading=12, textColor=BODY, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base, fontName="Helvetica", fontSize=8.5,
                                 leading=12, textColor=BODY, leftIndent=12, bulletIndent=2,
                                 spaceAfter=2),
        "bullet2": ParagraphStyle("bullet2", parent=base, fontName="Helvetica", fontSize=8.5,
                                  leading=12, textColor=BODY, leftIndent=26, bulletIndent=16,
                                  spaceAfter=2),
        "body_indent": ParagraphStyle("body_indent", parent=base, fontName="Helvetica",
                                      fontSize=8.5, leading=12, textColor=BODY, leftIndent=12,
                                      spaceAfter=4),
        "code": ParagraphStyle("code", parent=base, fontName="Courier", fontSize=7, leading=9,
                               textColor=INK, backColor=colors.HexColor("#f1f5f9"),
                               borderPadding=4, spaceBefore=2, spaceAfter=6),
        "th": ParagraphStyle("th", parent=base, fontName="Helvetica-Bold", fontSize=7.5,
                             leading=9.5, textColor=colors.white, alignment=1),
        "td": ParagraphStyle("td", parent=base, fontName="Helvetica", fontSize=7.5, leading=9.5,
                             textColor=colors.HexColor("#1e293b"), alignment=1),
    }


def inline(text):
    """Markdown inline markup to ReportLab's mini-HTML."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\$([^$]+)\$", lambda m: m.group(1).replace("\\", ""), text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return text


def table_flowable(rows, styles, width):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(inline(c), styles["th" if i == 0 else "td"]) for c in r]
            for i, r in enumerate(cells)]
    ncol = max(len(r) for r in data)
    t = Table(data, colWidths=[width / ncol] * ncol, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def markdown_story(md, styles, width):
    story, para, lines, i = [], [], md.splitlines(), 0
    para_style = ["body"]

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), styles[para_style[0]]))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(block), styles["code"]))
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            story.append(table_flowable(rows, styles, width))
            story.append(Spacer(1, 6))
            continue
        elif re.match(r"^#{1,3} ", stripped):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            story.append(Paragraph(inline(stripped[level:].strip()), styles["h%d" % level]))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=8))
        elif re.fullmatch(r"-{3,}", stripped):
            flush()
            story.append(Spacer(1, 4))
        elif re.match(r"^[*-] ", stripped):
            flush()
            nested = indent >= 2
            story.append(Paragraph(inline(stripped[2:]), styles["bullet2" if nested else "bullet"],
                                   bulletText="–" if nested else "•"))
        elif re.match(r"^\d+\. ", stripped):
            flush()
            num, rest = stripped.split(". ", 1)
            story.append(Paragraph(inline(rest), styles["bullet"], bulletText=num + "."))
        elif not stripped:
            flush()
        else:
            if not para:
                para_style[0] = "body_indent" if indent >= 2 else "body"
            para.append(stripped)
        i += 1
    flush()
    return story


def build_pdf():
    margin = 0.6 * inch
    doc = SimpleDocTemplate(PDF_PATH, pagesize=letter, leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin,
                            title="Let'sC - Technical Document", author="Team Let'sC")
    with open(SOURCE, encoding="utf-8") as f:
        md = f.read()
    doc.build(markdown_story(md, make_styles(), letter[0] - 2 * margin))
    shutil.copyfile(PDF_PATH, PDF_PATH_SAFE)
    print("Wrote", PDF_PATH)
    print("Wrote", PDF_PATH_SAFE)


if __name__ == "__main__":
    build_pdf()
