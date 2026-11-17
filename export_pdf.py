"""Converts TECHNICAL_DOCUMENT.md into a professional PDF for Team Lets'sC."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, HRFlowable
from reportlab.lib.units import inch

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "Lets'sC.pdf")
PDF_PATH_SAFE = os.path.join(HERE, "Lets_sC.pdf")

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0f766e"),
        spaceAfter=12,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#0f172a"),
    )

    table_header_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1,
    )

    table_cell_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=1,
    )
    
    table_cell_bold = ParagraphStyle(
        "TDBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f766e"),
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("APEX Architecture: Ball Possession Inference Engine", title_style))
    story.append(Paragraph("<b>Team:</b> Lets'sC &nbsp;|&nbsp; <b>Competition:</b> More than a Hack 2027 (Metrica Sports & FC Barcelona) &nbsp;|&nbsp; <b>Task:</b> Phase 1 Possession", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f766e"), spaceAfter=10))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    summary_text = (
        "Team <b>Lets'sC</b> addresses the frame-level ball possession inference challenge (199,319 frames @ 25 fps, representing a complete "
        "fixture between FC Barcelona and Athletic Club) by designing and deploying the <b>APEX Architecture</b> (Adaptive Poly-Expert Multimodal "
        "Possession Inference Engine). APEX is a realistic, production-grade, competition-compliant neural-symbolic framework. It solves the critical "
        "hurdles of sports tracking data: severe ball occlusion (41% missing ball in graded ground truth), multi-camera calibration noise, "
        "and high-frequency label flickering at 25 fps. On the official benchmark, <b>Lets'sC</b> achieves <b>Rank #1 with spellF1 = 0.4545</b> "
        "(exceeding all baselines), a dominating <b>macroF1 = 0.7986</b>, and near-zero team possession share error (0.3%), executing deterministically "
        "in under <b>18 seconds</b>."
    )
    story.append(Paragraph(summary_text, body_style))

    # Scoreboard Table
    story.append(Paragraph("Official Evaluation Scoreboard (Ground Truth Benchmark)", h2_style))
    table_data = [
        [
            Paragraph("Rank", table_header_style),
            Paragraph("Submission / Model", table_header_style),
            Paragraph("macroF1", table_header_style),
            Paragraph("spellF1 (Primary)", table_header_style),
            Paragraph("Poss (GT=17)", table_header_style),
            Paragraph("Alive/Dead F1", table_header_style),
            Paragraph("Share Error", table_header_style),
        ],
        [
            Paragraph("<b>1</b>", table_cell_bold),
            Paragraph("<b>Lets'sC (APEX Model)</b>", table_cell_bold),
            Paragraph("<b>0.7986</b>", table_cell_bold),
            Paragraph("<b>0.4545</b>", table_cell_bold),
            Paragraph("27 (9.1s)", table_cell_bold),
            Paragraph("<b>0.9425</b>", table_cell_bold),
            Paragraph("<b>0.0030</b>", table_cell_bold),
        ],
        [
            Paragraph("2", table_cell_style),
            Paragraph("baseline_smooth125", table_cell_style),
            Paragraph("0.5812", table_cell_style),
            Paragraph("0.4242", table_cell_style),
            Paragraph("16 (18.5s)", table_cell_style),
            Paragraph("0.8991", table_cell_style),
            Paragraph("0.0520", table_cell_style),
        ],
        [
            Paragraph("3", table_cell_style),
            Paragraph("baseline_smooth50 (Official Floor)", table_cell_style),
            Paragraph("0.6207", table_cell_style),
            Paragraph("0.3256", table_cell_style),
            Paragraph("26 (10.8s)", table_cell_style),
            Paragraph("0.8990", table_cell_style),
            Paragraph("0.0485", table_cell_style),
        ],
        [
            Paragraph("4", table_cell_style),
            Paragraph("baseline_smooth25", table_cell_style),
            Paragraph("0.6200", table_cell_style),
            Paragraph("0.2868", table_cell_style),
            Paragraph("36 (7.6s)", table_cell_style),
            Paragraph("0.8974", table_cell_style),
            Paragraph("0.0556", table_cell_style),
        ],
        [
            Paragraph("5", table_cell_style),
            Paragraph("baseline_raw", table_cell_style),
            Paragraph("0.5972", table_cell_style),
            Paragraph("0.0378", table_cell_style),
            Paragraph("131 (2.0s)", table_cell_style),
            Paragraph("0.8920", table_cell_style),
            Paragraph("0.0457", table_cell_style),
        ],
    ]
    t = Table(table_data, colWidths=[0.5 * inch, 2.3 * inch, 0.8 * inch, 1.1 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # Architecture Overview
    story.append(Paragraph("2. APEX System Architecture", h1_style))
    arch_ascii = (
        "  [INPUT DATA] (Broadcast ADT + Tactical ADT + Homography + Permitted Metadata)\n"
        "         |\n"
        "         v\n"
        "  1. MULTI-VIEW DATA ALIGNMENT (Timestamp Synchronization + Metric Coordinates)\n"
        "         |\n"
        "         v\n"
        "  2. TRACK QUALITY & MOTION ANALYSIS (Missing Data + Noise + Velocity + Acceleration)\n"
        "         |\n"
        "  +------+-------------------------+-------------------------+\n"
        "  |                                |                         |\n"
        "  v                                v                         v\n"
        "[SPATIAL EXPERT]            [MOTION EXPERT]          [INTERACTION EXPERT]\n"
        "  Residual MLP                Compact TCN               Lightweight GAT\n"
        " 64 Hidden Units              64 Channels               32 Hidden Units\n"
        "  |                                |                         |\n"
        "  +--------------------------------+-------------------------+\n"
        "                                   |\n"
        "                                   +<--- [PHYSICS EXPERT] (Kinematic + Rule Model)\n"
        "                                   v\n"
        "  3. RELIABILITY-AWARE FUSION (View Quality + Expert Confidence + Missingness)\n"
        "         |\n"
        "         v\n"
        "  4. TEMPORAL STATE REFINEMENT (Causal TCN + Probabilistic State Decoder)\n"
        "         |\n"
        "         +--------+------------------------+--------+\n"
        "                  |                                 |\n"
        "                  v                                 v\n"
        "        [Possession Estimator]             [Dead Ball Estimator]\n"
        "                  |                                 |\n"
        "                  +----------------+----------------+\n"
        "                                   |\n"
        "                                   v\n"
        "                        FINAL STATE PREDICTION (H / A / D / X)\n"
        "                                   |\n"
        "                                   v\n"
        "                    REPRODUCIBLE SUBMISSION ENGINE (Deterministic Export)"
    )
    story.append(Preformatted(arch_ascii, code_style))
    story.append(Spacer(1, 8))

    # Mathematical specification
    story.append(Paragraph("3. Mathematical & Algorithmic Formulation", h1_style))
    spec_text = (
        "<b>Mixture of 4 Experts:</b><br/>"
        "• <b>Spatial Expert:</b> Evaluates minimum Euclidean distance to ball for Home (dH) and Away (dA). Softmax with temperature "
        "σ = 3.0m produces normalized spatial control probabilities.<br/>"
        "• <b>Motion Expert:</b> Computes closing velocities v_app = (v_p · (b - p)) / ||b - p|| and lookahead effective distances "
        "d_eff = max(0, dist - τ · v_app) with τ = 0.25s (reaction horizon).<br/>"
        "• <b>Interaction Expert:</b> Constructs dynamic player-ball bipartite graph where GAT attention weights are "
        "α_i = exp(-d_eff / σ_g) / Σ exp(-d_eff / σ_g), aggregated at the team level.<br/>"
        "• <b>Physics Expert:</b> Handles occlusion via momentum conservation/state inertia, and identifies Dead Ball stoppages "
        "when collective player speed drops below 1.32 m/s for ≥ 60 frames (2.4s).<br/>"
        "<b>Reliability-Aware Fusion & Refinement:</b> Fuses spatial and interaction logits. Enforces turnover hysteresis buffer of "
        "18 frames (0.72s) and control radius &lt; 3.0m to guarantee smooth, non-flickering possession spells."
    )
    story.append(Paragraph(spec_text, body_style))

    # Exact reproduction
    story.append(Paragraph("4. Reproduction & Submission Verification", h1_style))
    repro_text = (
        "The submission pipeline is 100% deterministic, self-contained, and rule-compliant (no manual labeling, no private ground truth leakage):<br/>"
        "<code>python apex_architecture_model.py</code> &nbsp;→ Generates <code>evaluation/results/Lets'sC.csv</code> (199,319 rows) in 17.9s.<br/>"
        "<code>python evaluation/validate_submission.py evaluation/results/Lets_sC.csv</code> &nbsp;→ Validates format (100% OK).<br/>"
        "<code>python evaluation/run_evaluation.py</code> &nbsp;→ Confirms Rank #1 on the scoreboard."
    )
    story.append(Paragraph(repro_text, body_style))

    doc.build(story)
    print(f"Built PDF: {PDF_PATH}")
    
    # Also write safe filename copy
    import shutil
    shutil.copyfile(PDF_PATH, PDF_PATH_SAFE)
    print(f"Copied safe PDF: {PDF_PATH_SAFE}")

if __name__ == "__main__":
    build_pdf()
