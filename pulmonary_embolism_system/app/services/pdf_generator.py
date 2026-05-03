"""Generate medical-style PDF reports using ReportLab."""
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_scan_report(
    hospital_name: str,
    patient_name: str,
    patient_id: str,
    scan_date: str,
    doctor_name: str,
    confidence: float,
    processing_time: float,
    original_image_path: str | None,
    overlay_image_path: str | None,
    out_pdf_path: str,
) -> str:
    os.makedirs(os.path.dirname(out_pdf_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        out_pdf_path,
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "T",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0d47a1"),
        spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1565c0"),
        spaceBefore=10,
        spaceAfter=8,
    )
    body = ParagraphStyle("B", parent=styles["Normal"], alignment=TA_LEFT, fontSize=10, leading=14)

    story = []
    story.append(Paragraph(hospital_name, title_style))
    story.append(Paragraph("Pulmonary Embolism — Segmentation Report", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", body))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Patient Information", h2))
    data = [
        ["Patient ID", patient_id],
        ["Full Name", patient_name],
        ["Scan / Visit Date", scan_date],
        ["Referring Physician", doctor_name or "—"],
        ["Model Confidence", f"{confidence * 100:.2f}%"],
        ["Processing Time", f"{processing_time:.3f} s"],
    ]
    t = Table(data, colWidths=[2 * inch, 4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e3f2fd")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Imaging & Segmentation", h2))
    img_w = 3.2 * inch

    if original_image_path and os.path.isfile(original_image_path):
        story.append(Paragraph("<b>Original scan / slice</b>", body))
        story.append(Spacer(1, 0.08 * inch))
        story.append(RLImage(original_image_path, width=img_w, height=img_w * 0.75))
        story.append(Spacer(1, 0.15 * inch))

    if overlay_image_path and os.path.isfile(overlay_image_path):
        story.append(Paragraph("<b>Segmentation overlay</b>", body))
        story.append(Spacer(1, 0.08 * inch))
        story.append(RLImage(overlay_image_path, width=img_w, height=img_w * 0.75))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("<b>Physician signature / verification</b>", h2))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("_" * 50, body))
    story.append(Paragraph("Authorized clinician", body))

    doc.build(story)
    return out_pdf_path
