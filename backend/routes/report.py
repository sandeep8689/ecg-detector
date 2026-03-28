"""
/api/report — Generate downloadable PDF report
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import io
import datetime
import base64

router = APIRouter()


class ReportRequest(BaseModel):
    disease: str
    confidence: float
    risk_level: str
    clinical_info: Dict[str, Any]
    ranges_report: List[Dict[str, Any]]
    all_probabilities: Dict[str, float]
    overlay_base64: Optional[str] = None
    filename: Optional[str] = "ecg_scan"


@router.post("/report")
async def generate_report(data: ReportRequest):
    """Generate a downloadable PDF report from analysis results"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image as RLImage,
                                         HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=1.5*cm, bottomMargin=1.5*cm
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Colors
        DARK_BLUE = colors.HexColor('#0f172a')
        MED_BLUE = colors.HexColor('#1e40af')
        ACCENT = colors.HexColor('#3b82f6')
        RED = colors.HexColor('#ef4444')
        ORANGE = colors.HexColor('#f97316')
        GREEN = colors.HexColor('#22c55e')
        GRAY = colors.HexColor('#64748b')
        LIGHT_GRAY = colors.HexColor('#f1f5f9')
        
        risk_color_map = {"LOW": GREEN, "HIGH": ORANGE, "CRITICAL": RED}
        risk_color = risk_color_map.get(data.risk_level, GRAY)
        
        # ── Header ─────────────────────────────────────────────────────────
        title_style = ParagraphStyle('Title', parent=styles['Normal'],
            fontSize=22, textColor=DARK_BLUE, fontName='Helvetica-Bold',
            alignment=TA_CENTER, spaceAfter=4)
        sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
            fontSize=10, textColor=GRAY, alignment=TA_CENTER, spaceAfter=2)
        
        story.append(Paragraph("💓 ECG Heart Anomaly Analysis Report", title_style))
        story.append(Paragraph("AI-Powered Cardiac Screening — EfficientNet-B4 (96.4% accuracy)", sub_style))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%d %B %Y at %I:%M %p')} | File: {data.filename}",
            sub_style
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=12))
        
        # ── Main Result Banner ──────────────────────────────────────────────
        banner_style = ParagraphStyle('Banner', parent=styles['Normal'],
            fontSize=14, textColor=colors.white, fontName='Helvetica-Bold',
            alignment=TA_CENTER, backColor=risk_color,
            borderPad=8, spaceBefore=4, spaceAfter=4)
        
        story.append(Paragraph(
            f"DETECTION: {data.disease.upper()}  |  "
            f"CONFIDENCE: {data.confidence:.1f}%  |  "
            f"RISK: {data.risk_level}",
            banner_style
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # ── ECG Overlay Image ───────────────────────────────────────────────
        if data.overlay_base64:
            try:
                img_bytes = base64.b64decode(data.overlay_base64)
                img_buf = io.BytesIO(img_bytes)
                rl_img = RLImage(img_buf, width=5*inch, height=2.5*inch)
                
                caption_style = ParagraphStyle('Cap', parent=styles['Normal'],
                    fontSize=8, textColor=GRAY, alignment=TA_CENTER)
                
                story.append(rl_img)
                story.append(Paragraph(
                    "AI Attention Heatmap — Red/Yellow zones indicate areas that influenced the prediction",
                    caption_style
                ))
                story.append(Spacer(1, 0.15*inch))
            except Exception:
                pass
        
        # ── Abnormality Summary ─────────────────────────────────────────────
        heading_style = ParagraphStyle('H2', parent=styles['Normal'],
            fontSize=13, textColor=MED_BLUE, fontName='Helvetica-Bold',
            spaceBefore=8, spaceAfter=4)
        
        body_style = ParagraphStyle('Body', parent=styles['Normal'],
            fontSize=10, textColor=DARK_BLUE, leading=16)
        
        story.append(Paragraph("🔍 Abnormality Summary", heading_style))
        
        ab_count = data.clinical_info.get("abnormality_count", 0)
        ab_pct = data.clinical_info.get("abnormality_percentage", 0)
        total = data.clinical_info.get("total_checks", 8)
        
        summary_data = [
            ["Abnormalities Found", f"{ab_count} out of {total} checks"],
            ["Abnormality Rate", f"{ab_pct}%"],
            ["Risk Classification", data.risk_level],
            ["AI Confidence", f"{data.confidence:.1f}%"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), LIGHT_GRAY),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('TEXTCOLOR', (0,0), (-1,-1), DARK_BLUE),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, LIGHT_GRAY]),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.15*inch))
        
        # ── Detected Abnormalities ──────────────────────────────────────────
        abnormalities = data.clinical_info.get("abnormalities_found", [])
        if abnormalities:
            story.append(Paragraph("⚠️ Specific Abnormalities Detected", heading_style))
            for abn in abnormalities:
                story.append(Paragraph(f"• {abn}", body_style))
            story.append(Spacer(1, 0.15*inch))
        
        # ── Clinical Ranges Table ───────────────────────────────────────────
        story.append(Paragraph("📊 ECG Parameter Analysis", heading_style))
        
        range_header = [["Parameter", "Your Value", "Normal Range", "Status"]]
        range_rows = []
        
        for r in data.ranges_report:
            status = r["status"]
            status_color = GREEN if status == "NORMAL" else (RED if status == "CRITICAL" else ORANGE)
            range_rows.append([
                r["metric"],
                f"{r['value']} {r['unit']}",
                f"{r['normal_min']}–{r['normal_max']} {r['unit']}",
                status
            ])
        
        range_table_data = range_header + range_rows
        range_table = Table(range_table_data, colWidths=[2*inch, 1.5*inch, 2*inch, 1*inch])
        
        style_cmds = [
            ('BACKGROUND', (0,0), (-1,0), MED_BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GRAY]),
            ('PADDING', (0,0), (-1,-1), 7),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ]
        
        # Color status cells
        for i, r in enumerate(data.ranges_report, start=1):
            s = r["status"]
            cell_color = GREEN if s == "NORMAL" else (RED if s == "CRITICAL" else ORANGE)
            style_cmds.append(('TEXTCOLOR', (3, i), (3, i), cell_color))
            style_cmds.append(('FONTNAME', (3, i), (3, i), 'Helvetica-Bold'))
        
        range_table.setStyle(TableStyle(style_cmds))
        story.append(range_table)
        story.append(Spacer(1, 0.15*inch))
        
        # ── Clinical Info ───────────────────────────────────────────────────
        story.append(Paragraph("📋 Clinical Summary", heading_style))
        story.append(Paragraph(data.clinical_info.get("what_it_means", ""), body_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("🩺 Possible Symptoms", heading_style))
        for s in data.clinical_info.get("symptoms", []):
            story.append(Paragraph(f"• {s}", body_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("⏰ Recommended Action", heading_style))
        urgency_style = ParagraphStyle('Urgency', parent=styles['Normal'],
            fontSize=11, textColor=risk_color, fontName='Helvetica-Bold',
            backColor=colors.HexColor('#fef2f2') if data.risk_level != "LOW" else colors.HexColor('#f0fdf4'),
            borderPad=8, leading=16)
        story.append(Paragraph(data.clinical_info.get("urgency", ""), urgency_style))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("💡 Lifestyle Recommendations", heading_style))
        for tip in data.clinical_info.get("lifestyle_tips", []):
            story.append(Paragraph(f"• {tip}", body_style))
        
        # ── Probability Breakdown ───────────────────────────────────────────
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("📈 AI Confidence Breakdown", heading_style))
        
        prob_data = [["Condition", "Probability", "Bar"]]
        for cls, pct in data.all_probabilities.items():
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            prob_data.append([cls, f"{pct:.1f}%", bar])
        
        prob_table = Table(prob_data, colWidths=[2.5*inch, 1*inch, 3*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), MED_BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (2,1), (2,-1), 'Courier'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GRAY]),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(prob_table)
        
        # ── Disclaimer ──────────────────────────────────────────────────────
        story.append(Spacer(1, 0.2*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=GRAY))
        
        disclaimer_style = ParagraphStyle('Disc', parent=styles['Normal'],
            fontSize=7.5, textColor=GRAY, alignment=TA_CENTER, leading=12)
        story.append(Paragraph(
            "⚠️ IMPORTANT DISCLAIMER: This report is generated by artificial intelligence "
            "for informational and educational purposes only. It does NOT constitute a medical "
            "diagnosis, medical advice, or a substitute for professional medical evaluation. "
            "Always consult a qualified cardiologist or physician for proper diagnosis and treatment. "
            "In case of emergency, call 108 immediately.",
            disclaimer_style
        ))
        
        doc.build(story)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=ecg_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
