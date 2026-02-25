"""
PDF Report Exporter
====================
Generates a professional PDF interview report using reportlab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderPDF

NAVY = HexColor("#040812")
PANEL = HexColor("#0c1220")
CYAN = HexColor("#00d4ff")
AMBER = HexColor("#f59e0b")
GREEN = HexColor("#10b981")
RED = HexColor("#ef4444")
SLATE = HexColor("#64748b")
LIGHT = HexColor("#e2e8f0")


class ReportPDFExporter:

    def export(self, output_path: str, resume_profile, behavior_report, session_data: dict):
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm,
            title="AI Interview Report",
        )

        styles = self._build_styles()
        story = []

        # Cover section
        story.extend(self._build_header(resume_profile, session_data, styles))
        story.append(Spacer(1, 8*mm))

        # Score summary table
        story.extend(self._build_score_table(behavior_report, session_data, styles))
        story.append(Spacer(1, 8*mm))

        # Skill readiness
        if resume_profile and resume_profile.skill_matches:
            story.extend(self._build_skill_section(resume_profile.skill_matches, styles))
            story.append(Spacer(1, 8*mm))

        # Behavioral metrics
        story.extend(self._build_behavioral_section(behavior_report, styles))
        story.append(Spacer(1, 8*mm))

        # Recommendations
        story.extend(self._build_recommendations(behavior_report, styles))

        doc.build(story, onFirstPage=self._draw_background, onLaterPages=self._draw_background)

    def _draw_background(self, canvas, doc):
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)

        # Top accent bar
        canvas.setFillColor(CYAN)
        canvas.rect(0, A4[1] - 3, A4[0], 3, fill=1, stroke=0)

        # Page number
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(SLATE)
        canvas.drawRightString(A4[0] - 20*mm, 12*mm, f"Page {doc.page}")
        canvas.drawString(20*mm, 12*mm, "AI INTERVIEWER  ·  CONFIDENTIAL REPORT")

    def _build_styles(self) -> dict:
        return {
            "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26,
                                    textColor=LIGHT, spaceAfter=4),
            "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13,
                                       textColor=SLATE, spaceAfter=2),
            "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=9,
                                      textColor=CYAN, spaceBefore=6, spaceAfter=6,
                                      letterSpacing=2),
            "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10,
                                   textColor=LIGHT, leading=16),
            "mono": ParagraphStyle("mono", fontName="Courier", fontSize=9,
                                   textColor=CYAN),
            "rec": ParagraphStyle("rec", fontName="Helvetica", fontSize=10,
                                  textColor=LIGHT, leading=16, leftIndent=12),
        }

    def _build_header(self, profile, session_data: dict, styles: dict) -> list:
        elements = []
        name = getattr(profile, "candidate_name", "Candidate") if profile else "Candidate"
        role = getattr(profile, "job_role", "Unknown Role") if profile else "Unknown"
        duration = session_data.get("duration_seconds", 0)
        mins, secs = divmod(duration, 60)

        elements.append(Paragraph(f"◈ INTERVIEW REPORT", styles["mono"]))
        elements.append(Spacer(1, 3*mm))
        elements.append(Paragraph(name, styles["title"]))
        elements.append(Paragraph(f"Target Role: {role}  ·  Duration: {mins}m {secs}s  ·  "
                                   f"Questions: {session_data.get('total_questions', 0)}",
                                   styles["subtitle"]))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=CYAN, opacity=0.3,
                                    spaceAfter=4))
        return elements

    def _build_score_table(self, behavior_report, session_data: dict, styles: dict) -> list:
        elements = [Paragraph("PERFORMANCE SUMMARY", styles["section"])]

        readiness = session_data.get("readiness_score", 0)
        comm = getattr(behavior_report, "communication_score", 0) if behavior_report else 0
        fluency = getattr(behavior_report, "fluency_score", 0) if behavior_report else 0
        conf = getattr(behavior_report, "final_confidence_score", 0) if behavior_report else 0
        wpm = getattr(behavior_report, "average_wpm", 0) if behavior_report else 0

        def score_color(s):
            return GREEN if s >= 70 else AMBER if s >= 45 else RED

        data = [
            ["METRIC", "SCORE", "RATING"],
            ["Resume Readiness", f"{readiness:.0f}%", self._rating(readiness)],
            ["Communication Score", f"{comm:.0f}%", self._rating(comm)],
            ["Fluency Score", f"{fluency:.0f}%", self._rating(fluency)],
            ["Confidence Index", f"{conf:.0f}%", self._rating(conf)],
            ["Speaking Pace", f"{wpm:.0f} WPM", "Good" if 110 <= wpm <= 165 else "Review"],
        ]

        col_widths = [90*mm, 40*mm, 40*mm]
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0c1220")),
            ("TEXTCOLOR", (0, 0), (-1, 0), CYAN),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("TEXTCOLOR", (0, 1), (-1, -1), LIGHT),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("BACKGROUND", (0, 1), (-1, -1), HexColor("#060d1a")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#060d1a"), HexColor("#0a1020")]),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        elements.append(t)
        return elements

    def _rating(self, score: float) -> str:
        if score >= 80: return "Excellent"
        if score >= 65: return "Good"
        if score >= 50: return "Average"
        if score >= 35: return "Needs Work"
        return "Poor"

    def _build_skill_section(self, skill_matches: list, styles: dict) -> list:
        elements = [Paragraph("SKILL READINESS ANALYSIS", styles["section"])]
        data = [["SKILL", "MATCH SCORE", "ASSESSMENT"]]

        for m in skill_matches[:12]:
            score_pct = f"{m.score * 100:.0f}%"
            assessment = ("Strong Match" if m.score >= 0.6 else
                          "Partial Match" if m.score >= 0.35 else "Gap Detected")
            data.append([m.skill, score_pct, assessment])

        t = Table(data, colWidths=[95*mm, 35*mm, 40*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0c1220")),
            ("TEXTCOLOR", (0, 0), (-1, 0), CYAN),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("TEXTCOLOR", (0, 1), (-1, -1), LIGHT),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#060d1a"), HexColor("#0a1020")]),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        return elements

    def _build_behavioral_section(self, behavior_report, styles: dict) -> list:
        elements = [Paragraph("BEHAVIORAL ANALYSIS", styles["section"])]
        if not behavior_report:
            elements.append(Paragraph("No behavioral data recorded.", styles["body"]))
            return elements

        items = [
            f"Total Stutter Events: {behavior_report.total_stutter_events}",
            f"Average Speaking Rate: {behavior_report.average_wpm:.0f} WPM",
            f"Session Duration: {behavior_report.total_duration_sec:.0f}s",
            f"Prosody Score: {behavior_report.prosody_score:.0f}/100",
            f"Pace Consistency: {behavior_report.pace_score:.0f}/100",
        ]
        for item in items:
            elements.append(Paragraph(f"• {item}", styles["body"]))
        return elements

    def _build_recommendations(self, behavior_report, styles: dict) -> list:
        elements = [Paragraph("RECOMMENDATIONS", styles["section"])]
        recs = getattr(behavior_report, "recommendations", []) if behavior_report else []
        if not recs:
            recs = ["Complete the interview session to receive personalized recommendations."]
        for i, rec in enumerate(recs):
            prefix = "→ " if i == 0 else "  "
            elements.append(Paragraph(f"{prefix}{rec}", styles["rec"]))
            elements.append(Spacer(1, 2*mm))
        return elements
