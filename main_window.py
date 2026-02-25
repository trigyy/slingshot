"""
Module 4: PyQt6 GUI — AI Interviewer Interface
================================================
Three screens:
  1. Landing       — Resume upload + job role selector
  2. Interview     — Live waveform + transcript + phase tracker
  3. Report        — Radar chart, behavioral scores, recommendations

Aesthetic: Dark glassmorphic terminal-meets-dashboard
Colors: Deep navy (#0a0e1a), electric cyan (#00d4ff), 
        amber accent (#f59e0b), glass panels (#ffffff10)
Fonts: JetBrains Mono (monospace), Outfit (sans)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, QObject, QPropertyAnimation,
    QEasingCurve, QRect, QSize, pyqtProperty
)
from PyQt6.QtGui import (
    QColor, QFont, QPalette, QLinearGradient, QPainter, QPen,
    QBrush, QPixmap, QDragEnterEvent, QDropEvent, QPainterPath,
    QFontDatabase, QIcon
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QScrollArea,
    QFrame, QStackedWidget, QProgressBar, QSizePolicy, QSplitter,
    QTextEdit, QGraphicsDropShadowEffect
)

logger = logging.getLogger(__name__)

# ─── Color Palette ────────────────────────────────────────────────────────────

COLORS = {
    "bg_dark": "#f8f9fa",         # Light Grey Background
    "bg_panel": "#ffffff",        # Pure White Surface Cards
    "bg_glass": "#ffffff",        # Now solid white (legacy naming)
    "border": "#dadce0",          # Light Material Border
    "border_active": "#1a73e8",   # Blue Active
    "cyan": "#1a73e8",            # Primary Blue
    "cyan_dim": "#4285f4",        # Blue Lighter
    "amber": "#f29900",           # Yellow/Amber
    "green": "#1e8e3e",           # Green
    "red": "#d93025",             # Red
    "purple": "#9333ea",          # A generic deep purple
    "text_primary": "#202124",    # High-Contrast Text
    "text_secondary": "#5f6368",  # Muted Text
    "text_accent": "#1a73e8",     # Blue Accent
}

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f8f9fa;
    color: #202124;
    font-family: 'Roboto', 'Segoe UI', sans-serif;
}

QLabel { color: #202124; }
QLabel#heading { 
    font-size: 28px; font-weight: 500; 
    color: #202124; letter-spacing: -0.5px;
}
QLabel#subheading { 
    font-size: 14px; color: #5f6368; letter-spacing: 0px;
}
QLabel#accent { color: #1a73e8; font-weight: 500; }
QLabel#mono {
    font-family: 'Roboto Mono', 'Consolas', monospace;
    font-size: 12px; color: #1a73e8;
}

QPushButton {
    background: #1a73e8;
    color: white; font-weight: 500; font-size: 14px;
    border: none; border-radius: 20px; padding: 10px 24px;
    letter-spacing: 0.25px;
}
QPushButton:hover {
    background: #1557b0;
}
QPushButton:pressed { background: #124089; }
QPushButton:disabled { background: #e8eaed; color: #9aa0a6; }

QPushButton#danger {
    background: #d93025;
    color: white;
}
QPushButton#danger:hover {
    background: #a50e0e;
}
QPushButton#ghost {
    background: transparent; color: #1a73e8;
    border: 1px solid #dadce0; border-radius: 20px;
}
QPushButton#ghost:hover { border-color: #1a73e8; background: #e8f0fe; }

QComboBox {
    background: #ffffff; border: 1px solid #dadce0;
    border-radius: 8px; padding: 10px 16px; color: #202124;
    font-size: 14px; min-width: 200px;
}
QComboBox:hover { border-color: #1a73e8; }
QComboBox::drop-down { border: none; width: 30px; }
QComboBox::down-arrow { 
    image: none; border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #5f6368; margin-right: 10px;
}
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #dadce0;
    selection-background-color: #e8f0fe;
    color: #202124;
}

QScrollBar:vertical {
    background: transparent; width: 8px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #dadce0; border-radius: 4px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #bdc1c6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QTextEdit {
    background: #ffffff; border: 1px solid #dadce0;
    border-radius: 8px; color: #3c4043; font-size: 14px;
    padding: 12px; line-height: 1.6;
    font-family: 'Roboto', sans-serif;
}

QProgressBar {
    background: #f1f3f4; border: none; border-radius: 4px; height: 6px;
}
QProgressBar::chunk { background: #1a73e8; border-radius: 4px; }

QFrame#glass {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 12px;
}
QFrame#glass_active {
    background: #ffffff;
    border: 1px solid #1a73e8;
    border-radius: 12px;
}
"""


# ─── Reusable Material Card Widget ───────────────────────────────────────────────

class MaterialCard(QFrame):
    def __init__(self, parent=None, active=False):
        super().__init__(parent)
        self.setObjectName("glass_active" if active else "glass")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 20))  # Standard Material Elevation shadow
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def set_active(self, active: bool):
        self.setObjectName("glass_active" if active else "glass")
        self.style().unpolish(self)
        self.style().polish(self)


class MetricBadge(QWidget):
    """Compact metric display: value + label."""
    def __init__(self, label: str, value: str = "—", color: str = COLORS["cyan"], parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Roboto", 22, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {color};")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_text = QLabel(label.upper())
        self.label_text.setStyleSheet("color: #5f6368; font-size: 11px; font-weight: 500; letter-spacing: 1px;")
        self.label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_label)
        layout.addWidget(self.label_text)

    def update_value(self, value: str, color: str = None):
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")


# ─── Mic State Indicator Widget ──────────────────────────────────────────────

class MicIndicatorWidget(QWidget):
    """Animated microphone state indicator.
    - LISTENING: pulsing green mic icon
    - AI_SPEAKING: bouncing orange wave icon
    """
    LISTENING = "listening"
    AI_SPEAKING = "ai_speaking"
    IDLE = "idle"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 40)
        self._state = self.IDLE
        self._pulse = 0.0
        self._pulse_dir = 1

        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(40)  # 25 fps

    def set_listening(self):
        self._state = self.LISTENING
        self.update()

    def set_ai_speaking(self):
        self._state = self.AI_SPEAKING
        self.update()

    def set_idle(self):
        self._state = self.IDLE
        self.update()

    def _tick(self):
        self._pulse += 0.08 * self._pulse_dir
        if self._pulse >= 1.0:
            self._pulse = 1.0
            self._pulse_dir = -1
        elif self._pulse <= 0.0:
            self._pulse = 0.0
            self._pulse_dir = 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        if self._state == self.LISTENING:
            # Pulsing green ring + mic icon
            pulse_radius = int(12 + self._pulse * 5)
            cx, cy = 18, h // 2
            ring_color = QColor(30, 142, 62, int(80 - self._pulse * 60))
            painter.setBrush(QBrush(ring_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - pulse_radius, cy - pulse_radius,
                                pulse_radius * 2, pulse_radius * 2)

            painter.setBrush(QBrush(QColor(30, 142, 62)))
            painter.drawEllipse(cx - 9, cy - 9, 18, 18)

            # Mic symbol (white)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(cx - 4, cy - 7, 8, 10, 3, 3)
            painter.drawArc(cx - 6, cy - 2, 12, 8, 0, -180 * 16)
            painter.drawLine(cx, cy + 5, cx, cy + 8)

            # "LISTENING" text
            painter.setPen(QColor(30, 142, 62))
            painter.setFont(QFont("Roboto", 9, QFont.Weight.Medium))
            painter.drawText(32, 0, w - 32, h, Qt.AlignmentFlag.AlignVCenter, "LISTENING")

        elif self._state == self.AI_SPEAKING:
            # Bouncing orange bars (soundwave)
            cx, cy = 18, h // 2
            painter.setBrush(QBrush(QColor(242, 153, 0)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - 9, cy - 9, 18, 18)

            # White bars
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            bar_heights = [4, 8, 11, 8, 4]
            bar_w = 2
            start_x = cx - 7
            for i, bh in enumerate(bar_heights):
                anim_h = int(bh * (0.5 + 0.5 * abs(self._pulse - (i / 4))))
                anim_h = max(2, min(anim_h, 12))
                painter.drawRoundedRect(
                    start_x + i * 3, cy - anim_h // 2, bar_w, anim_h, 1, 1
                )

            painter.setPen(QColor(242, 153, 0))
            painter.setFont(QFont("Roboto", 9, QFont.Weight.Medium))
            painter.drawText(32, 0, w - 32, h, Qt.AlignmentFlag.AlignVCenter, "AI SPEAKING")

        else:
            # Idle grey
            cx, cy = 18, h // 2
            painter.setBrush(QBrush(QColor(218, 220, 224)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(cx - 9, cy - 9, 18, 18)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(cx - 4, cy - 7, 8, 10, 3, 3)
            painter.drawArc(cx - 6, cy - 2, 12, 8, 0, -180 * 16)
            painter.drawLine(cx, cy + 5, cx, cy + 8)
            painter.setPen(QColor(154, 160, 166))
            painter.setFont(QFont("Roboto", 9))
            painter.drawText(32, 0, w - 32, h, Qt.AlignmentFlag.AlignVCenter, "IDLE")


# ─── Live Waveform Widget ─────────────────────────────────────────────────────

class WaveformWidget(QWidget):
    """Animated waveform display for microphone input level."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._samples = np.zeros(120)
        self._is_active = False
        self._phase = 0.0

        self._timer = QTimer()
        self._timer.timeout.connect(self._animate)
        self._timer.start(40)  # 25fps

    def push_level(self, rms: float):
        """Update with current audio energy (0–1)."""
        self._samples = np.roll(self._samples, -1)
        self._samples[-1] = rms
        self._is_active = rms > 0.01
        self.update()

    def _animate(self):
        if not self._is_active:
            self._phase += 0.05  # Idle animation
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        mid = h // 2

        # Background
        painter.fillRect(0, 0, w, h, QColor(255, 255, 255))

        if not self._is_active:
            # Idle pulse (sinusoidal)
            self._draw_idle_wave(painter, w, h, mid)
        else:
            # Live audio waveform
            self._draw_audio_wave(painter, w, h, mid)

    def _draw_idle_wave(self, painter, w, h, mid):
        pen = QPen(QColor(26, 115, 232, 40))  # Blue fading
        pen.setWidth(1)
        painter.setPen(pen)
        step = w / 60
        for i in range(60):
            x = int(i * step)
            y = mid + int(np.sin(i * 0.2 + self._phase) * 6)
            painter.drawEllipse(x, y, 2, 2)

    def _draw_audio_wave(self, painter, w, h, mid):
        n = len(self._samples)
        step = w / n

        # Glow effect — draw 3 layers (Blue shadow)
        for glow_alpha, glow_width, glow_mult in [(15, 6, 1.2), (40, 3, 1.0), (200, 1, 0.9)]:
            pen = QPen(QColor(26, 115, 232, glow_alpha))
            pen.setWidth(glow_width)
            painter.setPen(pen)

            path = QPainterPath()
            for i, sample in enumerate(self._samples):
                x = i * step
                amp = sample * mid * 0.85 * glow_mult
                y_top = mid - amp
                y_bot = mid + amp
                if i == 0:
                    path.moveTo(x, y_top)
                else:
                    path.lineTo(x, y_top)

            # Mirror bottom
            for i in range(n - 1, -1, -1):
                sample = self._samples[i]
                x = i * step
                amp = sample * mid * 0.85 * glow_mult
                path.lineTo(x, mid + amp)

            path.closeSubpath()
            painter.drawPath(path)


# ─── Phase Progress Bar ───────────────────────────────────────────────────────

class PhaseTrackerWidget(QWidget):
    """Horizontal phase progress indicator."""

    PHASES = ["INTRO", "WARM UP", "TECHNICAL", "BEHAVIORAL", "DEEP DIVE", "WRAP UP"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self._current = 0

    def set_phase(self, phase_name: str):
        name_map = {
            "INTRO": 0, "WARM_UP": 1, "TECHNICAL_CORE": 2,
            "BEHAVIORAL": 3, "SKILL_PROBE": 4, "WRAP_UP": 5
        }
        self._current = name_map.get(phase_name.upper(), 0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self.PHASES)
        seg_w = w / n

        for i, phase in enumerate(self.PHASES):
            x = int(i * seg_w)
            is_done = i < self._current
            is_active = i == self._current

            # Connector line
            if i < n - 1:
                color = QColor(26, 115, 232, 120 if is_done else 25)
                painter.setPen(QPen(color, 1))
                painter.drawLine(int(x + seg_w * 0.55), h // 2,
                                 int(x + seg_w * 0.95), h // 2)

            # Circle
            cx = int(x + seg_w * 0.5)
            cy = h // 2

            if is_active:
                painter.setBrush(QBrush(QColor(26, 115, 232)))
                painter.setPen(QPen(QColor(26, 115, 232), 2))
                painter.drawEllipse(cx - 7, cy - 7, 14, 14)
            elif is_done:
                painter.setBrush(QBrush(QColor(30, 142, 62)))
                painter.setPen(QPen(QColor(30, 142, 62), 1))
                painter.drawEllipse(cx - 5, cy - 5, 10, 10)
            else:
                painter.setBrush(QBrush(QColor(241, 243, 244)))
                painter.setPen(QPen(QColor(218, 220, 224), 1))
                painter.drawEllipse(cx - 5, cy - 5, 10, 10)

            # Label
            font = QFont("Roboto", 9, QFont.Weight.Medium)
            painter.setFont(font)
            painter.setPen(QColor(26, 115, 232) if is_active else
                           QColor(154, 160, 166) if not is_done else QColor(30, 142, 62))
            painter.drawText(int(x), h - 2, int(seg_w), 12,
                             Qt.AlignmentFlag.AlignHCenter, phase)


# ─── Landing Screen ───────────────────────────────────────────────────────────

class LandingScreen(QWidget):
    start_interview = pyqtSignal(str, str, int)  # file_path, job_role, duration_minutes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path: Optional[str] = None
        self._setup_ui()
        self.setAcceptDrops(True)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 60, 80, 60)
        layout.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────
        header = QHBoxLayout()
        logo_label = QLabel("Slingshot Interviewer AI")
        logo_label.setStyleSheet(
            "font-family: 'Product Sans', 'Roboto', sans-serif; font-size: 20px;"
            "color: #5f6368; font-weight: 500; letter-spacing: 0px;"
        )
        version_label = QLabel("v2.0 Beta")
        version_label.setStyleSheet("color: #1a73e8; font-size: 12px; font-weight: 500;")
        header.addWidget(logo_label)
        header.addStretch()
        header.addWidget(version_label)
        layout.addLayout(header)
        layout.addSpacing(60)

        # ── Hero text ────────────────────────────────────────────────────
        hero_card = MaterialCard()
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(60, 50, 60, 50)
        hero_layout.setSpacing(16)

        title = QLabel("Your AI Interview\nSession Begins Here")
        title.setStyleSheet(
            "font-size: 40px; font-weight: 400; color: #202124; line-height: 1.2;"
        )
        title.setWordWrap(True)

        subtitle = QLabel(
            "Upload your resume · Select your target role · Begin your adaptive voice viva"
        )
        subtitle.setStyleSheet("font-size: 16px; color: #5f6368;")

        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addSpacing(32)

        # ── File Drop Zone ────────────────────────────────────────────
        self.drop_zone = DropZone()
        self.drop_zone.file_selected.connect(self._on_file_selected)
        hero_layout.addWidget(self.drop_zone)
        hero_layout.addSpacing(20)

        # ── Job Role Row ──────────────────────────────────────────────
        role_row = QHBoxLayout()
        role_label = QLabel("Target Role:")
        role_label.setStyleSheet("color: #5f6368; font-size: 14px; font-weight: 500;")

        self.role_combo = QComboBox()
        roles = [
            "Software Engineer", "ML Engineer", "Data Scientist",
            "Frontend Engineer", "Backend Engineer", "DevOps Engineer",
            "Full Stack Engineer"
        ]
        for r in roles:
            self.role_combo.addItem(r)

        role_row.addWidget(role_label)
        role_row.addWidget(self.role_combo)
        role_row.addSpacing(20)

        # ── Duration Row ──────────────────────────────────────────────
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet("color: #5f6368; font-size: 14px; font-weight: 500;")

        self.duration_combo = QComboBox()
        durations = [
            ("10 Minutes", 10),
            ("20 Minutes", 20),
            ("30 Minutes", 30),
            ("45 Minutes", 45),
        ]
        for label, val in durations:
            self.duration_combo.addItem(label, val)

        role_row.addWidget(duration_label)
        role_row.addWidget(self.duration_combo)
        role_row.addStretch()
        hero_layout.addLayout(role_row)
        hero_layout.addSpacing(28)

        # ── Start Button ──────────────────────────────────────────────
        self.start_btn = QPushButton("BEGIN INTERVIEW  →")
        self.start_btn.setFixedHeight(52)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        hero_layout.addWidget(self.start_btn)

        layout.addWidget(hero_card)
        layout.addStretch()

        # ── Status bar ─────────────────────────────────────────────────
        self.status_label = QLabel("Upload a PDF or DOCX resume to get started")
        self.status_label.setStyleSheet("color: #5f6368; font-size: 13px;")
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignCenter)

    def _on_file_selected(self, path: str):
        self._file_path = path
        self.start_btn.setEnabled(True)
        fname = Path(path).name
        self.status_label.setText(f"✓ Loaded: {fname}")
        self.status_label.setStyleSheet("color: #1e8e3e; font-size: 13px; font-weight: 500;")

    def _on_start(self):
        if self._file_path:
            duration = self.duration_combo.currentData()
            self.start_interview.emit(self._file_path, self.role_combo.currentText(), duration)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".pdf", ".docx")):
                self.drop_zone.file_selected.emit(path)
                break


class DropZone(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptDrops(True)
        self._hover = False

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon_label = QLabel("⬆")
        self.icon_label.setStyleSheet("font-size: 32px; color: #1e3a5f;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel("Drop resume here or click to browse")
        self.text_label.setStyleSheet("color: #475569; font-size: 13px;")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.type_label = QLabel("PDF · DOCX")
        self.type_label.setStyleSheet(
            "color: #1e3a5f; font-size: 10px; letter-spacing: 2px; "
            "font-family: monospace;"
        )
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.type_label)

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Resume", "", "Documents (*.pdf *.docx *.doc)"
        )
        if path:
            self.file_selected.emit(path)
            self.icon_label.setText("✓")
            self.icon_label.setStyleSheet("font-size: 32px; color: #10b981;")
            self.text_label.setText(Path(path).name)
            self.text_label.setStyleSheet("color: #10b981; font-size: 13px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        border_color = QColor(0, 212, 255, 60 if self._hover else 25)
        bg_color = QColor(0, 212, 255, 8 if self._hover else 3)

        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 1, Qt.PenStyle.DashLine))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 10, 10)

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self._hover = True
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._hover = False
        self.update()

    def dropEvent(self, event):
        self._hover = False
        self.update()
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".pdf", ".docx")):
                self.file_selected.emit(path)
                break


# ─── Interview Screen ─────────────────────────────────────────────────────────

class InterviewScreen(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # ── Top Bar ───────────────────────────────────────────────────
        top_bar = QHBoxLayout()

        self.candidate_label = QLabel("CANDIDATE: —")
        self.candidate_label.setStyleSheet(
            "font-family: 'Roboto Mono', monospace; font-size: 13px; "
            "color: #1a73e8; letter-spacing: 1px; font-weight: 500;"
        )

        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet(
            "font-family: 'Roboto Mono', monospace; font-size: 22px; "
            "color: #202124; font-weight: 700;"
        )

        self.phase_badge = QLabel("● INTRO")
        self.phase_badge.setStyleSheet(
            "font-size: 12px; color: #1a73e8; letter-spacing: 1px; font-weight: 500; "
            "background: #e8f0fe; padding: 6px 12px; border-radius: 6px;"
        )

        top_bar.addWidget(self.candidate_label)
        top_bar.addStretch()
        top_bar.addWidget(self.phase_badge)
        top_bar.addSpacing(16)
        top_bar.addWidget(self.timer_label)
        layout.addLayout(top_bar)

        # ── Phase Tracker ──────────────────────────────────────────────
        self.phase_tracker = PhaseTrackerWidget()
        layout.addWidget(self.phase_tracker)

        # ── Main Content Row ───────────────────────────────────────────
        content_row = QHBoxLayout()
        content_row.setSpacing(16)

        # Left: AI Question + Waveform
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        ai_card = MaterialCard()
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(20, 16, 20, 16)
        ai_label = QLabel("AI INTERVIEWER")
        ai_label.setStyleSheet(
            "font-size: 11px; color: #5f6368; letter-spacing: 2px; font-weight: 500;"
        )
        self.ai_text = QLabel("Initializing session...")
        self.ai_text.setWordWrap(True)
        self.ai_text.setStyleSheet(
            "font-size: 18px; color: #202124; line-height: 1.6; padding: 8px 0; font-weight: 400;"
        )
        self.ai_text.setMinimumHeight(100)
        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.ai_text)
        left_col.addWidget(ai_card)

        # Waveform card
        wave_card = MaterialCard()
        wave_layout = QVBoxLayout(wave_card)
        wave_layout.setContentsMargins(16, 12, 16, 12)
        wave_header = QHBoxLayout()
        wave_title = QLabel("MICROPHONE")
        wave_title.setStyleSheet("font-size: 11px; color: #5f6368; letter-spacing: 2px; font-weight: 500;")
        self.mic_status = QLabel("● LISTENING")
        self.mic_status.setStyleSheet("font-size: 10px; color: #10b981; letter-spacing: 1px;")
        wave_header.addWidget(wave_title)
        wave_header.addStretch()
        self.mic_indicator = MicIndicatorWidget()
        wave_header.addWidget(self.mic_indicator)
        self.waveform = WaveformWidget()
        wave_layout.addLayout(wave_header)
        wave_layout.addWidget(self.waveform)
        left_col.addWidget(wave_card)

        # Metrics row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)
        self.metric_confidence = MetricBadge("Confidence", "—%", COLORS["cyan"])
        self.metric_wpm = MetricBadge("WPM", "—", COLORS["amber"])
        self.metric_stutter = MetricBadge("Disfluency", "—/min", COLORS["green"])
        self.metric_phase = MetricBadge("Q Count", "0", COLORS["purple"])

        for m in [self.metric_confidence, self.metric_wpm, self.metric_stutter, self.metric_phase]:
            card = MaterialCard()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.addWidget(m)
            metrics_row.addWidget(card)

        left_col.addLayout(metrics_row)
        content_row.addLayout(left_col, 3)

        # Right: Live Transcript
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        transcript_card = MaterialCard()
        t_layout = QVBoxLayout(transcript_card)
        t_layout.setContentsMargins(16, 12, 16, 16)
        t_header = QHBoxLayout()
        t_title = QLabel("LIVE TRANSCRIPT")
        t_title.setStyleSheet("font-size: 11px; color: #5f6368; letter-spacing: 2px; font-weight: 500;")
        self.transcript_count = QLabel("0 exchanges")
        self.transcript_count.setStyleSheet("font-size: 10px; color: #334155;")
        t_header.addWidget(t_title)
        t_header.addStretch()
        t_header.addWidget(self.transcript_count)

        self.transcript_feed = QTextEdit()
        self.transcript_feed.setReadOnly(True)
        self.transcript_feed.setPlaceholderText("Transcript will appear here...")
        self.transcript_feed.setMinimumHeight(280)

        t_layout.addLayout(t_header)
        t_layout.addWidget(self.transcript_feed)
        right_col.addWidget(transcript_card, 1)

        # End Interview Button
        self.end_btn = QPushButton("END INTERVIEW")
        self.end_btn.setObjectName("danger")
        self.end_btn.setFixedHeight(44)
        right_col.addWidget(self.end_btn)

        content_row.addLayout(right_col, 2)
        layout.addLayout(content_row)

        # ── Timer ──────────────────────────────────────────────────────
        self._start_time = time.time()
        self._duration_seconds = 0
        self._exchange_count = 0
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_timer)
        self._timer.start(1000)

    def _update_timer(self):
        elapsed = int(time.time() - self._start_time)
        remaining = max(0, self._duration_seconds - elapsed)
        
        m, s = divmod(remaining, 60)
        self.timer_label.setText(f"{m:02d}:{s:02d}")
        
        # When time runs out, disable button and auto-end
        if remaining == 0 and self.end_btn.isEnabled():
            self.end_btn.setEnabled(False)
            self.end_btn.setText("GENERATING REPORT...")
            # Fire an event that we reached time
            parent = self.parent()
            while parent is not None and not hasattr(parent, 'on_end_interview'):
                parent = parent.parent()
            if parent:
                parent.on_end_interview()

    def set_candidate(self, name: str, role: str, duration_minutes: int):
        self.candidate_label.setText(f"CANDIDATE: {name.upper()}  ·  {role.upper()}")
        self._duration_seconds = duration_minutes * 60
        self._start_time = time.time()
        self.end_btn.setEnabled(True)
        self.end_btn.setText("END INTERVIEW")
        self.mic_indicator.set_listening()

    def set_phase(self, phase_name: str):
        self.phase_badge.setText(f"● {phase_name.replace('_', ' ')}")
        self.phase_tracker.set_phase(phase_name)

    def append_ai_text(self, chunk: str):
        """Append streamed text chunk to AI display."""
        current = self.ai_text.text()
        if current == "Initializing session...":
            current = ""
        self.ai_text.setText(current + chunk)

    def set_ai_text_complete(self, full_text: str):
        self.ai_text.setText(full_text)

    def append_transcript(self, speaker: str, text: str, is_interim: bool = False):
        if is_interim:
            cursor = self.transcript_feed.textCursor()
            # Replace last line if it's an interim result
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            color = "#475569"
            self.transcript_feed.append(
                f'<span style="color:{color}; font-style:italic;">{text}</span>'
            )
        else:
            if speaker == "AI":
                color = COLORS["cyan"]
                prefix = "◈ AI"
            else:
                color = "#94a3b8"
                prefix = "▸ YOU"
            self.transcript_feed.append(
                f'<br><b style="color:{color}; font-size:10px; '
                f'letter-spacing:1px;">{prefix}</b><br>'
                f'<span style="color:#cbd5e1;">{text}</span>'
            )
            self._exchange_count += 1
            self.transcript_feed.verticalScrollBar().setValue(
                self.transcript_feed.verticalScrollBar().maximum()
            )
            self.transcript_count.setText(f"{self._exchange_count} exchanges")

    def update_behavioral(self, snapshot):
        """Update live metrics from behavioral analyzer snapshot."""
        conf = snapshot.confidence_index
        color = (COLORS["green"] if conf >= 70 else
                 COLORS["amber"] if conf >= 45 else COLORS["red"])
        self.metric_confidence.update_value(f"{conf:.0f}%", color)
        self.metric_wpm.update_value(f"{snapshot.wpm:.0f}")
        self.metric_stutter.update_value(f"{snapshot.stutter_events_per_min:.1f}/m")
        self.waveform.push_level(min(1.0, snapshot.voice_energy / 100))

    def update_question_count(self, count: int):
        self.metric_phase.update_value(str(count))


# ─── Report Screen ────────────────────────────────────────────────────────────

class ReportScreen(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title = QLabel("Slingshot Interviewer AI Report")
        title.setStyleSheet(
            "font-family: 'Product Sans', 'Roboto', sans-serif; font-size: 20px; "
            "color: #1a73e8; font-weight: 500; letter-spacing: 0px;"
        )
        self.export_btn = QPushButton("EXPORT PDF")
        self.export_btn.setObjectName("ghost")
        self.export_btn.setFixedWidth(130)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.export_btn)
        outer.addLayout(header)

        # Scroll area for report content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget()
        scroll.setWidget(content)
        self.report_layout = QVBoxLayout(content)
        self.report_layout.setSpacing(16)
        outer.addWidget(scroll)

    def populate_report(self, behavior_report, session_data: dict, skill_matches: list,
                        ai_feedback: str = ""):
        """Render the full report with charts and metrics."""
        layout = self.report_layout

        # Clear existing
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # ── Score Summary Row ─────────────────────────────────────────
        scores_card = MaterialCard()
        scores_layout = QHBoxLayout(scores_card)
        scores_layout.setContentsMargins(24, 20, 24, 20)

        score_metrics = [
            ("Overall\nReadiness", f"{session_data.get('readiness_score', 0):.0f}",
             COLORS["cyan"], "%"),
            ("Communication\nScore", f"{behavior_report.communication_score:.0f}",
             COLORS["green"], "%"),
            ("Fluency\nScore", f"{behavior_report.fluency_score:.0f}",
             COLORS["amber"], "%"),
            ("Confidence\nIndex", f"{behavior_report.final_confidence_score:.0f}",
             COLORS["purple"], "%"),
            ("Avg Speed",
             f"{behavior_report.average_wpm:.0f}", COLORS["text_primary"], " wpm"),
        ]

        for label, val, color, unit in score_metrics:
            badge_widget = QWidget()
            badge_layout = QVBoxLayout(badge_widget)
            badge_layout.setContentsMargins(12, 0, 12, 0)

            val_label = QLabel(f"{val}{unit}")
            val_label.setStyleSheet(
                f"font-family: 'JetBrains Mono', monospace; font-size: 28px; "
                f"font-weight: 800; color: {color};"
            )
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 11px; color: #475569; letter-spacing: 0.5px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)

            badge_layout.addWidget(val_label)
            badge_layout.addWidget(lbl)
            scores_layout.addWidget(badge_widget)

        layout.addWidget(scores_card)

        # ── AI Feedback (What to Improve + Topics to Revise) ──────────
        if ai_feedback:
            self._add_ai_feedback(layout, ai_feedback)

        # ── Skill Readiness Chart ─────────────────────────────────────
        if skill_matches:
            self._add_skill_chart(layout, skill_matches)

        # ── Behavioral Timeline ────────────────────────────────────────
        if behavior_report.snapshots:
            self._add_confidence_chart(layout, behavior_report)

        # ── Recommendations ───────────────────────────────────────────
        self._add_recommendations(layout, behavior_report.recommendations)

    def _add_skill_chart(self, layout, skill_matches):
        """Add horizontal bar chart of skill readiness scores."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            fig, ax = plt.subplots(figsize=(9, 0.55 * min(len(skill_matches), 10) + 1.5))
            fig.patch.set_facecolor("#0c1220")
            ax.set_facecolor("#060d1a")

            skills = [m.skill for m in skill_matches[:10]][::-1]
            scores = [m.score * 100 for m in skill_matches[:10]][::-1]
            colors = ["#10b981" if s >= 60 else "#f59e0b" if s >= 35 else "#ef4444"
                      for s in scores]

            bars = ax.barh(skills, scores, color=colors, height=0.6, alpha=0.9)
            ax.set_xlim(0, 100)
            ax.axvline(x=60, color="rgba(0,212,255,0.3)", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlabel("Match Score (%)", color="#64748b", fontsize=10)
            ax.tick_params(colors="#64748b", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#1e293b")
            ax.set_title("SKILL READINESS", color="#00d4ff", fontsize=11,
                         fontfamily="monospace", pad=12, loc="left", fontweight="bold")

            for bar, score in zip(bars, scores):
                ax.text(min(score + 2, 95), bar.get_y() + bar.get_height() / 2,
                        f"{score:.0f}%", va="center", color="#94a3b8", fontsize=8)

            plt.tight_layout()

            card = MaterialCard()
            card_layout = QVBoxLayout(card)
            canvas = FigureCanvasQTAgg(fig)
            card_layout.addWidget(canvas)
            layout.addWidget(card)
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Skill chart render error: {e}")
            # Fallback text display
            card = MaterialCard()
            cl = QVBoxLayout(card)
            lbl = QLabel("SKILL ANALYSIS")
            lbl.setStyleSheet("font-size: 11px; color: #1e3a5f; letter-spacing: 3px; font-family: monospace;")
            cl.addWidget(lbl)
            for m in skill_matches[:8]:
                row = QHBoxLayout()
                skill_lbl = QLabel(m.skill)
                skill_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
                score_lbl = QLabel(f"{m.score*100:.0f}%")
                color = "#10b981" if m.score >= 0.6 else "#f59e0b" if m.score >= 0.35 else "#ef4444"
                score_lbl.setStyleSheet(f"color: {color}; font-family: monospace; font-size: 12px;")
                row.addWidget(skill_lbl)
                row.addStretch()
                row.addWidget(score_lbl)
                cl.addLayout(row)
            layout.addWidget(card)

    def _add_confidence_chart(self, layout, behavior_report):
        """Add confidence timeline chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

            snaps = behavior_report.snapshots
            times = [(s.timestamp - snaps[0].timestamp) for s in snaps]
            confidence = [s.confidence_index for s in snaps]
            wpm = [s.wpm for s in snaps]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
            fig.patch.set_facecolor("#0c1220")

            for ax in [ax1, ax2]:
                ax.set_facecolor("#060d1a")
                for spine in ax.spines.values():
                    spine.set_color("#1e293b")
                ax.tick_params(colors="#64748b", labelsize=9)

            # Confidence
            ax1.fill_between(times, confidence, alpha=0.15, color="#00d4ff")
            ax1.plot(times, confidence, color="#00d4ff", linewidth=1.5)
            ax1.set_ylabel("Confidence %", color="#64748b", fontsize=9)
            ax1.set_ylim(0, 100)
            ax1.axhline(60, color="#1e293b", linewidth=0.8, linestyle="--")
            ax1.set_title("BEHAVIORAL TIMELINE", color="#00d4ff", fontsize=11,
                          fontfamily="monospace", pad=10, loc="left", fontweight="bold")

            # WPM
            ax2.fill_between(times, wpm, alpha=0.15, color="#f59e0b")
            ax2.plot(times, wpm, color="#f59e0b", linewidth=1.5)
            ax2.set_ylabel("WPM", color="#64748b", fontsize=9)
            ax2.set_xlabel("Time (seconds)", color="#64748b", fontsize=9)
            ax2.axhline(140, color="#1e293b", linewidth=0.8, linestyle="--")

            plt.tight_layout()

            card = MaterialCard()
            card_layout = QVBoxLayout(card)
            canvas = FigureCanvasQTAgg(fig)
            card_layout.addWidget(canvas)
            layout.addWidget(card)
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Confidence chart error: {e}")

    def _add_ai_feedback(self, layout, ai_feedback: str):
        """Add AI-generated improvement feedback and topics to revise."""
        card = MaterialCard()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 24, 28, 24)
        cl.setSpacing(14)

        header = QLabel("🤖  AI INTERVIEW FEEDBACK")
        header.setFont(QFont("Roboto", 13, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {COLORS['cyan']}; letter-spacing: 1px;")
        cl.addWidget(header)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background: {COLORS['border']}; max-height: 1px;")
        cl.addWidget(sep)

        # Parse and display feedback lines with icons
        for line in ai_feedback.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Section headers
            if line.upper().startswith("WHAT") or line.upper().startswith("AREAS") or "IMPROVE" in line.upper():
                lbl = QLabel(f"🔧  {line}")
                lbl.setFont(QFont("Roboto", 12, QFont.Weight.Bold))
                lbl.setStyleSheet(f"color: {COLORS['text_primary']}; margin-top: 8px;")
                lbl.setWordWrap(True)
                cl.addWidget(lbl)
            elif line.upper().startswith("TOPIC") or "REVISE" in line.upper() or "STUDY" in line.upper():
                lbl = QLabel(f"📚  {line}")
                lbl.setFont(QFont("Roboto", 12, QFont.Weight.Bold))
                lbl.setStyleSheet(f"color: {COLORS['text_primary']}; margin-top: 8px;")
                lbl.setWordWrap(True)
                cl.addWidget(lbl)
            elif line.upper().startswith("WHAT YOU DID") or "WELL" in line.upper() or "STRENGTH" in line.upper():
                lbl = QLabel(f"✅  {line}")
                lbl.setFont(QFont("Roboto", 12, QFont.Weight.Bold))
                lbl.setStyleSheet(f"color: {COLORS['green']}; margin-top: 8px;")
                lbl.setWordWrap(True)
                cl.addWidget(lbl)
            elif line.startswith(("-", "•", "*", "·")):
                # Bullet points
                clean = line.lstrip("-•*· ")
                lbl = QLabel(f"    ▸  {clean}")
                lbl.setFont(QFont("Roboto", 11))
                lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 2px 0;")
                lbl.setWordWrap(True)
                cl.addWidget(lbl)
            elif line[0].isdigit() and "." in line[:4]:
                # Numbered items
                lbl = QLabel(f"    {line}")
                lbl.setFont(QFont("Roboto", 11))
                lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 2px 0;")
                lbl.setWordWrap(True)
                cl.addWidget(lbl)
            else:
                lbl = QLabel(line)
                lbl.setFont(QFont("Roboto", 11))
                lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 1px 0;")
                lbl.setWordWrap(True)
                cl.addWidget(lbl)

        layout.addWidget(card)

    def _add_recommendations(self, layout, recommendations: list[str]):
        card = MaterialCard()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(24, 20, 24, 20)
        cl.setSpacing(12)

        header = QLabel("RECOMMENDATIONS")
        header.setStyleSheet("font-size: 11px; color: #1e3a5f; letter-spacing: 3px; font-family: monospace;")
        cl.addWidget(header)

        for i, rec in enumerate(recommendations):
            icon = "●" if i == 0 else "○"
            color = COLORS["cyan"] if i == 0 else "#475569"
            lbl = QLabel(f"{icon}  {rec}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"font-size: 13px; color: {color}; padding: 4px 0; line-height: 1.6;")
            cl.addWidget(lbl)

        layout.addWidget(card)
        layout.addStretch()


# ─── Main Application Window ──────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Interviewer")
        self.setMinimumSize(1200, 750)
        self.resize(1360, 820)

        # Dark title bar on Windows
        self._setup_dark_titlebar()

        self.setStyleSheet(STYLESHEET)

        # Central stacked widget (screens)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Create screens
        self.landing = LandingScreen()
        self.interview = InterviewScreen()
        self.report = ReportScreen()

        self._stack.addWidget(self.landing)    # index 0
        self._stack.addWidget(self.interview)  # index 1
        self._stack.addWidget(self.report)     # index 2

        # Connect navigation signals
        self.landing.start_interview.connect(self.on_start_interview)
        self.interview.end_btn.clicked.connect(self.on_end_interview)
        self.report.export_btn.clicked.connect(self.on_export_pdf)

        # App state
        self._session_worker = None
        self._behavioral_analyzer = None
        self._audio_manager = None
        self._current_ai_response = ""
        self._resume_profile = None

    def _setup_dark_titlebar(self):
        """Enable dark title bar on Windows 10+."""
        try:
            if sys.platform == "win32":
                import ctypes
                HWND_OFFSET = 0
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                    ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
                )
        except Exception:
            pass

    # ── Screen Navigation ──────────────────────────────────────────────────────

    def show_landing(self):
        self._stack.setCurrentIndex(0)

    def show_interview(self):
        self._stack.setCurrentIndex(1)

    def show_report(self):
        self._stack.setCurrentIndex(2)

    # ── Session Lifecycle ──────────────────────────────────────────────────────

    def on_start_interview(self, file_path: str, job_role: str, duration_minutes: int = 20):
        """
        Called when user clicks BEGIN INTERVIEW.
        Runs resume processing + spins up the session worker.
        """
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
        import threading

        # ── Styled Progress Dialog ────────────────────────────────
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Analyzing Resume")
        progress_dialog.setFixedSize(460, 200)
        progress_dialog.setModal(True)
        progress_dialog.setStyleSheet(f"""
            QDialog {{
                background: {COLORS['bg_panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 16px;
            }}
        """)

        layout = QVBoxLayout(progress_dialog)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        # Title
        title_label = QLabel("📄  Analyzing Resume...")
        title_label.setFont(QFont("Roboto", 16, QFont.Weight.Bold))
        title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Percentage label
        pct_label = QLabel("0%")
        pct_label.setFont(QFont("Roboto", 28, QFont.Weight.Bold))
        pct_label.setStyleSheet(f"color: {COLORS['cyan']};")
        pct_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(pct_label)

        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(8)
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {COLORS['bg_dark']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['cyan']}, stop:1 {COLORS['cyan_dim']});
                border-radius: 4px;
            }}
        """)
        layout.addWidget(progress_bar)

        # Step description label
        step_label = QLabel("Preparing...")
        step_label.setFont(QFont("Roboto", 11))
        step_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(step_label)

        progress_dialog.show()
        QApplication.processEvents()

        # Thread-safe progress state
        progress_state = [0, "Preparing..."]  # [percent, message]

        def _progress_callback(pct, msg):
            progress_state[0] = pct
            progress_state[1] = msg

        def _process():
            try:
                from resume_engine import ResumeIntelligenceEngine
                engine = ResumeIntelligenceEngine()
                self._resume_profile = engine.process(
                    file_path, job_role, progress_callback=_progress_callback
                )
                return True
            except Exception as e:
                return str(e)

        result_holder = [None]

        def _worker():
            result_holder[0] = _process()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        last_pct = -1
        while t.is_alive():
            QApplication.processEvents()
            # Update UI from progress state
            if progress_state[0] != last_pct:
                last_pct = progress_state[0]
                progress_bar.setValue(last_pct)
                pct_label.setText(f"{last_pct}%")
                step_label.setText(progress_state[1])
                QApplication.processEvents()
            time.sleep(0.03)

        # Ensure 100% is shown briefly
        progress_bar.setValue(100)
        pct_label.setText("100%")
        step_label.setText("Resume analysis complete!")
        QApplication.processEvents()
        time.sleep(0.3)

        progress_dialog.close()

        if result_holder[0] is not True:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Resume Error", str(result_holder[0]))
            return

        self._launch_session(duration_minutes)

    def _launch_session(self, duration_minutes: int):
        from voice_pipeline import (
            InterviewSession, SessionWorker, build_system_prompt, PHASE_SEQUENCE
        )
        from behavioral_analyzer import BehavioralAnalyzer
        from audio_manager import AudioManager

        profile = self._resume_profile
        weak_skills = [
            m.skill for m in profile.skill_matches
            if m.score < 0.4
        ][:5]

        # Build interview session
        session = InterviewSession(
            candidate_name=profile.candidate_name,
            job_role=profile.job_role,
            system_prompt=build_system_prompt(profile.summary_for_llm, profile.job_role),
            weak_skills=weak_skills,
        )

        # Behavioral analyzer
        self._behavioral_analyzer = BehavioralAnalyzer(
            on_snapshot_callback=self._on_behavioral_snapshot,
            on_report_callback=self._on_behavioral_report,
        )
        self._behavioral_analyzer.start()

        # Audio manager
        self._audio_manager = AudioManager()
        self._audio_manager.initialize()

        # Session worker (QThread with asyncio event loop)
        self._session_worker = SessionWorker(session)
        self._session_worker_thread = QThread()
        self._session_worker.moveToThread(self._session_worker_thread)
        self._session_worker_thread.started.connect(self._session_worker.run)

        # Connect signals
        self._session_worker.interim_transcript.connect(self._on_interim_transcript)
        self._session_worker.final_transcript.connect(self._on_final_transcript)
        self._session_worker.ai_text_chunk.connect(self._on_ai_text_chunk)
        self._session_worker.audio_chunk_ready.connect(self._on_audio_chunk)
        self._session_worker.phase_changed.connect(self._on_phase_changed)
        self._session_worker.session_complete.connect(self._on_session_complete)
        self._session_worker.error_occurred.connect(self._on_error)

        # Register audio routing
        self._audio_manager.register_frame_callback(
            lambda b: self._session_worker.push_audio(b)
        )
        self._audio_manager.register_frame_callback(
            self._behavioral_analyzer.push_frame
        )

        # Update interview screen
        self.interview.set_candidate(profile.candidate_name, profile.job_role, duration_minutes)
        self.interview.update_question_count(0)
        self.show_interview()

        # Start session
        self._session_worker_thread.start()

    def on_end_interview(self):
        """Trigger an early generation of the final score, manually overriding the session loops."""
        self.interview.end_btn.setEnabled(False)
        self.interview.end_btn.setText("GENERATING REPORT...")
        if self._session_worker:
            self._session_worker.end_session()

    def on_export_pdf(self):
        """Export the report as a PDF file."""
        try:
            from utils.pdf_exporter import ReportPDFExporter
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Report", "interview_report.pdf", "PDF Files (*.pdf)"
            )
            if path:
                exporter = ReportPDFExporter()
                exporter.export(path, self._resume_profile, self._last_behavior_report,
                                self._last_session_data)
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Exported", f"Report saved to:\n{path}")
        except Exception as e:
            logger.error(f"PDF export error: {e}")

    # ── Signal Handlers ────────────────────────────────────────────────────────

    def _on_interim_transcript(self, text: str):
        self.interview.append_transcript("YOU", text, is_interim=True)

    def _on_final_transcript(self, text: str):
        self.interview.append_transcript("YOU", text, is_interim=False)
        # User finished speaking — AI will now respond. Keep mic LISTENING until AI speaks.
        self.interview.mic_indicator.set_listening()
        if self._behavioral_analyzer:
            self._behavioral_analyzer.update_word_count(text)

    def _on_ai_text_chunk(self, chunk: str):
        self._current_ai_response += chunk
        self.interview.append_ai_text(chunk)
        # Switch indicator to AI Speaking on first text chunk
        self.interview.mic_indicator.set_ai_speaking()

    def _on_audio_chunk(self, audio_bytes: bytes):
        if self._audio_manager:
            self._audio_manager.play_audio_chunk(audio_bytes)

    def _on_phase_changed(self, phase_name: str):
        self.interview.set_phase(phase_name)
        self._current_ai_response = ""
        # AI is generating next question — set to idle until audio starts
        self.interview.mic_indicator.set_idle()

    def _on_behavioral_snapshot(self, snapshot):
        self.interview.update_behavioral(snapshot)

    def _on_behavioral_report(self, report):
        self._last_behavior_report = report

    def _generate_ai_feedback(self, session_data: dict) -> str:
        """Use Groq LLM to generate personalized interview feedback."""
        try:
            import httpx
            conversation = session_data.get("conversation_history", [])
            if not conversation:
                return ""

            conv_text = ""
            for msg in conversation:
                role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
                conv_text += f"{role}: {msg['content']}\n"

            job_role = session_data.get("job_role", "Software Engineer")

            prompt = f"""You are an expert interview coach. Analyze the following {job_role} interview transcript and provide structured feedback.

Please respond in EXACTLY this format (use these exact headers):

What You Did Well:
- [list 2-3 specific strengths from the interview]

Areas for Improvement:
- [list 3-4 specific things the candidate should improve]

Topics to Revise:
- [list 4-6 specific technical topics the candidate should study based on weak answers]

Here is the interview transcript:
{conv_text[:4000]}
"""
            # Synchronous call since we're already in the main thread
            import os
            api_key = os.environ.get("GROQ_API_KEY", "")
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.4,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"AI feedback generation failed: {e}")
            return ""

    def _on_session_complete(self, session_data: dict):
        self._last_session_data = session_data
        if self._behavioral_analyzer:
            behavior_report = self._behavioral_analyzer.stop()
        else:
            behavior_report = None

        if self._audio_manager:
            self._audio_manager.shutdown()

        session_data["readiness_score"] = getattr(
            self._resume_profile, "overall_readiness", 0
        )

        # Record final AI response in transcript
        if self._current_ai_response:
            self.interview.append_transcript("AI", self._current_ai_response)

        # Generate AI feedback from conversation
        ai_feedback = self._generate_ai_feedback(session_data)

        # Build report
        if behavior_report and self._resume_profile:
            self.report.populate_report(
                behavior_report,
                session_data,
                self._resume_profile.skill_matches,
                ai_feedback=ai_feedback,
            )
        self.show_report()

    def _on_error(self, error_msg: str):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Session Error", error_msg)
        logger.error(f"Session error: {error_msg}")

    def closeEvent(self, event):
        if self._session_worker:
            self._session_worker.stop()
        if self._audio_manager:
            self._audio_manager.shutdown()
        if self._behavioral_analyzer:
            self._behavioral_analyzer.stop()
        event.accept()
