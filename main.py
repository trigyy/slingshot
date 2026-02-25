"""
AI Interviewer — Entry Point
==============================
Loads environment variables, validates API keys,
and launches the PyQt6 application.
"""

import sys
import os
import logging
from pathlib import Path

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai_interviewer")

# ── Load .env ─────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    logger.info(f"Loaded .env from {env_path}")
else:
    logger.warning(
        ".env file not found. Make sure GROQ_API_KEY, DEEPGRAM_API_KEY, "
        "ELEVENLABS_API_KEY are set as environment variables."
    )

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


def validate_env():
    """Validate required API keys are present."""
    required = {
        "GROQ_API_KEY": "https://console.groq.com",
        "DEEPGRAM_API_KEY": "https://console.deepgram.com",
        "ELEVENLABS_API_KEY": "https://elevenlabs.io/app/sign-up",
    }
    missing = []
    for key, url in required.items():
        if not os.environ.get(key):
            missing.append(f"  {key}  →  {url}")

    if missing:
        print("\n⚠  Missing API Keys (free tier signup required):\n")
        for m in missing:
            print(m)
        print("\nCreate a .env file with these values or set them as environment variables.")
        print("The system will start but voice features will be unavailable.\n")


def main():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont, QFontDatabase

    validate_env()

    app = QApplication(sys.argv)
    app.setApplicationName("AI Interviewer")
    app.setApplicationVersion("2.0.0")

    # Load custom fonts if available
    fonts_dir = Path(__file__).parent / "assets" / "fonts"
    if fonts_dir.exists():
        for font_file in fonts_dir.glob("*.ttf"):
            QFontDatabase.addApplicationFont(str(font_file))

    # Set default app font
    app.setFont(QFont("Outfit", 12))

    from main_window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("AI Interviewer launched successfully")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
