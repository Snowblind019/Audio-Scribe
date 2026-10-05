"""Entry point.

    python -m audioscribe [FILE]        open the app (optionally with a file)
    python -m audioscribe --check       verify that every part is installed
    python -m audioscribe --write-icons DIR
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from . import APP_ID, APP_NAME, ORG_NAME, __version__

COLORS = {
    "window": "#262B33",
    "panel": "#2E343D",
    "raised": "#353C46",
    "line": "#3F4651",
    "base": "#22272E",
    "alt": "#272D35",
    "text": "#DCE1E8",
    "muted": "#8D97A5",
    "accent": "#F0B23E",
    "accent_hover": "#F5C362",
    "accent_text": "#211A0C",
    "select": "#3E5F86",
}

STYLE = """
QWidget#Inspector {{ background: {panel}; }}
QFrame#InspectorFrame {{ background: {panel}; border: none; border-right: 1px solid {line}; }}
QScrollArea#InspectorPage {{ background: {panel}; border: none; }}
QFrame#Section {{ border: none; border-bottom: 1px solid {line}; }}
QLabel#SectionTitle {{ font-weight: 600; color: {text}; }}
QLabel#Hint, QLabel#Meta {{ color: {muted}; }}
QLabel#FieldName {{ color: {muted}; font-weight: 600; }}
QLabel#FileTitle {{ font-size: 14pt; font-weight: 600; }}
QWidget#Header {{ background: {window}; border-bottom: 1px solid {line}; }}
QWidget#Transport {{ background: {panel}; border-top: 1px solid {line}; }}
QPushButton#Primary {{
    background: {accent}; color: {accent_text}; border: none; border-radius: 5px;
    padding: 9px 14px; font-weight: 600;
}}
QPushButton#Primary:hover {{ background: {accent_hover}; }}
QPushButton#Primary:disabled {{ background: #5B5444; color: #9C9482; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {muted}; padding: 7px 16px; border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}
QSplitter::handle {{ background: {line}; }}
QTreeWidget {{ background: {base}; alternate-background-color: {alt}; border: none; }}
QHeaderView::section {{
    background: {panel}; color: {muted}; border: none; border-right: 1px solid {line};
    border-bottom: 1px solid {line}; padding: 4px 8px;
}}
QStatusBar {{ background: {panel}; color: {muted}; border-top: 1px solid {line}; }}
QStatusBar QLabel {{ color: {muted}; }}
QProgressBar {{ background: {base}; border: 1px solid {line}; border-radius: 3px; max-height: 10px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 2px; }}
QToolTip {{ background: {raised}; color: {text}; border: 1px solid {line}; padding: 5px; }}
QWidget#Toolbar {{ background: {window}; border-bottom: 1px solid {line}; }}
QToolButton#MuteButton, QToolButton#SoloButton, QToolButton#KeyButton {{
    background: {raised}; border: 1px solid {line}; border-radius: 4px; color: {muted}; font-weight: 700;
}}
QToolButton#MuteButton, QToolButton#SoloButton {{
    min-width: 24px; max-width: 24px; min-height: 22px; max-height: 22px;
}}
QToolButton#KeyButton {{ min-height: 24px; padding: 0px 2px; }}
QToolButton#MuteButton:checked {{ background: #D9645B; border-color: #D9645B; color: #22110F; }}
QToolButton#SoloButton:checked {{ background: {accent}; border-color: {accent}; color: {accent_text}; }}
QToolButton#KeyButton:checked {{ background: {select}; border-color: #5E86B3; color: #FFFFFF; }}
QToolButton#ModeButton {{
    background: {raised}; border: 1px solid {line}; border-radius: 4px; padding: 4px 12px; font-weight: 600;
}}
QToolButton#ModeButton:checked {{ background: {accent}; border-color: {accent}; color: {accent_text}; }}
QToolButton#LoopButton {{
    background: {raised}; border: 1px solid {line}; border-radius: 4px; padding: 4px 10px; font-weight: 600;
}}
QToolButton#LoopButton:checked {{ background: #5CC6C0; border-color: #5CC6C0; color: #0E2220; }}
QToolButton#LoopButton:disabled, QToolButton#ModeButton:disabled {{ color: #646D79; }}
""".format(**COLORS)


def cache_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    path = base / APP_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def _setup_logging() -> None:
    log_file = cache_dir() / f"{APP_ID}.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, mode="w", encoding="utf-8")]
    # pythonw on Windows has no console; libraries that print would crash without this.
    if sys.stdout is None or sys.stderr is None:
        stream = open(log_file.with_suffix(".out.log"), "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    else:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _privacy_defaults() -> None:
    """Model downloads are the only network traffic. Turn off the optional usage
    reporting that two of the libraries have."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import onnxruntime
        onnxruntime.disable_telemetry_events()  # only does anything on Windows builds
    except Exception:
        pass


def apply_theme(app) -> None:
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    c = {k: QColor(v) for k, v in COLORS.items()}
    pal = QPalette()
    pal.setColor(QPalette.Window, c["window"])
    pal.setColor(QPalette.WindowText, c["text"])
    pal.setColor(QPalette.Base, c["base"])
    pal.setColor(QPalette.AlternateBase, c["alt"])
    pal.setColor(QPalette.Text, c["text"])
    pal.setColor(QPalette.Button, c["raised"])
    pal.setColor(QPalette.ButtonText, c["text"])
    pal.setColor(QPalette.ToolTipBase, c["raised"])
    pal.setColor(QPalette.ToolTipText, c["text"])
    pal.setColor(QPalette.Highlight, c["select"])
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.PlaceholderText, c["muted"])
    pal.setColor(QPalette.Link, c["accent"])
    pal.setColor(QPalette.BrightText, c["accent"])
    pal.setColor(QPalette.Light, QColor("#4A525E"))
    pal.setColor(QPalette.Mid, c["line"])
    pal.setColor(QPalette.Dark, QColor("#1C2026"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor("#646D79"))
    pal.setColor(QPalette.Disabled, QPalette.Button, QColor("#2F353E"))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def run_check() -> int:
    """Import every component and report versions. Used by the installers."""
    ok = True

    def report(name: str, func) -> None:
        nonlocal ok
        try:
            print(f"  {name:<22} {func()}")
        except Exception as exc:  # report and keep going so the user sees everything
            ok = False
            print(f"  {name:<22} MISSING ({exc})")

    print(f"{APP_NAME} {__version__}, Python {sys.version.split()[0]}")
    report("Qt (PySide6)", lambda: __import__("PySide6").__version__)
    report("Qt audio playback", lambda: (__import__("PySide6.QtMultimedia"), "ok")[1])
    report("Audio decoding (PyAV)", lambda: __import__("av").__version__)
    report("Whisper", lambda: __import__("faster_whisper").__version__)
    report("ONNX Runtime", lambda: __import__("onnxruntime").__version__)
    report("librosa", lambda: __import__("librosa").__version__)

    def instrument_sounds() -> str:
        import numpy
        import scipy
        from . import synth
        if len(synth.render_single("piano", 60, 0.7, 0.3)) == 0:
            raise RuntimeError("the piano did not make any sound")
        return f"ok (numpy {numpy.__version__}, scipy {scipy.__version__})"

    report("Instrument sounds", instrument_sounds)

    def basic_pitch() -> str:
        logging.disable(logging.WARNING)
        try:
            from basic_pitch import ONNX_PRESENT
            from basic_pitch.inference import predict  # noqa: F401
        finally:
            logging.disable(logging.NOTSET)
        if not ONNX_PRESENT:
            raise RuntimeError("onnxruntime not found")
        return "ok"

    report("Basic Pitch", basic_pitch)

    from .engine import cuda_available, stems_available
    if stems_available():
        report("PyTorch", lambda: __import__("torch").__version__)
        report("Demucs (stems)", lambda: (__import__("demucs.apply"), "ok")[1])
    else:
        print(f"  {'Demucs (stems)':<22} not installed (optional)")
    print(f"  {'NVIDIA GPU for Whisper':<22} {'yes' if cuda_available() else 'no, using the CPU'}")
    print("All good." if ok else "Something is missing, see above.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    if "--version" in argv:
        print(f"{APP_NAME} {__version__}")
        return 0
    if "--check" in argv:
        return run_check()
    if "--write-icons" in argv:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QGuiApplication

        from .icon import write_icons
        _app = QGuiApplication(argv[:1])
        target = argv[argv.index("--write-icons") + 1] if len(argv) > argv.index("--write-icons") + 1 else "."
        for path in write_icons(target):
            print(path)
        return 0

    _setup_logging()
    _privacy_defaults()
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import QApplication

    from .icon import draw_icon
    from .window import MainWindow

    QApplication.setApplicationName(APP_NAME)
    QApplication.setOrganizationName(ORG_NAME)
    QApplication.setApplicationVersion(__version__)
    QApplication.setDesktopFileName(APP_ID)  # lets Wayland compositors match the menu icon
    app = QApplication(argv)
    apply_theme(app)
    app.setWindowIcon(QIcon(QPixmap.fromImage(draw_icon(256))))

    window = MainWindow()
    window.show()
    files = [a for a in argv[1:] if not a.startswith("-") and Path(a).is_file()]
    if files:
        window.open_file(files[0])
    return app.exec()
