"""The app icon, drawn in code so there is one source for every size.

The installers call `python -m audioscribe --write-icons DIR` to save a PNG
(Linux menu entry) and an ICO (Windows shortcut).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen


def draw_icon(size: int = 256) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    s = size / 256.0
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)

    # Body
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#2E343D"))
    p.drawRoundedRect(QRectF(10 * s, 10 * s, 236 * s, 236 * s), 46 * s, 46 * s)

    # Keyboard strip on the left
    p.setBrush(QColor("#C9CED6"))
    p.drawRoundedRect(QRectF(30 * s, 46 * s, 34 * s, 164 * s), 7 * s, 7 * s)
    p.setBrush(QColor("#30353D"))
    for y in (62, 94, 142, 174):
        p.drawRoundedRect(QRectF(30 * s, y * s, 20 * s, 14 * s), 3 * s, 3 * s)

    # Notes, one per track color
    notes = [(80, 58, 66, "#69A7E0"), (120, 98, 92, "#E58AA7"),
             (92, 142, 58, "#B39DEB"), (160, 178, 62, "#8CCB72")]
    for x, y, w, color in notes:
        p.setBrush(QColor(color))
        p.drawRoundedRect(QRectF(x * s, y * s, w * s, 22 * s), 6 * s, 6 * s)

    # Playhead
    p.setPen(QPen(QColor("#F0B23E"), 7 * s, Qt.SolidLine, Qt.RoundCap))
    p.drawLine(int(150 * s), int(40 * s), int(150 * s), int(216 * s))
    p.end()
    return img


def write_icons(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / "audio-scribe.png"
    ico = folder / "audio-scribe.ico"
    draw_icon(256).save(str(png), "PNG")
    draw_icon(256).save(str(ico), "ICO")
    return [png, ico]
