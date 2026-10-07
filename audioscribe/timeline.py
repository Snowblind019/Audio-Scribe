"""The Tracks view: every part of the song as a lane, like the arrangement window of a DAW.

    +----------------+------------------------------------------------+
    |                | time ruler                                     |
    +----------------+------------------------------------------------+
    | Vocals  M S    | ~~~waveform~~~ with the part's notes over it   |
    | vol ---o--     |                                                |
    +----------------+------------------------------------------------+
    | Bass ...       | ...                                            |

Each header has mute, solo, volume, pan and a level meter, the same controls as the Mixer.
Click a lane to move the playhead, drag across it to select a span (shared with the piano
roll), double-click a part to edit its notes in the piano roll. A take you recorded can be
dragged left or right to line it up with the song.

The waveforms are worked out once per part in the background (the loudest point in every
256 samples), then drawn at any zoom from that.
"""

from __future__ import annotations

import bisect

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QAbstractScrollArea, QFrame, QHBoxLayout, QLabel, QScrollArea, QToolButton,
                               QVBoxLayout, QWidget)

from . import playback
from .i18n import tr
from .mixer_view import Meter, PanSlider, VolumeSlider, db_text
from .music import format_time

HEADER_W = 200
RULER_H = 24
LANE_H = 76
BUCKET = 256                      # samples per waveform point
BUCKETS_PER_SECOND = playback.SR / BUCKET
TIME_STEPS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
MAX_PPS = 2000.0

COL = {
    "bg": QColor("#262B33"),
    "lane": QColor("#2A3038"),
    "lane_alt": QColor("#272D35"),
    "line": QColor("#3F4651"),
    "panel": QColor("#2E343D"),
    "ruler_text": QColor("#9AA4B2"),
    "text": QColor("#DCE1E8"),
    "muted": QColor("#7F8996"),
    "beat": QColor("#30363F"),
    "bar": QColor("#39414C"),
    "playhead": QColor("#F0B23E"),
    "span": QColor("#69A7E0"),
    "span_loop": QColor("#5CC6C0"),
    "after_end": QColor("#1F2329"),
    "record": QColor("#E0533B"),
}


# Waveform overviews -------------------------------------------------------------------------

def peaks_of(path: str, should_stop=None) -> np.ndarray:
    """The loudest sample (of both channels) in every BUCKET samples, as float16."""
    data = playback.open_wav(path)
    n = len(data) // BUCKET
    out = np.zeros(n + 1, np.float16)
    step = BUCKET * 4096
    for start in range(0, len(data), step):
        if should_stop and should_stop():
            break
        block = np.abs(data[start:start + step].astype(np.int32)).max(axis=1)
        k = len(block) // BUCKET
        if k:
            out[start // BUCKET:start // BUCKET + k] = block[:k * BUCKET].reshape(k, BUCKET).max(axis=1) / 32768.0
        if len(block) > k * BUCKET:
            out[start // BUCKET + k] = block[k * BUCKET:].max() / 32768.0
    return out


class PeaksThread(QThread):
    ready = Signal(str, object)        # file path, peaks

    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self.paths = paths
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for path in self.paths:
            if self._stop:
                return
            try:
                self.ready.emit(path, peaks_of(path, lambda: self._stop))
            except (OSError, ValueError):
                pass


# Headers ---------------------------------------------------------------------------------------

class TrackHeader(QFrame):
    """Name, mute, solo, volume, pan and a meter for one lane."""

    mixChanged = Signal(object)
    partsChanged = Signal(object)
    menuRequested = Signal(object, object)    # track, global position

    def __init__(self, track, parent=None):
        super().__init__(parent)
        self.track = track
        self.setObjectName("TrackHeader")
        self.setFixedHeight(LANE_H)
        self.setFixedWidth(HEADER_W)
        self.setStyleSheet("QFrame#TrackHeader { background: #2E343D; border-bottom: 1px solid #3F4651; "
                           "border-right: 1px solid #3F4651; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 6, 4)
        lay.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(5)
        bar = QFrame()
        bar.setFixedSize(4, 18)
        bar.setStyleSheet(f"background: {track.color}; border-radius: 2px;")
        self.name = QLabel()
        self.name.setTextFormat(Qt.PlainText)
        self.name.setObjectName("StripName")
        self.mute = QToolButton()
        self.mute.setObjectName("MuteButton")
        self.mute.setText("M")
        self.mute.setCheckable(True)
        self.mute.setToolTip(tr("Mute: hides this part's notes and silences it"))
        self.solo = QToolButton()
        self.solo.setObjectName("SoloButton")
        self.solo.setText("S")
        self.solo.setCheckable(True)
        self.solo.setToolTip(tr("Solo: show and hear only the soloed parts"))
        top.addSpacing(4)
        top.addWidget(bar)
        top.addWidget(self.name, 1)
        top.addWidget(self.mute)
        top.addWidget(self.solo)
        lay.addLayout(top)
        def tag(text: str) -> QLabel:
            label = QLabel(text)
            label.setObjectName("Hint")
            label.setFixedWidth(26)
            return label

        row = QHBoxLayout()
        row.setSpacing(4)
        row.addSpacing(8)
        row.addWidget(tag(tr("Vol")))
        self.volume = VolumeSlider()
        self.db_label = QLabel()
        self.db_label.setObjectName("Hint")
        self.db_label.setFixedWidth(52)
        self.db_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.db_label.setProperty("i18n_skip", True)
        row.addWidget(self.volume, 1)
        row.addWidget(self.db_label)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addSpacing(8)
        row.addWidget(tag(tr("Pan")))
        self.pan = PanSlider()
        self.meter = Meter(False)
        row.addWidget(self.pan, 2)
        row.addWidget(self.meter, 3)
        lay.addLayout(row)

        self.volume.dbChanged.connect(self._on_db)
        self.pan.panChanged.connect(self._on_pan)
        self.mute.toggled.connect(self._on_mute)
        self.solo.toggled.connect(self._on_solo)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self.menuRequested.emit(self.track, self.mapToGlobal(pos)))
        self.sync()

    def sync(self) -> None:
        t = self.track
        self.name.setText(tr(t.name))
        self.name.setToolTip(tr(t.name) + ("\n" + tr("Right-click for more") if t.take else ""))
        self.volume.set_db(t.volume_db)
        self.db_label.setText(db_text(t.volume_db))
        self.pan.set_pan(t.pan)
        for b, v in ((self.mute, t.muted), (self.solo, t.solo)):
            b.blockSignals(True)
            b.setChecked(v)
            b.blockSignals(False)

    def _on_db(self, db: float) -> None:
        self.track.volume_db = db
        self.db_label.setText(db_text(db))
        self.mixChanged.emit(self.track)

    def _on_pan(self, pan: float) -> None:
        self.track.pan = pan
        self.mixChanged.emit(self.track)

    def _on_mute(self, on: bool) -> None:
        self.track.muted = on
        self.partsChanged.emit(self.track)

    def _on_solo(self, on: bool) -> None:
        self.track.solo = on
        self.partsChanged.emit(self.track)


# Lanes -------------------------------------------------------------------------------------------

class LanesCanvas(QAbstractScrollArea):
    seekRequested = Signal(float)
    spanChanged = Signal(object, bool)
    trackActivated = Signal(int)            # double-click: edit this part's notes
    takeMoved = Signal(object, float)       # take, old offset (the new one is already set)
    takeMoving = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.viewport().setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tracks: list = []
        self.peaks: dict[str, np.ndarray] = {}
        self.duration = 0.0
        self.beats: list[float] = []
        self.meter = 4
        self.pps = 20.0
        self.playhead = 0.0
        self.follow = True
        self.span: tuple[float, float] | None = None
        self.loop_on = False
        self.snap = None                     # callable(t) -> t, from the window
        self.recording_from: float | None = None
        self._press: dict | None = None
        self.small_font = QFont(self.font())
        self.small_font.setPointSizeF(max(7.0, self.font().pointSizeF() * 0.85))

    # geometry
    def x_of(self, t: float) -> float:
        return t * self.pps - self.horizontalScrollBar().value()

    def t_of(self, x: float) -> float:
        return (x + self.horizontalScrollBar().value()) / self.pps

    def lane_top(self, i: int) -> float:
        return RULER_H + i * LANE_H - self.verticalScrollBar().value()

    def lane_at(self, y: float) -> int | None:
        if y < RULER_H:
            return None
        i = int((y - RULER_H + self.verticalScrollBar().value()) // LANE_H)
        return i if 0 <= i < len(self.tracks) else None

    def _clamp(self, t: float) -> float:
        return min(max(0.0, t), self.duration)

    def update_scrollbars(self) -> None:
        w = self.viewport().width()
        h = self.viewport().height() - RULER_H
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        hb.setRange(0, max(0, int(self.duration * self.pps - w)))
        hb.setPageStep(w)
        hb.setSingleStep(24)
        vb.setRange(0, max(0, len(self.tracks) * LANE_H - h))
        vb.setPageStep(max(1, h))
        vb.setSingleStep(LANE_H // 2)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.update_scrollbars()

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        self.viewport().update()

    def fit(self) -> None:
        if self.duration > 0:
            self.pps = max(0.5, self.viewport().width() / self.duration)
        self.update_scrollbars()
        self.horizontalScrollBar().setValue(0)
        self.viewport().update()

    def zoom_time(self, factor: float, anchor_x: float | None = None) -> None:
        if self.duration <= 0:
            return
        if anchor_x is None:
            anchor_x = self.viewport().width() / 2
        anchor_t = self.t_of(anchor_x)
        min_pps = min(5.0, self.viewport().width() / self.duration)
        self.pps = min(MAX_PPS, max(min_pps, self.pps * factor))
        self.update_scrollbars()
        self.horizontalScrollBar().setValue(int(anchor_t * self.pps - anchor_x))
        self.viewport().update()

    def set_playhead(self, t: float) -> None:
        self.playhead = t
        if self.follow and self.duration > 0 and self._press is None:
            x = self.x_of(t)
            if x > self.viewport().width() - 24 or x < 0:
                self.horizontalScrollBar().setValue(int(t * self.pps - self.viewport().width() * 0.1))
        self.viewport().update()

    # painting
    def paintEvent(self, event) -> None:  # noqa: N802
        vp = self.viewport()
        w, h = vp.width(), vp.height()
        p = QPainter(vp)
        p.fillRect(0, 0, w, h, COL["bg"])
        if self.duration <= 0:
            p.end()
            return
        t0, t1 = max(0.0, self.t_of(0)), min(self.duration, self.t_of(w))
        for i, track in enumerate(self.tracks):
            top = self.lane_top(i)
            if top + LANE_H < RULER_H or top > h:
                continue
            p.fillRect(QRectF(0, top, w, LANE_H), COL["lane"] if i % 2 == 0 else COL["lane_alt"])
        self._paint_grid(p, w, h, t0, t1)
        end_x = self.x_of(self.duration)
        if end_x < w:
            p.fillRect(QRectF(max(0.0, end_x), RULER_H, w - end_x, h), COL["after_end"])
        p.save()
        p.setClipRect(QRectF(0, RULER_H, w, h - RULER_H))
        for i, track in enumerate(self.tracks):
            top = self.lane_top(i)
            if top + LANE_H < RULER_H or top > h:
                continue
            self._paint_lane(p, track, top, w, t0, t1)
            p.setPen(COL["line"])
            p.drawLine(QPointF(0, top + LANE_H - 0.5), QPointF(w, top + LANE_H - 0.5))
        self._paint_recording(p, h)
        self._paint_span(p, w, h)
        x = self.x_of(self.playhead)
        p.setPen(QPen(COL["playhead"], 1.5))
        p.drawLine(QPointF(x, RULER_H), QPointF(x, h))
        p.restore()
        self._paint_ruler(p, w)
        p.end()

    def _paint_grid(self, p: QPainter, w: int, h: int, t0: float, t1: float) -> None:
        if len(self.beats) > 1:
            period = (self.beats[-1] - self.beats[0]) / (len(self.beats) - 1)
            every = 1 if period * self.pps >= 10 else self.meter
            i0, i1 = bisect.bisect_left(self.beats, t0), bisect.bisect_right(self.beats, t1)
            lines_bar, lines_beat = [], []
            for i in range(i0, i1):
                if i % every:
                    continue
                x = self.x_of(self.beats[i])
                (lines_bar if i % self.meter == 0 else lines_beat).append(QLineF(x, RULER_H, x, h))
            p.setPen(COL["beat"])
            p.drawLines(lines_beat)
            p.setPen(COL["bar"])
            p.drawLines(lines_bar)

    def _paint_lane(self, p: QPainter, track, top: float, w: int, t0: float, t1: float) -> None:
        color = QColor(track.color)
        alpha = 1.0 if track.visible else 0.3
        mid = top + LANE_H / 2
        half = LANE_H / 2 - 6
        peaks = self.peaks.get(track.audio or "")
        offset = track.offset
        if track.take and track.audio:
            # the region of the take
            region_a, region_b = self.x_of(offset), self.x_of(offset + self._take_length(track))
            fill = QColor(color)
            fill.setAlphaF(0.12 * alpha)
            p.fillRect(QRectF(region_a, top + 2, region_b - region_a, LANE_H - 4), fill)
            p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), int(160 * alpha)), 1))
            p.drawRect(QRectF(region_a, top + 2, region_b - region_a, LANE_H - 4))
        if peaks is not None and len(peaks):
            px0, px1 = max(0, int(self.x_of(max(t0, offset)))), min(w, int(self.x_of(t1)) + 1)
            if px1 > px0:
                xs = np.arange(px0, px1 + 1)
                times = (xs + self.horizontalScrollBar().value()) / self.pps - offset
                idx = np.clip((times * BUCKETS_PER_SECOND).astype(np.int64), 0, len(peaks))
                starts = idx[:-1][idx[:-1] < len(peaks)]
                lines = []
                levels = np.zeros(0, np.float32)
                if len(starts):
                    # the loudest point under each pixel (one point when zoomed in past the waveform's detail)
                    padded = np.append(peaks.astype(np.float32), np.float32(0))
                    levels = np.maximum.reduceat(padded, np.append(starts, len(peaks)))[:-1]
                if len(levels):
                    # a gentle curve so quiet parts still show
                    heights = np.sqrt(np.minimum(1.0, levels)) * half
                    for k, hgt in enumerate(heights):
                        if hgt > 0.4:
                            x = px0 + k + 0.5
                            lines.append(QLineF(x, mid - hgt, x, mid + hgt))
                wave = QColor(color)
                wave.setAlphaF(0.55 * alpha)
                p.setPen(QPen(wave, 1))
                p.drawLines(lines)
        if track.notes:
            pitches = [n.pitch for n in track.notes]
            lo, hi = min(pitches), max(pitches)
            span = max(12, hi - lo + 1)
            row = (LANE_H - 12) / span
            note_color = QColor(color).lighter(140)
            note_color.setAlphaF(0.9 * alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(note_color)
            starts = [n.start for n in track.notes]
            longest = max(n.end - n.start for n in track.notes)
            i0 = bisect.bisect_left(starts, t0 - longest)
            i1 = bisect.bisect_right(starts, t1)
            hgt = max(1.5, min(4.0, row))
            for n in track.notes[i0:i1]:
                x0, x1 = self.x_of(n.start), self.x_of(n.end)
                y = top + 6 + (hi + (span - (hi - lo + 1)) / 2 - n.pitch) * row
                p.drawRect(QRectF(x0, y, max(1.5, x1 - x0), hgt))
        p.setPen(COL["muted"] if not track.visible else COL["text"])
        p.setFont(self.small_font)
        label = tr(track.name) + ("" if track.visible else "  (" + tr("muted") + ")")
        p.drawText(QPointF(6, top + 14), label)

    def _take_length(self, track) -> float:
        peaks = self.peaks.get(track.audio or "")
        if peaks is not None:
            return len(peaks) / BUCKETS_PER_SECOND
        return 0.0

    def _paint_recording(self, p: QPainter, h: int) -> None:
        if self.recording_from is None:
            return
        a, b = self.x_of(self.recording_from), self.x_of(max(self.playhead, self.recording_from))
        top = self.lane_top(len(self.tracks))
        fill = QColor(COL["record"])
        fill.setAlpha(70)
        p.fillRect(QRectF(a, top + 2, max(2.0, b - a), LANE_H - 4), fill)
        p.setPen(COL["record"])
        p.drawText(QPointF(a + 6, top + 16), tr("Recording..."))

    def _paint_span(self, p: QPainter, w: int, h: int) -> None:
        if not self.span:
            return
        a, b = self.span
        x0, x1 = max(0.0, self.x_of(a)), min(float(w), self.x_of(b))
        if x1 <= 0 or x0 >= w:
            return
        color = COL["span_loop"] if self.loop_on else COL["span"]
        band = QColor(color)
        band.setAlpha(40)
        p.fillRect(QRectF(x0, RULER_H, x1 - x0, h - RULER_H), band)
        p.setPen(QPen(color, 1.5))
        for t in (a, b):
            x = self.x_of(t)
            p.drawLine(QPointF(x, RULER_H), QPointF(x, h))

    def _paint_ruler(self, p: QPainter, w: int) -> None:
        p.fillRect(QRectF(0, 0, w, RULER_H), COL["panel"])
        p.setPen(COL["line"])
        p.drawLine(QPointF(0, RULER_H - 0.5), QPointF(w, RULER_H - 0.5))
        step = next((s for s in TIME_STEPS if s * self.pps >= 70), TIME_STEPS[-1])
        decimals = 2 if step < 0.1 else (1 if step < 1 else 0)
        t0, t1 = max(0.0, self.t_of(0)), min(self.duration, self.t_of(w))
        p.setFont(self.small_font)
        k = int(t0 // step)
        while k * step <= t1 + step:
            x = self.x_of(k * step)
            p.setPen(COL["ruler_text"])
            p.drawLine(QPointF(x, RULER_H - 8), QPointF(x, RULER_H - 1))
            p.drawText(QPointF(x + 4, RULER_H - 9), format_time(k * step, decimals))
            k += 1
        if self.span:
            color = QColor(COL["span_loop"] if self.loop_on else COL["span"])
            color.setAlpha(80)
            xa, xb = self.x_of(self.span[0]), self.x_of(self.span[1])
            p.fillRect(QRectF(xa, 0, max(1.0, xb - xa), RULER_H - 1), color)
        x = self.x_of(self.playhead)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(COL["playhead"])
        p.drawPolygon(QPolygonF([QPointF(x - 5, RULER_H - 9), QPointF(x + 5, RULER_H - 9), QPointF(x, RULER_H - 1)]))

    # mouse
    def _snap(self, t: float) -> float:
        return self.snap(t) if self.snap else t

    def _take_at(self, x: float, y: float):
        i = self.lane_at(y)
        if i is None:
            return None
        track = self.tracks[i]
        if not track.take:
            return None
        t = self.t_of(x)
        if track.offset <= t <= track.offset + self._take_length(track):
            return track
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton or self.duration <= 0:
            return
        x, y = event.position().x(), event.position().y()
        t = self._clamp(self.t_of(x))
        if y < RULER_H:
            self._press = {"kind": "scrub"}
            self.seekRequested.emit(t)
            return
        take = self._take_at(x, y)
        if take is not None and event.modifiers() & Qt.AltModifier:
            self._press = {"kind": "take", "track": take, "t": t, "offset": take.offset, "moved": False}
            return
        self._press = {"kind": "empty", "x": x, "t": t}

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x, y = event.position().x(), event.position().y()
        press = self._press
        if press is None:
            take = self._take_at(x, y)
            self.viewport().setCursor(Qt.OpenHandCursor if take is not None and event.modifiers() & Qt.AltModifier
                                      else Qt.ArrowCursor)
            if take is not None:
                self.setToolTip(tr("Hold Alt and drag to move this take"))
            else:
                self.setToolTip("")
            return
        t = self._clamp(self.t_of(x))
        if press["kind"] == "scrub":
            self.seekRequested.emit(t)
        elif press["kind"] == "take":
            track = press["track"]
            new = press["offset"] + (t - press["t"])
            new = self._snap(new) if self.snap else new
            track.offset = min(max(0.0, new), max(0.0, self.duration - 0.05))
            press["moved"] = True
            self.takeMoving.emit(track)
            self.viewport().update()
        elif press["kind"] in ("empty", "span"):
            if press["kind"] == "empty" and abs(x - press["x"]) < 4:
                return
            press["kind"] = "span"
            a, b = sorted((self._snap(press["t"]), self._snap(t)))
            self.span = (a, b)
            self.spanChanged.emit(self.span, False)
            self.viewport().update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        press, self._press = self._press, None
        if press is None:
            return
        if press["kind"] == "empty":
            self.seekRequested.emit(press["t"])
        elif press["kind"] == "span":
            if self.span and self.span[1] - self.span[0] < 0.05:
                self.span = None
                self.spanChanged.emit(None, True)
            else:
                self.spanChanged.emit(self.span, True)
                if self.span:
                    self.seekRequested.emit(self.span[0])
            self.viewport().update()
        elif press["kind"] == "take" and press["moved"]:
            self.takeMoved.emit(press["track"], press["offset"])

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        i = self.lane_at(event.position().y())
        if i is not None:
            self.trackActivated.emit(i)

    def wheelEvent(self, event) -> None:  # noqa: N802
        mods = event.modifiers()
        d = event.angleDelta()
        delta = d.y() or d.x()
        if mods & Qt.ControlModifier:
            self.zoom_time(1.2 ** (delta / 120), event.position().x())
        elif mods & Qt.ShiftModifier or (d.x() and not d.y()):
            hb = self.horizontalScrollBar()
            hb.setValue(hb.value() - delta)
        else:
            super().wheelEvent(event)
            return
        event.accept()


class TracksView(QWidget):
    """Headers on the left, lanes on the right, scrolling together."""

    seekRequested = Signal(float)
    spanChanged = Signal(object, bool)
    trackActivated = Signal(int)
    mixChanged = Signal(object)
    partsChanged = Signal(object)
    takeMenuRequested = Signal(object, object)
    takeMoved = Signal(object, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        left = QWidget()
        left.setFixedWidth(HEADER_W)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)
        corner = QLabel(tr("Parts"))
        corner.setFixedHeight(RULER_H)
        corner.setContentsMargins(10, 0, 0, 0)
        corner.setObjectName("Meta")
        corner.setStyleSheet("background: #2E343D; border-bottom: 1px solid #3F4651; border-right: 1px solid #3F4651;")
        ll.addWidget(corner)
        self.header_scroll = QScrollArea()
        self.header_scroll.setFrameShape(QFrame.NoFrame)
        self.header_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header_scroll.setWidgetResizable(True)
        self.header_box = QWidget()
        self.header_col = QVBoxLayout(self.header_box)
        self.header_col.setContentsMargins(0, 0, 0, 0)
        self.header_col.setSpacing(0)
        self.header_col.addStretch(1)
        self.header_scroll.setWidget(self.header_box)
        ll.addWidget(self.header_scroll, 1)
        lay.addWidget(left)
        self.canvas = LanesCanvas()
        lay.addWidget(self.canvas, 1)
        self.canvas.verticalScrollBar().valueChanged.connect(self.header_scroll.verticalScrollBar().setValue)
        self.canvas.seekRequested.connect(self.seekRequested)
        self.canvas.spanChanged.connect(self.spanChanged)
        self.canvas.trackActivated.connect(self.trackActivated)
        self.canvas.takeMoved.connect(self.takeMoved)
        self.headers: list[TrackHeader] = []
        self._threads: set[PeaksThread] = set()
        self.empty = QLabel(tr("The parts show up here after an analysis."), self.canvas.viewport())
        self.empty.setObjectName("Hint")
        self.empty.move(20, RULER_H + 20)
        self.empty.resize(500, 30)

    def set_data(self, duration: float, tracks, beats, meter: int) -> None:
        c = self.canvas
        first = c.duration <= 0
        c.duration = max(0.0, duration)
        c.tracks = list(tracks)
        c.beats, c.meter = list(beats or []), max(1, meter)
        for hdr in self.headers:
            self.header_col.removeWidget(hdr)
            hdr.deleteLater()
        self.headers = []
        for i, t in enumerate(tracks):
            hdr = TrackHeader(t)
            hdr.mixChanged.connect(self.mixChanged)
            hdr.partsChanged.connect(self.partsChanged)
            hdr.menuRequested.connect(self.takeMenuRequested)
            self.header_col.insertWidget(i, hdr)
            self.headers.append(hdr)
        # room for a take being recorded below the last part
        self.header_box.setMinimumHeight(len(tracks) * LANE_H + LANE_H)
        self.empty.setVisible(not tracks)
        self.load_peaks([t.audio for t in tracks if t.audio and t.audio not in c.peaks])
        c.update_scrollbars()
        if first or c.pps * c.duration < c.viewport().width() * 0.2:
            if self.isVisible():
                c.fit()
            else:
                self._needs_fit = True      # the width is only known once it is shown
        c.viewport().update()

    _needs_fit = False

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._needs_fit:
            self._needs_fit = False
            self.canvas.fit()

    def load_peaks(self, paths: list[str]) -> None:
        paths = [p for p in dict.fromkeys(paths) if p]
        if not paths:
            return
        thread = PeaksThread(paths, self)
        thread.ready.connect(self._on_peaks)
        thread.finished.connect(lambda t=thread: self._threads.discard(t))
        thread.finished.connect(thread.deleteLater)
        self._threads.add(thread)
        thread.start()

    def shutdown(self) -> None:
        for thread in list(self._threads):
            thread.stop()
            thread.wait(10000)

    def _on_peaks(self, path: str, peaks) -> None:
        self.canvas.peaks[path] = peaks
        self.canvas.viewport().update()

    def clear(self) -> None:
        self.canvas.peaks.clear()
        self.canvas.span = None
        self.set_data(0.0, [], [], 4)

    def set_beats(self, beats, meter: int) -> None:
        self.canvas.beats, self.canvas.meter = list(beats or []), max(1, meter)
        self.canvas.viewport().update()

    def sync_headers(self) -> None:
        for hdr in self.headers:
            hdr.sync()
        self.canvas.viewport().update()

    def refresh(self) -> None:
        self.canvas.viewport().update()

    def set_span(self, span, loop_on: bool) -> None:
        self.canvas.span = span
        self.canvas.loop_on = loop_on
        self.canvas.viewport().update()

    def set_playhead(self, t: float) -> None:
        self.canvas.set_playhead(t)

    def push_meters(self, peaks: dict) -> None:
        if not self.isVisible():
            return
        for hdr in self.headers:
            hdr.meter.push(peaks.get(hdr.track.uid, 0.0))

    def reset_meters(self) -> None:
        for hdr in self.headers:
            hdr.meter.reset()
