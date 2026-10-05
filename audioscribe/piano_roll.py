"""A Cubase style piano roll.

Layout inside the widget:

    +--------+----------------------------------------------+
    |        | time ruler                                   |
    | corner +----------------------------------------------+
    |        | lyrics lane (only when words were found)     |
    +--------+----------------------------------------------+
    | keys   | note grid                                    |
    |        |                                              |
    +--------+----------------------------------------------+

Grid rows and notes are drawn into a cached pixmap that is only rebuilt on
scroll, zoom, or data changes. The playhead, ruler, lyrics, and keyboard are
drawn on top every frame, which keeps playback smooth with thousands of notes.
"""

from __future__ import annotations

import bisect

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QAbstractScrollArea, QFrame, QToolTip

from .music import format_time, is_black, note_name

KEY_W = 58
RULER_H = 24
LYRIC_H = 26
TIME_STEPS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)

COL = {
    "bg": QColor("#2A3038"),
    "bg_black": QColor("#252A31"),
    "row_line": QColor("#30363F"),
    "octave_line": QColor("#454D59"),
    "beat": QColor("#323944"),
    "bar": QColor("#3E4653"),
    "after_end": QColor("#1F2329"),
    "panel": QColor("#2E343D"),
    "panel_line": QColor("#3F4651"),
    "ruler_text": QColor("#9AA4B2"),
    "text": QColor("#DCE1E8"),
    "muted": QColor("#7F8996"),
    "white_key": QColor("#C9CED6"),
    "white_key_line": QColor("#9AA1AB"),
    "black_key": QColor("#30353D"),
    "key_label": QColor("#525A66"),
    "playhead": QColor("#F0B23E"),
    "word_bar": QColor("#4B5563"),
    "note_text": QColor("#15191E"),
}


class _TrackIndex:
    """Sorted start times per track so visible notes can be found with bisect."""

    def __init__(self, track):
        self.track = track
        self.starts = [n.start for n in track.notes]
        self.max_len = max((n.end - n.start for n in track.notes), default=0.0)
        self.color = QColor(track.color)

    def in_range(self, t0: float, t1: float) -> range:
        lo = bisect.bisect_left(self.starts, t0 - self.max_len)
        hi = bisect.bisect_right(self.starts, t1)
        return range(lo, hi)


class PianoRoll(QAbstractScrollArea):
    seekRequested = Signal(float)
    noteClicked = Signal(int, int)  # track index, note index
    pitchToggled = Signal(object)  # MIDI pitch or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setFocusPolicy(Qt.ClickFocus)
        self.viewport().setMouseTracking(True)
        self.horizontalScrollBar().setSingleStep(24)

        self.duration = 0.0
        self.indexes: list[_TrackIndex] = []
        self.lane: list[tuple[float, float, str]] = []
        self.lane_starts: list[float] = []
        self.lane_max = 0.0
        self.beats: list[float] = []
        self.pps = 60.0  # pixels per second
        self.row_h = 12
        self.lo, self.hi = 48, 84
        self.playhead = 0.0
        self.follow = True
        self.selected: tuple[int, int] | None = None
        self.highlight_pitch: int | None = None
        self.message = "Open a file or drop one here to get started."

        self._version = 0
        self._cache: QPixmap | None = None
        self._cache_key = None
        self._active: dict[int, QColor] = {}
        self._scrubbing = False

        self.small_font = QFont(self.font())
        self.small_font.setPointSizeF(max(7.0, self.font().pointSizeF() * 0.85))
        self._update_scrollbars()

    # Geometry ---------------------------------------------------------------

    @property
    def header_h(self) -> int:
        return RULER_H + (LYRIC_H if self.lane else 0)

    def _rows(self) -> int:
        return self.hi - self.lo + 1

    def _view_w(self) -> int:
        return max(1, self.viewport().width() - KEY_W)

    def _view_h(self) -> int:
        return max(1, self.viewport().height() - self.header_h)

    def x_of(self, t: float) -> float:
        return KEY_W + t * self.pps - self.horizontalScrollBar().value()

    def t_of(self, x: float) -> float:
        return (x - KEY_W + self.horizontalScrollBar().value()) / self.pps

    def y_of(self, pitch: int) -> float:
        return self.header_h + (self.hi - pitch) * self.row_h - self.verticalScrollBar().value()

    def pitch_at(self, y: float) -> int:
        row = int((y - self.header_h + self.verticalScrollBar().value()) // self.row_h)
        return self.hi - row

    def _clamp_t(self, t: float) -> float:
        return min(max(0.0, t), self.duration)

    # Data -------------------------------------------------------------------

    def set_duration(self, seconds: float) -> None:
        """Used before analysis so the ruler and seeking already work."""
        if not self.indexes and not self.lane:
            self.duration = max(0.0, seconds)
            self._initial_zoom()

    def set_data(self, duration: float, tracks, segments, beats) -> None:
        self.duration = max(0.0, duration)
        self.indexes = [_TrackIndex(t) for t in tracks]
        self.beats = list(beats or [])

        words = [(w.start, w.end, w.text) for s in segments for w in (s.words or [])]
        if not words:
            words = [(s.start, s.end, s.text) for s in segments if s.text]
        self.lane = sorted(words)
        self.lane_starts = [w[0] for w in self.lane]
        self.lane_max = max((w[1] - w[0] for w in self.lane), default=0.0)

        pitches = [n.pitch for idx in self.indexes for n in idx.track.notes]
        if pitches:
            self.lo = max(21, min(pitches) - 3)
            self.hi = min(108, max(pitches) + 3)
            if self.hi - self.lo < 24:
                pad = (24 - (self.hi - self.lo)) // 2
                self.lo, self.hi = max(21, self.lo - pad), min(108, self.hi + pad)
            self.message = ""
        else:
            self.lo, self.hi = 48, 84
            self.message = "No notes to show." if tracks else ""
        self.selected = None
        self.highlight_pitch = None
        self.playhead = 0.0
        self._active = {}
        self._initial_zoom()

    def clear(self, message: str = "") -> None:
        self.indexes, self.lane, self.lane_starts, self.beats = [], [], [], []
        self.duration = 0.0
        self.lo, self.hi = 48, 84
        self.selected = None
        self.highlight_pitch = None
        self.playhead = 0.0
        self.message = message
        self._active = {}
        self._update_scrollbars()
        self.viewport().update()

    def set_message(self, text: str) -> None:
        self.message = text
        self.refresh()

    def refresh(self) -> None:
        self._version += 1
        self._active = self._compute_active(self.playhead)
        self.viewport().update()

    def set_highlight(self, pitch: int | None) -> None:
        self.highlight_pitch = pitch
        if pitch is not None and self.lo <= pitch <= self.hi:
            y = self.y_of(pitch)
            if y < self.header_h or y > self.viewport().height() - self.row_h:
                vb = self.verticalScrollBar()
                vb.setValue(int(vb.value() + y - self.header_h - self._view_h() / 2))
        self.refresh()

    def select_note(self, track_index: int, note_index: int) -> None:
        self.selected = (track_index, note_index)
        note = self.indexes[track_index].track.notes[note_index]
        x, y = self.x_of(note.start), self.y_of(note.pitch)
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        if x < KEY_W or x > self.viewport().width() - 40:
            hb.setValue(int(note.start * self.pps - self._view_w() * 0.25))
        if y < self.header_h or y > self.viewport().height() - self.row_h:
            vb.setValue(int(vb.value() + y - self.header_h - self._view_h() / 2))
        self.refresh()

    # Zoom and scroll ----------------------------------------------------------

    def _initial_zoom(self) -> None:
        if self.duration > 0:
            window = self.duration if self.duration <= 60 else 30.0
            self.pps = min(400.0, max(2.0, self._view_w() / window))
        self.row_h = int(min(18, max(9, self._view_h() / max(1, self._rows()))))
        self._update_scrollbars()
        self.horizontalScrollBar().setValue(0)
        vb = self.verticalScrollBar()
        vb.setValue((vb.maximum() + vb.minimum()) // 2)
        self.refresh()

    def fit(self) -> None:
        if self.duration > 0:
            self.pps = max(1.0, self._view_w() / self.duration)
        self.row_h = int(min(18, max(6, self._view_h() / max(1, self._rows()))))
        self._update_scrollbars()
        self.horizontalScrollBar().setValue(0)
        self.refresh()

    def zoom_time(self, factor: float, anchor_x: float | None = None) -> None:
        if self.duration <= 0:
            return
        if anchor_x is None:
            anchor_x = KEY_W + self._view_w() / 2
        anchor_t = self.t_of(anchor_x)
        min_pps = min(20.0, self._view_w() / self.duration)
        self.pps = min(800.0, max(min_pps, self.pps * factor))
        self._update_scrollbars()
        self.horizontalScrollBar().setValue(int(anchor_t * self.pps - (anchor_x - KEY_W)))
        self.refresh()

    def zoom_rows(self, factor: float) -> None:
        center = self.pitch_at(self.header_h + self._view_h() / 2)
        new = int(round(self.row_h * factor))
        if new == self.row_h:
            new += 1 if factor > 1 else -1
        self.row_h = min(32, max(6, new))
        self._update_scrollbars()
        vb = self.verticalScrollBar()
        vb.setValue(int((self.hi - center) * self.row_h - self._view_h() / 2))
        self.refresh()

    def _update_scrollbars(self) -> None:
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        hb.setRange(0, max(0, int(self.duration * self.pps - self._view_w())))
        hb.setPageStep(self._view_w())
        vb.setRange(0, max(0, int(self._rows() * self.row_h - self._view_h())))
        vb.setPageStep(self._view_h())
        vb.setSingleStep(self.row_h)

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802 (Qt name)
        self.viewport().update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_scrollbars()

    # Playhead -----------------------------------------------------------------

    def _compute_active(self, t: float) -> dict[int, QColor]:
        active: dict[int, QColor] = {}
        for idx in self.indexes:
            if not idx.track.visible:
                continue
            notes = idx.track.notes
            for i in idx.in_range(t, t):
                n = notes[i]
                if n.start <= t < n.end:
                    active[n.pitch] = idx.color
        return active

    def set_playhead(self, t: float) -> None:
        self.playhead = t
        self._active = self._compute_active(t)
        if self.follow and self.duration > 0 and not self._scrubbing:
            x = self.x_of(t)
            if x > self.viewport().width() - 24 or x < KEY_W:
                self.horizontalScrollBar().setValue(int(t * self.pps - self._view_w() * 0.1))
        self.viewport().update()

    # Painting -----------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        vp = self.viewport()
        w, h = vp.width(), vp.height()
        p = QPainter(vp)
        p.drawPixmap(0, 0, self._static_layer(w, h))
        if self.duration > 0:
            self._paint_playhead(p, h)
            if self.lane:
                self._paint_lane(p, w)
            self._paint_ruler(p, w)
        else:
            p.fillRect(KEY_W, 0, w - KEY_W, self.header_h, COL["panel"])
        self._paint_keyboard(p, h)
        self._paint_corner(p)
        p.end()

    def _static_layer(self, w: int, h: int) -> QPixmap:
        dpr = self.viewport().devicePixelRatioF()
        key = (w, h, dpr, self.horizontalScrollBar().value(), self.verticalScrollBar().value(),
               self.pps, self.row_h, self.header_h, self._version)
        if self._cache is not None and self._cache_key == key:
            return self._cache
        pm = QPixmap(max(1, int(w * dpr)), max(1, int(h * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(COL["bg"])
        p = QPainter(pm)
        p.setClipRect(KEY_W, self.header_h, w - KEY_W, h - self.header_h)
        self._paint_rows(p, w, h)
        self._paint_grid(p, w, h)
        self._paint_notes(p, w, h)
        p.setClipping(False)
        if self.message:
            p.setPen(COL["muted"])
            p.drawText(QRectF(KEY_W, self.header_h, w - KEY_W, h - self.header_h),
                       Qt.AlignCenter, self.message)
        p.end()
        self._cache, self._cache_key = pm, key
        return pm

    def _visible_pitches(self, h: int) -> range:
        top = min(self.hi, self.pitch_at(self.header_h))
        bottom = max(self.lo, self.pitch_at(h))
        return range(top, bottom - 1, -1)

    def _paint_rows(self, p: QPainter, w: int, h: int) -> None:
        rh = self.row_h
        for pitch in self._visible_pitches(h):
            y = self.y_of(pitch)
            if is_black(pitch):
                p.fillRect(QRectF(KEY_W, y, w - KEY_W, rh), COL["bg_black"])
            p.setPen(COL["octave_line"] if pitch % 12 == 0 else COL["row_line"])
            p.drawLine(QPointF(KEY_W, y + rh - 0.5), QPointF(w, y + rh - 0.5))
        end_x = self.x_of(self.duration)
        if self.duration > 0 and end_x < w:
            p.fillRect(QRectF(max(KEY_W, end_x), self.header_h, w - end_x, h), COL["after_end"])

    def _time_step(self) -> float:
        for step in TIME_STEPS:
            if step * self.pps >= 70:
                return step
        return TIME_STEPS[-1]

    def _paint_grid(self, p: QPainter, w: int, h: int) -> None:
        if self.duration <= 0:
            return
        t0, t1 = max(0.0, self.t_of(KEY_W)), min(self.duration, self.t_of(w))
        top = self.header_h
        beat_px = 0.0
        if len(self.beats) > 1:
            beat_px = (self.beats[-1] - self.beats[0]) / (len(self.beats) - 1) * self.pps
        if beat_px >= 7:
            i0 = bisect.bisect_left(self.beats, t0)
            i1 = bisect.bisect_right(self.beats, t1)
            for i in range(i0, i1):
                x = self.x_of(self.beats[i])
                p.setPen(COL["bar"] if i % 4 == 0 else COL["beat"])
                p.drawLine(QPointF(x, top), QPointF(x, h))
        else:
            step = self._time_step()
            k = int(t0 // step)
            while k * step <= t1:
                x = self.x_of(k * step)
                p.setPen(COL["beat"])
                p.drawLine(QPointF(x, top), QPointF(x, h))
                k += 1

    def _paint_notes(self, p: QPainter, w: int, h: int) -> None:
        t0, t1 = self.t_of(KEY_W), self.t_of(w)
        rh = self.row_h
        fm = QFontMetrics(self.small_font)
        p.setFont(self.small_font)
        p.setRenderHint(QPainter.Antialiasing, True)
        amber = QPen(COL["playhead"], 1.6)
        for ti, idx in enumerate(self.indexes):
            if not idx.track.visible:
                continue
            base = idx.color
            edge = QPen(base.lighter(125), 1.0)
            notes = idx.track.notes
            for ni in idx.in_range(t0, t1):
                n = notes[ni]
                if n.end < t0:
                    continue
                y = self.y_of(n.pitch)
                if y + rh < self.header_h or y > h:
                    continue
                x0, x1 = self.x_of(n.start), self.x_of(n.end)
                rect = QRectF(x0, y + 1, max(2.0, x1 - x0), rh - 2)
                dim = self.highlight_pitch is not None and n.pitch != self.highlight_pitch
                fill = QColor(base)
                fill.setAlphaF(0.14 if dim else 0.45 + 0.55 * n.velocity)
                p.setBrush(fill)
                marked = self.selected == (ti, ni) or (self.highlight_pitch == n.pitch)
                p.setPen(amber if marked else (Qt.NoPen if dim else edge))
                p.drawRoundedRect(rect, 2, 2)
                if not dim and rh >= 11 and rect.width() > 26:
                    label = note_name(n.pitch)
                    if fm.horizontalAdvance(label) + 6 < rect.width():
                        p.setPen(COL["note_text"])
                        p.drawText(rect.adjusted(3, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, label)

    def _paint_playhead(self, p: QPainter, h: int) -> None:
        x = self.x_of(self.playhead)
        if x >= KEY_W:
            p.setPen(QPen(COL["playhead"], 1.5))
            p.drawLine(QPointF(x, RULER_H), QPointF(x, h))

    def _paint_lane(self, p: QPainter, w: int) -> None:
        top = RULER_H
        p.fillRect(QRectF(KEY_W, top, w - KEY_W, LYRIC_H), COL["panel"])
        p.setPen(COL["panel_line"])
        p.drawLine(QPointF(KEY_W, top + LYRIC_H - 0.5), QPointF(w, top + LYRIC_H - 0.5))
        p.save()
        p.setClipRect(QRectF(KEY_W, top, w - KEY_W, LYRIC_H))
        fm = QFontMetrics(self.font())
        p.setFont(self.font())
        t0, t1 = self.t_of(KEY_W), self.t_of(w)
        i0 = bisect.bisect_left(self.lane_starts, t0 - self.lane_max)
        i1 = bisect.bisect_right(self.lane_starts, t1)
        text_end = -1e9
        bar_y = top + LYRIC_H - 5
        for start, end, text in self.lane[i0:i1]:
            if end < t0:
                continue
            x0, x1 = self.x_of(start), self.x_of(end)
            current = start <= self.playhead < end
            p.fillRect(QRectF(x0, bar_y, max(2.0, x1 - x0 - 1), 2),
                       COL["playhead"] if current else COL["word_bar"])
            if x0 >= text_end:
                width = fm.horizontalAdvance(text) + 2
                room = max(x1 - x0, width) if self.lane_max < 3 else max(x1 - x0, 30)
                shown = text if room >= width else fm.elidedText(text, Qt.ElideRight, int(room))
                p.setPen(COL["playhead"] if current else COL["text"])
                p.drawText(QRectF(x0 + 1, top + 2, room + 2, LYRIC_H - 8),
                           Qt.AlignLeft | Qt.AlignVCenter, shown)
                text_end = x0 + fm.horizontalAdvance(shown) + 8
        p.restore()

    def _paint_ruler(self, p: QPainter, w: int) -> None:
        p.fillRect(QRectF(KEY_W, 0, w - KEY_W, RULER_H), COL["panel"])
        p.setPen(COL["panel_line"])
        p.drawLine(QPointF(KEY_W, RULER_H - 0.5), QPointF(w, RULER_H - 0.5))
        p.save()
        p.setClipRect(QRectF(KEY_W, 0, w - KEY_W, RULER_H))
        p.setFont(self.small_font)
        step = self._time_step()
        minor = step / (5 if step in (0.05, 0.5, 5, 10, 60, 300, 600) else 4)
        decimals = 2 if step < 0.1 else (1 if step < 1 else 0)
        t0, t1 = max(0.0, self.t_of(KEY_W)), min(self.duration, self.t_of(w))
        k = int(t0 // minor)
        while k * minor <= t1 + minor:
            t = k * minor
            x = self.x_of(t)
            major = abs(t / step - round(t / step)) < 1e-6
            p.setPen(COL["ruler_text"] if major else COL["panel_line"])
            p.drawLine(QPointF(x, RULER_H - (8 if major else 4)), QPointF(x, RULER_H - 1))
            if major:
                p.drawText(QPointF(x + 4, RULER_H - 9), format_time(t, decimals))
            k += 1
        x = self.x_of(self.playhead)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(COL["playhead"])
        p.drawPolygon(QPolygonF([QPointF(x - 5, RULER_H - 9), QPointF(x + 5, RULER_H - 9),
                                 QPointF(x, RULER_H - 1)]))
        p.restore()

    def _paint_keyboard(self, p: QPainter, h: int) -> None:
        p.save()
        p.setClipRect(QRectF(0, self.header_h, KEY_W, h - self.header_h))
        p.fillRect(QRectF(0, self.header_h, KEY_W, h - self.header_h), COL["panel"])
        rh = self.row_h
        black_w = KEY_W * 0.6
        p.setFont(self.small_font)
        for pitch in self._visible_pitches(h):
            y = self.y_of(pitch)
            lit = self._active.get(pitch)
            if is_black(pitch):
                p.fillRect(QRectF(0, y, KEY_W, rh), COL["white_key"])
                p.fillRect(QRectF(0, y, black_w, rh), lit if lit else COL["black_key"])
            else:
                p.fillRect(QRectF(0, y, KEY_W, rh), lit.lighter(115) if lit else COL["white_key"])
                if pitch % 12 in (0, 5):
                    p.setPen(COL["white_key_line"])
                    p.drawLine(QPointF(0, y + rh - 0.5), QPointF(KEY_W, y + rh - 0.5))
            if pitch == self.highlight_pitch:
                p.fillRect(QRectF(KEY_W - 5, y, 5, rh), COL["playhead"])
            if pitch % 12 == 0 and rh >= 8:
                p.setPen(COL["key_label"])
                p.drawText(QRectF(0, y, KEY_W - 7, rh), Qt.AlignRight | Qt.AlignVCenter, note_name(pitch))
        p.setPen(COL["panel_line"])
        p.drawLine(QPointF(KEY_W - 0.5, self.header_h), QPointF(KEY_W - 0.5, h))
        p.restore()

    def _paint_corner(self, p: QPainter) -> None:
        p.fillRect(QRectF(0, 0, KEY_W, self.header_h), COL["panel"])
        p.setPen(COL["panel_line"])
        p.drawLine(QPointF(0, self.header_h - 0.5), QPointF(KEY_W, self.header_h - 0.5))
        p.drawLine(QPointF(KEY_W - 0.5, 0), QPointF(KEY_W - 0.5, self.header_h))
        if self.lane:
            p.setPen(COL["muted"])
            p.setFont(self.small_font)
            p.drawText(QRectF(0, RULER_H, KEY_W - 6, LYRIC_H), Qt.AlignRight | Qt.AlignVCenter, "Lyrics")

    # Mouse --------------------------------------------------------------------

    def note_at(self, x: float, y: float) -> tuple[int, int] | None:
        if y < self.header_h or x < KEY_W:
            return None
        pitch = self.pitch_at(y)
        t = self.t_of(x)
        slack = 3 / self.pps
        for ti in range(len(self.indexes) - 1, -1, -1):
            idx = self.indexes[ti]
            if not idx.track.visible:
                continue
            for ni in idx.in_range(t - slack, t + slack):
                n = idx.track.notes[ni]
                if n.pitch == pitch and n.start - slack <= t <= max(n.end, n.start + slack):
                    return ti, ni
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        x, y = event.position().x(), event.position().y()
        if x < KEY_W:
            if y >= self.header_h:
                pitch = self.pitch_at(y)
                if self.lo <= pitch <= self.hi:
                    self.set_highlight(None if self.highlight_pitch == pitch else pitch)
                    self.pitchToggled.emit(self.highlight_pitch)
            return
        if self.duration <= 0:
            return
        if y < self.header_h:
            self._scrubbing = True
            self.seekRequested.emit(self._clamp_t(self.t_of(x)))
            return
        hit = self.note_at(x, y)
        if hit:
            self.selected = hit
            self.refresh()
            self.noteClicked.emit(*hit)
            self.seekRequested.emit(self.indexes[hit[0]].track.notes[hit[1]].start)
        else:
            if self.selected:
                self.selected = None
                self.refresh()
            self.seekRequested.emit(self._clamp_t(self.t_of(x)))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x, y = event.position().x(), event.position().y()
        if self._scrubbing:
            self.seekRequested.emit(self._clamp_t(self.t_of(x)))
            return
        text = ""
        if x < KEY_W and y >= self.header_h:
            pitch = self.pitch_at(y)
            if self.lo <= pitch <= self.hi:
                text = f"{note_name(pitch)}\nClick to highlight every {note_name(pitch)}"
        else:
            hit = self.note_at(x, y)
            if hit:
                track = self.indexes[hit[0]].track
                n = track.notes[hit[1]]
                text = (f"{note_name(n.pitch)} ({track.name})\n"
                        f"Starts at {format_time(n.start, 2)}, lasts {n.end - n.start:.2f} s")
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self.viewport())
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._scrubbing = False

    def wheelEvent(self, event) -> None:  # noqa: N802
        mods = event.modifiers()
        d = event.angleDelta()
        delta = d.y() or d.x()
        if mods & Qt.ControlModifier and mods & Qt.ShiftModifier:
            self.zoom_rows(1.15 if delta > 0 else 1 / 1.15)
        elif mods & Qt.ControlModifier:
            self.zoom_time(1.2 ** (delta / 120), event.position().x())
        elif mods & Qt.ShiftModifier or (d.x() and not d.y()):
            hb = self.horizontalScrollBar()
            hb.setValue(hb.value() - delta)
        else:
            super().wheelEvent(event)
            return
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape and (self.highlight_pitch is not None or self.selected):
            self.selected = None
            self.set_highlight(None)
            self.pitchToggled.emit(None)
            return
        super().keyPressEvent(event)
