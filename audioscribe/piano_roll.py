"""A Cubase style piano roll.

Layout inside the widget:

    +--------+----------------------------------------------+
    |        | time ruler                                   |
    | corner +----------------------------------------------+
    |        | lyrics lane (only when words were found)     |
    |        | chord lane (only when chords were found)     |
    +--------+----------------------------------------------+
    | keys   | note grid                                    |
    |        |                                              |
    +--------+----------------------------------------------+

Grid rows and notes are drawn into a cached pixmap that is only rebuilt on
scroll, zoom, or data changes. The playhead, ruler, lyrics, keyboard, the selected
span, and the selection box are drawn on top every frame, which keeps playback
smooth with thousands of notes.

Two modes:
  * View (the default): click to move the playhead, drag across the grid to select a
    span of time, drag the span's edges to change it.
  * Edit: move, resize, add and delete notes, and drop chords from the Chords tab.
    Nothing can be changed by accident because none of this works until Edit mode
    is switched on.
"""

from __future__ import annotations

import bisect
import json

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QAbstractScrollArea, QFrame, QToolTip

from .engine import Note
from .i18n import tr
from .music import NoteFilter, format_time, is_black, label_pc, note_label, note_name, pitch_hz

MIME_CHORDS = "application/x-audioscribe-chords"

KEY_W = 58
RULER_H = 24
LYRIC_H = 26
CHORD_H = 22
TIME_STEPS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
LOWEST, HIGHEST = 21, 108   # the range of an 88 key piano
EDGE_PX = 6                 # how close to the edge of a note or span counts as "the edge"
DRAG_PX = 4                 # a mouse move smaller than this is still a click
MIN_NOTE = 0.03             # shortest note in seconds
MIN_SPAN = 0.05
MAX_PPS = 3000.0

COL = {
    "bg": QColor("#2A3038"),
    "bg_black": QColor("#252A31"),
    "row_line": QColor("#30363F"),
    "octave_line": QColor("#454D59"),
    "beat": QColor("#323944"),
    "bar": QColor("#3E4653"),
    "sub": QColor("#2D343D"),
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
    "span": QColor("#69A7E0"),
    "span_loop": QColor("#5CC6C0"),
    "edit": QColor("#F0B23E"),
    "chord_block": QColor("#363E49"),
    "chord_line": QColor("#4A5361"),
    "ghost": QColor("#F0B23E"),
}


class _TrackIndex:
    """Sorted start times per track so visible notes can be found with bisect."""

    def __init__(self, track):
        self.track = track
        self.is_drum = track.name == "Drums"
        self.color = QColor(track.color)
        self.dirty = False   # true while notes are being dragged and the order is out of date
        self.starts: list[float] = []
        self.max_len = 0.0
        self.rebuild()

    def rebuild(self) -> None:
        notes = self.track.notes
        self.starts = [n.start for n in notes]
        self.max_len = max((n.end - n.start for n in notes), default=0.0)
        self.dirty = False

    def in_range(self, t0: float, t1: float) -> range:
        if self.dirty:
            return range(len(self.track.notes))
        lo = bisect.bisect_left(self.starts, t0 - self.max_len)
        hi = bisect.bisect_right(self.starts, t1)
        return range(lo, hi)


class PianoRoll(QAbstractScrollArea):
    seekRequested = Signal(float)
    noteClicked = Signal(int, int)        # track index, note index
    pitchToggled = Signal(object)         # MIDI pitch or None
    spanChanged = Signal(object, bool)    # (start, end) or None, and True once the drag is finished
    selectionChanged = Signal()           # which notes are selected changed (Edit mode)
    editStarted = Signal()                # about to change notes: the moment to take an undo snapshot
    notesEdited = Signal()                # notes were moved, resized, added or deleted
    auditionRequested = Signal(int, int)  # pitch, track index: play this note so it can be heard
    chordsDropped = Signal(object, float) # chords dragged in from the Chords tab, and the time
    dropRefused = Signal()                # chords dragged in while Edit mode is off
    chordClicked = Signal(float, object)  # a chord in the chord lane was clicked: its start and Chord

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
        self.meter = 4              # beats per bar, for the stronger grid lines
        self.chord_lane: list[tuple] = []   # (start, end, Chord or None)
        self.chord_starts: list[float] = []
        self.drop_preview = None    # callback (payload, time) -> [(start, end, pitch, strength)]
        self._ghost: list[tuple] | None = None
        self.pps = 60.0  # pixels per second
        self.row_h = 12
        self.lo, self.hi = 48, 84
        self.playhead = 0.0
        self.follow = True
        self.selected: set[Note] = set()
        self.highlight_pitch: int | None = None
        self.message = "Open a file or drop one here to get started."
        self.filter = NoteFilter()

        self.span: tuple[float, float] | None = None
        self.loop_on = False
        self.edit_mode = False
        self.snap_div = 0           # 0 off, 1 beat, 2 half a beat, 4 quarter of a beat
        self.edit_track = 0         # where double-click adds notes
        self.saved_view: tuple | None = None

        self._version = 0
        self._cache: QPixmap | None = None
        self._cache_key = None
        self._active: dict[int, QColor] = {}
        self._scrubbing = False
        self._press: dict | None = None
        self._marquee: tuple[float, float, float, float] | None = None
        self._headroom = False

        self.small_font = QFont(self.font())
        self.small_font.setPointSizeF(max(7.0, self.font().pointSizeF() * 0.85))
        self.bold_font = QFont(self.font())
        self.bold_font.setBold(True)
        self.viewport().setAcceptDrops(True)
        self._update_scrollbars()

    # Geometry ---------------------------------------------------------------

    @property
    def header_h(self) -> int:
        return RULER_H + (LYRIC_H if self.lane else 0) + (CHORD_H if self.chord_lane else 0)

    @property
    def chord_top(self) -> int:
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

    # Beats and snapping -----------------------------------------------------

    def beat_period(self) -> float | None:
        if len(self.beats) > 1:
            return (self.beats[-1] - self.beats[0]) / (len(self.beats) - 1)
        return None

    def snap_time(self, t: float) -> float:
        """Nearest grid line for the current snap setting, or t itself when snapping is off."""
        if not self.snap_div:
            return t
        period = self.beat_period()
        if period is None:
            step = 0.5 / self.snap_div
            return round(t / step) * step
        b = self.beats
        i = bisect.bisect_right(b, t) - 1
        if i < 0:
            origin, step = b[0], period / self.snap_div
        elif i >= len(b) - 1:
            origin, step = b[-1], period / self.snap_div
        else:
            origin, step = b[i], (b[i + 1] - b[i]) / self.snap_div
        return origin + round((t - origin) / step) * step

    def default_note_length(self) -> float:
        period = self.beat_period()
        return (period / 2.0) if period else 0.25

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
            self.lo = max(LOWEST, min(pitches) - 3)
            self.hi = min(HIGHEST, max(pitches) + 3)
            if self.hi - self.lo < 24:
                pad = (24 - (self.hi - self.lo)) // 2
                self.lo, self.hi = max(LOWEST, self.lo - pad), min(HIGHEST, self.hi + pad)
            self.message = ""
        else:
            self.lo, self.hi = 48, 84
            self.message = tr("No notes to show.") if tracks else ""
        self.selected = set()
        self.highlight_pitch = None
        self.span = None
        self.saved_view = None
        self._headroom = False
        self._press = None
        self._marquee = None
        self.playhead = 0.0
        self._active = {}
        self._initial_zoom()
        if self.edit_mode:
            self._add_headroom()

    def set_beats(self, beats, meter: int = 4) -> None:
        self.beats = list(beats or [])
        self.meter = max(1, int(meter))
        self.refresh()

    def set_chords(self, chords) -> None:
        """The chord lane: (start, end, Chord or None) for the whole song, or [] to hide it."""
        before = bool(self.chord_lane)
        self.chord_lane = [c for c in (chords or []) if c[1] > c[0]]
        self.chord_starts = [c[0] for c in self.chord_lane]
        if before != bool(self.chord_lane):
            self._update_scrollbars()
        self.refresh()

    def clear(self, message: str = "") -> None:
        self.indexes, self.lane, self.lane_starts, self.beats = [], [], [], []
        self.chord_lane, self.chord_starts = [], []
        self.duration = 0.0
        self.lo, self.hi = 48, 84
        self.selected = set()
        self.highlight_pitch = None
        self.span = None
        self.saved_view = None
        self._headroom = False
        self._press = None
        self._marquee = None
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

    def reindex(self) -> None:
        """Call after notes were changed from outside (or by an edit): re-sorts every
        track and rebuilds the lookup tables."""
        for idx in self.indexes:
            idx.track.notes.sort(key=lambda n: (n.start, n.pitch))
            idx.rebuild()
        self.refresh()

    def set_filter(self, note_filter: NoteFilter) -> None:
        self.filter = note_filter
        if self.selected:
            keep = {n for n in self.selected if self.filter.allows(n.pitch, self._is_drum_note(n))}
            if keep != self.selected:
                self.selected = keep
                self.selectionChanged.emit()
        self.refresh()

    def set_highlight(self, pitch: int | None) -> None:
        self.highlight_pitch = pitch
        if pitch is not None and self.lo <= pitch <= self.hi:
            y = self.y_of(pitch)
            if y < self.header_h or y > self.viewport().height() - self.row_h:
                vb = self.verticalScrollBar()
                vb.setValue(int(vb.value() + y - self.header_h - self._view_h() / 2))
        self.refresh()

    def select_note(self, track_index: int, note_index: int) -> None:
        note = self.indexes[track_index].track.notes[note_index]
        self.selected = {note}
        x, y = self.x_of(note.start), self.y_of(note.pitch)
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        if x < KEY_W or x > self.viewport().width() - 40:
            hb.setValue(int(note.start * self.pps - self._view_w() * 0.25))
        if y < self.header_h or y > self.viewport().height() - self.row_h:
            vb.setValue(int(vb.value() + y - self.header_h - self._view_h() / 2))
        self.refresh()
        self.selectionChanged.emit()

    def selected_notes(self) -> list[Note]:
        return sorted(self.selected, key=lambda n: (n.start, n.pitch))

    def clear_selection(self) -> None:
        if self.selected:
            self.selected = set()
            self.refresh()
            self.selectionChanged.emit()

    def _is_drum_note(self, note: Note) -> bool:
        return any(idx.is_drum and note in idx.track.notes for idx in self.indexes)

    def _shown(self, idx: _TrackIndex, n: Note) -> bool:
        return idx.track.visible and self.filter.allows(n.pitch, idx.is_drum)

    def shown_notes(self):
        """Every (track index, note) that is currently visible."""
        for ti, idx in enumerate(self.indexes):
            for n in idx.track.notes:
                if self._shown(idx, n):
                    yield ti, n

    def _track_of(self, note: Note) -> int:
        for ti, idx in enumerate(self.indexes):
            if note in idx.track.notes:
                return ti
        return 0

    # Span (the selected stretch of time) ------------------------------------

    def set_span(self, start: float, end: float, final: bool = True) -> None:
        a, b = sorted((self._clamp_t(start), self._clamp_t(end)))
        self.span = (a, b)
        self.viewport().update()
        self.spanChanged.emit(self.span, final)

    def clear_span(self) -> None:
        if self.span is not None:
            self.span = None
            self.saved_view = None
            self.viewport().update()
            self.spanChanged.emit(None, True)

    def set_loop_visual(self, on: bool) -> None:
        self.loop_on = on
        self.viewport().update()

    def zoom_to_span(self, pad: float = 0.05) -> None:
        """Fills the view with the selected span, tall enough rows to read the notes."""
        if not self.span or self.duration <= 0:
            return
        if self.saved_view is None:
            self.saved_view = (self.pps, self.row_h, self.horizontalScrollBar().value(),
                               self.verticalScrollBar().value())
        a, b = self.span
        length = max(b - a, 0.2)
        self.pps = min(MAX_PPS, max(2.0, self._view_w() / (length * (1 + 2 * pad))))
        pitches = [n.pitch for _, n in self.shown_notes() if n.end >= a and n.start <= b]
        if pitches:
            top, bottom = min(HIGHEST, max(pitches) + 2), max(LOWEST, min(pitches) - 2)
            rows = max(top - bottom + 1, 14)
            self.row_h = int(min(32, max(9, self._view_h() / rows)))
            centre = (top + bottom) / 2
        else:
            centre = (self.hi + self.lo) / 2
        self._update_scrollbars()
        self.horizontalScrollBar().setValue(int((a - length * pad) * self.pps))
        vb = self.verticalScrollBar()
        vb.setValue(int((self.hi - centre) * self.row_h - self._view_h() / 2))
        self.refresh()

    def restore_view(self) -> None:
        if self.saved_view is None:
            return
        pps, row_h, hx, vy = self.saved_view
        self.saved_view = None
        self.pps, self.row_h = pps, row_h
        self._update_scrollbars()
        self.horizontalScrollBar().setValue(hx)
        self.verticalScrollBar().setValue(vy)
        self.refresh()

    # Edit mode ---------------------------------------------------------------

    def set_edit_mode(self, on: bool) -> None:
        self.edit_mode = on
        self._press = None
        self._marquee = None
        if on:
            self._add_headroom()
        else:
            self.clear_selection()
        self.viewport().setCursor(Qt.ArrowCursor)
        self.viewport().update()

    def _add_headroom(self) -> None:
        """Extra empty rows above and below, so notes can be moved up or down."""
        if self._headroom or not self.indexes:
            return
        self._headroom = True
        self._extend_range(self.lo - 9, self.hi + 9)

    def _extend_range(self, lo: int, hi: int) -> None:
        lo, hi = max(LOWEST, min(self.lo, lo)), min(HIGHEST, max(self.hi, hi))
        if (lo, hi) == (self.lo, self.hi):
            return
        added_on_top = hi - self.hi
        self.lo, self.hi = lo, hi
        self._update_scrollbars()
        vb = self.verticalScrollBar()
        vb.setValue(vb.value() + added_on_top * self.row_h)
        self.refresh()

    def _begin_edit(self) -> None:
        self.editStarted.emit()

    def _finish_edit(self) -> None:
        self.reindex()
        self.notesEdited.emit()
        self.selectionChanged.emit()

    def select_all(self) -> None:
        self.selected = {n for _, n in self.shown_notes()}
        self.refresh()
        self.selectionChanged.emit()

    def delete_selected(self) -> None:
        if not self.selected:
            return
        self._begin_edit()
        gone = self.selected
        for idx in self.indexes:
            idx.track.notes[:] = [n for n in idx.track.notes if n not in gone]
        self.selected = set()
        self._finish_edit()

    def nudge(self, seconds: float = 0.0, semitones: int = 0) -> None:
        """Move the selected notes (arrow keys)."""
        notes = self.selected_notes()
        if not notes:
            return
        lo_p, hi_p = min(n.pitch for n in notes), max(n.pitch for n in notes)
        semitones = max(LOWEST - lo_p, min(semitones, HIGHEST - hi_p))
        first, last = min(n.start for n in notes), max(n.end for n in notes)
        seconds = max(-first, min(seconds, max(0.0, self.duration - last)))
        if not semitones and abs(seconds) < 1e-9:
            return
        self._begin_edit()
        for n in notes:
            n.start += seconds
            n.end += seconds
            n.pitch += semitones
            n.edited = True
        self._extend_range(lo_p + semitones - 2, hi_p + semitones + 2)
        self._finish_edit()
        self.auditionRequested.emit(notes[0].pitch, self._track_of(notes[0]))

    def set_selected_strength(self, value: float, push: bool = True) -> None:
        notes = self.selected_notes()
        if not notes:
            return
        if push:
            self._begin_edit()
        for n in notes:
            n.velocity = min(1.0, max(0.05, value))
            n.edited = True
        self.reindex()
        self.notesEdited.emit()

    def add_note_at(self, t: float, pitch: int) -> None:
        shown = [i for i, idx in enumerate(self.indexes) if idx.track.visible]
        if not shown:
            return
        ti = self.edit_track if self.edit_track in shown else shown[0]
        pitch = max(LOWEST, min(HIGHEST, pitch))
        start = self._clamp_t(self.snap_time(t))
        length = self.default_note_length()
        if self.snap_div and self.beat_period():
            length = self.beat_period() / self.snap_div
        end = min(self.duration, start + length)
        if end - start < MIN_NOTE:
            start = max(0.0, end - MIN_NOTE)
        self._begin_edit()
        note = Note(start, end, pitch, 0.75, edited=True)
        self.indexes[ti].track.notes.append(note)
        self.selected = {note}
        self._finish_edit()
        self.auditionRequested.emit(pitch, ti)

    def insert_notes(self, track_index: int, rows) -> None:
        """Add notes as one edit (used for chords from the Chords tab): (start, end, pitch, strength)."""
        if not (0 <= track_index < len(self.indexes)) or not rows:
            return
        self._begin_edit()
        new = []
        for start, end, pitch, vel in rows:
            start = self._clamp_t(start)
            end = min(self.duration, max(end, start + MIN_NOTE))
            new.append(Note(start, end, max(LOWEST, min(HIGHEST, int(pitch))), vel, edited=True))
        self.indexes[track_index].track.notes.extend(new)
        self.selected = set(new)
        pitches = [n.pitch for n in new]
        self._extend_range(min(pitches) - 2, max(pitches) + 2)
        self._finish_edit()

    # Dragging chords in from the Chords tab ---------------------------------------

    def _drop_payload(self, event) -> dict | None:
        mime = event.mimeData()
        if not mime.hasFormat(MIME_CHORDS):
            return None
        try:
            data = json.loads(bytes(mime.data(MIME_CHORDS)).decode("utf-8"))
            return data if isinstance(data, dict) and isinstance(data.get("chords"), list) else None
        except (ValueError, UnicodeDecodeError):
            return None

    def _drop_time(self, x: float) -> float:
        t = self._clamp_t(self.t_of(x))
        if self.snap_div:
            return self._clamp_t(self.snap_time(t))
        div, self.snap_div = self.snap_div, 1      # chords land on a beat even with snapping off
        t = self.snap_time(t)
        self.snap_div = div
        return self._clamp_t(t)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._drop_payload(event) is not None and self.duration > 0:
            event.acceptProposedAction()
            if not self.edit_mode:
                self.dropRefused.emit()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        payload = self._drop_payload(event)
        if payload is None or self.duration <= 0:
            event.ignore()
            return
        if not self.edit_mode:
            self._ghost = None
            event.ignore()
            return
        t = self._drop_time(event.position().x())
        self._ghost = list(self.drop_preview(payload, t)) if self.drop_preview else []
        event.acceptProposedAction()
        self.viewport().update()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._ghost = None
        self.viewport().update()

    def dropEvent(self, event) -> None:  # noqa: N802
        payload = self._drop_payload(event)
        self._ghost = None
        self.viewport().update()
        if payload is None or not self.edit_mode:
            event.ignore()
            if payload is not None:
                self.dropRefused.emit()
            return
        event.acceptProposedAction()
        self.chordsDropped.emit(payload, self._drop_time(event.position().x()))

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
        self.saved_view = None
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
        self.pps = min(MAX_PPS, max(min_pps, self.pps * factor))
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
                if n.start <= t < n.end and self.filter.allows(n.pitch, idx.is_drum):
                    active[n.pitch] = idx.color
        return active

    def set_playhead(self, t: float) -> None:
        self.playhead = t
        self._active = self._compute_active(t)
        if self.follow and self.duration > 0 and not self._scrubbing and self._press is None:
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
            self._paint_span(p, w, h)
            self._paint_marquee(p, w, h)
            self._paint_ghost(p, w, h)
            self._paint_playhead(p, h)
            if self.lane:
                self._paint_lane(p, w)
            if self.chord_lane:
                self._paint_chord_lane(p, w)
            self._paint_ruler(p, w)
        else:
            p.fillRect(KEY_W, 0, w - KEY_W, self.header_h, COL["panel"])
        self._paint_keyboard(p, h)
        self._paint_corner(p)
        if self.edit_mode:
            p.setPen(QPen(COL["edit"], 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(1, 1, w - 2, h - 2)
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

    def _row_kept(self, pitch: int) -> bool:
        """For shading: is this key's note shown by the scale filter?"""
        f = self.filter
        if f.classes is None:
            return True
        return ((pitch % 12) in f.classes) != f.outside

    def _paint_rows(self, p: QPainter, w: int, h: int) -> None:
        rh = self.row_h
        shade = self.filter.classes is not None
        for pitch in self._visible_pitches(h):
            y = self.y_of(pitch)
            if is_black(pitch):
                p.fillRect(QRectF(KEY_W, y, w - KEY_W, rh), COL["bg_black"])
            if shade and not self._row_kept(pitch):
                p.fillRect(QRectF(KEY_W, y, w - KEY_W, rh), QColor(10, 12, 15, 150))
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
        period = self.beat_period()
        if period:
            beat_px = period * self.pps
        if beat_px >= 7:
            i0 = bisect.bisect_left(self.beats, t0)
            i1 = bisect.bisect_right(self.beats, t1)
            if beat_px >= 56:  # zoomed in: also show half and quarter beats
                sub = 4 if beat_px >= 160 else 2
                p.setPen(COL["sub"])
                for i in range(max(i0 - 1, 0), min(i1 + 1, len(self.beats) - 1)):
                    b0, b1 = self.beats[i], self.beats[i + 1]
                    for k in range(1, sub):
                        x = self.x_of(b0 + (b1 - b0) * k / sub)
                        p.drawLine(QPointF(x, top), QPointF(x, h))
            for i in range(i0, i1):
                x = self.x_of(self.beats[i])
                p.setPen(COL["bar"] if i % self.meter == 0 else COL["beat"])
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
        edited = QPen(QColor(255, 255, 255, 215), 1.3)
        flt = self.filter
        for idx in self.indexes:
            if not idx.track.visible:
                continue
            base = idx.color
            edge = QPen(base.lighter(125), 1.0)
            notes = idx.track.notes
            part = idx.track.name
            for ni in idx.in_range(t0, t1):
                n = notes[ni]
                if n.end < t0 or not flt.allows(n.pitch, idx.is_drum):
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
                marked = n in self.selected or (self.highlight_pitch == n.pitch)
                if marked:
                    p.setPen(amber)
                elif dim:
                    p.setPen(Qt.NoPen)
                else:
                    p.setPen(edited if n.edited else edge)
                p.drawRoundedRect(rect, 2, 2)
                if not dim and rh >= 11 and rect.width() > 26:
                    label = note_label(n.pitch, part)
                    if rect.width() > 110 and rh >= 13:
                        label += f"  {n.velocity * 100:.0f}%"
                    if fm.horizontalAdvance(label) + 6 < rect.width():
                        p.setPen(COL["note_text"])
                        p.drawText(rect.adjusted(3, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, label)
                    elif fm.horizontalAdvance(note_label(n.pitch, part)) + 6 < rect.width():
                        p.setPen(COL["note_text"])
                        p.drawText(rect.adjusted(3, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft,
                                   note_label(n.pitch, part))

    def _paint_span(self, p: QPainter, w: int, h: int) -> None:
        if not self.span:
            return
        a, b = self.span
        x0, x1 = max(KEY_W, self.x_of(a)), min(w, self.x_of(b))
        if x1 <= KEY_W or x0 >= w:
            return
        color = COL["span_loop"] if self.loop_on else COL["span"]
        band = QColor(color)
        band.setAlpha(46 if self.loop_on else 38)
        p.fillRect(QRectF(x0, self.header_h, x1 - x0, h - self.header_h), band)
        p.setPen(QPen(color, 1.5))
        for t in (a, b):
            x = self.x_of(t)
            if KEY_W <= x <= w:
                p.drawLine(QPointF(x, self.header_h), QPointF(x, h))
        # The length goes in a small label at the top of the band, clear of the time ruler.
        label = f"{'LOOP  ' if self.loop_on else ''}{b - a:.2f} s"
        fm = QFontMetrics(self.small_font)
        pill_w = fm.horizontalAdvance(label) + 14
        if x1 - x0 > pill_w + 8:
            cx = (x0 + x1) / 2
            rect = QRectF(cx - pill_w / 2, self.header_h + 5, pill_w, fm.height() + 4)
            p.setPen(Qt.NoPen)
            fill = QColor(COL["panel"])
            fill.setAlpha(225)
            p.setBrush(fill)
            p.drawRoundedRect(rect, 4, 4)
            p.setFont(self.small_font)
            p.setPen(COL["text"])
            p.drawText(rect, Qt.AlignCenter, label)

    def _paint_marquee(self, p: QPainter, w: int, h: int) -> None:
        if not self._marquee:
            return
        x0, y0, x1, y1 = self._marquee
        rect = QRectF(QPointF(min(x0, x1), min(y0, y1)), QPointF(max(x0, x1), max(y0, y1)))
        p.setPen(QPen(COL["edit"], 1, Qt.DashLine))
        fill = QColor(COL["edit"])
        fill.setAlpha(28)
        p.setBrush(fill)
        p.drawRect(rect)

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

    def _paint_chord_lane(self, p: QPainter, w: int) -> None:
        top = self.chord_top
        p.fillRect(QRectF(KEY_W, top, w - KEY_W, CHORD_H), COL["panel"])
        p.setPen(COL["panel_line"])
        p.drawLine(QPointF(KEY_W, top + CHORD_H - 0.5), QPointF(w, top + CHORD_H - 0.5))
        p.save()
        p.setClipRect(QRectF(KEY_W, top, w - KEY_W, CHORD_H))
        p.setRenderHint(QPainter.Antialiasing, True)
        fm = QFontMetrics(self.bold_font)
        p.setFont(self.bold_font)
        t0, t1 = self.t_of(KEY_W), self.t_of(w)
        i0 = max(0, bisect.bisect_right(self.chord_starts, t0) - 1)
        i1 = bisect.bisect_right(self.chord_starts, t1)
        for start, end, chord in self.chord_lane[i0:i1]:
            if chord is None or end < t0:
                continue
            x0, x1 = self.x_of(start), self.x_of(end)
            current = start <= self.playhead < end
            rect = QRectF(x0 + 1, top + 3, max(2.0, x1 - x0 - 2), CHORD_H - 6)
            p.setPen(Qt.NoPen)
            p.setBrush(COL["chord_block"] if not current else QColor(240, 178, 62, 60))
            p.drawRoundedRect(rect, 3, 3)
            name = chord.name()
            if rect.width() >= 16:
                if fm.horizontalAdvance(name) + 6 > rect.width():
                    name = fm.elidedText(name, Qt.ElideRight, int(rect.width() - 6))
                p.setPen(COL["playhead"] if current else COL["text"])
                p.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, name)
        p.restore()

    def _paint_ghost(self, p: QPainter, w: int, h: int) -> None:
        """Where dragged-in chords would land."""
        if not self._ghost:
            return
        p.save()
        p.setClipRect(QRectF(KEY_W, self.header_h, w - KEY_W, h - self.header_h))
        fill = QColor(COL["ghost"])
        fill.setAlpha(70)
        p.setBrush(fill)
        p.setPen(QPen(COL["ghost"], 1, Qt.DashLine))
        rh = self.row_h
        for start, end, pitch, _vel in self._ghost:
            y = self.y_of(pitch)
            x0, x1 = self.x_of(start), self.x_of(end)
            p.drawRect(QRectF(x0, y + 1, max(2.0, x1 - x0), rh - 2))
        p.restore()

    def _paint_ruler(self, p: QPainter, w: int) -> None:
        p.fillRect(QRectF(KEY_W, 0, w - KEY_W, RULER_H), COL["panel"])
        p.setPen(COL["panel_line"])
        p.drawLine(QPointF(KEY_W, RULER_H - 0.5), QPointF(w, RULER_H - 0.5))
        p.save()
        p.setClipRect(QRectF(KEY_W, 0, w - KEY_W, RULER_H))
        if self.span:
            a, b = self.span
            color = QColor(COL["span_loop"] if self.loop_on else COL["span"])
            xa, xb = self.x_of(a), self.x_of(b)
            color.setAlpha(80)
            p.fillRect(QRectF(xa, 0, max(1.0, xb - xa), RULER_H - 1), color)
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
        if self.span:
            a, b = self.span
            xa, xb = self.x_of(a), self.x_of(b)
            p.setPen(QPen(COL["span_loop"] if self.loop_on else COL["span"], 2))
            p.drawLine(QPointF(xa, 2), QPointF(xa, RULER_H - 2))
            p.drawLine(QPointF(xb, 2), QPointF(xb, RULER_H - 2))
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
            if self.filter.classes is not None and not self._row_kept(pitch):
                p.fillRect(QRectF(0, y, KEY_W, rh), QColor(10, 12, 15, 120))
            if pitch == self.highlight_pitch:
                p.fillRect(QRectF(KEY_W - 5, y, 5, rh), COL["playhead"])
            if pitch % 12 == label_pc() and rh >= 8:
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
        if self.edit_mode:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setPen(Qt.NoPen)
            p.setBrush(COL["edit"])
            p.drawRoundedRect(QRectF(6, 4, KEY_W - 14, RULER_H - 9), 3, 3)
            p.setPen(COL["note_text"])
            p.setFont(self.small_font)
            p.drawText(QRectF(6, 4, KEY_W - 14, RULER_H - 9), Qt.AlignCenter, "EDIT")
        if self.lane:
            p.setPen(COL["muted"])
            p.setFont(self.small_font)
            p.drawText(QRectF(0, RULER_H, KEY_W - 6, LYRIC_H), Qt.AlignRight | Qt.AlignVCenter, tr("Lyrics"))
        if self.chord_lane:
            p.setPen(COL["muted"])
            p.setFont(self.small_font)
            p.drawText(QRectF(0, self.chord_top, KEY_W - 6, CHORD_H), Qt.AlignRight | Qt.AlignVCenter, tr("Chords"))

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
                if n.pitch == pitch and self.filter.allows(n.pitch, idx.is_drum) \
                        and n.start - slack <= t <= max(n.end, n.start + slack):
                    return ti, ni
        return None

    def _span_edge_at(self, x: float) -> str | None:
        if not self.span:
            return None
        a, b = self.span
        if abs(x - self.x_of(a)) <= EDGE_PX:
            return "a"
        if abs(x - self.x_of(b)) <= EDGE_PX:
            return "b"
        return None

    def _near_note_end(self, note: Note, x: float) -> bool:
        x0, x1 = self.x_of(note.start), self.x_of(note.end)
        zone = min(EDGE_PX, max(2.0, (x1 - x0) * 0.4))
        return x >= x1 - zone

    def _marquee_notes(self, x0: float, y0: float, x1: float, y1: float) -> set[Note]:
        ta, tb = sorted((self.t_of(x0), self.t_of(x1)))
        pa, pb = sorted((self.pitch_at(y0), self.pitch_at(y1)))
        found: set[Note] = set()
        for idx in self.indexes:
            if not idx.track.visible:
                continue
            notes = idx.track.notes
            for ni in idx.in_range(ta, tb):
                n = notes[ni]
                if n.end >= ta and n.start <= tb and pa <= n.pitch <= pb \
                        and self.filter.allows(n.pitch, idx.is_drum):
                    found.add(n)
        return found

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
                    if self.highlight_pitch is not None:
                        self.auditionRequested.emit(pitch, -1)
            return
        if self.duration <= 0:
            return
        if self.chord_lane and self.chord_top <= y < self.chord_top + CHORD_H:
            t = self.t_of(x)
            i = bisect.bisect_right(self.chord_starts, t) - 1
            if 0 <= i < len(self.chord_lane) and self.chord_lane[i][0] <= t < self.chord_lane[i][1]:
                start, _end, chord = self.chord_lane[i]
                self.chordClicked.emit(start, chord)
            return
        if y < self.header_h:
            self._scrubbing = True
            self.seekRequested.emit(self._clamp_t(self.t_of(x)))
            return
        self.setFocus()
        mods = event.modifiers()
        hit = self.note_at(x, y)
        t = self.t_of(x)

        if self.edit_mode and hit:
            ti, ni = hit
            note = self.indexes[ti].track.notes[ni]
            additive = bool(mods & (Qt.ControlModifier | Qt.ShiftModifier))
            if additive:
                if note in self.selected:
                    self.selected.discard(note)
                    self.refresh()
                    self.selectionChanged.emit()
                    return
                self.selected.add(note)
            elif note not in self.selected:
                self.selected = {note}
            self._press = {
                "kind": "resize" if self._near_note_end(note, x) else "move",
                "x": x, "y": y, "t": t, "pitch": self.pitch_at(y), "note": note, "ti": ti,
                "orig": [(n, n.start, n.end, n.pitch) for n in self.selected],
                "begun": False, "last_pitch": note.pitch,
            }
            self.refresh()
            self.selectionChanged.emit()
            self.auditionRequested.emit(note.pitch, ti)
            return

        edge = self._span_edge_at(x)
        if edge:
            self._press = {"kind": "edge", "edge": edge, "x": x, "y": y}
            return
        self._press = {"kind": "empty", "x": x, "y": y, "t": t, "hit": hit, "mods": mods,
                       "shift": bool(mods & Qt.ShiftModifier), "base": set(self.selected)}

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x, y = event.position().x(), event.position().y()
        if self._scrubbing:
            self.seekRequested.emit(self._clamp_t(self.t_of(x)))
            return
        press = self._press
        if press is not None:
            if not (event.buttons() & Qt.LeftButton):  # the release never arrived
                self.mouseReleaseEvent(event)
                return
            self._drag(press, x, y)
            return
        self._hover(event, x, y)

    def _drag(self, press: dict, x: float, y: float) -> None:
        kind = press["kind"]
        if kind == "empty":
            if abs(x - press["x"]) < DRAG_PX and abs(y - press["y"]) < DRAG_PX:
                return
            # In Edit mode a drag on empty space draws a selection box (hold Shift for a span).
            kind = "marquee" if (self.edit_mode and not press["shift"]) else "span"
            press["kind"] = kind
        if kind == "span":
            self.set_span(self.snap_time(press["t"]), self.snap_time(self._clamp_t(self.t_of(x))), final=False)
        elif kind == "edge":
            a, b = self.span
            t = self.snap_time(self._clamp_t(self.t_of(x)))
            if press["edge"] == "a":
                self.set_span(t, b, final=False)
            else:
                self.set_span(a, t, final=False)
            # dragging an edge past the other end flips which edge is being dragged
            press["edge"] = "a" if abs(t - self.span[0]) < 1e-9 else "b"
        elif kind == "marquee":
            self._marquee = (press["x"], press["y"], x, y)
            found = self._marquee_notes(*self._marquee)
            mods = press["mods"]
            self.selected = (press["base"] | found) if mods & (Qt.ControlModifier | Qt.ShiftModifier) else found
            self.refresh()
        elif kind in ("move", "resize"):
            if not press["begun"]:
                if abs(x - press["x"]) < DRAG_PX and abs(y - press["y"]) < DRAG_PX:
                    return
                press["begun"] = True
                self._begin_edit()
                for idx in self.indexes:
                    idx.dirty = True
            if kind == "move":
                self._apply_move(press, x, y)
            else:
                self._apply_resize(press, x)
            self.refresh()

    def _apply_move(self, press: dict, x: float, y: float) -> None:
        orig = press["orig"]
        dt = self.t_of(x) - press["t"]
        dp = self.pitch_at(y) - press["pitch"]
        first = min(o[1] for o in orig)
        last = max(o[2] for o in orig)
        dt = self.snap_time(first + dt) - first
        dt = max(-first, min(dt, max(0.0, self.duration - last)))
        lo_p, hi_p = min(o[3] for o in orig), max(o[3] for o in orig)
        dp = max(LOWEST - lo_p, min(dp, HIGHEST - hi_p))
        for n, s, e, p in orig:
            n.start, n.end, n.pitch = s + dt, e + dt, p + dp
            n.edited = n.edited or dt != 0 or dp != 0
        self._extend_range(lo_p + dp - 2, hi_p + dp + 2)
        note = press["note"]
        if note.pitch != press["last_pitch"]:
            press["last_pitch"] = note.pitch
            self.auditionRequested.emit(note.pitch, press["ti"])

    def _apply_resize(self, press: dict, x: float) -> None:
        orig = press["orig"]
        anchor = next(o for o in orig if o[0] is press["note"])
        new_end = self.snap_time(anchor[2] + (self.t_of(x) - press["t"]))
        dt = new_end - anchor[2]
        for n, s, e, p in orig:
            n.end = max(s + MIN_NOTE, min(self.duration, e + dt))
            n.edited = n.edited or n.end != e

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._scrubbing = False
        press, self._press = self._press, None
        if press is None:
            return
        kind = press["kind"]
        if kind == "empty":
            self._click(press)
        elif kind == "span":
            if self.span and self.span[1] - self.span[0] < MIN_SPAN:
                self.clear_span()
            elif self.span:
                self.spanChanged.emit(self.span, True)
                self.seekRequested.emit(self.span[0])
        elif kind == "edge":
            if self.span:
                self.spanChanged.emit(self.span, True)
        elif kind == "marquee":
            self._marquee = None
            self.refresh()
            self.selectionChanged.emit()
        elif kind in ("move", "resize"):
            for idx in self.indexes:
                idx.dirty = False
            if press["begun"]:
                self._finish_edit()
            else:
                mods = event.modifiers()
                if not (mods & (Qt.ControlModifier | Qt.ShiftModifier)) and len(self.selected) > 1:
                    self.selected = {press["note"]}
                    self.refresh()
                    self.selectionChanged.emit()

    def _click(self, press: dict) -> None:
        """A press and release in the grid without a drag."""
        hit, t = press["hit"], press["t"]
        if hit and not self.edit_mode:
            note = self.indexes[hit[0]].track.notes[hit[1]]
            self.selected = {note}
            self.refresh()
            self.noteClicked.emit(*hit)
            self.seekRequested.emit(note.start)
            return
        if self.selected and not (press["mods"] & (Qt.ControlModifier | Qt.ShiftModifier)):
            self.selected = set()
            self.refresh()
            self.selectionChanged.emit()
        self.seekRequested.emit(self._clamp_t(t))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        x, y = event.position().x(), event.position().y()
        if x < KEY_W or y < self.header_h or self.duration <= 0:
            return
        if self.edit_mode:
            if not self.note_at(x, y):
                self.add_note_at(self.t_of(x), self.pitch_at(y))
        elif self.span and self.span[0] <= self.t_of(x) <= self.span[1]:
            self.zoom_to_span()

    def _hover(self, event, x: float, y: float) -> None:
        cursor = Qt.ArrowCursor
        text = ""
        in_grid = x >= KEY_W and y >= self.header_h
        hit = self.note_at(x, y) if in_grid else None
        if self.edit_mode and hit:
            note = self.indexes[hit[0]].track.notes[hit[1]]
            cursor = Qt.SizeHorCursor if self._near_note_end(note, x) else Qt.SizeAllCursor
        elif in_grid and self._span_edge_at(x):
            cursor = Qt.SizeHorCursor
        self.viewport().setCursor(cursor)
        if x < KEY_W and y >= self.header_h:
            pitch = self.pitch_at(y)
            if self.lo <= pitch <= self.hi:
                text = (f"{note_name(pitch)}  ({pitch_hz(pitch):.1f} Hz)\n"
                        + tr("Click to highlight every {name} and hear it").format(name=note_name(pitch)))
        elif self.chord_lane and self.chord_top <= y < self.chord_top + CHORD_H and x >= KEY_W:
            t = self.t_of(x)
            i = bisect.bisect_right(self.chord_starts, t) - 1
            if 0 <= i < len(self.chord_lane) and self.chord_lane[i][2] is not None:
                start, end, chord = self.chord_lane[i]
                text = (f"{chord.name()}  {format_time(start, 1)} - {format_time(end, 1)}\n"
                        + tr("Click to hear it"))
        elif hit:
            track = self.indexes[hit[0]].track
            n = track.notes[hit[1]]
            period = self.beat_period()
            length = n.end - n.start
            beats = "  " + tr("({n} beats)").format(n=f"{length / period:.2f}") if period else ""
            name = note_label(n.pitch, track.name)
            if name != note_name(n.pitch):
                name = f"{name}  ({note_name(n.pitch)})"
            text = (f"{name}, {tr(track.name)}\n"
                    f"MIDI {n.pitch}, {pitch_hz(n.pitch):.1f} Hz\n"
                    + tr("Start {start}   End {end}").format(start=format_time(n.start, 2),
                                                             end=format_time(n.end, 2)) + "\n"
                    + tr("Length {length} s").format(length=f"{length:.2f}") + beats + "\n"
                    + tr("Strength {n}%").format(n=f"{n.velocity * 100:.0f}"))
            if n.edited:
                text += "\n" + tr("Changed by hand")
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self.viewport())
        else:
            QToolTip.hideText()

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
        key, mods = event.key(), event.modifiers()
        if key == Qt.Key_Escape:
            if self.highlight_pitch is not None or self.selected:
                self.selected = set()
                self.set_highlight(None)
                self.pitchToggled.emit(None)
                self.selectionChanged.emit()
            elif self.span:
                self.clear_span()
            return
        if self.edit_mode:
            period = self.beat_period()
            step = (period / self.snap_div) if (period and self.snap_div) else 0.05
            big = bool(mods & Qt.ShiftModifier)
            if key in (Qt.Key_Delete, Qt.Key_Backspace):
                self.delete_selected()
                return
            if key == Qt.Key_A and mods & Qt.ControlModifier:
                self.select_all()
                return
            if key == Qt.Key_Up:
                self.nudge(semitones=12 if big else 1)
                return
            if key == Qt.Key_Down:
                self.nudge(semitones=-12 if big else -1)
                return
            if key == Qt.Key_Left:
                self.nudge(seconds=-step * (4 if big else 1))
                return
            if key == Qt.Key_Right:
                self.nudge(seconds=step * (4 if big else 1))
                return
        super().keyPressEvent(event)
