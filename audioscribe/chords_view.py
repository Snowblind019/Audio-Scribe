"""The Chords tab: a palette of chords in a key that can be heard and dragged onto the piano
roll, plus a progression library, what could come next, substitutes, chords from outside the
key, ways to change key, a circle of fifths, and a sketch pad.

Everything that knows music theory lives in theory.py. This file is only the interface.
"""

from __future__ import annotations

import json
import math

from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QTabWidget,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import theory
from .i18n import tr
from .music import format_time, key_name, pc_name
from .piano_roll import MIME_CHORDS
from .synth import INSTRUMENTS
from .theory import Chord

LENGTHS = [("1 beat", 1), ("2 beats", 2), ("1 bar", 4), ("2 bars", 8), ("4 bars", 16)]
SKETCH_SLOTS = 7


def chord_mime(chords: list[Chord]) -> QMimeData:
    mime = QMimeData()
    mime.setData(MIME_CHORDS, json.dumps({"chords": [c.to_dict() for c in chords]}).encode("utf-8"))
    mime.setText(" ".join(c.name() for c in chords))
    return mime


def chords_from_payload(payload: dict) -> list[Chord]:
    out = []
    for d in payload.get("chords", [])[:64]:
        try:
            out.append(Chord.from_dict(d))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _chip_pixmap(text: str, font: QFont) -> QPixmap:
    fm = QFontMetrics(font)
    w, h = fm.horizontalAdvance(text) + 18, fm.height() + 10
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#F0B23E"))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(0, 0, w, h), 5, 5)
    p.setPen(QColor("#211A0C"))
    p.setFont(font)
    p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, text)
    p.end()
    return pm


def start_chord_drag(source: QWidget, chords: list[Chord]) -> None:
    if not chords:
        return
    drag = QDrag(source)
    drag.setMimeData(chord_mime(chords))
    label = " ".join(c.name() for c in chords[:6]) + (" ..." if len(chords) > 6 else "")
    font = QFont(source.font())
    font.setBold(True)
    pm = _chip_pixmap(label, font)
    drag.setPixmap(pm)
    drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
    drag.exec(Qt.CopyAction)


class ChordButton(QToolButton):
    """A chord chip: click to hear it (and pick it), drag it onto the piano roll."""

    picked = Signal(object)   # Chord

    def __init__(self, chord: Chord, caption: str = "", parent=None):
        super().__init__(parent)
        self.chord = chord
        self.caption = caption
        self.setObjectName("ChordChip")
        self.setProperty("i18n_skip", True)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._press: QPoint | None = None
        self.refresh()

    def refresh(self) -> None:
        name = self.chord.name()
        self.setText(f"{name}\n{self.caption}" if self.caption else name)
        self.setToolTip(tr("Click to hear {name}. Drag it onto the piano roll (in Edit mode) to add it.").format(
            name=name))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press is not None and (event.buttons() & Qt.LeftButton):
            if (event.position().toPoint() - self._press).manhattanLength() >= QApplication.startDragDistance():
                self._press = None
                self.setDown(False)
                start_chord_drag(self, [self.chord])
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        was_click = self._press is not None
        self._press = None
        super().mouseReleaseEvent(event)
        if was_click and event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.picked.emit(self.chord)


class DragHandle(QToolButton):
    """A small "Drag" button that drags a whole row of chords."""

    def __init__(self, get_chords, parent=None):
        super().__init__(parent)
        self.get_chords = get_chords
        self.setText(tr("Drag"))
        self.setToolTip(tr("Drag all these chords onto the piano roll (in Edit mode)"))
        self.setCursor(Qt.OpenHandCursor)
        self._press = None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press is not None and (event.position().toPoint() - self._press).manhattanLength() >= 4:
            self._press = None
            self.setDown(False)
            start_chord_drag(self, self.get_chords())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._press = None
        super().mouseReleaseEvent(event)


class ChordRow(QFrame):
    """A row of chord chips with Play, Drag and (optionally) more buttons."""

    picked = Signal(object)       # Chord
    playRequested = Signal(object)  # list[Chord]

    def __init__(self, chords: list[Chord], captions: list[str] | None = None, title: str = "",
                 note: str = "", parent=None):
        super().__init__(parent)
        self.chords = chords
        self.setObjectName("ChordRow")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(3)
        if title:
            t = QLabel(title)
            t.setObjectName("FieldName")
            t.setProperty("i18n_skip", True)
            lay.addWidget(t)
        if note:
            n = QLabel(note)
            n.setObjectName("Hint")
            n.setWordWrap(True)
            n.setProperty("i18n_skip", True)
            lay.addWidget(n)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.buttons = []
        for i, c in enumerate(chords):
            b = ChordButton(c, captions[i] if captions else "")
            b.picked.connect(self.picked)
            self.buttons.append(b)
            row.addWidget(b)
        row.addStretch(1)
        play = QToolButton()
        play.setText(tr("Play"))
        play.setToolTip(tr("Hear these chords one after another"))
        play.clicked.connect(lambda: self.playRequested.emit(self.chords))
        row.addWidget(play)
        row.addWidget(DragHandle(lambda: self.chords))
        lay.addLayout(row)


class CircleOfFifths(QWidget):
    """Major keys around the outside, their relative minors inside. Click a key to use it,
    the chords of the current key are lit."""

    keyPicked = Signal(int, str)    # tonic, "major" or "minor"
    chordPicked = Signal(object)    # Chord

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tonic, self.mode = 0, "major"
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setToolTip(tr("Click a key to use it. Right-click to hear its chord."))

    def set_key(self, tonic: int, mode: str) -> None:
        self.tonic, self.mode = tonic, mode
        self.update()

    def _geometry(self):
        side = min(self.width(), self.height()) - 8
        cx, cy = self.width() / 2, self.height() / 2
        return cx, cy, side / 2, side / 2 * 0.62, side / 2 * 0.30

    def _hit(self, pos: QPointF):
        cx, cy, r_out, r_mid, r_in = self._geometry()
        dx, dy = pos.x() - cx, pos.y() - cy
        dist = math.hypot(dx, dy)
        if dist > r_out or dist < r_in:
            return None
        angle = (math.degrees(math.atan2(dx, -dy)) + 15) % 360
        i = int(angle // 30)
        major_root = theory.CIRCLE[i]
        if dist >= r_mid:
            return major_root, "major"
        return (major_root + 9) % 12, "minor"

    def mousePressEvent(self, event) -> None:  # noqa: N802
        hit = self._hit(event.position())
        if hit is None:
            return
        tonic, mode = hit
        if event.button() == Qt.RightButton:
            self.chordPicked.emit(Chord(tonic, "maj" if mode == "major" else "min"))
        else:
            self.keyPicked.emit(tonic, mode)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r_out, r_mid, r_in = self._geometry()
        key_chords = theory.diatonic(self.tonic, "Major" if self.mode == "major" else "Natural minor")
        lit = {(c.root, "minor" if c.quality in theory.MINORISH else "major") for c in key_chords if c.quality != "dim"}
        font = QFont(self.font())
        font.setBold(True)
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, self.font().pointSizeF() * 0.85))
        for i, major_root in enumerate(theory.CIRCLE):
            a0 = -90 - 15 + i * 30
            for ring, (r1, r2), root, mode in (("outer", (r_mid, r_out), major_root, "major"),
                                                ("inner", (r_in, r_mid), (major_root + 9) % 12, "minor")):
                path = QPainterPath()
                path.arcMoveTo(QRectF(cx - r2, cy - r2, 2 * r2, 2 * r2), -a0)
                path.arcTo(QRectF(cx - r2, cy - r2, 2 * r2, 2 * r2), -a0, -30)
                path.arcTo(QRectF(cx - r1, cy - r1, 2 * r1, 2 * r1), -(a0 + 30), 30)
                path.closeSubpath()
                current = root == self.tonic and mode == self.mode
                if current:
                    fill = QColor("#F0B23E")
                elif (root, mode) in lit:
                    fill = QColor("#3E5F86")
                else:
                    fill = QColor("#2E343D") if ring == "outer" else QColor("#272D35")
                p.setPen(QPen(QColor("#3F4651"), 1))
                p.setBrush(fill)
                p.drawPath(path)
                mid_angle = math.radians(a0 + 15)
                rr = (r1 + r2) / 2
                tx, ty = cx + rr * math.cos(mid_angle), cy + rr * math.sin(mid_angle)
                label = pc_name(root, absolute=True) + ("" if mode == "major" else "m")
                p.setFont(font if ring == "outer" else small)
                p.setPen(QColor("#211A0C") if current else QColor("#DCE1E8"))
                p.drawText(QRectF(tx - 30, ty - 10, 60, 20), Qt.AlignCenter, label)
        p.setPen(QColor("#8D97A5"))
        p.setFont(small)
        p.drawText(QRectF(cx - r_in, cy - 12, 2 * r_in, 24), Qt.AlignCenter, key_name(self.tonic, self.mode))


class ProgressionTree(QTreeWidget):
    """The progression library. Drag a row onto the piano roll, double-click to hear it."""

    def __init__(self, panel: "ChordsPanel"):
        super().__init__()
        self.panel = panel
        self.setHeaderLabels(["Progression", "Chords", "Mood"])
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QTreeWidget.DragOnly)
        self.setColumnWidth(0, 170)
        self.setColumnWidth(1, 260)

    def startDrag(self, actions) -> None:  # noqa: N802
        item = self.currentItem()
        chords = self.panel.progression_chords(item) if item else []
        if chords:
            start_chord_drag(self, chords)


class ChordsPanel(QWidget):
    """The whole Chords tab."""

    auditionRequested = Signal(object)    # list[Chord]
    insertRequested = Signal(object)      # list[Chord], insert at the playhead
    keyChanged = Signal(int, str)         # the song key changed: tonic, "major"/"minor"
    laneSourceChanged = Signal(str)       # "recording", "notes" or "off"
    chordSeek = Signal(float)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.current: Chord | None = None
        self.edit_mode = False
        self.sketch: list[list[Chord]] = [[] for _ in range(SKETCH_SLOTS)]
        self._song_key: tuple[int, str] | None = None
        self._busy = False
        self._lane: list = []

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 6, 6, 6)
        outer.setSpacing(10)
        outer.addWidget(self._build_left())
        self.tabs = QTabWidget()
        self.tabs.setObjectName("ChordTabs")
        self.tabs.setDocumentMode(True)
        self._build_right()
        outer.addWidget(self.tabs, 1)
        self._load_settings()
        self.rebuild()

    # Building ----------------------------------------------------------------------------------

    def _build_left(self) -> QWidget:
        col = QWidget()
        col.setFixedWidth(400)
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel("Key")
        label.setObjectName("FieldName")
        self.root_box = QComboBox()
        self.root_box.setProperty("i18n_skip_items", True)
        for pc in range(12):
            self.root_box.addItem(pc_name(pc, absolute=True), pc)
        self.scale_box = QComboBox()
        for name, _steps in theory.KEY_SCALES:
            self.scale_box.addItem(name, name)
        self.song_key_btn = QToolButton()
        self.song_key_btn.setText("Song key")
        self.song_key_btn.setToolTip("Use the key the analysis found")
        self.song_key_btn.clicked.connect(self._use_song_key)
        row.addWidget(label)
        row.addWidget(self.root_box)
        row.addWidget(self.scale_box, 1)
        row.addWidget(self.song_key_btn)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel("Chords")
        label.setObjectName("FieldName")
        self.kind_box = QComboBox()
        for key, text in theory.CHORD_KINDS:
            self.kind_box.addItem(text, key)
        row.addWidget(label)
        row.addWidget(self.kind_box, 1)
        lay.addLayout(row)

        self.palette_box = QWidget()
        self.palette_layout = QHBoxLayout(self.palette_box)
        self.palette_layout.setContentsMargins(0, 2, 0, 2)
        self.palette_layout.setSpacing(3)
        lay.addWidget(self.palette_box)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(5)
        self.voicing_box = QComboBox()
        for key, text in theory.VOICINGS:
            self.voicing_box.addItem(text, key)
        self.octave_spin = QSpinBox()
        self.octave_spin.setRange(2, 6)
        self.octave_spin.setPrefix(tr("Octave") + " ")
        self.octave_spin.setProperty("i18n_skip", True)
        self.octave_spin.setToolTip("Which octave the chord sits in (4 is around middle C)")
        self.lead_chk = QCheckBox("Voice leading")
        self.lead_chk.setToolTip("Each chord picks the inversion closest to the one before it, so the notes move as little as possible")
        self.bass_chk = QCheckBox("Bass note")
        self.bass_chk.setToolTip("Add the root an octave below")
        self.pattern_box = QComboBox()
        for key, text in theory.PATTERNS:
            self.pattern_box.addItem(text, key)
        self.pattern_box.setToolTip("How each chord is played: held, pulsed, as an arpeggio, strummed, or as a bass line")
        self.length_box = QComboBox()
        for text, beats in LENGTHS:
            self.length_box.addItem(text, beats)
        self.length_box.setToolTip("How long each chord lasts")
        self.sound_box = QComboBox()
        self.sound_box.addItem("Sound of the part", "")
        for inst in INSTRUMENTS:
            self.sound_box.addItem(inst.label, inst.key)
        self.sound_box.setToolTip("What chords sound like when you click them")
        labels = [QLabel(t) for t in ("Voicing", "Pattern", "Length", "Hear with")]
        for lab in labels:
            lab.setObjectName("Hint")
        grid.addWidget(labels[0], 0, 0)
        grid.addWidget(self.voicing_box, 0, 1)
        grid.addWidget(self.octave_spin, 0, 2)
        grid.addWidget(self.lead_chk, 1, 1)
        grid.addWidget(self.bass_chk, 1, 2)
        grid.addWidget(labels[1], 2, 0)
        grid.addWidget(self.pattern_box, 2, 1, 1, 2)
        grid.addWidget(labels[2], 3, 0)
        grid.addWidget(self.length_box, 3, 1)
        grid.addWidget(labels[3], 4, 0)
        grid.addWidget(self.sound_box, 4, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.picked_label = QLabel("")
        self.picked_label.setObjectName("Meta")
        self.picked_label.setProperty("i18n_skip", True)
        self.insert_btn = QPushButton("Insert at playhead")
        self.insert_btn.setToolTip("Add the picked chord where the playhead is (Edit mode)")
        self.insert_btn.clicked.connect(lambda: self.current and self.insertRequested.emit([self.current]))
        self.add_sketch_btn = QPushButton("Add to sketch")
        self.add_sketch_btn.setToolTip("Add the picked chord to the chosen sketch pad line")
        self.add_sketch_btn.clicked.connect(self._add_to_sketch)
        row.addWidget(self.picked_label, 1)
        row.addWidget(self.insert_btn)
        row.addWidget(self.add_sketch_btn)
        lay.addLayout(row)
        hint = QLabel("Click a chord to hear it. Turn on Edit notes, then drag chords onto the piano roll.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addStretch(1)

        for box in (self.root_box, self.scale_box):
            box.currentIndexChanged.connect(self._on_key_changed)
        self.kind_box.currentIndexChanged.connect(lambda _i: self.rebuild())
        for w in (self.voicing_box, self.pattern_box, self.length_box, self.sound_box):
            w.currentIndexChanged.connect(lambda _i: self._save_settings())
        for w in (self.lead_chk, self.bass_chk):
            w.toggled.connect(lambda _on: self._save_settings())
        self.octave_spin.valueChanged.connect(lambda _v: self._save_settings())
        return col

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _add_tab(self, page: QWidget, title: str, tip: str) -> None:
        index = self.tabs.addTab(page, title)
        self.tabs.setTabToolTip(index, tip)

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        return page, lay

    def _build_right(self) -> None:
        # Song chords: what was found in the song
        page, lay = self._page()
        row = QHBoxLayout()
        label = QLabel("Chord lane")
        label.setObjectName("Hint")
        self.lane_box = QComboBox()
        self.lane_box.addItem("From the recording", "recording")
        self.lane_box.addItem("From the notes", "notes")
        self.lane_box.addItem("Hidden", "off")
        self.lane_box.setToolTip("Where the chord names above the piano roll come from")
        self.lane_box.currentIndexChanged.connect(self._on_lane_source)
        row.addWidget(label)
        row.addWidget(self.lane_box)
        row.addStretch(1)
        lay.addLayout(row)
        self.song_tree = QTreeWidget()
        self.song_tree.setHeaderLabels(["Time", "Chord", "In the key"])
        self.song_tree.setRootIsDecorated(False)
        self.song_tree.setAlternatingRowColors(True)
        self.song_tree.setColumnWidth(0, 80)
        self.song_tree.setColumnWidth(1, 110)
        self.song_tree.itemClicked.connect(self._on_song_chord)
        lay.addWidget(self.song_tree, 1)
        self._add_tab(page, "Song", "The chords found in the song")

        # Progressions
        page, lay = self._page()
        row = QHBoxLayout()
        label = QLabel("Mood")
        label.setObjectName("Hint")
        self.mood_box = QComboBox()
        self.mood_box.addItem("Any mood", "")
        for mood in theory.MOODS:
            self.mood_box.addItem(mood, mood)
        self.mood_box.currentIndexChanged.connect(lambda _i: self._fill_progressions())
        play = QPushButton("Play")
        play.setToolTip("Hear the selected progression")
        play.clicked.connect(self._play_progression)
        ins = QPushButton("Insert at playhead")
        ins.setToolTip("Add the selected progression where the playhead is (Edit mode)")
        ins.clicked.connect(self._insert_progression)
        sk = QPushButton("To sketch")
        sk.setToolTip("Copy the selected progression to the chosen sketch pad line")
        sk.clicked.connect(self._progression_to_sketch)
        self.prog_insert_btn = ins
        row.addWidget(label)
        row.addWidget(self.mood_box)
        row.addStretch(1)
        row.addWidget(play)
        row.addWidget(ins)
        row.addWidget(sk)
        lay.addLayout(row)
        self.prog_tree = ProgressionTree(self)
        self.prog_tree.itemDoubleClicked.connect(lambda _item, _col: self._play_progression())
        lay.addWidget(self.prog_tree, 1)
        self._add_tab(page, "Progressions", "Ready-made chord progressions to play or drag in")

        # Next chord / Substitutes / Explore / Change key: rebuilt as things change
        self.next_page, self.next_lay = self._page()
        self._add_tab(self._scroll(self.next_page), "Next", "Chords that sound good after the picked one")
        self.subs_page, self.subs_lay = self._page()
        self._add_tab(self._scroll(self.subs_page), "Substitutes", "Chords that can stand in for the picked one")
        self.explore_page, self.explore_lay = self._page()
        self._add_tab(self._scroll(self.explore_page), "Outside key", "Chords from outside the key that still fit")

        page, lay = self._page()
        row = QHBoxLayout()
        label = QLabel("Change to")
        label.setObjectName("Hint")
        self.to_root = QComboBox()
        self.to_root.setProperty("i18n_skip_items", True)
        for pc in range(12):
            self.to_root.addItem(pc_name(pc, absolute=True), pc)
        self.to_mode = QComboBox()
        self.to_mode.addItem("major", "major")
        self.to_mode.addItem("minor", "minor")
        self.to_root.setCurrentIndex(7)
        for box in (self.to_root, self.to_mode):
            box.currentIndexChanged.connect(lambda _i: self._fill_modulations())
        row.addWidget(label)
        row.addWidget(self.to_root)
        row.addWidget(self.to_mode)
        row.addStretch(1)
        lay.addLayout(row)
        self.mod_box = QWidget()
        self.mod_lay = QVBoxLayout(self.mod_box)
        self.mod_lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.mod_box)
        lay.addStretch(1)
        self._add_tab(self._scroll(page), "Key change", "Ways to move into another key")

        page, lay = self._page()
        self.circle = CircleOfFifths()
        self.circle.keyPicked.connect(self.set_key)
        self.circle.chordPicked.connect(self.pick)
        circle_hint = QLabel("Keys next to each other share six of their seven notes, so moving one step "
                             "around the circle sounds natural. The lit keys are the chords of your key.")
        circle_hint.setObjectName("Hint")
        circle_hint.setWordWrap(True)
        lay.addWidget(self.circle, 1)
        lay.addWidget(circle_hint)
        self._add_tab(page, "Circle", "The circle of fifths: every key next to its closest neighbors")

        page, lay = self._page()
        hint = QLabel("Collect ideas here. Pick a line, then use Add to sketch, or To sketch in Progressions.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.sketch_group = QButtonGroup(self)
        self.sketch_rows: list[QHBoxLayout] = []
        self.sketch_box = QWidget()
        self.sketch_lay = QVBoxLayout(self.sketch_box)
        self.sketch_lay.setContentsMargins(0, 0, 0, 0)
        self.sketch_lay.setSpacing(2)
        lay.addWidget(self.sketch_box)
        lay.addStretch(1)
        self._add_tab(self._scroll(page), "Sketch", "Write down chord ideas, then play them or drag them in")

    # Settings ------------------------------------------------------------------------------

    def _load_settings(self) -> None:
        s = self.settings

        def pick(box, key, default):
            i = box.findData(s.value(key, default))
            box.setCurrentIndex(i if i >= 0 else max(0, box.findData(default)))

        self._busy = True
        pick(self.voicing_box, "chords/voicing", "close")
        pick(self.pattern_box, "chords/pattern", "block")
        pick(self.length_box, "chords/length", 4)
        pick(self.sound_box, "chords/sound", "")
        pick(self.kind_box, "chords/kind", "triads")
        pick(self.lane_box, "chords/lane", "recording")
        try:
            self.octave_spin.setValue(int(s.value("chords/octave", 4)))
        except (TypeError, ValueError):
            self.octave_spin.setValue(4)
        self.lead_chk.setChecked(str(s.value("chords/lead", "true")).lower() in ("true", "1"))
        self.bass_chk.setChecked(str(s.value("chords/bass", "false")).lower() in ("true", "1"))
        self._busy = False

    def _save_settings(self) -> None:
        if self._busy:
            return
        s = self.settings
        s.setValue("chords/voicing", self.voicing_box.currentData())
        s.setValue("chords/pattern", self.pattern_box.currentData())
        s.setValue("chords/length", self.length_box.currentData())
        s.setValue("chords/sound", self.sound_box.currentData())
        s.setValue("chords/kind", self.kind_box.currentData())
        s.setValue("chords/lane", self.lane_box.currentData())
        s.setValue("chords/octave", self.octave_spin.value())
        s.setValue("chords/lead", self.lead_chk.isChecked())
        s.setValue("chords/bass", self.bass_chk.isChecked())

    # Key -----------------------------------------------------------------------------------

    def key(self) -> tuple[int, str, str]:
        scale = self.scale_box.currentData()
        return self.root_box.currentData(), theory.mode_of(scale), scale

    def set_key(self, tonic: int, mode: str) -> None:
        self._busy = True
        self.root_box.setCurrentIndex(tonic % 12)
        self.scale_box.setCurrentIndex(self.scale_box.findData("Major" if mode == "major" else "Natural minor"))
        self._busy = False
        self._on_key_changed()

    def set_song_key(self, tonic: int | None, mode: str | None) -> None:
        """The key the analysis found. Becomes the panel's key."""
        self._song_key = (tonic, mode) if tonic is not None else None
        self.song_key_btn.setEnabled(self._song_key is not None)
        if self._song_key:
            self.set_key(*self._song_key)

    def _use_song_key(self) -> None:
        if self._song_key:
            self.set_key(*self._song_key)

    def _on_key_changed(self) -> None:
        if self._busy:
            return
        tonic, mode, _scale = self.key()
        self.keyChanged.emit(tonic, mode)
        self.rebuild()

    def _on_lane_source(self) -> None:
        self._save_settings()
        self.laneSourceChanged.emit(self.lane_box.currentData())

    def lane_source(self) -> str:
        return self.lane_box.currentData()

    # Rebuilding the views --------------------------------------------------------------------

    def rebuild(self) -> None:
        """Everything that depends on the key, the note names or the language."""
        tonic, mode, scale = self.key()
        for box in (self.root_box, self.to_root):
            for pc in range(12):
                box.setItemText(pc, pc_name(pc, absolute=True))
        self.octave_spin.setPrefix(tr("Octave") + " ")
        self.circle.set_key(tonic, mode)
        while self.palette_layout.count():
            item = self.palette_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.palette_buttons = []
        for chord in theory.diatonic(tonic, scale, self.kind_box.currentData()):
            b = ChordButton(chord, theory.numeral(chord, tonic))
            b.setMinimumHeight(44)
            b.picked.connect(self.pick)
            self.palette_layout.addWidget(b)
            self.palette_buttons.append(b)
        self._fill_progressions()
        self._fill_suggestions()
        self._fill_explore()
        self._fill_modulations()
        self._fill_sketch()
        self._fill_song_chords()
        self._update_picked()

    def _clear(self, lay) -> None:
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear(item.layout())

    def _row(self, chords, captions=None, title="", note="") -> ChordRow:
        row = ChordRow(chords, captions, title, note)
        row.picked.connect(self.pick)
        row.playRequested.connect(self.auditionRequested)
        return row

    def _fill_progressions(self) -> None:
        tonic, _mode, _scale = self.key()
        mood = self.mood_box.currentData()
        tree = self.prog_tree
        expanded = {tree.topLevelItem(i).data(0, Qt.UserRole) for i in range(tree.topLevelItemCount())
                    if tree.topLevelItem(i).isExpanded()}
        tree.clear()
        cats: dict[str, QTreeWidgetItem] = {}
        for cat, name, nums, prog_mood in theory.PROGRESSIONS:
            if mood and prog_mood != mood:
                continue
            if cat not in cats:
                parent = QTreeWidgetItem([tr(cat), "", ""])
                parent.setData(0, Qt.UserRole, cat)
                parent.setFlags(parent.flags() & ~Qt.ItemIsDragEnabled)
                tree.addTopLevelItem(parent)
                cats[cat] = parent
            chords = theory.parse_numerals(nums, tonic)
            item = QTreeWidgetItem([tr(name), "  ".join(c.name() for c in chords), tr(prog_mood)])
            item.setData(0, Qt.UserRole, nums)
            item.setToolTip(1, nums)
            cats[cat].addChild(item)
        for cat, item in cats.items():
            item.setExpanded(cat in expanded or bool(mood) or not expanded)

    def progression_chords(self, item: QTreeWidgetItem) -> list[Chord]:
        nums = item.data(0, Qt.UserRole) if item else None
        if not nums or item.childCount():
            return []
        return theory.parse_numerals(nums, self.key()[0])

    def _selected_progression(self) -> list[Chord]:
        return self.progression_chords(self.prog_tree.currentItem())

    def _play_progression(self) -> None:
        chords = self._selected_progression()
        if chords:
            self.auditionRequested.emit(chords)

    def _insert_progression(self) -> None:
        chords = self._selected_progression()
        if chords:
            self.insertRequested.emit(chords)

    def _progression_to_sketch(self) -> None:
        chords = self._selected_progression()
        if chords:
            self.sketch[self._sketch_slot()] = list(chords)
            self._fill_sketch()

    def _fill_suggestions(self) -> None:
        self._clear(self.next_lay)
        self._clear(self.subs_lay)
        tonic, mode, _scale = self.key()
        chord = self.current
        if chord is None:
            for lay in (self.next_lay, self.subs_lay):
                hint = QLabel(tr("Click a chord anywhere in this tab first."))
                hint.setObjectName("Hint")
                lay.addWidget(hint)
                lay.addStretch(1)
            return
        kind = self.kind_box.currentData()
        title = QLabel(tr("After {chord}, these sound natural (best first):").format(chord=chord.name()))
        title.setObjectName("FieldName")
        self.next_lay.addWidget(title)
        for c, reason in theory.suggest_next(chord, tonic, mode, kind):
            self.next_lay.addWidget(self._row([c], [theory.numeral(c, tonic)], note=reason))
        self.next_lay.addStretch(1)
        title = QLabel(tr("Chords that can stand in for {chord}:").format(chord=chord.name()))
        title.setObjectName("FieldName")
        self.subs_lay.addWidget(title)
        for c, reason in theory.substitutes(chord, tonic, mode):
            self.subs_lay.addWidget(self._row([c], [theory.numeral(c, tonic)], note=reason))
        self.subs_lay.addStretch(1)

    def _fill_explore(self) -> None:
        self._clear(self.explore_lay)
        tonic, mode, _scale = self.key()
        for title, items in theory.explore(tonic, mode):
            if items:
                self.explore_lay.addWidget(self._row([c for c, _ in items], [n for _, n in items], title=title))
        self.explore_lay.addStretch(1)

    def _fill_modulations(self) -> None:
        self._clear(self.mod_lay)
        tonic, mode, _scale = self.key()
        to_tonic, to_mode = self.to_root.currentData(), self.to_mode.currentData()
        if to_tonic is None:
            return
        if (to_tonic, to_mode) == (tonic, mode):
            hint = QLabel(tr("Pick a different key to move to."))
            hint.setObjectName("Hint")
            self.mod_lay.addWidget(hint)
            return
        for method, how, chords in theory.modulations(tonic, mode, to_tonic, to_mode):
            self.mod_lay.addWidget(self._row(chords, title=method, note=how))

    def _sketch_slot(self) -> int:
        checked = self.sketch_group.checkedId()
        return checked if checked >= 0 else 0

    def _fill_sketch(self) -> None:
        current = self._sketch_slot()
        self._clear(self.sketch_lay)
        for b in self.sketch_group.buttons():
            self.sketch_group.removeButton(b)
        for i in range(SKETCH_SLOTS):
            line = QWidget()
            row = QHBoxLayout(line)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            radio = QRadioButton(str(i + 1))
            radio.setProperty("i18n_skip", True)
            self.sketch_group.addButton(radio, i)
            radio.setChecked(i == current)
            row.addWidget(radio)
            chords = self.sketch[i]
            if chords:
                for c in chords[:12]:
                    b = ChordButton(c)
                    b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                    b.picked.connect(self.pick)
                    row.addWidget(b)
            else:
                empty = QLabel(tr("empty"))
                empty.setObjectName("Hint")
                row.addWidget(empty)
            row.addStretch(1)
            play = QToolButton()
            play.setText(tr("Play"))
            play.setEnabled(bool(chords))
            play.clicked.connect(lambda _c=False, k=i: self.sketch[k] and self.auditionRequested.emit(self.sketch[k]))
            undo = QToolButton()
            undo.setText(tr("Remove last"))
            undo.setEnabled(bool(chords))
            undo.clicked.connect(lambda _c=False, k=i: self._sketch_pop(k))
            clear = QToolButton()
            clear.setText(tr("Clear"))
            clear.setEnabled(bool(chords))
            clear.clicked.connect(lambda _c=False, k=i: self._sketch_clear(k))
            row.addWidget(play)
            row.addWidget(DragHandle(lambda k=i: self.sketch[k]))
            row.addWidget(undo)
            row.addWidget(clear)
            self.sketch_lay.addWidget(line)

    def _sketch_pop(self, k: int) -> None:
        if self.sketch[k]:
            self.sketch[k].pop()
            self._fill_sketch()

    def _sketch_clear(self, k: int) -> None:
        self.sketch[k] = []
        self._fill_sketch()

    def _add_to_sketch(self) -> None:
        if self.current:
            self.sketch[self._sketch_slot()].append(self.current)
            self._fill_sketch()
            self.tabs.setCurrentIndex(self.tabs.count() - 1)

    # The song's own chords ------------------------------------------------------------------

    def set_song_chords(self, lane: list) -> None:
        self._lane = list(lane or [])
        self._fill_song_chords()

    def _fill_song_chords(self) -> None:
        self.song_tree.clear()
        tonic, _mode, _scale = self.key()
        items = []
        for start, _end, chord in self._lane:
            if chord is None:
                continue
            item = QTreeWidgetItem([format_time(start, 1), chord.name(), theory.numeral(chord, tonic)])
            item.setData(0, Qt.UserRole, start)
            item.setData(1, Qt.UserRole, chord.to_dict())
            items.append(item)
        self.song_tree.addTopLevelItems(items)

    def _on_song_chord(self, item: QTreeWidgetItem) -> None:
        self.chordSeek.emit(float(item.data(0, Qt.UserRole)))
        self.pick(Chord.from_dict(item.data(1, Qt.UserRole)))

    # Picking and playing --------------------------------------------------------------------

    def pick(self, chord: Chord) -> None:
        """A chord was clicked somewhere: hear it, and make it the one the other views follow."""
        self.current = chord
        self.auditionRequested.emit([chord])
        self._fill_suggestions()
        self._update_picked()

    def _update_picked(self) -> None:
        tonic = self.key()[0]
        if self.current:
            self.picked_label.setText(tr("Picked: {chord} ({numeral})").format(
                chord=self.current.name(), numeral=theory.numeral(self.current, tonic)))
        else:
            self.picked_label.setText(tr("Nothing picked yet"))
        self.insert_btn.setEnabled(self.current is not None and self.edit_mode)
        self.prog_insert_btn.setEnabled(self.edit_mode)
        self.add_sketch_btn.setEnabled(self.current is not None)

    def set_edit_mode(self, on: bool) -> None:
        self.edit_mode = on
        self._update_picked()

    # Turning chords into notes ----------------------------------------------------------------

    def chord_beats(self) -> int:
        return int(self.length_box.currentData() or 4)

    def sound(self) -> str:
        return self.sound_box.currentData() or ""

    def render_notes(self, chords: list[Chord], start: float, beat: float,
                     velocity: float = 0.7) -> list[tuple[float, float, int, float]]:
        """Notes for a list of chords played one after another from `start`."""
        length = self.chord_beats() * beat
        voicing = self.voicing_box.currentData()
        octave = self.octave_spin.value()
        lead = self.lead_chk.isChecked()
        bass = self.bass_chk.isChecked()
        pattern = self.pattern_box.currentData()
        out = []
        prev = None
        for i, chord in enumerate(chords):
            pitches = theory.voice(chord, voicing, octave, prev=prev, lead=lead and prev is not None, bass=bass)
            nxt = chords[i + 1] if i + 1 < len(chords) else None
            out += theory.pattern_notes(chord, pitches, start + i * length, length, beat, pattern, velocity, nxt)
            prev = [p for p in pitches if not bass or p != min(pitches)] or pitches
        return out

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(900, 280)
