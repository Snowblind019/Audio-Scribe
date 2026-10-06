"""Smaller widgets used by the main window."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QComboBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QSpinBox, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .engine import LANGUAGE_NAMES
from .i18n import tr, tr_n
from .music import (SCALES, NoteFilter, estimate_key, format_time, match_scale, note_label, note_name, pc_name,
                    pitch_class_weights, scale_classes, used_notes)
from .synth import DRUM_KEY, INSTRUMENTS
from .theory import guess_chords

SORT_ROLE = Qt.UserRole + 1
DATA_ROLE = Qt.UserRole


class ElidedLabel(QLabel):
    """A label that shortens its text with "..." when there is no room, instead of
    making the whole window wider."""

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt name)
        return QSize(30, super().minimumSizeHint().height())

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, rect.width())
        self.style().drawItemText(painter, rect, int(self.alignment()) | Qt.TextSingleLine, self.palette(),
                                  self.isEnabled(), text, self.foregroundRole())


class SortItem(QTreeWidgetItem):
    """Sorts by a hidden value per column instead of by display text."""

    def __lt__(self, other) -> bool:
        tree = self.treeWidget()
        col = tree.sortColumn() if tree else 0
        a, b = self.data(col, SORT_ROLE), other.data(col, SORT_ROLE)
        if a is not None and b is not None:
            return a < b
        return self.text(col) < other.text(col)


def swatch_icon(color: str, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawRoundedRect(QRectF(1, 1, size - 2, size - 2), 3, 3)
    p.end()
    return QIcon(pm)


def sureness(confidence: str) -> str:
    return tr({"high": "fairly sure", "medium": "somewhat sure", "low": "a rough guess"}[confidence])


class PitchClassChart(QWidget):
    """Twelve bars, C to B, showing how much time each note gets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.weights = [0.0] * 12
        self.key = None
        self.setMinimumSize(300, 170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, weights, key) -> None:
        self.weights = list(weights)
        self.key = key
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        total = sum(self.weights)
        if total <= 0:
            p.setPen(QColor("#7F8996"))
            p.drawText(self.rect(), Qt.AlignCenter, tr("No notes yet"))
            return
        top, bottom = 22, 24
        slot = w / 12
        bar_w = slot * 0.62
        peak = max(self.weights)
        scale = self.key.scale if self.key else None
        fm = p.fontMetrics()
        for pc in range(12):
            share = self.weights[pc] / total
            bh = (h - top - bottom) * (self.weights[pc] / peak)
            x = pc * slot + (slot - bar_w) / 2
            y = h - bottom - bh
            if self.key and pc == self.key.tonic:
                color = QColor("#F0B23E")
            elif scale is None or pc in scale:
                color = QColor("#69A7E0")
            else:
                color = QColor("#4C5664")
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x, y, bar_w, max(bh, 1.5)), 3, 3)
            p.setPen(QColor("#DCE1E8"))
            p.drawText(QRectF(pc * slot, h - bottom + 4, slot, bottom - 4),
                       Qt.AlignHCenter | Qt.AlignTop, pc_name(pc))
            if share >= 0.005:
                p.setPen(QColor("#8D97A5"))
                label = f"{share * 100:.0f}%"
                p.drawText(QRectF(pc * slot, y - fm.height() - 2, slot, fm.height()),
                           Qt.AlignHCenter | Qt.AlignBottom, label)


class SummaryView(QWidget):
    pitchSelected = Signal(object)  # MIDI pitch or None

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 12, 12, 12)
        lay.setSpacing(24)

        facts = QWidget()
        form = QFormLayout(facts)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft)
        self.fields = {}
        for key, label in (("key", "Key"), ("tempo", "Tempo"), ("range", "Range"),
                           ("notes", "Notes found"), ("words", "Words")):
            value = QLabel("Not analyzed yet")
            value.setTextFormat(Qt.PlainText)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            name = QLabel(label)
            name.setObjectName("FieldName")
            form.addRow(name, value)
            self.fields[key] = value
        lay.addWidget(facts, 3)

        middle = QVBoxLayout()
        middle.setSpacing(4)
        title = QLabel("Time spent on each note")
        title.setObjectName("FieldName")
        self.chart_hint = QLabel("")
        self.chart_hint.setObjectName("Hint")
        self.chart = PitchClassChart()
        middle.addWidget(title)
        middle.addWidget(self.chart_hint)
        middle.addWidget(self.chart, 1)
        lay.addLayout(middle, 4)

        right = QVBoxLayout()
        right.setSpacing(4)
        title = QLabel("Every note used")
        title.setObjectName("FieldName")
        hint = QLabel("Click one to highlight it in the piano roll.")
        hint.setObjectName("Hint")
        self.used = QTreeWidget()
        self.used.setHeaderLabels(["Note", "Times", "Total time"])
        self.used.setRootIsDecorated(False)
        self.used.setAlternatingRowColors(True)
        self.used.setSortingEnabled(True)
        self.used.itemSelectionChanged.connect(self._on_select)
        self.used.setColumnWidth(0, 70)
        self.used.setColumnWidth(1, 92)
        right.addWidget(title)
        right.addWidget(hint)
        right.addWidget(self.used, 1)
        lay.addLayout(right, 3)

    def clear(self) -> None:
        for value in self.fields.values():
            value.setText(tr("Not analyzed yet"))
        self.chart.set_data([0.0] * 12, None)
        self.chart_hint.setText("")
        self.used.clear()

    def set_result(self, r) -> None:
        pitched = [n for t in r.tracks if t.name != "Drums" for n in t.notes]
        all_notes = [n for t in r.tracks for n in t.notes]

        if r.key:
            text = f"{r.key.name} ({sureness(r.key.confidence)})"
            if r.key.alternative:
                text += tr(", could also be {key}").format(key=r.key.alternative_name)
            self.fields["key"].setText(text)
        elif r.notes_requested:
            self.fields["key"].setText(tr("Not enough notes to tell"))
        else:
            self.fields["key"].setText(tr("Turn on Find notes to estimate the key"))

        if r.tempo:
            text = tr("Around {bpm} BPM").format(bpm=f"{r.tempo:.0f}")
            if r.meter and r.meter != 4:
                text += tr(", {n} beats per bar").format(n=r.meter)
            self.fields["tempo"].setText(text)
        else:
            self.fields["tempo"].setText(tr("No steady beat found"))

        if pitched:
            lo, hi = min(n.pitch for n in pitched), max(n.pitch for n in pitched)
            self.fields["range"].setText(tr("{low} to {high}").format(low=note_name(lo), high=note_name(hi)))
        else:
            self.fields["range"].setText(tr("No notes"))

        if r.tracks:
            parts = ", ".join(f"{tr(t.name)} {len(t.notes)}" for t in r.tracks)
            total = len(all_notes)
            self.fields["notes"].setText(f"{total} ({parts})" if len(r.tracks) > 1 else str(total))
        else:
            self.fields["notes"].setText(tr("Note finding was off"))

        if r.words_requested:
            words = sum(len(s.words) or len(s.text.split()) for s in r.segments)
            lang = tr(LANGUAGE_NAMES.get(r.language or "", (r.language or "unknown").upper()))
            sure = tr(", {n}% sure").format(n=f"{r.language_probability * 100:.0f}") if r.language_probability else ""
            self.fields["words"].setText(
                tr("{words} words in {lines} lines. Language: {lang}").format(words=words, lines=len(r.segments),
                                                                              lang=lang) + sure)
        else:
            self.fields["words"].setText(tr("Transcription was off"))

        self.chart.set_data(pitch_class_weights(pitched), r.key)
        self.chart_hint.setText(tr("Brighter bars are notes in the estimated key.") if r.key else "")

        self.used.setSortingEnabled(False)
        self.used.clear()
        items = []
        for pitch, (count, seconds) in used_notes(pitched).items():
            item = SortItem([note_name(pitch), str(count), f"{seconds:.1f} s"])
            item.setData(0, SORT_ROLE, pitch)
            item.setData(1, SORT_ROLE, count)
            item.setData(2, SORT_ROLE, seconds)
            item.setData(0, DATA_ROLE, pitch)
            items.append(item)
        self.used.addTopLevelItems(items)
        self.used.setSortingEnabled(True)
        self.used.sortItems(0, Qt.DescendingOrder)

    def select_pitch(self, pitch) -> None:
        self.used.blockSignals(True)
        self.used.clearSelection()
        if pitch is not None:
            for i in range(self.used.topLevelItemCount()):
                item = self.used.topLevelItem(i)
                if item.data(0, DATA_ROLE) == pitch:
                    item.setSelected(True)
                    self.used.scrollToItem(item)
                    break
        self.used.blockSignals(False)

    def _on_select(self) -> None:
        items = self.used.selectedItems()
        self.pitchSelected.emit(items[0].data(0, DATA_ROLE) if items else None)



class PartRow(QFrame):
    """One part of the music in the Parts list: name, Mute, Solo, and its instrument."""

    changed = Signal()
    instrumentPicked = Signal(str)

    def __init__(self, track, parent=None):
        super().__init__(parent)
        self.track = track
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(6)
        swatch = QLabel()
        swatch.setPixmap(swatch_icon(track.color, 12).pixmap(12, 12))
        self.name = QLabel()
        self.name.setTextFormat(Qt.PlainText)
        self.mute = QToolButton()
        self.mute.setObjectName("MuteButton")
        self.mute.setText("M")
        self.mute.setCheckable(True)
        self.mute.setChecked(track.muted)
        self.mute.setToolTip("Mute: hides this part's notes and silences it")
        self.solo = QToolButton()
        self.solo.setObjectName("SoloButton")
        self.solo.setText("S")
        self.solo.setCheckable(True)
        self.solo.setChecked(track.solo)
        self.solo.setToolTip("Solo: show and hear only the soloed parts")
        top.addWidget(swatch)
        top.addWidget(self.name, 1)
        top.addWidget(self.mute)
        top.addWidget(self.solo)
        lay.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel("Sound")
        label.setObjectName("Hint")
        self.combo = QComboBox()
        for inst in INSTRUMENTS:
            # Drum hits only make sense as drum sounds. Every other part can be played on anything.
            if track.name != "Drums" or inst.key == DRUM_KEY:
                self.combo.addItem(inst.label, inst.key)
        index = self.combo.findData(track.instrument)
        self.combo.setCurrentIndex(max(0, index))
        self.combo.setToolTip("The instrument used when the notes of this part are played back.")
        row.addWidget(label)
        row.addWidget(self.combo, 1)
        lay.addLayout(row)
        self.refresh_name()

        self.mute.toggled.connect(self._on_mute)
        self.solo.toggled.connect(self._on_solo)
        self.combo.activated.connect(self._on_instrument)

    def refresh_name(self) -> None:
        count = len(self.track.notes)
        if not count:
            what = tr("no notes")
        elif self.track.name == "Drums":
            what = tr_n(count, "{n} hit", "{n} hits")
        else:
            what = tr_n(count, "{n} note", "{n} notes")
        self.name.setText(f"{tr(self.track.name)}  ({what})")

    def _on_mute(self, on: bool) -> None:
        self.track.muted = on
        self.changed.emit()

    def _on_solo(self, on: bool) -> None:
        self.track.solo = on
        self.changed.emit()

    def _on_instrument(self, index: int) -> None:
        key = self.combo.itemData(index)
        self.track.instrument = key
        self.instrumentPicked.emit(key)
        self.changed.emit()


class NoteSpin(QSpinBox):
    """A spin box that shows note names (C4) instead of numbers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(12, 120)

    def textFromValue(self, value: int) -> str:  # noqa: N802
        return note_name(value)

    def valueFromText(self, text: str) -> int:  # noqa: N802
        text = text.strip().upper().replace("♯", "#").replace("♭", "B")
        for pitch in range(self.minimum(), self.maximum() + 1):
            if note_name(pitch).upper() == text:
                return pitch
        try:
            return int(text)
        except ValueError:
            return self.value()


class ScaleFilterBox(QWidget):
    """Choose which notes to show: a scale (or any set of notes), and a range of notes."""

    changed = Signal(object)  # NoteFilter

    CUSTOM = "Custom"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(7)
        self._busy = False
        self._detected = None

        row = QHBoxLayout()
        row.setSpacing(6)
        self.root = QComboBox()
        self.root.setProperty("i18n_skip_items", True)   # note names, set by refresh_names()
        for pc in range(12):
            self.root.addItem(pc_name(pc, absolute=True), pc)
        self.root.setMaximumWidth(78)
        self.kind = QComboBox()
        for name, _ in SCALES:
            self.kind.addItem(name, name)
        self.kind.addItem(self.CUSTOM, self.CUSTOM)
        self.kind.setToolTip("Show only the notes of a scale or chord. Or click the keys below to pick your own.")
        row.addWidget(self.root)
        row.addWidget(self.kind, 1)
        lay.addLayout(row)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)
        self.keys: list[QToolButton] = []
        white = {0: 0, 2: 2, 4: 4, 5: 6, 7: 8, 9: 10, 11: 12}
        black = {1: 1, 3: 3, 6: 7, 8: 9, 10: 11}
        buttons = {}
        for pc in range(12):
            b = QToolButton()
            b.setObjectName("KeyButton")
            b.setProperty("i18n_skip", True)
            b.setCheckable(True)
            b.setChecked(True)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.toggled.connect(self._on_key)
            buttons[pc] = b
        for pc, col in black.items():
            grid.addWidget(buttons[pc], 0, col, 1, 2)
        for pc, col in white.items():
            grid.addWidget(buttons[pc], 1, col, 1, 2)
        self.keys = [buttons[pc] for pc in range(12)]
        lay.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel("Show")
        label.setObjectName("Hint")
        self.mode = QComboBox()
        self.mode.addItem("Notes in the scale", False)
        self.mode.addItem("Notes outside the scale", True)
        self.mode.setToolTip("Outside shows only the notes that are NOT in the scale, "
                             "which is a quick way to find the odd ones out.")
        row.addWidget(label)
        row.addWidget(self.mode, 1)
        lay.addLayout(row)

        self.key_btn = QPushButton("Use the detected key")
        self.key_btn.setProperty("i18n_skip", True)      # its text includes the key, set in code
        self.key_btn.setEnabled(False)
        lay.addWidget(self.key_btn)

        row = QHBoxLayout()
        row.setSpacing(6)
        low_label, high_label = QLabel("From"), QLabel("to")
        low_label.setObjectName("Hint")
        high_label.setObjectName("Hint")
        self.low = NoteSpin()
        self.low.setValue(12)
        self.low.setToolTip("Lowest note to show")
        self.high = NoteSpin()
        self.high.setValue(120)
        self.high.setToolTip("Highest note to show")
        row.addWidget(low_label)
        row.addWidget(self.low, 1)
        row.addWidget(high_label)
        row.addWidget(self.high, 1)
        lay.addLayout(row)

        self.reset_btn = QPushButton("Show every note")
        lay.addWidget(self.reset_btn)

        self.root.currentIndexChanged.connect(self._on_scale)
        self.kind.currentIndexChanged.connect(self._on_scale)
        self.mode.currentIndexChanged.connect(self._emit)
        self.low.valueChanged.connect(self._emit)
        self.high.valueChanged.connect(self._emit)
        self.key_btn.clicked.connect(self._use_detected)
        self.reset_btn.clicked.connect(self.reset)
        self.refresh_names()
        self._sync_enabled()

    def refresh_names(self) -> None:
        """Note names follow the chosen style (C D E or Do Re Mi) and language."""
        for pc in range(12):
            self.root.setItemText(pc, pc_name(pc, absolute=True))
            self.keys[pc].setText(pc_name(pc))
        # the spin boxes show note_name() text
        for spin in (self.low, self.high):
            spin.lineEdit().setText(spin.textFromValue(spin.value()))
        self.set_detected_key(self._detected)

    # Reading and writing the state ---------------------------------------------------

    def _classes(self) -> frozenset[int] | None:
        picked = frozenset(i for i, b in enumerate(self.keys) if b.isChecked())
        return None if len(picked) == 12 else picked

    def current(self) -> NoteFilter:
        classes = self._classes()
        low, high = self.low.value(), self.high.value()
        kind = self.kind.currentData()
        named = classes is not None and kind not in (self.CUSTOM, SCALES[0][0])
        return NoteFilter(classes, bool(self.mode.currentData()), 0 if low <= 12 else low, 127 if high >= 120 else high,
                          self.root.currentIndex() if named else None, kind if named else "")

    def _set_keys(self, classes: frozenset[int] | None) -> None:
        for pc, b in enumerate(self.keys):
            b.blockSignals(True)
            b.setChecked(classes is None or pc in classes)
            b.blockSignals(False)

    def _sync_enabled(self) -> None:
        scaled = self._classes() is not None
        self.mode.setEnabled(scaled)
        self.root.setEnabled(self.kind.currentData() not in (SCALES[0][0], self.CUSTOM))

    def _emit(self) -> None:
        if not self._busy:
            self._sync_enabled()
            self.changed.emit(self.current())

    def _on_scale(self) -> None:
        if self._busy:
            return
        name = self.kind.currentData()
        if name == self.CUSTOM:
            self._sync_enabled()
            self.changed.emit(self.current())
            return
        steps = dict(SCALES)[name]
        self._set_keys(scale_classes(self.root.currentIndex(), steps))
        self._emit()

    def _on_key(self) -> None:
        if self._busy:
            return
        self._busy = True
        found = match_scale(self._classes())
        if found is None:
            self.kind.setCurrentIndex(self.kind.count() - 1)
        else:
            root, index = found
            self.kind.setCurrentIndex(index)
            if index:
                self.root.setCurrentIndex(root)
        self._busy = False
        self._emit()

    def set_detected_key(self, key) -> None:
        """key: a KeyEstimate or None."""
        self._detected = key
        self.key_btn.setEnabled(key is not None)
        self.key_btn.setText(tr("Use the detected key ({key})").format(key=key.name) if key
                             else tr("Use the detected key"))

    def use_key(self, tonic: int, mode: str) -> None:
        names = [n for n, _ in SCALES]
        self._busy = True
        self.root.setCurrentIndex(tonic)
        self.kind.setCurrentIndex(names.index("Major" if mode == "major" else "Natural minor"))
        self._set_keys(scale_classes(tonic, dict(SCALES)[self.kind.currentData()]))
        self._busy = False
        self._emit()

    def _use_detected(self) -> None:
        if self._detected:
            self.use_key(self._detected.tonic, self._detected.mode)

    def reset(self) -> None:
        self._busy = True
        self.kind.setCurrentIndex(0)
        self.mode.setCurrentIndex(0)
        self._set_keys(None)
        self.low.setValue(12)
        self.high.setValue(120)
        self._busy = False
        self._emit()


def _clipped_seconds(note, a: float, b: float) -> float:
    return max(0.0, min(note.end, b) - max(note.start, a))


class SelectionView(QWidget):
    """Details about the span of time that was selected in the piano roll."""

    pitchSelected = Signal(object)  # MIDI pitch or None
    seekRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.empty = QLabel("Drag across the piano roll to select a span of time. "
                            "Details about it show up here.\n\n"
                            "Then use Zoom in the bar above the piano roll to see it up close, "
                            "or Loop to play it over and over.")
        self.empty.setObjectName("Hint")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        outer.addWidget(self.empty)

        self.body = QWidget()
        lay = QHBoxLayout(self.body)
        lay.setContentsMargins(16, 12, 12, 12)
        lay.setSpacing(22)
        outer.addWidget(self.body)

        facts = QWidget()
        form = QFormLayout(facts)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft)
        self.fields = {}
        for key, label in (("span", "Span"), ("length", "Length"), ("key", "Key here"), ("range", "Range"),
                           ("notes", "Notes"), ("busiest", "Most played"), ("words", "Words")):
            value = QLabel("")
            value.setTextFormat(Qt.PlainText)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            name = QLabel(label)
            name.setObjectName("FieldName")
            name.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            form.addRow(name, value)
            self.fields[key] = value
        lay.addWidget(facts, 4)

        middle = QVBoxLayout()
        middle.setSpacing(4)
        title = QLabel("Time spent on each note")
        title.setObjectName("FieldName")
        self.chart = PitchClassChart()
        self.chart.setMinimumSize(240, 150)
        middle.addWidget(title)
        middle.addWidget(self.chart, 1)
        lay.addLayout(middle, 4)

        notes_col = QVBoxLayout()
        notes_col.setSpacing(4)
        title = QLabel("Every note used here")
        title.setObjectName("FieldName")
        self.used = QTreeWidget()
        self.used.setHeaderLabels(["Note", "Times", "Total time"])
        self.used.setRootIsDecorated(False)
        self.used.setAlternatingRowColors(True)
        self.used.setSortingEnabled(True)
        self.used.setColumnWidth(0, 60)
        self.used.setColumnWidth(1, 92)
        self.used.itemSelectionChanged.connect(self._on_note_select)
        notes_col.addWidget(title)
        notes_col.addWidget(self.used, 1)
        lay.addLayout(notes_col, 3)

        chords_col = QVBoxLayout()
        chords_col.setSpacing(4)
        title = QLabel("Chords (rough guess)")
        title.setObjectName("FieldName")
        self.chords = QTreeWidget()
        self.chords.setHeaderLabels(["Time", "Chord"])
        self.chords.setRootIsDecorated(False)
        self.chords.setAlternatingRowColors(True)
        self.chords.setColumnWidth(0, 80)
        self.chords.itemClicked.connect(self._on_chord_clicked)
        chords_col.addWidget(title)
        chords_col.addWidget(self.chords, 1)
        lay.addLayout(chords_col, 3)
        self.show_none()

    def show_none(self) -> None:
        self.empty.show()
        self.body.hide()

    def _on_note_select(self) -> None:
        items = self.used.selectedItems()
        self.pitchSelected.emit(items[0].data(0, DATA_ROLE) if items else None)

    def _on_chord_clicked(self, item) -> None:
        self.seekRequested.emit(item.data(0, DATA_ROLE))

    def select_pitch(self, pitch) -> None:
        self.used.blockSignals(True)
        self.used.clearSelection()
        if pitch is not None:
            for i in range(self.used.topLevelItemCount()):
                item = self.used.topLevelItem(i)
                if item.data(0, DATA_ROLE) == pitch:
                    item.setSelected(True)
                    self.used.scrollToItem(item)
                    break
        self.used.blockSignals(False)

    def show_span(self, a: float, b: float, tracks, flt: NoteFilter, beats, segments, meter: int = 4,
                  lane=None) -> None:
        """Fills in the details for [a, b]. `tracks` are the parts, only visible ones count."""
        self.empty.hide()
        self.body.show()
        length = b - a
        period = (beats[-1] - beats[0]) / (len(beats) - 1) if len(beats) > 1 else None

        shown = [(t, [n for n in t.notes if flt.allows(n.pitch, t.name == "Drums") and n.end > a and n.start < b])
                 for t in tracks if t.visible]
        pitched = [n for t, notes in shown if t.name != "Drums" for n in notes]
        drums = [n for t, notes in shown if t.name == "Drums" for n in notes]

        self.fields["span"].setText(tr("{low} to {high}").format(low=format_time(a, 2), high=format_time(b, 2)))
        text = f"{length:.2f} s"
        if period:
            beat_count = length / period
            text += tr(", about {beats} beats ({bars} bars)").format(beats=f"{beat_count:.1f}",
                                                                     bars=f"{beat_count / max(1, meter):.1f}")
        self.fields["length"].setText(text)

        weights = [0.0] * 12
        for n in pitched:
            weights[n.pitch % 12] += _clipped_seconds(n, a, b) * (0.5 + n.velocity)
        key = estimate_key(weights) if pitched else None
        if key:
            self.fields["key"].setText(f"{key.name} ({sureness(key.confidence)})")
        else:
            self.fields["key"].setText(tr("Not enough notes to tell"))
        self.chart.set_data(weights, key)

        if pitched:
            lo, hi = min(n.pitch for n in pitched), max(n.pitch for n in pitched)
            self.fields["range"].setText(tr("{low} to {high}").format(low=note_name(lo), high=note_name(hi)))
        else:
            self.fields["range"].setText(tr("No notes"))

        parts = [f"{tr(t.name)} {len(notes)}" for t, notes in shown if notes]
        total = sum(len(notes) for _, notes in shown)
        self.fields["notes"].setText(f"{total}" + (f" ({', '.join(parts)})" if len(parts) > 1 else "")
                                     if total else tr("None"))

        clipped: dict[int, list] = {}  # pitch -> [times played, seconds inside the span]
        for n in pitched:
            e = clipped.setdefault(n.pitch, [0, 0.0])
            e[0] += 1
            e[1] += _clipped_seconds(n, a, b)
        busiest = max(clipped.items(), key=lambda kv: kv[1][1], default=None)
        if drums:
            kinds: dict[str, int] = {}
            for n in drums:
                name = note_label(n.pitch, "Drums")
                kinds[name] = kinds.get(name, 0) + 1
            drum_text = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
            self.fields["busiest"].setText((f"{note_name(busiest[0])}. " if busiest else "")
                                           + tr("Drums: {list}").format(list=drum_text))
        else:
            self.fields["busiest"].setText(
                f"{note_name(busiest[0])}, " + tr_n(busiest[1][0], "{n} time", "{n} times") if busiest else "-")

        words = [w.text for s in segments for w in (s.words or []) if a <= w.start < b]
        if not words:
            words = [s.text for s in segments if s.text and s.start < b and s.end > a]
        text = " ".join(words)
        self.fields["words"].setText((text[:300] + "...") if len(text) > 300 else (text or tr("None")))

        self.used.setSortingEnabled(False)
        self.used.clear()
        items = []
        for pitch, (count, seconds) in clipped.items():
            item = SortItem([note_name(pitch), str(count), f"{seconds:.2f} s"])
            item.setData(0, SORT_ROLE, pitch)
            item.setData(1, SORT_ROLE, count)
            item.setData(2, SORT_ROLE, seconds)
            item.setData(0, DATA_ROLE, pitch)
            items.append(item)
        self.used.addTopLevelItems(items)
        self.used.setSortingEnabled(True)
        self.used.sortItems(0, Qt.DescendingOrder)

        self.chords.clear()
        step = period * 2 if period else 1.5
        rows = []
        if lane:
            found = [(max(s, a), e, c) for s, e, c in lane if e > a and s < b]
        else:
            found = guess_chords(pitched, a, b, step)
        for start, _end, chord in found:
            item = QTreeWidgetItem([format_time(start, 1), chord.name() if chord else "?"])
            item.setData(0, DATA_ROLE, start)
            rows.append(item)
        self.chords.addTopLevelItems(rows)
