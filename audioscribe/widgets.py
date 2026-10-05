"""Smaller widgets used by the main window."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QLabel, QSizePolicy, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .engine import LANGUAGE_NAMES
from .music import NOTE_NAMES, note_name, pitch_class_weights, used_notes

SORT_ROLE = Qt.UserRole + 1
DATA_ROLE = Qt.UserRole


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
            p.drawText(self.rect(), Qt.AlignCenter, "No notes yet")
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
                       Qt.AlignHCenter | Qt.AlignTop, NOTE_NAMES[pc])
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
        self.used.setColumnWidth(1, 60)
        right.addWidget(title)
        right.addWidget(hint)
        right.addWidget(self.used, 1)
        lay.addLayout(right, 3)

    def clear(self) -> None:
        for value in self.fields.values():
            value.setText("Not analyzed yet")
        self.chart.set_data([0.0] * 12, None)
        self.chart_hint.setText("")
        self.used.clear()

    def set_result(self, r) -> None:
        pitched = [n for t in r.tracks if t.name != "Drums" for n in t.notes]
        all_notes = [n for t in r.tracks for n in t.notes]

        if r.key:
            sure = {"high": "fairly sure", "medium": "somewhat sure", "low": "a rough guess"}[r.key.confidence]
            text = f"{r.key.name} ({sure})"
            if r.key.alternative:
                text += f", could also be {r.key.alternative}"
            self.fields["key"].setText(text)
        elif r.notes_requested:
            self.fields["key"].setText("Not enough notes to tell")
        else:
            self.fields["key"].setText("Turn on Find notes to estimate the key")

        self.fields["tempo"].setText(f"Around {r.tempo:.0f} BPM" if r.tempo else "No steady beat found")

        if pitched:
            lo, hi = min(n.pitch for n in pitched), max(n.pitch for n in pitched)
            self.fields["range"].setText(f"{note_name(lo)} to {note_name(hi)}")
        else:
            self.fields["range"].setText("No notes")

        if r.tracks:
            parts = ", ".join(f"{t.name} {len(t.notes)}" for t in r.tracks)
            total = len(all_notes)
            self.fields["notes"].setText(f"{total} ({parts})" if len(r.tracks) > 1 else str(total))
        else:
            self.fields["notes"].setText("Note finding was off")

        if r.words_requested:
            words = sum(len(s.words) or len(s.text.split()) for s in r.segments)
            lang = LANGUAGE_NAMES.get(r.language or "", (r.language or "unknown").upper())
            sure = f", {r.language_probability * 100:.0f}% sure" if r.language_probability else ""
            self.fields["words"].setText(f"{words} words in {len(r.segments)} lines. Language: {lang}{sure}")
        else:
            self.fields["words"].setText("Transcription was off")

        self.chart.set_data(pitch_class_weights(pitched), r.key)
        self.chart_hint.setText("Brighter bars are notes in the estimated key." if r.key else "")

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

