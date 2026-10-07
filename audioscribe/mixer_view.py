"""The Mixer tab: one channel strip per part (fader, pan, mute, solo, level meter) and the
master. The small controls here (volume slider, pan slider, meter) are also used in the
Parts list and in the Tracks view, so a part's volume can be changed wherever you are.

Every control changes the Track directly and then emits a signal; the window passes the new
values to the live mixer, which applies them to the sound that is playing, and tells the
other views to show them.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QSlider, QToolButton,
                               QVBoxLayout, QWidget)

from .i18n import tr

MIN_DB, MAX_DB = -60.0, 6.0
METER_FLOOR = -48.0


# Fader law ------------------------------------------------------------------------------------

def db_to_pos(db: float) -> int:
    """Slider position (0..1000) for a volume. 0 dB sits at about 70% of the travel, like a
    mixing desk, so the useful range around it gets most of the slider."""
    if db <= MIN_DB:
        return 0
    gain = 10 ** (db / 20.0)
    return int(round(1000 * math.sqrt(gain / 2.0)))


def pos_to_db(pos: int) -> float:
    if pos <= 0:
        return MIN_DB
    x = pos / 1000.0
    return max(MIN_DB, min(MAX_DB, 20 * math.log10(2 * x * x)))


def db_text(db: float) -> str:
    if db <= MIN_DB:
        return "-∞ dB"
    return f"{db:+.1f} dB"


def pan_text(pan: float) -> str:
    v = int(round(pan * 100))
    if v == 0:
        return tr("Centre")
    return (tr("L {n}") if v < 0 else tr("R {n}")).format(n=abs(v))


class VolumeSlider(QSlider):
    """A fader in dB. Double-click puts it back to 0 dB."""

    dbChanged = Signal(float)

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setRange(0, 1000)
        self.setPageStep(50)
        self.setValue(db_to_pos(0.0))
        self.valueChanged.connect(lambda v: self.dbChanged.emit(pos_to_db(v)))
        self.setToolTip(tr("Volume. Double-click for 0 dB."))

    def db(self) -> float:
        return pos_to_db(self.value())

    def set_db(self, db: float) -> None:
        if self.isSliderDown():      # being dragged: it already shows the value
            return
        self.blockSignals(True)
        self.setValue(db_to_pos(db))
        self.blockSignals(False)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.setValue(db_to_pos(0.0))


class PanSlider(QSlider):
    """Left to right. Double-click centres it."""

    panChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setRange(-100, 100)
        self.setPageStep(10)
        self.valueChanged.connect(lambda v: self.panChanged.emit(v / 100.0))
        self.setToolTip(tr("Pan: left or right. Double-click to centre."))

    def set_pan(self, pan: float) -> None:
        if self.isSliderDown():
            return
        self.blockSignals(True)
        self.setValue(int(round(pan * 100)))
        self.blockSignals(False)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.setValue(0)


class Meter(QWidget):
    """A level meter in dB with a peak marker that falls back slowly."""

    def __init__(self, vertical: bool = True, parent=None):
        super().__init__(parent)
        self.vertical = vertical
        self.level = 0.0       # 0..1 position of the bar
        self.hold = 0.0
        self.hold_age = 0
        if vertical:
            self.setFixedWidth(10)
            self.setMinimumHeight(60)
        else:
            self.setFixedHeight(6)
            self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Fixed if vertical else QSizePolicy.Expanding,
                           QSizePolicy.Expanding if vertical else QSizePolicy.Fixed)

    @staticmethod
    def _pos(peak: float) -> float:
        if peak <= 1e-6:
            return 0.0
        db = 20 * math.log10(peak)
        return min(1.0, max(0.0, (db - METER_FLOOR) / (3.0 - METER_FLOOR)))

    def push(self, peak: float) -> None:
        target = self._pos(peak)
        # rises at once, falls smoothly (about 20 dB a second at 30 updates a second)
        self.level = target if target > self.level else max(target, self.level - 0.013)
        if target >= self.hold:
            self.hold, self.hold_age = target, 0
        else:
            self.hold_age += 1
            if self.hold_age > 30:
                self.hold = max(self.level, self.hold - 0.01)
        self.update()

    def reset(self) -> None:
        self.level = self.hold = 0.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        r = self.rect()
        p.fillRect(r, QColor("#15181D"))
        length = r.height() if self.vertical else r.width()
        filled = int(length * self.level)
        if self.vertical:
            grad = QLinearGradient(0, r.height(), 0, 0)
        else:
            grad = QLinearGradient(0, 0, r.width(), 0)
        grad.setColorAt(0.0, QColor("#3FA34D"))
        grad.setColorAt(0.75, QColor("#B5C93A"))
        grad.setColorAt(0.9, QColor("#E3A33B"))
        grad.setColorAt(1.0, QColor("#E0533B"))
        if filled > 0:
            if self.vertical:
                p.fillRect(QRectF(0, r.height() - filled, r.width(), filled), grad)
            else:
                p.fillRect(QRectF(0, 0, filled, r.height()), grad)
        if self.hold > 0.01:
            h = int(length * self.hold)
            p.setPen(QPen(QColor("#E8ECF1"), 1))
            if self.vertical:
                p.drawLine(0, r.height() - h, r.width(), r.height() - h)
            else:
                p.drawLine(h, 0, h, r.height())


def _toggle(text: str, name: str, tip: str) -> QToolButton:
    b = QToolButton()
    b.setObjectName(name)
    b.setText(text)
    b.setCheckable(True)
    b.setToolTip(tip)
    return b


class ChannelStrip(QFrame):
    """One part in the Mixer: name, meter and fader, pan, mute and solo."""

    mixChanged = Signal(object)        # the track: volume or pan changed
    partsChanged = Signal(object)      # the track: mute or solo changed

    def __init__(self, track, parent=None):
        super().__init__(parent)
        self.track = track
        self.setObjectName("ChannelStrip")
        self.setFixedWidth(92)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        bar = QFrame()
        bar.setFixedHeight(4)
        bar.setStyleSheet(f"background: {track.color}; border-radius: 2px;")
        self.name = QLabel()
        self.name.setTextFormat(Qt.PlainText)
        self.name.setAlignment(Qt.AlignCenter)
        self.name.setObjectName("StripName")
        lay.addWidget(bar)
        lay.addWidget(self.name)

        mid = QHBoxLayout()
        mid.setSpacing(4)
        self.meter = Meter(True)
        self.fader = VolumeSlider(Qt.Vertical)
        self.fader.setMinimumHeight(110)
        mid.addStretch(1)
        mid.addWidget(self.meter)
        mid.addWidget(self.fader)
        mid.addStretch(1)
        lay.addLayout(mid, 1)
        self.db_label = QLabel()
        self.db_label.setAlignment(Qt.AlignCenter)
        self.db_label.setObjectName("Hint")
        self.db_label.setProperty("i18n_skip", True)
        lay.addWidget(self.db_label)
        self.pan = PanSlider()
        lay.addWidget(self.pan)
        self.pan_label = QLabel()
        self.pan_label.setAlignment(Qt.AlignCenter)
        self.pan_label.setObjectName("Hint")
        self.pan_label.setProperty("i18n_skip", True)
        lay.addWidget(self.pan_label)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.mute = _toggle("M", "MuteButton", tr("Mute: hides this part's notes and silences it"))
        self.solo = _toggle("S", "SoloButton", tr("Solo: show and hear only the soloed parts"))
        row.addWidget(self.mute)
        row.addWidget(self.solo)
        lay.addLayout(row)

        self.fader.dbChanged.connect(self._on_db)
        self.pan.panChanged.connect(self._on_pan)
        self.mute.toggled.connect(self._on_mute)
        self.solo.toggled.connect(self._on_solo)
        self.sync()

    def sync(self) -> None:
        t = self.track
        self.name.setText(tr(t.name))
        self.name.setToolTip(tr(t.name))
        self.fader.set_db(t.volume_db)
        self.pan.set_pan(t.pan)
        self.db_label.setText(db_text(t.volume_db))
        self.pan_label.setText(pan_text(t.pan))
        for b, v in ((self.mute, t.muted), (self.solo, t.solo)):
            b.blockSignals(True)
            b.setChecked(v)
            b.blockSignals(False)
        if self.property("dimmed") != (not t.visible):
            self.setProperty("dimmed", not t.visible)
            self.style().unpolish(self)
            self.style().polish(self)

    def _on_db(self, db: float) -> None:
        self.track.volume_db = db
        self.db_label.setText(db_text(db))
        self.mixChanged.emit(self.track)

    def _on_pan(self, pan: float) -> None:
        self.track.pan = pan
        self.pan_label.setText(pan_text(pan))
        self.mixChanged.emit(self.track)

    def _on_mute(self, on: bool) -> None:
        self.track.muted = on
        self.partsChanged.emit(self.track)

    def _on_solo(self, on: bool) -> None:
        self.track.solo = on
        self.partsChanged.emit(self.track)


class MasterStrip(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ChannelStrip")
        self.setFixedWidth(92)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        name = QLabel(tr("Master"))
        name.setAlignment(Qt.AlignCenter)
        name.setObjectName("StripName")
        lay.addWidget(name)
        mid = QHBoxLayout()
        mid.setSpacing(3)
        self.meter = Meter(True)
        mid.addStretch(1)
        mid.addWidget(self.meter)
        mid.addStretch(1)
        lay.addLayout(mid, 1)
        hint = QLabel(tr("The volume in the bottom bar is the master."))
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)


class MixerPanel(QWidget):
    """The Mixer tab."""

    mixChanged = Signal(object)
    partsChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.strips: list[ChannelStrip] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.empty = QLabel(tr("The parts of the song show up here once the notes have been found "
                               "(or the song was split into stems)."))
        self.empty.setObjectName("Hint")
        self.empty.setWordWrap(True)
        self.empty.setContentsMargins(16, 16, 16, 16)
        outer.addWidget(self.empty)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.inner = QWidget()
        self.row = QHBoxLayout(self.inner)
        self.row.setContentsMargins(10, 8, 10, 8)
        self.row.setSpacing(8)
        self.master = MasterStrip()
        self.row.addStretch(1)
        self.row.addWidget(self.master)
        self.scroll.setWidget(self.inner)
        outer.addWidget(self.scroll, 1)
        self.scroll.hide()

    def set_tracks(self, tracks) -> None:
        for strip in self.strips:
            self.row.removeWidget(strip)
            strip.deleteLater()
        self.strips = []
        for i, t in enumerate(tracks):
            strip = ChannelStrip(t)
            strip.mixChanged.connect(self.mixChanged)
            strip.partsChanged.connect(self.partsChanged)
            self.row.insertWidget(i, strip)
            self.strips.append(strip)
        self.empty.setVisible(not tracks)
        self.scroll.setVisible(bool(tracks))
        self.master.meter.reset()

    def sync(self) -> None:
        for strip in self.strips:
            strip.sync()

    def push_meters(self, peaks: dict, master: float) -> None:
        for strip in self.strips:
            strip.meter.push(peaks.get(strip.track.uid, 0.0))
        self.master.meter.push(master)

    def reset_meters(self) -> None:
        for strip in self.strips:
            strip.meter.reset()
        self.master.meter.reset()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(600, 260)
