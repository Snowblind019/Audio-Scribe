"""Record from a microphone (sing, hum, or play), then open the take in the app.

Uses Qt's audio input directly and writes a plain WAV file, so nothing extra is needed.
Recordings go to a folder you can pick (Music/Audio Scribe by default).
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QVBoxLayout)

from .i18n import tr
from .music import format_time


def default_folder() -> Path:
    music = Path.home() / "Music"
    return (music if music.is_dir() else Path.home()) / "Audio Scribe"


def to_int16(raw: bytes, fmt: QAudioFormat) -> np.ndarray:
    """Converts whatever sample format the device gives into 16-bit samples."""
    sf = fmt.sampleFormat()
    if sf == QAudioFormat.Int16:
        return np.frombuffer(raw, dtype="<i2").copy()
    if sf == QAudioFormat.Int32:
        return (np.frombuffer(raw, dtype="<i4") >> 16).astype("<i2")
    if sf == QAudioFormat.Float:
        x = np.clip(np.frombuffer(raw, dtype="<f4"), -1.0, 1.0)
        return (x * 32767.0).astype("<i2")
    if sf == QAudioFormat.UInt8:
        return ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).astype("<i2")
    raise ValueError("Unsupported microphone sample format")


class TakeRecorder(QObject):
    """Records the microphone into a WAV file while the song plays, for a take on its own
    track. The window decides where the take starts in the song."""

    level = Signal(float)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.source: QAudioSource | None = None
        self.device_io = None
        self.wav: wave.Wave_write | None = None
        self.frames = 0
        self.fmt: QAudioFormat | None = None

    def start(self) -> str:
        """Starts recording. Returns an error message, or "" when it is recording."""
        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            return tr("No microphone was found. Plug one in and try again.")
        fmt = device.preferredFormat()
        if fmt.sampleFormat() not in (QAudioFormat.Int16, QAudioFormat.Int32, QAudioFormat.Float, QAudioFormat.UInt8):
            fmt.setSampleFormat(QAudioFormat.Int16)
        self.fmt = fmt
        self.wav = wave.open(str(self.path), "wb")
        self.wav.setnchannels(max(1, fmt.channelCount()))
        self.wav.setsampwidth(2)
        self.wav.setframerate(fmt.sampleRate())
        self.source = QAudioSource(device, fmt, self)
        self.source.setBufferSize(int(fmt.sampleRate() * 0.05) * max(1, fmt.bytesPerFrame()))
        self.device_io = self.source.start()
        if self.device_io is None:
            self.stop()
            self.path.unlink(missing_ok=True)
            return tr("The microphone could not be opened.")
        self.device_io.readyRead.connect(self._read)
        return ""

    def _read(self) -> None:
        if self.device_io is None or self.wav is None:
            return
        raw = bytes(self.device_io.readAll())
        if not raw:
            return
        try:
            samples = to_int16(raw, self.fmt)
        except ValueError:
            return
        self.wav.writeframes(samples.tobytes())
        self.frames += len(samples) // max(1, self.fmt.channelCount())
        if len(samples):
            self.level.emit(float(np.abs(samples).max()) / 32768.0)

    def seconds(self) -> float:
        return self.frames / max(1, self.fmt.sampleRate()) if self.fmt else 0.0

    def stop(self) -> float:
        """Stops and closes the file. Returns how long the take is, in seconds."""
        if self.source is not None:
            self._read()
            self.source.stop()
            self.source = None
            self.device_io = None
        if self.wav is not None:
            self.wav.close()
            self.wav = None
        return self.seconds()


class RecordDialog(QDialog):
    """Record, stop, and hand the file to the main window."""

    recorded = Signal(str, bool, bool)   # file path, find notes, transcribe words

    def __init__(self, parent, folder: Path):
        super().__init__(parent)
        self.setWindowTitle(tr("Record from microphone"))
        self.folder = Path(folder)
        self.source: QAudioSource | None = None
        self.device_io = None
        self.wav: wave.Wave_write | None = None
        self.path: Path | None = None
        self.started = 0.0
        self.frames = 0
        self.level = 0.0

        lay = QVBoxLayout(self)
        intro = QLabel(tr("Sing, hum, whistle or play an instrument. When you stop, the recording opens here "
                          "and its notes are found, like humming a melody into a MIDI keyboard."))
        intro.setWordWrap(True)
        intro.setObjectName("Hint")
        lay.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Microphone")))
        self.device_box = QComboBox()
        self.device_box.setProperty("i18n_skip_items", True)
        for dev in QMediaDevices.audioInputs():
            self.device_box.addItem(dev.description(), dev)
        default = QMediaDevices.defaultAudioInput()
        for i in range(self.device_box.count()):
            if self.device_box.itemData(i) == default:
                self.device_box.setCurrentIndex(i)
        row.addWidget(self.device_box, 1)
        lay.addLayout(row)

        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        lay.addWidget(self.meter)
        self.clock = QLabel("0:00.0")
        self.clock.setObjectName("FileTitle")
        lay.addWidget(self.clock)

        row = QHBoxLayout()
        self.folder_label = QLabel()
        self.folder_label.setTextFormat(Qt.PlainText)
        self.folder_label.setObjectName("Hint")
        self.folder_label.setWordWrap(True)
        self.folder_btn = QPushButton(tr("Change folder..."))
        self.folder_btn.clicked.connect(self._pick_folder)
        row.addWidget(self.folder_label, 1)
        row.addWidget(self.folder_btn)
        lay.addLayout(row)
        self._show_folder()

        self.notes_chk = QCheckBox(tr("Find the notes when I stop"))
        self.notes_chk.setChecked(True)
        self.words_chk = QCheckBox(tr("Also transcribe the words"))
        lay.addWidget(self.notes_chk)
        lay.addWidget(self.words_chk)

        row = QHBoxLayout()
        self.record_btn = QPushButton(tr("Record"))
        self.record_btn.setObjectName("Primary")
        self.record_btn.clicked.connect(self._toggle)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(close)
        row.addWidget(self.record_btn)
        lay.addLayout(row)

        self.status = QLabel("")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        if self.device_box.count() == 0:
            self.record_btn.setEnabled(False)
            self.status.setText(tr("No microphone was found. Plug one in, then open this window again."))

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

    def _show_folder(self) -> None:
        self.folder_label.setText(tr("Saved in {folder}").format(folder=str(self.folder)))

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Where should recordings go?"), str(self.folder))
        if folder:
            self.folder = Path(folder)
            self._show_folder()

    def _toggle(self) -> None:
        if self.source is None:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        device = self.device_box.currentData()
        if device is None:
            return
        fmt = device.preferredFormat()
        if fmt.sampleFormat() not in (QAudioFormat.Int16, QAudioFormat.Int32, QAudioFormat.Float, QAudioFormat.UInt8):
            fmt.setSampleFormat(QAudioFormat.Int16)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status.setText(tr("Could not use that folder: {error}").format(error=str(exc)))
            return
        stamp = time.strftime("%Y-%m-%d %H-%M-%S")
        self.path = self.folder / f"{tr('Recording')} {stamp}.wav"
        self.wav = wave.open(str(self.path), "wb")
        self.wav.setnchannels(max(1, fmt.channelCount()))
        self.wav.setsampwidth(2)
        self.wav.setframerate(fmt.sampleRate())
        self.fmt = fmt
        self.frames = 0
        self.source = QAudioSource(device, fmt, self)
        self.device_io = self.source.start()
        if self.device_io is None:
            self._close_file()
            self.source = None
            self.status.setText(tr("The microphone could not be opened."))
            return
        self.device_io.readyRead.connect(self._read)
        self.started = time.monotonic()
        self.timer.start()
        self.record_btn.setText(tr("Stop"))
        self.device_box.setEnabled(False)
        self.folder_btn.setEnabled(False)
        self.status.setText(tr("Recording..."))

    def _read(self) -> None:
        if self.device_io is None or self.wav is None:
            return
        raw = bytes(self.device_io.readAll())
        if not raw:
            return
        try:
            samples = to_int16(raw, self.fmt)
        except ValueError:
            return
        self.wav.writeframes(samples.tobytes())
        self.frames += len(samples) // max(1, self.fmt.channelCount())
        if len(samples):
            peak = float(np.abs(samples).max()) / 32768.0
            self.level = max(peak, self.level * 0.85)

    def _tick(self) -> None:
        self.clock.setText(format_time(time.monotonic() - self.started, 1))
        self.meter.setValue(int(min(1.0, self.level) * 100))
        self.level *= 0.9

    def _close_file(self) -> None:
        if self.wav is not None:
            self.wav.close()
            self.wav = None

    def stop(self) -> None:
        if self.source is None:
            return
        self._read()
        self.source.stop()
        self.source = None
        self.device_io = None
        self.timer.stop()
        self._close_file()
        self.record_btn.setText(tr("Record"))
        self.device_box.setEnabled(True)
        self.folder_btn.setEnabled(True)
        if self.frames < self.fmt.sampleRate() * 0.3:
            self.status.setText(tr("That was too short to use."))
            if self.path:
                self.path.unlink(missing_ok=True)
            return
        self.recorded.emit(str(self.path), self.notes_chk.isChecked(), self.words_chk.isChecked())
        self.accept()

    def reject(self) -> None:
        if self.source is not None:
            self.source.stop()
            self.source = None
            self.timer.stop()
            self._close_file()
            if self.path:
                self.path.unlink(missing_ok=True)
        super().reject()
