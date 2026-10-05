"""The main window."""

from __future__ import annotations

import bisect
import html
import logging
import shutil
import tempfile
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QFontDatabase, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMenu,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider,
                               QSpinBox, QSplitter, QStyle, QTabWidget, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import APP_NAME, exporters
from .app import cache_dir
from .engine import (LANGUAGE_NAMES, LANGUAGES, STEM_ORDER, WHISPER_MODELS, Analyzer, Cancelled,
                     Options, Result, cuda_available, friendly_error, remove_dir, stems_available)
from .music import format_time, note_name
from .piano_roll import PianoRoll
from .widgets import DATA_ROLE, SORT_ROLE, SortItem, SummaryView, swatch_icon

log = logging.getLogger(__name__)

OPEN_FILTER = ("Audio and video (*.mp3 *.wav *.flac *.ogg *.oga *.opus *.m4a *.aac *.wma *.aif "
               "*.aiff *.alac *.mp4 *.m4v *.mkv *.webm *.mov *.avi);;All files (*)")
PLAY_ORDER = ["Original", "Vocals", "Bass", "Other", "Drums"]


def _work_root() -> Path:
    root = cache_dir() / "work"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _clean_old_work_dirs(max_age_hours: float = 12) -> None:
    cutoff = time.time() - max_age_hours * 3600
    for child in _work_root().iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass


class AnalyzeThread(QThread):
    progressed = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, path: Path, options: Options, work_dir: Path, parent=None):
        super().__init__(parent)
        self.path, self.options, self.work_dir = path, options, work_dir
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            analyzer = Analyzer(self.path, self.options, self.work_dir,
                                report=lambda f, t: self.progressed.emit(f, t),
                                should_stop=lambda: self._stop)
            self.succeeded.emit(analyzer.run())
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            log.exception("Analysis failed")
            self.failed.emit(friendly_error(exc), traceback.format_exc())


def show_message(parent, icon, text: str, details: str | None = None, informative: str | None = None) -> None:
    """Message box that never interprets file names or error text as HTML."""
    box = QMessageBox(icon, APP_NAME, text, QMessageBox.Ok, parent)
    box.setTextFormat(Qt.PlainText)
    if informative:
        box.setInformativeText(html.escape(informative))
    if details:
        box.setDetailedText(details)
    box.exec()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        self.settings = QSettings()

        self.source_path: Path | None = None
        self.result: Result | None = None
        self.thread: AnalyzeThread | None = None
        self._work_dir: Path | None = None
        self._pending_seek: int | None = None
        self._resume_after_load = False
        self._slider_held = False
        self._seg_starts: list[float] = []
        self._current_seg = -1
        self._stems_ok = stems_available()
        self._cuda_ok = cuda_available()

        self._build_ui()
        self._build_player()
        self._build_shortcuts()
        self._load_settings()
        self._update_controls()
        _clean_old_work_dirs()

    # UI construction ------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_inspector())

        right = QWidget()
        col = QVBoxLayout(right)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._build_header())

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(1)
        self.roll = PianoRoll()
        self.roll.seekRequested.connect(self.seek)
        self.roll.noteClicked.connect(self._on_roll_note_clicked)
        self.roll.pitchToggled.connect(self._on_pitch_toggled)
        self.splitter.addWidget(self.roll)
        self.splitter.addWidget(self._build_tabs())
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([520, 300])
        col.addWidget(self.splitter, 1)
        col.addWidget(self._build_transport())
        outer.addWidget(right, 1)
        self.setCentralWidget(root)

        self.status_text = QLabel("Ready")

        self.status_text.setTextFormat(Qt.PlainText)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(240)
        self.progress.hide()
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().setSizeGripEnabled(False)

    def _section(self, title: str, hint: str | None = None):
        box = QFrame()
        box.setObjectName("Section")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(7)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        lay.addWidget(label)
        if hint:
            h = QLabel(hint)
            h.setObjectName("Hint")
            h.setWordWrap(True)
            lay.addWidget(h)
        return box, lay

    @staticmethod
    def _compact(box: QComboBox) -> QComboBox:
        box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        box.setMinimumContentsLength(8)
        return box

    @staticmethod
    def _form() -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 2, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return form

    def _build_inspector(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("Inspector")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # File
        box, l = self._section("File")
        self.file_label = QLabel("No file open yet.")
        self.file_label.setTextFormat(Qt.PlainText)
        self.file_label.setObjectName("Hint")
        self.file_label.setWordWrap(True)
        self.open_btn = QPushButton("Open file...")
        self.open_btn.clicked.connect(self._open_dialog)
        drop = QLabel("You can also drop a file anywhere in this window.")
        drop.setObjectName("Hint")
        drop.setWordWrap(True)
        l.addWidget(self.file_label)
        l.addWidget(self.open_btn)
        l.addWidget(drop)
        lay.addWidget(box)

        # Words
        box, l = self._section("Words")
        self.chk_words = QCheckBox("Transcribe words")
        self.chk_words.toggled.connect(self._update_controls)
        l.addWidget(self.chk_words)
        form = self._form()
        self.model_box = self._compact(QComboBox())
        for key, label in WHISPER_MODELS:
            self.model_box.addItem(label, key)
        self.model_box.setToolTip("Bigger models are more accurate but slower and take more memory.\n"
                                  "Each model downloads once, the first time you use it.")
        self.lang_box = self._compact(QComboBox())
        for code, label in LANGUAGES:
            self.lang_box.addItem(label, code)
        self.lang_box.setToolTip("Picking the language helps with short clips and songs.")
        form.addRow("Model", self.model_box)
        form.addRow("Language", self.lang_box)
        l.addLayout(form)
        self.chk_translate = QCheckBox("Translate to English")
        self.chk_vad = QCheckBox("Skip silent parts")
        self.chk_vad.setToolTip("Speeds things up and avoids made-up text in quiet parts.\n"
                                "Turn it off if quiet singing gets skipped.")
        l.addWidget(self.chk_translate)
        l.addWidget(self.chk_vad)
        lay.addWidget(box)

        # Notes
        box, l = self._section("Notes")
        self.chk_notes = QCheckBox("Find notes")
        self.chk_notes.toggled.connect(self._update_controls)
        l.addWidget(self.chk_notes)
        form = self._form()
        sens_row = QWidget()
        sr = QHBoxLayout(sens_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.setSpacing(6)
        self.sens = QSlider(Qt.Horizontal)
        self.sens.setRange(0, 100)
        self.sens.setToolTip("Higher finds quieter notes, but also more stray ones.")
        fewer, more = QLabel("Fewer"), QLabel("More")
        fewer.setObjectName("Hint")
        more.setObjectName("Hint")
        sr.addWidget(fewer)
        sr.addWidget(self.sens, 1)
        sr.addWidget(more)
        self.min_len = QSpinBox()
        self.min_len.setRange(30, 1000)
        self.min_len.setSingleStep(10)
        self.min_len.setSuffix(" ms")
        self.min_len.setToolTip("Notes shorter than this are ignored.")
        form.addRow("Sensitivity", sens_row)
        form.addRow("Shortest note", self.min_len)
        l.addLayout(form)
        lay.addWidget(box)

        # Stems
        hint = ("Separates vocals, bass, drums, and the rest before analyzing. Slower, "
                "but gives much cleaner words and notes on full songs.")
        if not self._stems_ok:
            hint = "Not installed. Run the installer again and choose stem separation to turn this on."
        box, l = self._section("Stems", hint)
        self.chk_stems = QCheckBox("Split into stems first")
        self.chk_stems.toggled.connect(self._update_controls)
        l.insertWidget(1, self.chk_stems)
        self.stems_label = QLabel("Find notes in")
        self.stems_label.setObjectName("Hint")
        l.addWidget(self.stems_label)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        self.stem_checks: dict[str, QCheckBox] = {}
        for i, name in enumerate(STEM_ORDER):
            chk = QCheckBox(name)
            if name == "Drums":
                chk.setToolTip("Drums have no real pitch, so this mostly finds noise.")
            self.stem_checks[name] = chk
            grid.addWidget(chk, i // 2, i % 2)
        l.addLayout(grid)
        lay.addWidget(box)

        # Processing
        box, l = self._section("Processing")
        form = self._form()
        self.device_box = self._compact(QComboBox())
        self.device_box.addItem("CPU", "cpu")
        self.device_box.addItem("NVIDIA GPU", "cuda")
        if not self._cuda_ok:
            self.device_box.model().item(1).setEnabled(False)
            self.device_box.setToolTip("No NVIDIA GPU with CUDA was found, so everything runs on the CPU.")
        form.addRow("Run on", self.device_box)
        l.addLayout(form)
        lay.addWidget(box)

        run = QWidget()
        rl = QVBoxLayout(run)
        rl.setContentsMargins(16, 16, 16, 16)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setObjectName("Primary")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.cancel_btn.hide()
        rl.addWidget(self.analyze_btn)
        rl.addWidget(self.cancel_btn)
        lay.addWidget(run)

        # Tracks (filled after analysis)
        self.tracks_box, self.tracks_layout = self._section("Tracks", "Show or hide each part in the piano roll.")
        self.tracks_box.hide()
        lay.addWidget(self.tracks_box)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("InspectorScroll")
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(316)
        return scroll

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("Header")
        l = QHBoxLayout(header)
        l.setContentsMargins(18, 10, 12, 10)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title_label = QLabel(APP_NAME)
        self.title_label.setTextFormat(Qt.PlainText)
        self.title_label.setObjectName("FileTitle")
        self.meta_label = QLabel("Open a song, a voice memo, or a video to see its words and notes.")
        self.meta_label.setTextFormat(Qt.PlainText)
        self.meta_label.setObjectName("Meta")
        text.addWidget(self.title_label)
        text.addWidget(self.meta_label)
        l.addLayout(text, 1)

        self.export_btn = QToolButton()
        self.export_btn.setText("Export")
        self.export_btn.setPopupMode(QToolButton.InstantPopup)
        self.export_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        menu = QMenu(self.export_btn)
        self.act_txt = menu.addAction("Transcript as text (.txt)", lambda: self._export_text("txt"))
        self.act_srt = menu.addAction("Transcript as subtitles (.srt)", lambda: self._export_text("srt"))
        self.act_lrc = menu.addAction("Lyrics with timing (.lrc)", lambda: self._export_text("lrc"))
        menu.addSeparator()
        self.act_mid = menu.addAction("Notes as MIDI (.mid)", self._export_midi)
        self.act_csv = menu.addAction("Notes as spreadsheet (.csv)", self._export_csv)
        menu.addSeparator()
        self.act_stems = menu.addAction("Stems as WAV files...", self._export_stems)
        self.export_btn.setMenu(menu)
        l.addWidget(self.export_btn)
        return header

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        page = QWidget()
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(14, 6, 10, 6)
        self.lang_label = QLabel("Click a line to jump to it.")
        self.lang_label.setTextFormat(Qt.PlainText)
        self.lang_label.setObjectName("Meta")
        self.copy_btn = QPushButton("Copy text")
        self.copy_btn.clicked.connect(self._copy_transcript)
        bar.addWidget(self.lang_label, 1)
        bar.addWidget(self.copy_btn)
        pl.addLayout(bar)
        self.transcript = QTreeWidget()
        self.transcript.setHeaderLabels(["Time", "Words"])
        self.transcript.setRootIsDecorated(False)
        self.transcript.setUniformRowHeights(False)
        self.transcript.setWordWrap(True)
        self.transcript.setAlternatingRowColors(True)
        self.transcript.setColumnWidth(0, 80)
        self.transcript.itemClicked.connect(self._on_transcript_clicked)
        pl.addWidget(self.transcript, 1)
        tabs.addTab(page, "Transcript")

        self.notes_table = QTreeWidget()
        self.notes_table.setHeaderLabels(["Note", "Track", "Start", "Length", "Strength"])
        self.notes_table.setRootIsDecorated(False)
        self.notes_table.setAlternatingRowColors(True)
        self.notes_table.setUniformRowHeights(True)
        self.notes_table.setSortingEnabled(True)
        self.notes_table.itemClicked.connect(self._on_note_row_clicked)
        tabs.addTab(self.notes_table, "Notes")

        self.summary = SummaryView()
        self.summary.pitchSelected.connect(self._on_summary_pitch)
        tabs.addTab(self.summary, "Summary")
        self.tabs = tabs
        return tabs

    def _build_transport(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Transport")
        l = QHBoxLayout(bar)
        l.setContentsMargins(10, 6, 12, 6)
        l.setSpacing(6)
        st = self.style()

        def tool(icon, tip, slot):
            b = QToolButton()
            b.setIcon(st.standardIcon(icon))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            return b

        self.start_btn = tool(QStyle.SP_MediaSkipBackward, "Go to start (Home)", lambda: self.seek(0.0))
        self.play_btn = tool(QStyle.SP_MediaPlay, "Play or pause (Space)", self.toggle_play)
        self.stop_btn = tool(QStyle.SP_MediaStop, "Stop", self.stop_playback)
        for b in (self.start_btn, self.play_btn, self.stop_btn):
            l.addWidget(b)

        self.clock = QLabel("0:00.0 / 0:00.0")
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSizeF(max(9.0, self.font().pointSizeF() + 0.5))
        self.clock.setFont(mono)
        self.clock.setMinimumWidth(150)
        self.clock.setAlignment(Qt.AlignCenter)
        l.addWidget(self.clock)

        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 0)
        self.pos_slider.sliderPressed.connect(lambda: setattr(self, "_slider_held", True))
        self.pos_slider.sliderReleased.connect(self._on_slider_released)
        self.pos_slider.sliderMoved.connect(lambda v: self.seek(v / 1000.0))
        l.addWidget(self.pos_slider, 1)

        self.source_label = QLabel("Play")
        self.source_label.setObjectName("Meta")
        self.source_box = QComboBox()
        self.source_box.currentIndexChanged.connect(self._on_source_changed)
        l.addWidget(self.source_label)
        l.addWidget(self.source_box)

        vol_icon = QLabel()
        vol_icon.setPixmap(st.standardIcon(QStyle.SP_MediaVolume).pixmap(16, 16))
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setFixedWidth(90)
        self.volume.setToolTip("Volume")
        self.volume.valueChanged.connect(lambda v: self.audio_out.setVolume(v / 100.0))
        l.addSpacing(6)
        l.addWidget(vol_icon)
        l.addWidget(self.volume)

        self.follow_chk = QCheckBox("Follow")
        self.follow_chk.setToolTip("Keep the playhead in view while playing")
        self.follow_chk.toggled.connect(lambda on: setattr(self.roll, "follow", on))
        l.addSpacing(8)
        l.addWidget(self.follow_chk)

        l.addSpacing(8)
        for text, tip, slot in (("-", "Zoom out (Ctrl+minus, or Ctrl+scroll)", lambda: self.roll.zoom_time(1 / 1.5)),
                                ("+", "Zoom in (Ctrl+plus, or Ctrl+scroll)", lambda: self.roll.zoom_time(1.5)),
                                ("Fit", "Fit the whole file (Ctrl+0)", self.roll.fit)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            b.setMinimumWidth(28)
            l.addWidget(b)
        return bar

    def _build_player(self) -> None:
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.playbackStateChanged.connect(self._on_play_state)
        self.player.durationChanged.connect(self._on_duration)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_player_error)
        self.ticker = QTimer(self)
        self.ticker.setInterval(33)
        self.ticker.timeout.connect(self._on_tick)

    def _build_shortcuts(self) -> None:
        def add(keys, slot):
            for key in keys if isinstance(keys, (list, tuple)) else [keys]:
                sc = QShortcut(QKeySequence(key), self)
                sc.setContext(Qt.WindowShortcut)
                sc.activated.connect(slot)

        add("Space", self.toggle_play)
        add("Home", lambda: self.seek(0.0))
        add(QKeySequence.Open, self._open_dialog)
        add(["Ctrl+Return", "Ctrl+Enter", "F5"], self.start_analysis)
        add([QKeySequence.ZoomIn, "Ctrl+="], lambda: self.roll.zoom_time(1.5))
        add(QKeySequence.ZoomOut, lambda: self.roll.zoom_time(1 / 1.5))
        add("Ctrl+0", self.roll.fit)

    # Settings -------------------------------------------------------------------

    def _load_settings(self) -> None:
        s = self.settings

        def get_bool(key, default):
            value = s.value(key, default)
            return value in (True, "true", "1", 1) if not isinstance(value, bool) else value

        self.chk_words.setChecked(get_bool("words/enabled", True))
        self._set_combo(self.model_box, s.value("words/model", "small"))
        self._set_combo(self.lang_box, s.value("words/language", ""))
        self.chk_translate.setChecked(get_bool("words/translate", False))
        self.chk_vad.setChecked(get_bool("words/vad", True))
        self.chk_notes.setChecked(get_bool("notes/enabled", True))
        self.sens.setValue(int(s.value("notes/sensitivity", 50)))
        self.min_len.setValue(int(s.value("notes/min_ms", 120)))
        self.chk_stems.setChecked(get_bool("stems/enabled", False) and self._stems_ok)
        chosen = s.value("stems/note_stems", "Vocals,Bass,Other")
        chosen = chosen if isinstance(chosen, str) else ",".join(chosen)
        for name, chk in self.stem_checks.items():
            chk.setChecked(name in chosen.split(","))
        device = s.value("device", "cpu")
        self._set_combo(self.device_box, device if (device != "cuda" or self._cuda_ok) else "cpu")
        self.volume.setValue(int(s.value("volume", 85)))
        self.audio_out.setVolume(self.volume.value() / 100.0)
        self.follow_chk.setChecked(get_bool("follow", True))
        geometry = s.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        splitter = s.value("window/splitter")
        if splitter:
            self.splitter.restoreState(splitter)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("words/enabled", self.chk_words.isChecked())
        s.setValue("words/model", self.model_box.currentData())
        s.setValue("words/language", self.lang_box.currentData())
        s.setValue("words/translate", self.chk_translate.isChecked())
        s.setValue("words/vad", self.chk_vad.isChecked())
        s.setValue("notes/enabled", self.chk_notes.isChecked())
        s.setValue("notes/sensitivity", self.sens.value())
        s.setValue("notes/min_ms", self.min_len.value())
        s.setValue("stems/enabled", self.chk_stems.isChecked())
        s.setValue("stems/note_stems", ",".join(n for n, c in self.stem_checks.items() if c.isChecked()))
        s.setValue("device", self.device_box.currentData())
        s.setValue("volume", self.volume.value())
        s.setValue("follow", self.follow_chk.isChecked())
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/splitter", self.splitter.saveState())

    @staticmethod
    def _set_combo(box: QComboBox, data) -> None:
        index = box.findData(data)
        if index >= 0:
            box.setCurrentIndex(index)

    # State --------------------------------------------------------------------

    def _busy(self) -> bool:
        return self.thread is not None

    def _update_controls(self) -> None:
        busy = self._busy()
        for w in (self.open_btn, self.chk_words, self.chk_notes, self.device_box):
            w.setEnabled(not busy)
        words = self.chk_words.isChecked() and not busy
        for w in (self.model_box, self.lang_box, self.chk_translate, self.chk_vad):
            w.setEnabled(words)
        notes = self.chk_notes.isChecked() and not busy
        self.sens.setEnabled(notes)
        self.min_len.setEnabled(notes)
        self.chk_stems.setEnabled(self._stems_ok and not busy)
        stems = self.chk_stems.isChecked() and self._stems_ok and not busy
        self.stems_label.setEnabled(stems and notes)
        for chk in self.stem_checks.values():
            chk.setEnabled(stems and notes)
        self.analyze_btn.setVisible(not busy)
        self.cancel_btn.setVisible(busy)
        self.analyze_btn.setEnabled(self.source_path is not None)

        has_media = self.source_path is not None
        for w in (self.start_btn, self.play_btn, self.stop_btn, self.pos_slider):
            w.setEnabled(has_media)
        r = self.result
        has_words = bool(r and r.segments)
        has_notes = bool(r and any(t.notes for t in r.tracks))
        has_stems = bool(r and len(r.audio_files) > 1)
        for act in (self.act_txt, self.act_srt, self.act_lrc):
            act.setEnabled(has_words)
        self.act_mid.setEnabled(has_notes)
        self.act_csv.setEnabled(has_notes)
        self.act_stems.setEnabled(has_stems)
        self.export_btn.setEnabled(bool(r) and not busy)
        self.copy_btn.setEnabled(has_words)
        self.source_label.setVisible(has_stems)
        self.source_box.setVisible(has_stems)

    # Opening files --------------------------------------------------------------

    def _open_dialog(self) -> None:
        if self._busy():
            return
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open audio or video", start, OPEN_FILTER)
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> None:
        if self._busy():
            return
        path = Path(path)
        if not path.is_file():
            show_message(self, QMessageBox.Warning, f"Could not find {path}.")
            return
        self.settings.setValue("last_dir", str(path.parent))
        self.source_path = path
        self._clear_results()
        self.title_label.setText(path.name)
        self.meta_label.setText("Press Analyze to find the words and notes.")
        self.file_label.setText(path.name)
        self.setWindowTitle(f"{path.name} - {APP_NAME}")
        self.roll.clear("Press Analyze to find the words and notes.")
        self.status_text.setText("File opened. You can play it now or press Analyze.")
        self._set_source(str(path), keep_position=False)
        self._update_controls()

    def _clear_results(self) -> None:
        old = self.result
        self.result = None
        self.transcript.clear()
        self.notes_table.clear()
        self.summary.clear()
        self._seg_starts, self._current_seg = [], -1
        self.lang_label.setText("Click a line to jump to it.")
        self._fill_tracks([])
        self.source_box.blockSignals(True)
        self.source_box.clear()
        self.source_box.blockSignals(False)
        if old:
            QTimer.singleShot(500, lambda d=old.work_dir: remove_dir(d))

    # Analysis -------------------------------------------------------------------

    def _options(self) -> Options:
        return Options(
            transcribe=self.chk_words.isChecked(),
            model=self.model_box.currentData(),
            language=self.lang_box.currentData() or None,
            translate=self.chk_translate.isChecked(),
            vad=self.chk_vad.isChecked(),
            notes=self.chk_notes.isChecked(),
            sensitivity=self.sens.value(),
            min_note_ms=self.min_len.value(),
            separate=self.chk_stems.isChecked() and self._stems_ok,
            note_stems=[n for n, c in self.stem_checks.items() if c.isChecked()],
            device=self.device_box.currentData(),
        )

    def start_analysis(self) -> None:
        if self._busy():
            return
        if self.source_path is None:
            self._open_dialog()
            return
        o = self._options()
        if not o.transcribe and not o.notes:
            show_message(self, QMessageBox.Information, "Turn on Transcribe words or Find notes first.")
            return
        if o.notes and o.separate and not o.note_stems:
            show_message(self, QMessageBox.Information, "Pick at least one stem to find notes in.")
            return
        self._save_settings()
        self._work_dir = Path(tempfile.mkdtemp(prefix="run-", dir=_work_root()))
        self.thread = AnalyzeThread(self.source_path, o, self._work_dir, self)
        self.thread.progressed.connect(self._on_progress)
        self.thread.succeeded.connect(self._on_success)
        self.thread.failed.connect(self._on_failed)
        self.thread.cancelled.connect(self._on_cancelled)
        self.thread.finished.connect(self._on_thread_finished)
        self.progress.setValue(0)
        self.progress.show()
        self.status_text.setText("Starting")
        self.thread.start()
        self._update_controls()

    def cancel_analysis(self) -> None:
        if self.thread:
            self.thread.stop()
            self.cancel_btn.setEnabled(False)
            self.status_text.setText("Stopping after the current step")

    def _on_progress(self, fraction: float, text: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        self.status_text.setText(text)

    def _on_success(self, result: Result) -> None:
        self._work_dir = None
        self._show_result(result)
        secs = int(round(result.elapsed))
        took = f"{secs // 60} min {secs % 60} s" if secs >= 60 else f"{secs} s"
        self.status_text.setText(f"Done in {took}.")

    def _on_failed(self, message: str, details: str) -> None:
        remove_dir(self._work_dir)
        self._work_dir = None
        self.status_text.setText("Analysis failed.")
        show_message(self, QMessageBox.Critical, message, details=details,
                     informative="Your file was not changed. Full details are below.")

    def _on_cancelled(self) -> None:
        remove_dir(self._work_dir)
        self._work_dir = None
        self.status_text.setText("Analysis cancelled.")

    def _on_thread_finished(self) -> None:
        if self.thread:
            self.thread.deleteLater()
        self.thread = None
        self.progress.hide()
        self.cancel_btn.setEnabled(True)
        self._update_controls()

    # Showing results --------------------------------------------------------------

    def _show_result(self, r: Result) -> None:
        self._clear_results()
        self.result = r
        self.roll.set_data(r.duration, r.tracks, r.segments, r.beats)
        if not r.tracks and not r.segments:
            self.roll.set_message("Nothing was found in this file.")
        elif not r.tracks:
            self.roll.set_message("Turn on Find notes to see them here.")

        parts = [f"{format_time(r.duration, 0)} long"]
        if r.key:
            parts.append(f"probably {r.key.name}")
        if r.tempo:
            parts.append(f"around {r.tempo:.0f} BPM")
        text = ", ".join(parts)
        self.meta_label.setText(text[:1].upper() + text[1:])

        self._fill_transcript(r)
        self._fill_notes_table(r)
        self.summary.set_result(r)
        self._fill_tracks(r.tracks)

        self.source_box.blockSignals(True)
        for name in PLAY_ORDER:
            if name in r.audio_files:
                self.source_box.addItem(name, r.audio_files[name])
        self.source_box.blockSignals(False)
        self._set_source(r.audio_files["Original"], keep_position=True)
        self.pos_slider.setRange(0, int(r.duration * 1000))
        self._sync_position(self.player.position() / 1000.0)
        self._update_controls()

    def _fill_transcript(self, r: Result) -> None:
        items = []
        for i, seg in enumerate(r.segments):
            item = QTreeWidgetItem([format_time(seg.start, 1), seg.text])
            item.setData(0, DATA_ROLE, i)
            item.setTextAlignment(0, Qt.AlignRight | Qt.AlignTop)
            items.append(item)
        self.transcript.addTopLevelItems(items)
        self._seg_starts = [s.start for s in r.segments]
        if r.words_requested:
            if r.segments:
                lang = LANGUAGE_NAMES.get(r.language or "", (r.language or "").upper())
                how = "Translated to English from" if r.translated else "Language:"
                self.lang_label.setText(f"{how} {lang}. Click a line to jump to it.")
            else:
                self.lang_label.setText("No words were found.")
        else:
            self.lang_label.setText("Transcription was off for this run.")

    def _fill_notes_table(self, r: Result) -> None:
        self.notes_table.setSortingEnabled(False)
        items = []
        for ti, track in enumerate(r.tracks):
            for ni, n in enumerate(track.notes):
                length = n.end - n.start
                item = SortItem([note_name(n.pitch), track.name, format_time(n.start, 2),
                                 f"{length:.2f} s", f"{n.velocity * 100:.0f}%"])
                item.setData(0, SORT_ROLE, n.pitch)
                item.setData(1, SORT_ROLE, ti)
                item.setData(2, SORT_ROLE, n.start)
                item.setData(3, SORT_ROLE, length)
                item.setData(4, SORT_ROLE, n.velocity)
                item.setData(0, DATA_ROLE, (ti, ni))
                item.setIcon(1, swatch_icon(track.color, 10))
                items.append(item)
        self.notes_table.addTopLevelItems(items)
        self.notes_table.setSortingEnabled(True)
        self.notes_table.sortItems(2, Qt.AscendingOrder)
        for col, width in ((0, 70), (1, 110), (2, 90), (3, 80)):
            self.notes_table.setColumnWidth(col, width)

    def _fill_tracks(self, tracks) -> None:
        while self.tracks_layout.count() > 2:
            item = self.tracks_layout.takeAt(2)
            if item.widget():
                item.widget().deleteLater()
        for track in tracks:
            chk = QCheckBox(f"{track.name}  ({len(track.notes)} notes)")
            chk.setIcon(swatch_icon(track.color))
            chk.setChecked(track.visible)
            chk.toggled.connect(lambda on, t=track: self._set_track_visible(t, on))
            self.tracks_layout.addWidget(chk)
        self.tracks_box.setVisible(bool(tracks))

    def _set_track_visible(self, track, on: bool) -> None:
        track.visible = on
        self.roll.refresh()

    # Interaction between views ------------------------------------------------------

    def _on_transcript_clicked(self, item: QTreeWidgetItem) -> None:
        if self.result:
            self.seek(self.result.segments[item.data(0, DATA_ROLE)].start)

    def _on_note_row_clicked(self, item: QTreeWidgetItem) -> None:
        if not self.result:
            return
        ti, ni = item.data(0, DATA_ROLE)
        self.roll.select_note(ti, ni)
        self.seek(self.result.tracks[ti].notes[ni].start)

    def _on_roll_note_clicked(self, ti: int, ni: int) -> None:
        n = self.result.tracks[ti].notes[ni] if self.result else None
        if n:
            self.status_text.setText(f"{note_name(n.pitch)} at {format_time(n.start, 2)}, "
                                     f"{n.end - n.start:.2f} s long ({self.result.tracks[ti].name})")

    def _on_pitch_toggled(self, pitch) -> None:
        self.summary.select_pitch(pitch)
        self._describe_pitch(pitch)

    def _on_summary_pitch(self, pitch) -> None:
        self.roll.set_highlight(pitch)
        self._describe_pitch(pitch)

    def _describe_pitch(self, pitch) -> None:
        if pitch is None or not self.result:
            self.status_text.setText("Highlight cleared.")
            return
        count = sum(1 for t in self.result.tracks if t.visible for n in t.notes if n.pitch == pitch)
        times = "time" if count == 1 else "times"
        self.status_text.setText(f"Highlighting {note_name(pitch)}, played {count} {times}. "
                                 "Click the key again or press Esc to clear.")

    # Playback ---------------------------------------------------------------------

    def _set_source(self, path: str, keep_position: bool = True) -> None:
        playing = self.player.playbackState() == QMediaPlayer.PlayingState
        position = self.player.position()
        self._pending_seek = position if keep_position and position > 0 else None
        self._resume_after_load = playing and keep_position
        self.player.setSource(QUrl.fromLocalFile(path))

    def _on_media_status(self, status) -> None:
        if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia):
            if self._pending_seek is not None:
                self.player.setPosition(self._pending_seek)
                self._pending_seek = None
            if self._resume_after_load:
                self._resume_after_load = False
                self.player.play()
        elif status == QMediaPlayer.EndOfMedia:
            self._sync_position(self._duration())

    def _on_source_changed(self, index: int) -> None:
        path = self.source_box.itemData(index)
        if path:
            self._set_source(path, keep_position=True)

    def _on_duration(self, ms: int) -> None:
        if not self.result and ms > 0:
            self.pos_slider.setRange(0, ms)
            self.roll.set_duration(ms / 1000.0)
            self._sync_position(self.player.position() / 1000.0)

    def _on_player_error(self, error, message: str) -> None:
        if error != QMediaPlayer.NoError:
            self.status_text.setText(f"Playback problem: {message}. You can still analyze the file.")

    def _duration(self) -> float:
        if self.result:
            return self.result.duration
        return max(0.0, self.player.duration() / 1000.0)

    def toggle_play(self) -> None:
        if self.source_path is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop_playback(self) -> None:
        self.player.stop()
        self._sync_position(0.0)

    def seek(self, seconds: float) -> None:
        if self.source_path is None:
            return
        seconds = min(max(0.0, seconds), self._duration() or seconds)
        self.player.setPosition(int(seconds * 1000))
        self._sync_position(seconds)

    def _on_slider_released(self) -> None:
        self._slider_held = False
        self.seek(self.pos_slider.value() / 1000.0)

    def _on_play_state(self, state) -> None:
        playing = state == QMediaPlayer.PlayingState
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))
        if playing:
            self.ticker.start()
        else:
            self.ticker.stop()
            self._sync_position(self.player.position() / 1000.0)

    def _on_tick(self) -> None:
        self._sync_position(self.player.position() / 1000.0)

    def _sync_position(self, t: float) -> None:
        self.roll.set_playhead(t)
        self.clock.setText(f"{format_time(t)} / {format_time(self._duration())}")
        if not self._slider_held:
            self.pos_slider.blockSignals(True)
            self.pos_slider.setValue(int(t * 1000))
            self.pos_slider.blockSignals(False)
        self._highlight_segment(t)

    def _highlight_segment(self, t: float) -> None:
        if not self._seg_starts:
            return
        index = bisect.bisect_right(self._seg_starts, t) - 1
        if index == self._current_seg:
            return
        self._current_seg = index
        if index < 0:
            self.transcript.clearSelection()
            return
        item = self.transcript.topLevelItem(index)
        if item:
            self.transcript.setCurrentItem(item)
            if self.follow_chk.isChecked():
                self.transcript.scrollToItem(item)

    # Export -----------------------------------------------------------------------

    def _save_path(self, title: str, ext: str, filter_text: str) -> Path | None:
        stem = self.source_path.stem if self.source_path else "audio"
        start = Path(self.settings.value("export_dir", str(self.source_path.parent if self.source_path else Path.home())))
        path, _ = QFileDialog.getSaveFileName(self, title, str(start / f"{stem}.{ext}"), filter_text)
        if not path:
            return None
        path = Path(path)
        if path.suffix.lower() != f".{ext}":
            path = path.with_name(path.name + f".{ext}")
        self.settings.setValue("export_dir", str(path.parent))
        return path

    def _export(self, func, path: Path | None) -> None:
        if path is None:
            return
        try:
            func(path)
            self.status_text.setText(f"Saved {path}")
        except Exception as exc:
            show_message(self, QMessageBox.Critical, "Could not save the file.", informative=str(exc))

    def _export_text(self, kind: str) -> None:
        segs = self.result.segments
        title = self.source_path.stem if self.source_path else None
        makers = {"txt": (exporters.transcript_text, "Text files (*.txt)"),
                  "srt": (exporters.transcript_srt, "Subtitles (*.srt)"),
                  "lrc": (lambda s: exporters.transcript_lrc(s, title), "Timed lyrics (*.lrc)")}
        make, filt = makers[kind]
        path = self._save_path("Save transcript", kind, filt)
        self._export(lambda p: exporters.write_text(p, make(segs)), path)

    def _visible_tracks(self):
        return [t for t in self.result.tracks if t.visible and t.notes]

    def _export_midi(self) -> None:
        path = self._save_path("Save notes as MIDI", "mid", "MIDI files (*.mid)")
        self._export(lambda p: exporters.save_midi(self._visible_tracks(), p, self.result.tempo), path)

    def _export_csv(self) -> None:
        path = self._save_path("Save notes as a spreadsheet", "csv", "CSV files (*.csv)")
        self._export(lambda p: exporters.write_text(p, exporters.notes_csv(self._visible_tracks())), path)

    def _export_stems(self) -> None:
        start = self.settings.value("export_dir", str(self.source_path.parent))
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder for the stems", start)
        if not folder:
            return
        stem = self.source_path.stem
        try:
            for name, src in self.result.audio_files.items():
                if name != "Original":
                    shutil.copyfile(src, Path(folder) / f"{stem} - {name}.wav")
            self.settings.setValue("export_dir", folder)
            self.status_text.setText(f"Saved stems to {folder}")
        except Exception as exc:
            show_message(self, QMessageBox.Critical, "Could not save the stems.", informative=str(exc))

    def _copy_transcript(self) -> None:
        if self.result and self.result.segments:
            QApplication.clipboard().setText(exporters.transcript_text(self.result.segments))
            self.status_text.setText("Transcript copied.")

    # Window events ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                if self._busy():
                    self.status_text.setText("Wait for the current analysis to finish first.")
                else:
                    self.open_file(url.toLocalFile())
                break

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.thread:
            answer = QMessageBox.question(self, APP_NAME, "An analysis is still running. Quit anyway?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.thread.stop()
            self.hide()
            self.thread.wait()
        self._save_settings()
        self.player.stop()
        self.player.setSource(QUrl())
        if self.result:
            remove_dir(self.result.work_dir)
        remove_dir(self._work_dir)
        event.accept()
