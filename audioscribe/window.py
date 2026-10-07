"""The main window."""

from __future__ import annotations

import bisect
import dataclasses
import html
import logging
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QFontDatabase, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QSplitter, QStackedWidget, QStyle, QTabWidget, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import APP_NAME, audio, exporters, i18n, mixer, playback, project, synth
from .app import cache_dir
from .engine import (LANGUAGE_NAMES, LANGUAGES, SIX_ONLY, STEM_ORDER, WHISPER_MODELS, Analyzer, Cancelled, Note,
                     Options, Result, cuda_available, friendly_error, remove_dir, stems_available)
from .i18n import tr, tr_n
from .music import NoteFilter, estimate_key, format_time, note_label, note_name, pitch_class_weights, set_naming
from .mixer_view import MixerPanel
from .piano_roll import PianoRoll
from .timeline import TracksView
from .widgets import (DATA_ROLE, SORT_ROLE, ElidedLabel, PartRow, ScaleFilterBox, SelectionView, SortItem,
                      SummaryView, swatch_icon)
from .update_ui import UpdateMixin
from .window_extras import ExtrasMixin

log = logging.getLogger(__name__)

OPEN_FILTER = ("Audio, video and projects (*.mp3 *.wav *.flac *.ogg *.oga *.opus *.m4a *.aac *.wma *.aif "
               "*.aiff *.alac *.mp4 *.m4v *.mkv *.webm *.mov *.avi *.ascribe);;Audio Scribe projects (*.ascribe);;"
               "All files (*)")
TAKE_OFFSETS = "~take offsets"     # key in an undo snapshot (part uids never look like this)
SNAP_CHOICES = [("Snap: off", 0), ("Snap: beat", 1), ("Snap: 1/2 beat", 2), ("Snap: 1/4 beat", 4)]
UNDO_LIMIT = 60


def _remove_file(path: str | Path) -> None:
    """Delete one of the app's own temporary files. Windows refuses while a player still has the
    file open, so a failure is fine: the whole work folder is cleared when the file is closed."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


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


class LaneRenderer(QThread):
    """Renders the parts' notes on their instruments (and the click track) into WAV files the
    live mixer plays, one at a time, in the background. A newer request for the same part
    replaces an older one, and stops it if it is already rendering."""

    rendered = Signal(str, str, str)    # uid, signature, file path
    progressed = Signal(float)
    failed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._jobs: dict[str, tuple[str, object, Path]] = {}     # uid -> (signature, job, file)
        self._want: dict[str, str] = {}
        self._quit = False

    def request(self, job, signature: str, path: Path) -> None:
        with self._cond:
            self._jobs[job.uid] = (signature, job, path)
            self._want[job.uid] = signature
            self._cond.notify()

    def forget_all(self) -> None:
        with self._cond:
            self._jobs.clear()
            self._want.clear()

    def busy(self) -> bool:
        with self._cond:
            return bool(self._jobs) or self._current is not None

    def stop(self) -> None:
        with self._cond:
            self._quit = True
            self._cond.notify()

    _current = None

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._jobs and not self._quit:
                    self._cond.wait()
                if self._quit:
                    return
                uid = next(iter(self._jobs))
                sig, job, path = self._jobs.pop(uid)
                self._current = uid

            def stale() -> bool:
                return self._quit or self._want.get(uid) != sig

            try:
                mixer.render(job, path, report=self.progressed.emit, should_stop=stale)
                if not stale():
                    self.rendered.emit(uid, sig, str(path))
            except mixer.Stopped:
                _remove_file(path)
            except Exception as exc:
                log.exception("Building the sound failed")
                _remove_file(path)
                self.failed.emit(str(exc), traceback.format_exc())
            finally:
                with self._cond:
                    self._current = None


class DecodeThread(QThread):
    """Turns the opened file into a WAV the player can read, before any analysis."""

    decoded = Signal(str, str, float)       # source path, wav path, duration
    failed = Signal(str, str)

    def __init__(self, source: Path, out: Path, parent=None):
        super().__init__(parent)
        self.source, self.out = source, out

    def run(self) -> None:
        try:
            data = audio.decode(self.source, playback.SR, 2)
            audio.write_wav(self.out, data, playback.SR)
            self.decoded.emit(str(self.source), str(self.out), data.shape[1] / playback.SR)
        except Exception as exc:
            log.warning("Could not prepare the file for playback: %s", exc)
            self.failed.emit(str(self.source), str(exc))


def show_message(parent, icon, text: str, details: str | None = None, informative: str | None = None) -> None:
    """Message box that never interprets file names or error text as HTML."""
    box = QMessageBox(icon, APP_NAME, text, QMessageBox.Ok, parent)
    box.setTextFormat(Qt.PlainText)
    if informative:
        box.setInformativeText(html.escape(informative))
    if details:
        box.setDetailedText(details)
    box.exec()


class Auditioner:
    """Plays a single note, or a short phrase, so edits and instruments can be heard right
    away. It plays through the live mixer, on top of the song if it is playing."""

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.bank = synth.Bank()
        self._cache: dict[tuple, object] = {}

    def _play(self, sound) -> None:
        self.window.player.audition(sound)

    def note(self, key: str, pitch: int, velocity: float = 0.75) -> None:
        try:
            k = ("note", key, pitch, int(velocity * 4))
            if k not in self._cache:
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[k] = synth.render_single(key, pitch, velocity, bank=self.bank)
            self._play(self._cache[k])
        except Exception:
            log.exception("Could not play a note")

    def rows(self, key: str, rows) -> None:
        """Play any list of (start, end, pitch, strength), for example chords from the Chords tab."""
        try:
            self._play(synth.render_preview(key, list(rows), bank=self.bank))
        except Exception:
            log.exception("Could not play the chords")

    def phrase(self, key: str) -> None:
        try:
            k = ("phrase", key)
            if k not in self._cache:
                self._cache[k] = synth.render_preview(key, bank=self.bank)
            self._play(self._cache[k])
        except Exception:
            log.exception("Could not play the instrument preview")


class MainWindow(ExtrasMixin, UpdateMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        self.settings = QSettings()
        i18n.set_language(str(self.settings.value("ui/language", "en")))
        set_naming(style=str(self.settings.value("ui/note_names", "letters")))
        self._init_extras_state()

        self.source_path: Path | None = None
        self.result: Result | None = None
        self.thread: AnalyzeThread | None = None
        self._work_dir: Path | None = None
        self._slider_held = False
        self._seg_starts: list[float] = []
        self._current_seg = -1
        self._stems_ok = stems_available()
        self._cuda_ok = cuda_available()

        self.note_filter = NoteFilter()
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._original: dict | None = None
        self._notes_dirty = False
        self._last_strength_push = 0.0

        self._decoders: set[DecodeThread] = set()
        self._clips: dict[tuple, playback.Clip] = {}     # recordings the mixer reads, by (file, start)
        self._notes_clips: dict[str, playback.Clip] = {}  # each part's notes on its instrument, by part uid
        self._lane_files: dict[str, str] = {}             # rendered file per signature
        self._lane_want: dict[str, str] = {}              # signature each part should sound like
        self._lane_loaded: dict[str, str] = {}            # signature each part sounds like now
        self._lane_serial = 0
        self._edit_hint_shown = False
        self._edit_count = 0     # goes up with every change to the notes
        self._saved_count = 0    # the value of _edit_count when the notes were last exported

        self._build_ui()
        self._wire_roll_extras()
        self._build_player()
        self._build_shortcuts()
        self._load_settings()
        self._update_controls()
        _clean_old_work_dirs()
        if i18n.language() != "en":
            i18n.retranslate(self, i18n.language())
            self._retranslate_dynamic()
        self._start_update_checks()

        self.audition = Auditioner(self)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self._do_refresh)
        self._sound_timer = QTimer(self)
        self._sound_timer.setSingleShot(True)
        self._sound_timer.setInterval(350)
        self._sound_timer.timeout.connect(self._apply_sound)

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
        col.addWidget(self._build_toolbar())

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(1)
        self.roll = PianoRoll()
        self.roll.seekRequested.connect(self.seek)
        self.roll.noteClicked.connect(self._on_roll_note_clicked)
        self.roll.pitchToggled.connect(self._on_pitch_toggled)
        self.roll.spanChanged.connect(self._on_span_changed)
        self.roll.selectionChanged.connect(self._on_note_selection_changed)
        self.roll.editStarted.connect(self._on_edit_started)
        self.roll.notesEdited.connect(self._on_notes_edited)
        self.roll.auditionRequested.connect(self._on_audition)
        self.roll.toolChanged.connect(self._set_tool)
        self.roll.status.connect(self.status_text_set)
        self.tracks_view = TracksView()
        self.tracks_view.seekRequested.connect(self.seek)
        self.tracks_view.spanChanged.connect(self._on_tracks_span)
        self.tracks_view.trackActivated.connect(self._edit_track_in_roll)
        self.tracks_view.mixChanged.connect(self._on_track_mix)
        self.tracks_view.partsChanged.connect(lambda _t: self._on_parts_changed())
        self.tracks_view.takeMenuRequested.connect(self._take_menu)
        self.tracks_view.takeMoved.connect(self._on_take_moved)
        self.tracks_view.canvas.snap = lambda t: self.roll.snap_time(t)
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.roll)
        self.view_stack.addWidget(self.tracks_view)
        self.splitter.addWidget(self.view_stack)
        self.splitter.addWidget(self._build_tabs())
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([520, 300])
        col.addWidget(self.splitter, 1)
        col.addWidget(self._build_transport())
        outer.addWidget(right, 1)
        self.setCentralWidget(root)

        self.status_text = QLabel(tr("Ready"))
        self.status_text.setProperty("i18n_skip", True)

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
        self.file_label.setProperty("i18n_skip", True)
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
        self._build_file_extras(l)
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
        self.chk_chords = QCheckBox("Find chords")
        self.chk_chords.setToolTip("Names the chords in the recording. They show above the piano roll and in the Chords tab.")
        l.addWidget(self.chk_chords)
        lay.addWidget(box)

        # Stems
        hint = ("Separates vocals, bass, drums, and the rest (and with 6 parts also piano and guitar) "
                "before analyzing. Slower, but gives much cleaner words and notes on full songs. "
                "Strings like violins stay in Other: no model can split them out yet.")
        if not self._stems_ok:
            hint = "Not installed. Run the installer again and choose stem separation to turn this on."
        box, l = self._section("Stems", hint)
        self.chk_stems = QCheckBox("Split into stems first")
        self.chk_stems.toggled.connect(self._update_controls)
        l.insertWidget(1, self.chk_stems)
        form = self._form()
        self.stem_count_box = self._compact(QComboBox())
        self.stem_count_box.addItem("4 parts", 4)
        self.stem_count_box.addItem("6 parts (adds piano, guitar)", 6)
        self.stem_count_box.setToolTip("4 parts: vocals, bass, drums, and everything else.\n"
                                       "6 parts: also takes piano and guitar out of everything else. "
                                       "Slower, and the first time it downloads another model.")
        self.stem_count_box.currentIndexChanged.connect(lambda _i: self._update_controls())
        form.addRow("Split into", self.stem_count_box)
        l.addLayout(form)
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
                chk.setToolTip("Finds drum hits (kick, snare, hi-hat) instead of pitches.")
            elif name in SIX_ONLY:
                chk.setToolTip("Only with 6 parts")
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
        self._build_update_section(lay)

        lay.addStretch(1)

        # The second page: what to show and hear. It fills in after an analysis.
        play = QWidget()
        play.setObjectName("Inspector")
        play_lay = QVBoxLayout(play)
        play_lay.setContentsMargins(0, 0, 0, 0)
        play_lay.setSpacing(0)

        self.play_empty = QLabel("The parts of the song, the scale filter, and the instrument choices show up "
                                 "here once the notes have been found. Press Analyze with Find notes turned on.")
        self.play_empty.setObjectName("Hint")
        self.play_empty.setWordWrap(True)
        self.play_empty.setContentsMargins(16, 16, 16, 16)
        play_lay.addWidget(self.play_empty)

        self.parts_box, self.parts_layout = self._section(
            "Parts", "M mutes a part (hides it and silences it). S solos it. "
                     "The instrument is what its notes sound like when played back.")
        self.parts_box.hide()
        play_lay.addWidget(self.parts_box)

        self.filter_box, l = self._section("Show only", "Choose a scale or a range of notes. "
                                                         "The Notes tab, the piano roll, and note playback follow it.")
        self.scale_filter = ScaleFilterBox()
        self.scale_filter.changed.connect(self._on_filter_changed)
        l.addWidget(self.scale_filter)
        self.filter_box.hide()
        play_lay.addWidget(self.filter_box)

        self._build_grid_section(play_lay)

        self.sound_box, l = self._section(
            "Sound", "Choose what plays in the bar at the bottom: the recording, "
                     "the notes on instruments, or both.")
        form = self._form()
        self.notes_level = QSlider(Qt.Horizontal)
        self.notes_level.setRange(0, 100)
        self.notes_level.setToolTip("How loud the instruments are when playing together with the recording.")
        self.notes_level.valueChanged.connect(lambda _v: self._queue_sound())
        form.addRow("Notes level", self.notes_level)
        l.addLayout(form)
        self.room_chk = QCheckBox("Add room sound")
        self.room_chk.setToolTip("A little reverb on the instruments, so they don't sound dry and flat.")
        self.room_chk.toggled.connect(lambda _on: self._queue_sound())
        l.addWidget(self.room_chk)
        self._build_click_controls(l, form)
        self.sound_box.hide()
        play_lay.addWidget(self.sound_box)
        play_lay.addStretch(1)

        def page(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setObjectName("InspectorPage")
            scroll.setWidget(widget)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            return scroll

        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setObjectName("InspectorTabs")
        self.inspector_tabs.setDocumentMode(True)
        self.inspector_tabs.tabBar().setExpanding(True)
        self.inspector_tabs.addTab(page(panel), "Analyze")
        self.inspector_tabs.addTab(page(play), "Parts and sound")
        frame = QFrame()
        frame.setObjectName("InspectorFrame")
        frame.setFixedWidth(316)
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.inspector_tabs)
        return frame

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
        self.title_label.setProperty("i18n_skip", True)
        self.meta_label = QLabel("Open a song, a voice memo, or a video to see its words and notes.")
        self.meta_label.setTextFormat(Qt.PlainText)
        self.meta_label.setObjectName("Meta")
        text.addWidget(self.title_label)
        text.addWidget(self.meta_label)
        l.addLayout(text, 1)
        self._build_header_extras(l)

        self.export_btn = QToolButton()
        self.export_btn.setText("Export")
        self.export_btn.setPopupMode(QToolButton.InstantPopup)
        self.export_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        menu = QMenu(self.export_btn)
        self.act_txt = menu.addAction("Transcript as text (.txt)", lambda: self._export_text("txt"))
        self.act_srt = menu.addAction("Transcript as subtitles (.srt)", lambda: self._export_text("srt"))
        self.act_lrc = menu.addAction("Lyrics with timing (.lrc)", lambda: self._export_text("lrc"))
        self.act_chord_txt = menu.addAction("Lyrics with chords (.txt)", lambda: self._export_chords("txt"))
        self.act_chordpro = menu.addAction("Lyrics with chords, ChordPro (.cho)", lambda: self._export_chords("cho"))
        menu.addSeparator()
        self.act_mid = menu.addAction("Notes as MIDI (.mid)", self._export_midi)
        self.act_csv = menu.addAction("Notes as spreadsheet (.csv)", self._export_csv)
        self.act_sheet = menu.addAction("Sheet music (PDF, print, MusicXML)...", self._open_sheet)
        menu.addSeparator()
        self.act_stems = menu.addAction("Stems as WAV files...", self._export_stems)
        menu.addAction("Bounce: mix or parts as WAV or FLAC (what you hear)...", self._export_sound)
        self.act_sound = menu.actions()[-1]
        menu.addSeparator()
        self.act_save_project = menu.addAction("Project (everything, to open again later)...", self._save_project)
        self.export_btn.setMenu(menu)
        self._fit_export_button()
        l.addWidget(self.export_btn)
        return header

    def _fit_export_button(self) -> None:
        """Leaves room for the little menu arrow next to the text."""
        text_w = self.export_btn.fontMetrics().horizontalAdvance(self.export_btn.text())
        self.export_btn.setMinimumWidth(text_w + 44)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Toolbar")
        l = QHBoxLayout(bar)
        l.setContentsMargins(12, 6, 12, 6)
        l.setSpacing(8)

        self.roll_view_btn = QToolButton()
        self.roll_view_btn.setObjectName("ViewButton")
        self.roll_view_btn.setText("Piano roll")
        self.roll_view_btn.setCheckable(True)
        self.roll_view_btn.setChecked(True)
        self.roll_view_btn.setToolTip("The notes of every part on one keyboard (F1)")
        self.tracks_view_btn = QToolButton()
        self.tracks_view_btn.setObjectName("ViewButton")
        self.tracks_view_btn.setText("Tracks")
        self.tracks_view_btn.setCheckable(True)
        self.tracks_view_btn.setToolTip("Every part as a lane with its waveform, volume and pan, like a DAW (F2)")
        self.roll_view_btn.clicked.connect(lambda: self._show_view(0))
        self.tracks_view_btn.clicked.connect(lambda: self._show_view(1))
        views = QHBoxLayout()
        views.setSpacing(0)
        views.addWidget(self.roll_view_btn)
        views.addWidget(self.tracks_view_btn)
        l.addLayout(views)
        l.addSpacing(6)

        self.edit_btn = QToolButton()
        self.edit_btn.setObjectName("ModeButton")
        self.edit_btn.setText("Edit notes")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setToolTip("Switch on to move, resize, add and delete notes (E).\n"
                                 "Nothing in the piano roll can be changed while this is off.")
        self.edit_btn.toggled.connect(self._set_edit_mode)
        l.addWidget(self.edit_btn)

        self.edit_box = QWidget()
        self.edit_box.setObjectName("Toolbar")
        el = QHBoxLayout(self.edit_box)
        el.setContentsMargins(12, 0, 12, 6)
        el.setSpacing(6)
        from PySide6.QtWidgets import QButtonGroup
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons: dict[str, QToolButton] = {}
        for i, (key, text, tip) in enumerate((
                ("select", "Select", "Select, move and resize notes. Drag on empty space to select a box (1)"),
                ("draw", "Draw", "Click to add a note, drag to set its length (2)"),
                ("erase", "Erase", "Click or drag over notes to remove them (3)"),
                ("split", "Split", "Click a note to cut it in two there (4)"),
                ("glue", "Glue", "Click a note to join it to the next note of the same pitch (5)"))):
            b = QToolButton()
            b.setObjectName("ToolButton")
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, k=key: self._set_tool(k))
            self.tool_group.addButton(b, i)
            self.tool_buttons[key] = b
            el.addWidget(b)
        self.tool_buttons["select"].setChecked(True)
        el.addSpacing(8)
        add_label = QLabel("Part")
        add_label.setObjectName("Hint")
        self.add_to_box = QComboBox()
        self.add_to_box.setToolTip("The part being edited. New and pasted notes go into it.")
        self.lock_chk = QCheckBox("Only this part")
        self.lock_chk.setChecked(True)
        self.lock_chk.setToolTip("Other parts are dimmed and can't be changed by accident")
        self.lock_chk.toggled.connect(lambda on: self.roll.set_lock_others(on))
        self.vel_chk = QCheckBox("Strength lane")
        self.vel_chk.setChecked(True)
        self.vel_chk.setToolTip("Show each note's strength as a bar under the notes. Drag the bars to change it.")
        self.vel_chk.toggled.connect(lambda on: self.roll.set_show_velocity(on))
        self.quantize_btn = QToolButton()
        self.quantize_btn.setText("Quantize")
        self.quantize_btn.setToolTip("Line the selected notes up with the grid (Q). Uses the Snap setting, "
                                     "or half beats when Snap is off.")
        self.quantize_btn.clicked.connect(lambda: self.roll._report_count(self.roll.quantize_selected(), "Quantized"))
        self.add_to_box.currentIndexChanged.connect(self._on_add_to_changed)
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(5, 100)
        self.strength_spin.setSuffix("%")
        self.strength_spin.setToolTip("Strength (how loud) of the selected notes")
        self.strength_spin.setEnabled(False)
        self.strength_spin.valueChanged.connect(self._on_strength_changed)
        self.undo_btn = QToolButton()
        self.undo_btn.setText("Undo")
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn = QToolButton()
        self.redo_btn.setText("Redo")
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo)
        self.revert_btn = QToolButton()
        self.revert_btn.setText("Revert all")
        self.revert_btn.setToolTip("Put every note back the way it was found")
        self.revert_btn.clicked.connect(self._revert_all)
        self.cleanup_btn = QToolButton()
        self.cleanup_btn.setText("Clean up...")
        self.cleanup_btn.setToolTip("Remove stray notes, join broken ones, line notes up with the beat")
        self.cleanup_btn.clicked.connect(self._open_cleanup)
        self.add_to_box.setProperty("i18n_skip_items", True)
        for w in (add_label, self.add_to_box, self.lock_chk, self.strength_spin, self.vel_chk, self.quantize_btn,
                  self.undo_btn, self.redo_btn, self.revert_btn, self.cleanup_btn):
            el.addWidget(w)
        el.addStretch(1)
        self.edit_box.hide()

        self.snap_box = QComboBox()
        for label, value in SNAP_CHOICES:
            self.snap_box.addItem(label, value)
        self.snap_box.setToolTip("Snap the selected span and moved notes to the beat grid")
        self.snap_box.currentIndexChanged.connect(self._on_snap_changed)
        l.addWidget(self.snap_box)
        l.addStretch(1)

        self.span_label = ElidedLabel("Drag across the piano roll to select a span")
        self.span_label.setProperty("i18n_skip", True)
        self.span_label.setObjectName("Meta")
        self.span_label.setTextFormat(Qt.PlainText)
        self.zoom_btn = QToolButton()
        self.zoom_btn.setText("Zoom to span")
        self.zoom_btn.setToolTip("Zoom in to the selected span to see it in more detail (Z)")
        self.zoom_btn.clicked.connect(self._toggle_zoom_span)
        self.clear_span_btn = QToolButton()
        self.clear_span_btn.setText("Clear")
        self.clear_span_btn.setToolTip("Clear the selected span (Esc)")
        self.clear_span_btn.clicked.connect(lambda: self.roll.clear_span())
        self.loop_btn = QToolButton()
        self.loop_btn.setObjectName("LoopButton")
        self.loop_btn.setText("Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setToolTip("Play the selected span over and over (L)")
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        for w in (self.span_label, self.zoom_btn, self.clear_span_btn, self.loop_btn):
            l.addWidget(w)
        self._update_span_controls(None)
        both = QWidget()
        both.setObjectName("Toolbar")
        rows = QVBoxLayout(both)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(0)
        bar.setStyleSheet("QWidget#Toolbar { border-bottom: none; }")
        rows.addWidget(bar)
        rows.addWidget(self.edit_box)
        return both

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
        self.lang_label.setProperty("i18n_skip", True)
        self.copy_btn = QPushButton("Copy text")
        self.copy_btn.clicked.connect(self._copy_transcript)
        bar.addWidget(self.lang_label, 1)
        bar.addWidget(self.copy_btn)
        self._build_transcript_extras(bar)
        pl.addLayout(bar)
        self.transcript = QTreeWidget()
        self.transcript.setHeaderLabels(["Time", "Words", "Translation"])
        self.transcript.setColumnHidden(2, True)
        self.transcript.setRootIsDecorated(False)
        self.transcript.setUniformRowHeights(False)
        self.transcript.setWordWrap(True)
        self.transcript.setAlternatingRowColors(True)
        self.transcript.setColumnWidth(0, 80)
        self.transcript.itemClicked.connect(self._on_transcript_clicked)
        pl.addWidget(self.transcript, 1)
        tabs.addTab(page, "Transcript")

        self.mixer_panel = MixerPanel()
        self.mixer_panel.mixChanged.connect(self._on_track_mix)
        self.mixer_panel.partsChanged.connect(lambda _t: self._on_parts_changed())
        tabs.addTab(self.mixer_panel, "Mixer")

        notes_page = QWidget()
        nl = QVBoxLayout(notes_page)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(0)
        nbar = QHBoxLayout()
        nbar.setContentsMargins(14, 6, 10, 6)
        self.notes_label = QLabel("")
        self.notes_label.setProperty("i18n_skip", True)
        self.notes_label.setTextFormat(Qt.PlainText)
        self.notes_label.setObjectName("Meta")
        self.span_only_chk = QCheckBox("Only the selected span")
        self.span_only_chk.setChecked(True)
        self.span_only_chk.setToolTip("With a span selected, list only the notes inside it")
        self.span_only_chk.toggled.connect(lambda _on: self._schedule_refresh())
        nbar.addWidget(self.notes_label, 1)
        nbar.addWidget(self.span_only_chk)
        nl.addLayout(nbar)
        self.notes_table = QTreeWidget()
        self.notes_table.setHeaderLabels(["Note", "Part", "Start", "Length", "Strength"])
        self.notes_table.setRootIsDecorated(False)
        self.notes_table.setAlternatingRowColors(True)
        self.notes_table.setUniformRowHeights(True)
        self.notes_table.setSortingEnabled(True)
        self.notes_table.itemClicked.connect(self._on_note_row_clicked)
        nl.addWidget(self.notes_table, 1)
        tabs.addTab(notes_page, "Notes")

        self.summary = SummaryView()
        self.summary.pitchSelected.connect(self._on_summary_pitch)
        tabs.addTab(self.summary, "Summary")

        self.selection = SelectionView()
        self.selection.pitchSelected.connect(self._on_summary_pitch)
        self.selection.seekRequested.connect(self.seek)
        tabs.addTab(self.selection, "Selection")
        self._build_chords_tab(tabs)
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
        self._build_record_button(l)

        self.clock = QLabel("0:00.0 / 0:00.0")
        self.clock.setProperty("i18n_skip", True)
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
        self.mode_box = QComboBox()
        for key, label in mixer.MODES:
            self.mode_box.addItem(label, key)
        self.mode_box.setToolTip("Recording: the original (or the parts that are not muted).\n"
                                 "Notes on instruments: the notes played with the instruments chosen in Parts.\n"
                                 "Recording and notes: both together, to check the notes against the song.")
        self.mode_box.currentIndexChanged.connect(lambda _i: self._on_mode_changed())
        l.addWidget(self.source_label)
        l.addWidget(self.mode_box)
        l.addSpacing(6)
        self._build_speed_controls(l)

        vol_icon = QLabel()
        vol_icon.setPixmap(st.standardIcon(QStyle.SP_MediaVolume).pixmap(16, 16))
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setFixedWidth(90)
        self.volume.setToolTip("Volume")
        self.volume.valueChanged.connect(lambda v: self.player.set_volume(v / 100.0))
        l.addSpacing(6)
        l.addWidget(vol_icon)
        l.addWidget(self.volume)

        self.follow_chk = QCheckBox("Follow")
        self.follow_chk.setToolTip("Keep the playhead in view while playing")
        self.follow_chk.toggled.connect(lambda on: (setattr(self.roll, "follow", on),
                                                    setattr(self.tracks_view.canvas, "follow", on)))
        l.addSpacing(8)
        l.addWidget(self.follow_chk)

        l.addSpacing(8)
        for text, tip, slot in (("-", "Zoom out (Ctrl+minus, or Ctrl+scroll)", lambda: self._zoom(1 / 1.5)),
                                ("+", "Zoom in (Ctrl+plus, or Ctrl+scroll)", lambda: self._zoom(1.5)),
                                ("Fit", "Fit the whole file (Ctrl+0)", self._fit_view)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            b.setMinimumWidth(28)
            l.addWidget(b)
        return bar

    def _build_player(self) -> None:
        self.player = playback.Player(self)
        self.player.stateChanged.connect(self._on_play_state)
        self.player.finished.connect(self._on_finished)
        self.player.failed.connect(self._on_player_error)
        self.lanes = LaneRenderer(self)
        self.lanes.rendered.connect(self._on_lane_rendered)
        self.lanes.progressed.connect(self._on_render_progress)
        self.lanes.failed.connect(self._on_render_failed)
        self.lanes.start()
        QApplication.instance().aboutToQuit.connect(lambda: (self.lanes.stop(), self.lanes.wait(10000)))
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
        add([QKeySequence.ZoomIn, "Ctrl+="], lambda: self._zoom(1.5))
        add(QKeySequence.ZoomOut, lambda: self._zoom(1 / 1.5))
        add("F1", lambda: self._show_view(0))
        add("F2", lambda: self._show_view(1))
        add("Ctrl+0", self._fit_view)

        def letter(slot):
            """Single letters stay out of the way while a list or box has the keyboard (they pick items there)."""
            def run():
                if isinstance(QApplication.focusWidget(), (QComboBox, QAbstractSpinBox, QLineEdit)):
                    return
                slot()
            return run

        add("E", letter(lambda: self.edit_btn.isEnabled() and self.edit_btn.toggle()))
        add("L", letter(lambda: self.loop_btn.isEnabled() and self.loop_btn.toggle()))
        add("Z", letter(self._toggle_zoom_span))
        add("Ctrl+Z", self.undo)
        add(["Ctrl+Y", "Ctrl+Shift+Z"], self.redo)
        add(QKeySequence.Save, self._save_project)
        add("T", letter(lambda: self.result and self._tap()))
        add("R", letter(lambda: self.rec_btn.isEnabled() and self._toggle_take()))

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
        self.chk_chords.setChecked(get_bool("notes/chords", True))
        self.click_chk.setChecked(get_bool("sound/click", False))
        self.click_level.setValue(int(s.value("sound/click_level", 60)))
        self._set_combo(self.names_box, str(s.value("ui/note_names", "letters")))
        self.lang_combo.blockSignals(True)
        self._set_combo(self.lang_combo, i18n.language())
        self.lang_combo.blockSignals(False)
        self.chk_stems.setChecked(get_bool("stems/enabled", False) and self._stems_ok)
        self._set_combo(self.stem_count_box, int(s.value("stems/count", 4)))
        chosen = s.value("stems/note_stems", "Vocals,Piano,Guitar,Bass,Other,Drums")
        chosen = chosen if isinstance(chosen, str) else ",".join(chosen)
        if s.value("stems/count") is None:      # saved by a version without piano and guitar stems
            chosen += ",Piano,Guitar"
        for name, chk in self.stem_checks.items():
            chk.setChecked(name in chosen.split(","))
        device = s.value("device", "cpu")
        self._set_combo(self.device_box, device if (device != "cuda" or self._cuda_ok) else "cpu")
        self.volume.setValue(int(s.value("volume", 85)))
        self.player.set_volume(self.volume.value() / 100.0)
        self.follow_chk.setChecked(get_bool("follow", True))
        self._set_combo(self.snap_box, int(s.value("edit/snap", 0)))
        self.roll.snap_div = self.snap_box.currentData()
        self.notes_level.setValue(int(s.value("sound/notes_level", 70)))
        self.room_chk.setChecked(get_bool("sound/room", True))
        self.loop_btn.setChecked(get_bool("loop", False))
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
        s.setValue("notes/chords", self.chk_chords.isChecked())
        s.setValue("sound/click", self.click_chk.isChecked())
        s.setValue("sound/click_level", self.click_level.value())
        s.setValue("stems/enabled", self.chk_stems.isChecked())
        s.setValue("stems/note_stems", ",".join(n for n, c in self.stem_checks.items() if c.isChecked()))
        s.setValue("stems/count", self.stem_count_box.currentData())
        s.setValue("device", self.device_box.currentData())
        s.setValue("volume", self.volume.value())
        s.setValue("follow", self.follow_chk.isChecked())
        s.setValue("edit/snap", self.snap_box.currentData())
        s.setValue("sound/notes_level", self.notes_level.value())
        s.setValue("sound/room", self.room_chk.isChecked())
        s.setValue("loop", self.loop_btn.isChecked())
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

    def _tracks(self):
        return self.result.tracks if self.result else []

    def _has_notes(self) -> bool:
        return any(t.notes for t in self._tracks())

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
        self.chk_chords.setEnabled(not busy)
        self.chk_stems.setEnabled(self._stems_ok and not busy)
        stems = self.chk_stems.isChecked() and self._stems_ok and not busy
        self.stems_label.setEnabled(stems and notes)
        self.stem_count_box.setEnabled(stems)
        six = self.stem_count_box.currentData() == 6
        for name, chk in self.stem_checks.items():
            chk.setEnabled(stems and notes and (six or name not in SIX_ONLY))
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
        self.act_sound.setEnabled(bool(r) and not busy)
        self.export_btn.setEnabled(bool(r) and not busy)
        self.copy_btn.setEnabled(has_words)

        has_tracks = bool(r and r.tracks)
        self.mode_box.setEnabled(has_notes)
        self.source_label.setVisible(has_tracks)
        self.mode_box.setVisible(has_tracks)
        self.edit_btn.setEnabled(has_tracks and not busy)
        if not self.edit_btn.isEnabled() and self.edit_btn.isChecked():
            self.edit_btn.setChecked(False)
        for box in (self.filter_box, self.sound_box, self.parts_box):
            box.setVisible(has_tracks)
        self.play_empty.setVisible(not has_tracks)
        self._update_undo_buttons()
        self._extras_update_controls()

    # Opening files --------------------------------------------------------------

    def _open_dialog(self) -> None:
        if self._busy():
            return
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, tr("Open audio, video or a project"), start, tr(OPEN_FILTER))
        if path:
            self.open_file(path)

    def open_file(self, path: str | Path) -> None:
        if self._busy():
            return
        path = Path(path)
        if not path.is_file():
            show_message(self, QMessageBox.Warning, tr("Could not find {path}.").format(path=path))
            return
        if path.suffix.lower() == project.SUFFIX:
            self._open_project(path)
            return
        if not self._confirm_discard_edits():
            return
        self.settings.setValue("last_dir", str(path.parent))
        self.source_path = path
        self._clear_results()
        self.title_label.setText(path.name)
        self.meta_label.setText(tr("Press Analyze to find the words and notes."))
        self.file_label.setText(path.name)
        self.setWindowTitle(f"{path.name} - {APP_NAME}")
        self.roll.clear(tr("Press Analyze to find the words and notes."))
        self._prepare_preview(path)
        self._update_controls()

    def _clear_results(self) -> None:
        old = self.result
        self.result = None
        self.lanes.forget_all()
        self.player.mix.set_lanes([])
        with self.player.mix.lock:
            self.player.mix.click = None
        self._clips.clear()
        self._notes_clips.clear()
        self._lane_files.clear()
        self._lane_want.clear()
        self._lane_loaded.clear()
        self._undo.clear()
        self._redo.clear()
        self._original = None
        self._notes_dirty = False
        self._edit_hint_shown = False
        self._edit_count = self._saved_count = 0
        self.edit_btn.setChecked(False)
        self.transcript.clear()
        self.notes_table.clear()
        self.notes_label.setText("")
        self.summary.clear()
        self.selection.show_none()
        self._seg_starts, self._current_seg = [], -1
        self.lang_label.setText(tr("Click a line to jump to it."))
        self._fill_parts([])
        self.tracks_view.clear()
        self._extras_clear()
        self.inspector_tabs.setCurrentIndex(0)
        self.note_filter = NoteFilter()
        self.scale_filter.reset()
        self.scale_filter.set_detected_key(None)
        self.roll.set_filter(self.note_filter)
        self._update_span_controls(None)
        if old:
            QTimer.singleShot(2000, lambda d=old.work_dir: remove_dir(d))

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
            note_stems=[n for n, c in self.stem_checks.items() if c.isChecked()
                        and (self.stem_count_box.currentData() == 6 or n not in SIX_ONLY)],
            stem_count=self.stem_count_box.currentData() or 4,
            chords=self.chk_chords.isChecked(),
            device=self.device_box.currentData(),
        )

    def start_analysis(self) -> None:
        if self._busy():
            return
        if self.source_path is None:
            self._open_dialog()
            return
        o = self._options()
        if not o.transcribe and not o.notes and not o.chords:
            show_message(self, QMessageBox.Information, tr("Turn on Transcribe words, Find notes or Find chords first."))
            return
        if o.notes and o.separate and not o.note_stems:
            show_message(self, QMessageBox.Information, tr("Pick at least one stem to find notes in."))
            return
        if self.source_path is not None and self.source_path.suffix.lower() == project.SUFFIX:
            show_message(self, QMessageBox.Information, tr("This is a project, so it is already analyzed. "
                                                           "To analyze again, open the original audio file."))
            return
        if not self._confirm_discard_edits(tr("Analyzing again replaces the notes you edited.")):
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
        self.status_text.setText(tr("Starting"))
        self.thread.start()
        self._update_controls()

    def cancel_analysis(self) -> None:
        if self.thread:
            self.thread.stop()
            self.cancel_btn.setEnabled(False)
            self.status_text.setText(tr("Stopping after the current step"))

    def _on_progress(self, fraction: float, text: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        self.status_text.setText(tr(text))

    def _on_success(self, result: Result) -> None:
        self._work_dir = None
        self._show_result(result)
        secs = int(round(result.elapsed))
        took = f"{secs // 60} min {secs % 60} s" if secs >= 60 else f"{secs} s"
        self.status_text.setText(tr("Done in {time}.").format(time=took))

    def _on_failed(self, message: str, details: str) -> None:
        remove_dir(self._work_dir)
        self._work_dir = None
        self.status_text.setText(tr("Analysis failed."))
        show_message(self, QMessageBox.Critical, tr(message), details=details,
                     informative=tr("Your file was not changed. Full details are below."))

    def _on_cancelled(self) -> None:
        remove_dir(self._work_dir)
        self._work_dir = None
        self.status_text.setText(tr("Analysis cancelled."))

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
        for t in r.tracks:
            saved = self.settings.value(f"instrument/{t.name}")
            if saved in synth.BY_KEY and (t.name != "Drums" or saved == synth.DRUM_KEY):
                t.instrument = saved
        self._apply_visibility(refresh=False)
        self.roll.set_data(r.duration, r.tracks, r.segments, r.beats)
        self.roll.set_beats(r.beats, r.meter)
        self.roll.set_filter(self.note_filter)
        if not r.tracks and not r.segments and not r.chords:
            self.roll.set_message(tr("Nothing was found in this file."))
        elif not any(t.notes for t in r.tracks):
            self.roll.set_message(tr("Turn on Find notes to see them here."))

        self._update_meta()
        self._fill_transcript(r)
        self.summary.set_result(r)
        self._fill_parts(r.tracks)
        self.tracks_view.set_data(r.duration, r.tracks, r.beats, r.meter)
        if r.tracks:
            self.inspector_tabs.setCurrentIndex(1)
        self.scale_filter.set_detected_key(r.key)
        self._original = self._snapshot()
        self._refresh_edit_tracks()
        self._fill_notes_table()

        self.player.set_duration(r.duration)
        self._set_combo(self.mode_box, "recording")
        self._rebuild_lanes()
        self.pos_slider.setRange(0, int(r.duration * 1000))
        self._sync_position(self.player.position())
        self._extras_show_result(r)
        self._update_controls()

    def _update_meta(self) -> None:
        r = self.result
        if not r:
            return
        parts = [tr("{time} long").format(time=format_time(r.duration, 0))]
        if self.song_key:
            from .music import key_name
            parts.append(tr("in {key}").format(key=key_name(*self.song_key)))
        elif r.key:
            parts.append(tr("probably {key}").format(key=r.key.name))
        if r.tempo:
            parts.append(tr("around {bpm} BPM").format(bpm=f"{r.tempo:.0f}"))
            if r.meter != 4:
                parts.append(f"{r.meter}/4")
        text = ", ".join(parts)
        self.meta_label.setText(text[:1].upper() + text[1:])

    def _fill_transcript(self, r: Result) -> None:
        self.transcript.clear()
        items = []
        lines = self.translation["lines"] if self.translation else []
        for i, seg in enumerate(r.segments):
            item = QTreeWidgetItem([format_time(seg.start, 1), seg.text, lines[i] if i < len(lines) else ""])
            item.setData(0, DATA_ROLE, i)
            item.setTextAlignment(0, Qt.AlignRight | Qt.AlignTop)
            items.append(item)
        self.transcript.addTopLevelItems(items)
        self.transcript.setColumnWidth(1, 520 if lines else 800)
        self._seg_starts = [s.start for s in r.segments]
        self._current_seg = -1
        self._apply_transcript_view()
        self._refresh_transcript_label()

    def _refresh_transcript_label(self) -> None:
        r = self.result
        if not r:
            return
        if r.words_requested:
            if r.segments:
                lang = tr(LANGUAGE_NAMES.get(r.language or "", (r.language or "").upper()))
                how = tr("Translated to English from {lang}.") if r.translated else tr("Language: {lang}.")
                self.lang_label.setText(how.format(lang=lang) + " " + tr("Click a line to jump to it."))
            else:
                self.lang_label.setText(tr("No words were found."))
        else:
            self.lang_label.setText(tr("Transcription was off for this run."))

    def _fill_notes_table(self) -> None:
        """Lists the notes that are shown: not muted, passing the scale filter, and (when
        'Only the selected span' is on) inside the span."""
        r = self.result
        table = self.notes_table
        table.setUpdatesEnabled(False)
        table.setSortingEnabled(False)
        table.clear()
        if not r:
            table.setUpdatesEnabled(True)
            self.notes_label.setText("")
            return
        span = self.roll.span if self.span_only_chk.isChecked() else None
        flt = self.note_filter
        items = []
        total = 0
        for ti, track in enumerate(r.tracks):
            if not track.visible:
                continue
            for ni, n in enumerate(track.notes):
                total += 1
                if not flt.allows(n.pitch, track.name == "Drums"):
                    continue
                if span and (n.end < span[0] or n.start > span[1]):
                    continue
                length = n.end - n.start
                item = SortItem([note_label(n.pitch, track.name), tr(track.name), format_time(n.start, 2),
                                 f"{length:.2f} s", f"{n.velocity * 100:.0f}%"])
                item.setData(0, SORT_ROLE, n.pitch)
                item.setData(1, SORT_ROLE, ti)
                item.setData(2, SORT_ROLE, n.start)
                item.setData(3, SORT_ROLE, length)
                item.setData(4, SORT_ROLE, n.velocity)
                item.setData(0, DATA_ROLE, (ti, ni))
                item.setIcon(1, swatch_icon(track.color, 10))
                items.append(item)
        table.addTopLevelItems(items)
        table.setSortingEnabled(True)
        table.sortItems(2, Qt.AscendingOrder)
        for col, width in ((0, 80), (1, 110), (2, 90), (3, 80)):
            table.setColumnWidth(col, width)
        table.setUpdatesEnabled(True)

        text = tr_n(len(items), "{n} note shown", "{n} notes shown")
        hidden = []
        if len(items) != total:
            text += " " + tr("of {total}").format(total=total)
        if flt.active:
            hidden.append(flt.describe())
        if span:
            hidden.append(tr("inside {a} to {b}").format(a=format_time(span[0], 1), b=format_time(span[1], 1)))
        if hidden:
            text += " (" + ", ".join(hidden) + ")"
        self.notes_label.setText(text)

    # Parts: mute, solo, instrument -------------------------------------------------

    def _fill_parts(self, tracks) -> None:
        while self.parts_layout.count() > 2:
            item = self.parts_layout.takeAt(2)
            if item.widget():
                item.widget().deleteLater()
        self.part_rows: list[PartRow] = []
        for track in tracks:
            row = PartRow(track)
            if i18n.language() != "en":
                i18n.retranslate(row, i18n.language())
            row.changed.connect(self._on_parts_changed)
            row.mixChanged.connect(self._on_track_mix)
            row.instrumentPicked.connect(lambda key, t=track: self._on_instrument_picked(t, key))
            self.parts_layout.addWidget(row)
            self.part_rows.append(row)
        self.mixer_panel.set_tracks(tracks)
        if i18n.language() != "en":
            i18n.retranslate(self.mixer_panel, i18n.language())
        self.parts_box.setVisible(bool(tracks))
        self.play_empty.setVisible(not tracks)

    def _apply_visibility(self, refresh: bool = True) -> None:
        tracks = self._tracks()
        solo = any(t.solo for t in tracks)
        for t in tracks:
            t.visible = (not t.muted) and (t.solo or not solo)
        if refresh:
            self.roll.refresh()
            self.tracks_view.refresh()

    def _on_parts_changed(self) -> None:
        self._apply_visibility()
        self._refresh_edit_tracks()
        self._schedule_refresh()
        self._queue_sound()
        self._sync_mix_views()

    def _on_track_mix(self, _track) -> None:
        """A part's volume or pan moved: heard at once, and shown everywhere it appears."""
        self._apply_mix()
        self._sync_mix_views()

    def _sync_mix_views(self) -> None:
        for row in getattr(self, "part_rows", []):
            row.sync()
        self.mixer_panel.sync()
        if hasattr(self, "tracks_view"):
            self.tracks_view.sync_headers()

    def _on_instrument_picked(self, track, key: str) -> None:
        self.settings.setValue(f"instrument/{track.name}", key)
        self.audition.phrase(key)

    def _refresh_edit_tracks(self) -> None:
        box = self.add_to_box
        current = box.currentData()
        box.blockSignals(True)
        box.clear()
        tracks = self._tracks()
        for t in tracks:
            if t.visible:
                box.addItem(tr(t.name), t.uid)       # by uid: indexes change when a take is removed
        pick = box.findData(current) if current is not None else -1
        if pick < 0:
            drums = {t.uid for t in tracks if t.drums}
            pitched = [i for i in range(box.count()) if box.itemData(i) not in drums]
            pick = pitched[0] if pitched else 0
        if box.count():
            box.setCurrentIndex(pick)
        box.blockSignals(False)
        self.roll.set_edit_track(self._edit_index())

    def _edit_index(self) -> int:
        """Index of the part chosen in Part (the one being edited)."""
        uid = self.add_to_box.currentData()
        for ti, t in enumerate(self._tracks()):
            if t.uid == uid:
                return ti
        return 0

    def _on_add_to_changed(self, _index: int) -> None:
        self.roll.set_edit_track(self._edit_index())

    def _set_tool(self, key: str) -> None:
        self.roll.set_tool(key)
        self.tool_buttons[key].setChecked(True)
        hints = {"select": "Select tool: drag notes to move them, drag their right edge to resize.",
                 "draw": "Draw tool: click to add a note to the part being edited, drag to set its length.",
                 "erase": "Erase tool: click or drag over notes to remove them.",
                 "split": "Split tool: click a note where it should be cut in two.",
                 "glue": "Glue tool: click a note to join it to the next note of the same pitch."}
        self.status_text.setText(tr(hints[key]))

    # Filter ---------------------------------------------------------------------

    def _on_filter_changed(self, flt: NoteFilter) -> None:
        self.note_filter = flt
        self.roll.set_filter(flt)
        self._schedule_refresh()
        self._queue_sound()
        if self.result and (self.chords.lane_source() == "notes" or not self.result.chords):
            self._update_chord_lane()
        if flt.active:
            self.status_text.setText(tr("Showing {what}.").format(what=flt.describe()))
        else:
            self.status_text.setText(tr("Showing every note."))

    # Span, loop, zoom ---------------------------------------------------------------

    def _on_span_changed(self, span, final: bool) -> None:
        self._update_span_controls(span)
        if not final:
            return
        self._update_selection_details()
        self._schedule_refresh()
        if span:
            self.status_text.setText(tr("Span {a} to {b} selected. See the Selection tab for details, "
                                        "press Z to zoom, L to loop.").format(a=format_time(span[0], 2),
                                                                             b=format_time(span[1], 2)))
        else:
            self.status_text.setText(tr("Span cleared."))

    def _update_span_controls(self, span) -> None:
        has = span is not None
        if hasattr(self, "tracks_view"):
            self.tracks_view.set_span(span, self.loop_btn.isChecked() and has)
        self.zoom_btn.setEnabled(has or self.roll_saved_view())
        self.clear_span_btn.setEnabled(has)
        self.loop_btn.setEnabled(has)
        if has:
            a, b = span
            self.span_label.setText(tr("Span {a} to {b}").format(a=format_time(a, 2), b=format_time(b, 2))
                                    + f"  ({b - a:.2f} s)")
        else:
            self.span_label.setText(tr("Drag across the piano roll to select a span"))
        self.span_label.setToolTip(self.span_label.text())
        self.zoom_btn.setText(tr("Zoom back") if self.roll_saved_view() else tr("Zoom to span"))
        if hasattr(self, "roll"):
            self.roll.set_loop_visual(self.loop_btn.isChecked() and has)
        if hasattr(self, "player"):
            self.player.set_loop(self._loop_span())

    def roll_saved_view(self) -> bool:
        return hasattr(self, "roll") and self.roll.saved_view is not None

    def _toggle_zoom_span(self) -> None:
        if self.roll.saved_view is not None:
            self.roll.restore_view()
        elif self.roll.span:
            self.roll.zoom_to_span()
        self._update_span_controls(self.roll.span)

    def _fit_view(self) -> None:
        if self.view_stack.currentIndex() == 1:
            self.tracks_view.canvas.fit()
            return
        self.roll.fit()
        self._update_span_controls(self.roll.span)

    def _zoom(self, factor: float) -> None:
        if self.view_stack.currentIndex() == 1:
            self.tracks_view.canvas.zoom_time(factor)
        else:
            self.roll.zoom_time(factor)

    def _show_view(self, index: int) -> None:
        self.view_stack.setCurrentIndex(index)
        self.roll_view_btn.setChecked(index == 0)
        self.tracks_view_btn.setChecked(index == 1)
        if index == 1:
            self.tracks_view.set_playhead(self.player.position())

    def _edit_track_in_roll(self, ti: int) -> None:
        """Double-click on a part in the Tracks view: edit its notes in the piano roll."""
        tracks = self._tracks()
        if not (0 <= ti < len(tracks)):
            return
        self._show_view(0)
        if not tracks[ti].visible:
            self.status_text.setText(tr("{part} is muted. Unmute it to edit its notes.").format(part=tr(tracks[ti].name)))
            return
        if self.edit_btn.isEnabled() and not self.edit_btn.isChecked():
            self.edit_btn.setChecked(True)
        i = self.add_to_box.findData(tracks[ti].uid)
        if i >= 0:
            self.add_to_box.setCurrentIndex(i)

    def _on_tracks_span(self, span, final: bool) -> None:
        if span is None:
            self.roll.clear_span()
        else:
            self.roll.set_span(span[0], span[1], final)

    def _on_loop_toggled(self, on: bool) -> None:
        self._update_span_controls(self.roll.span)
        if on and self.roll.span:
            self.status_text.setText(tr("Looping the selected span. Press L or click Loop again to turn it off."))
        elif not on:
            self.status_text.setText(tr("Loop off."))

    def _loop_span(self) -> tuple[float, float] | None:
        if self.loop_btn.isChecked() and self.roll.span:
            return self.roll.span
        return None

    def _update_selection_details(self) -> None:
        r = self.result
        span = self.roll.span
        if not r or not span:
            self.selection.show_none()
            return
        self.selection.show_span(span[0], span[1], r.tracks, self.note_filter, r.beats, r.segments, r.meter,
                                 self._current_lane())

    # Editing --------------------------------------------------------------------

    def status_text_set(self, text: str) -> None:
        self.status_text.setText(text)

    def _set_edit_mode(self, on: bool) -> None:
        self.edit_box.setVisible(on)
        self.roll.set_edit_mode(on)
        self.chords.set_edit_mode(on)
        if on:
            self._refresh_edit_tracks()
            self.status_text.setText(tr(
                "Edit mode. Drag a note to move it, drag its right edge to resize, double-click empty space "
                "to add one, right-click for copy, paste, quantize and more. Hold Shift and drag for a span."))
        else:
            self.status_text.setText(tr("Edit mode off."))
        self._update_undo_buttons()

    def _on_snap_changed(self, _index: int) -> None:
        self.roll.snap_div = self.snap_box.currentData() or 0

    def _snapshot(self) -> dict:
        """Every part's notes, by part uid (names can change: a take can be renamed), and
        where each take starts, since moving a take moves its notes too."""
        snap = {t.uid: [(n.start, n.end, n.pitch, n.velocity, n.edited) for n in t.notes] for t in self._tracks()}
        snap[TAKE_OFFSETS] = {t.uid: t.offset for t in self._tracks() if t.take}
        return snap

    def _restore(self, snap: dict) -> None:
        moved = False
        offsets = snap.get(TAKE_OFFSETS, {})
        for t in self._tracks():
            if t.uid in snap:
                t.notes[:] = [Note(*row) for row in snap[t.uid]]
            if t.uid in offsets and abs(offsets[t.uid] - t.offset) > 1e-9:
                t.offset = offsets[t.uid]
                moved = True
        if moved:
            self._rebuild_lanes()
        self.roll.selected = set()
        self.roll.reindex()
        self.roll.selectionChanged.emit()
        self._after_notes_changed()

    def _on_edit_started(self) -> None:
        self._undo.append(self._snapshot())
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()
        self._update_undo_buttons()

    def _on_notes_edited(self) -> None:
        self._after_notes_changed()
        if not self._edit_hint_shown and self.mode_box.currentData() == "recording":
            # The recording itself can't change, so tell people where their edits can be heard.
            self._edit_hint_shown = True
            self.status_text.setText(tr("Your changes show in the piano roll. Set Play (bottom bar) to "
                                        "Notes on instruments to hear them."))

    def _after_notes_changed(self) -> None:
        self.tracks_view.refresh()
        self._edit_count += 1
        self._notes_dirty = True
        self._schedule_refresh()
        self._queue_sound()
        self._update_undo_buttons()

    def undo(self) -> None:
        if not self.roll.edit_mode or not self._undo:
            return
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        self.status_text.setText(tr("Undone."))

    def redo(self) -> None:
        if not self.roll.edit_mode or not self._redo:
            return
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        self.status_text.setText(tr("Redone."))

    def _revert_all(self) -> None:
        if not self._original or not self._undo:
            return
        answer = QMessageBox.question(self, APP_NAME, tr("Put every note back the way it was found? "
                                                         "You can still undo this."))
        if answer != QMessageBox.Yes:
            return
        self._on_edit_started()
        self._restore(self._original)
        self.status_text.setText(tr("Every note is back the way it was found."))

    def _update_undo_buttons(self) -> None:
        editing = self.roll.edit_mode if hasattr(self, "roll") else False
        self.undo_btn.setEnabled(editing and bool(self._undo))
        self.redo_btn.setEnabled(editing and bool(self._redo))
        self.revert_btn.setEnabled(editing and bool(self._undo))

    def _has_edits(self) -> bool:
        """True when notes were changed and not exported since."""
        return self._edit_count != self._saved_count

    def _confirm_discard_edits(self, reason: str | None = None) -> bool:
        if not self._has_edits():
            return True
        reason = reason or tr("Opening another file closes this one.")
        box = QMessageBox(QMessageBox.Warning, APP_NAME, tr("You have edited notes that are not saved."),
                          QMessageBox.NoButton, self)
        box.setInformativeText(reason + " " + tr("Save the project (Ctrl+S) or export the notes first, "
                                                 "or continue and lose the changes."))
        keep = box.addButton(tr("Go back"), QMessageBox.RejectRole)
        box.addButton(tr("Continue and lose the changes"), QMessageBox.DestructiveRole)
        box.setDefaultButton(keep)
        box.exec()
        return box.clickedButton() is not keep

    def _on_note_selection_changed(self) -> None:
        notes = self.roll.selected_notes()
        self.strength_spin.setEnabled(bool(notes) and self.roll.edit_mode)
        if notes:
            self.strength_spin.blockSignals(True)
            self.strength_spin.setValue(int(round(sum(n.velocity for n in notes) / len(notes) * 100)))
            self.strength_spin.blockSignals(False)
        if self.roll.edit_mode and notes:
            if len(notes) == 1:
                n = notes[0]
                self.status_text.setText(tr("{note} at {time}, {length} s long. Drag to move, drag the right edge "
                                            "to resize, Delete to remove.").format(
                    note=note_name(n.pitch), time=format_time(n.start, 2), length=f"{n.end - n.start:.2f}"))
            else:
                self.status_text.setText(tr("{n} notes selected. Drag to move them together, "
                                            "arrow keys nudge them, Delete removes them.").format(n=len(notes)))

    def _on_strength_changed(self, value: int) -> None:
        if not self.roll.selected:
            return
        # Changing the strength by clicking the arrows many times in a row counts as one
        # undo step, so only save the "before" state when the last change was a while ago.
        now = time.monotonic()
        if now - self._last_strength_push > 1.2:
            self._on_edit_started()
        self._last_strength_push = now
        self.roll.set_selected_strength(value / 100.0, push=False)
        self._after_notes_changed()

    def _on_audition(self, pitch: int, track_index: int) -> None:
        tracks = self._tracks()
        if 0 <= track_index < len(tracks):
            key = tracks[track_index].instrument
        else:
            visible = [t for t in tracks if t.visible and t.name != "Drums"]
            key = visible[0].instrument if visible else synth.DEFAULT_INSTRUMENT
        self.audition.note(key, pitch)

    # Showing and hiding work --------------------------------------------------------

    def _schedule_refresh(self) -> None:
        if hasattr(self, "_refresh_timer"):
            self._refresh_timer.start()

    def _do_refresh(self) -> None:
        r = self.result
        if not r:
            return
        self._fill_notes_table()
        if hasattr(self, "part_rows"):
            for row in self.part_rows:
                row.refresh_name()
        if self.roll.span:
            self._update_selection_details()
        if self._notes_dirty:
            self._notes_dirty = False
            pitched = [n for t in r.tracks if t.name != "Drums" for n in t.notes]
            r.key = estimate_key(pitch_class_weights(pitched)) if pitched else None
            self.summary.set_result(r)
            self.scale_filter.set_detected_key(r.key)
            self.chords._song_key = (r.key.tonic, r.key.mode) if r.key else self.chords._song_key
            self._update_meta()
            if self.chords.lane_source() == "notes" or not r.chords:
                self._update_chord_lane()

    # Interaction between views ------------------------------------------------------

    def _on_transcript_clicked(self, item: QTreeWidgetItem) -> None:
        if self.result:
            self.seek(self.result.segments[item.data(0, DATA_ROLE)].start)

    def _on_note_row_clicked(self, item: QTreeWidgetItem) -> None:
        if not self.result:
            return
        ti, ni = item.data(0, DATA_ROLE)
        tracks = self.result.tracks
        if ti >= len(tracks) or ni >= len(tracks[ti].notes):
            return
        self.roll.select_note(ti, ni)
        self.seek(tracks[ti].notes[ni].start)

    def _on_roll_note_clicked(self, ti: int, ni: int) -> None:
        n = self.result.tracks[ti].notes[ni] if self.result else None
        if n:
            track = self.result.tracks[ti]
            self.status_text.setText(tr("{note} at {time}, {length} s long ({part})").format(
                note=note_label(n.pitch, track.name), time=format_time(n.start, 2),
                length=f"{n.end - n.start:.2f}", part=tr(track.name)))

    def _on_pitch_toggled(self, pitch) -> None:
        self.summary.select_pitch(pitch)
        self.selection.select_pitch(pitch)
        self._describe_pitch(pitch)

    def _on_summary_pitch(self, pitch) -> None:
        self.roll.set_highlight(pitch)
        self._describe_pitch(pitch)

    def _describe_pitch(self, pitch) -> None:
        if pitch is None or not self.result:
            self.status_text.setText(tr("Highlight cleared."))
            return
        count = sum(1 for t in self.result.tracks if t.visible for n in t.notes if n.pitch == pitch)
        self.status_text.setText(tr("Highlighting {note}, played {times}. Click the key again or press Esc to clear.")
                                 .format(note=note_name(pitch), times=tr_n(count, "{n} time", "{n} times")))

    # Playback ---------------------------------------------------------------------

    def _prepare_preview(self, path: Path) -> None:
        """Before an analysis the file itself plays. It is turned into a WAV first."""
        self.player.pause()
        self.player.mix.set_lanes([])
        self.player.set_duration(0.0)
        self.player.seek(0.0)
        self.pos_slider.setRange(0, 0)
        folder = _work_root() / "preview"
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.wav"):
            _remove_file(old)
        out = folder / f"preview-{int(time.time() * 1000)}.wav"
        thread = DecodeThread(path, out, self)
        thread.decoded.connect(self._on_preview_ready)
        thread.failed.connect(self._on_preview_failed)
        thread.finished.connect(lambda t=thread: self._decoders.discard(t))
        thread.finished.connect(thread.deleteLater)
        self._decoders.add(thread)
        self.status_text.setText(tr("Getting the file ready to play..."))
        thread.start()

    def _on_preview_ready(self, source: str, wav: str, duration: float) -> None:
        if self.source_path is None or str(self.source_path) != source or self.result is not None:
            _remove_file(wav)
            return
        try:
            self.player.mix.set_lanes([playback.Lane("preview", audio=playback.Clip.from_wav(wav))])
        except (OSError, ValueError) as exc:
            self._on_preview_failed(source, str(exc))
            return
        self.player.set_duration(duration)
        self.pos_slider.setRange(0, int(duration * 1000))
        self.roll.set_duration(duration)
        self._sync_position(0.0)
        self.status_text.setText(tr("File opened. You can play it now or press Analyze."))

    def _on_preview_failed(self, source: str, message: str) -> None:
        if self.source_path is not None and str(self.source_path) == source and self.result is None:
            self.status_text.setText(tr("This file can't be played here, but you can still analyze it."))

    def _on_mode_changed(self) -> None:
        self._queue_sound()

    def _queue_sound(self) -> None:
        """Something that changes what you hear changed. Volumes, mutes and the Play setting
        apply at once; new renders of the notes start shortly, once things settle."""
        if not hasattr(self, "_sound_timer") or not self.result:
            return
        self._apply_mix()
        self._sound_timer.start()

    def _clip(self, path: str, offset: float = 0.0) -> playback.Clip | None:
        key = (path, round(offset, 5))
        clip = self._clips.get(key)
        if clip is None:
            try:
                clip = playback.Clip.from_wav(path, offset)
            except (OSError, ValueError) as exc:
                log.warning("Could not open %s for playback: %s", path, exc)
                return None
            self._clips[key] = clip
        return clip

    def _rebuild_lanes(self) -> None:
        """Sets up one mixer lane per part (or one for the recording when there are no parts)."""
        r = self.result
        lanes = []
        if r:
            for t in r.tracks:
                lanes.append(playback.Lane(t.uid, audio=self._clip(t.audio, t.offset) if t.audio else None,
                                           notes=self._notes_clips.get(t.uid), always_audio=t.take))
            if not r.tracks:
                lanes.append(playback.Lane("original", audio=self._clip(r.audio_files["Original"])))
        self.player.mix.set_lanes(lanes)
        self._apply_mix()
        self._apply_sound()

    def _apply_mix(self) -> None:
        """Volume, pan, mute, solo, the Play setting and the click: heard right away."""
        mix = self.player.mix
        with mix.lock:
            mix.mode = self.mode_box.currentData() or "recording"
            mix.notes_level = self.notes_level.value() / 100.0
            mix.click_on = self.click_chk.isChecked()
            mix.click_level = self.click_level.value() / 100.0
            for t in self._tracks():
                mix.update_lane(t.uid, gain=playback.db_to_gain(t.volume_db), pan=t.pan, audible=t.visible)

    def _notes_job(self, t) -> mixer.NotesJob:
        flt = self.note_filter
        notes = [(n.start, n.end, n.pitch, n.velocity) for n in t.notes if flt.allows(n.pitch, t.drums)]
        return mixer.NotesJob(t.uid, t.instrument, notes, self.result.duration, self.room_chk.isChecked())

    def _apply_sound(self) -> None:
        """Starts rendering whatever the mix needs and doesn't have yet: the notes of each part
        on its instrument (only when the Play setting uses them) and the click track."""
        r = self.result
        if not r:
            return
        folder = Path(r.work_dir) / "lanes"
        folder.mkdir(parents=True, exist_ok=True)
        jobs = []
        if (self.mode_box.currentData() or "recording") != "recording":
            jobs += [self._notes_job(t) for t in r.tracks]
        if self.click_chk.isChecked() and r.beats:
            jobs.append(mixer.ClickJob(list(r.beats), r.meter, r.duration))
        waiting = False
        for job in jobs:
            if not job.notes and isinstance(job, mixer.NotesJob):
                self._lane_want[job.uid] = ""
                self._use_lane(job.uid, "", None)
                continue
            sig = job.signature()
            if self._lane_want.get(job.uid) == sig:
                waiting = waiting or self._lane_loaded.get(job.uid) != sig
                continue
            self._lane_want[job.uid] = sig
            path = self._lane_files.get(sig)
            if path and Path(path).exists():
                self._use_lane(job.uid, sig, path)
            else:
                # every render gets a file of its own, so one never writes over a file that is playing
                self._lane_serial += 1
                self.lanes.request(job, sig, folder / f"{job.uid}-{sig[:12]}-{self._lane_serial}.wav")
                waiting = True
        if waiting:
            self.status_text.setText(tr("Preparing the sound..."))

    def _use_lane(self, uid: str, sig: str, path: str | None) -> None:
        old_sig = self._lane_loaded.get(uid)
        clip = None
        if path:
            try:
                clip = playback.Clip.from_wav(path)
            except (OSError, ValueError) as exc:
                log.warning("Could not open a rendered part: %s", exc)
                return
            self._lane_files[sig] = path
        self._lane_loaded[uid] = sig
        mix = self.player.mix
        if uid == "click":
            with mix.lock:
                mix.click = clip
        else:
            if clip is None:
                self._notes_clips.pop(uid, None)
            else:
                self._notes_clips[uid] = clip
            mix.update_lane(uid, notes=clip)
        # the sound it replaced is not needed again unless the edit is undone, which renders it anew
        if old_sig and old_sig != sig and old_sig not in self._lane_want.values() \
                and old_sig not in self._lane_loaded.values():
            old_path = self._lane_files.pop(old_sig, None)
            if old_path:
                QTimer.singleShot(1500, lambda p=old_path: _remove_file(p))

    def _on_lane_rendered(self, uid: str, sig: str, path: str) -> None:
        if not self.result or self._lane_want.get(uid) != sig or not Path(path).is_relative_to(self.result.work_dir) \
                or self._lane_loaded.get(uid) == sig:     # (an identical render is already playing)
            _remove_file(path)
            return
        self._use_lane(uid, sig, path)
        if not self.lanes.busy():
            self.status_text.setText(tr("Sound ready."))

    def _on_render_progress(self, fraction: float) -> None:
        self.status_text.setText(tr("Preparing the sound... {n}%").format(n=f"{fraction * 100:.0f}"))

    def _on_render_failed(self, message: str, details: str) -> None:
        self.status_text.setText(tr("Could not build the sound."))
        show_message(self, QMessageBox.Warning, tr("Could not build the sound for playback."),
                     details=details, informative=message)

    def _on_player_error(self, message: str) -> None:
        self.status_text.setText(tr("Playback problem: {message}. You can still analyze the file.").format(
            message=tr(message)))

    def _on_finished(self) -> None:
        self._sync_position(self._duration())

    def _duration(self) -> float:
        if self.result:
            return self.result.duration
        return self.player.duration

    def toggle_play(self) -> None:
        if self.source_path is None:
            return
        if self.player.is_playing():
            self.player.pause()
        else:
            span = self._loop_span()
            if span:
                t = self.player.position()
                if t < span[0] - 0.02 or t >= span[1] - 0.02:
                    self.seek(span[0])
            self.player.play()

    def stop_playback(self) -> None:
        self.player.pause()
        span = self._loop_span()
        self.seek(span[0] if span else 0.0)

    def seek(self, seconds: float) -> None:
        if self.source_path is None:
            return
        seconds = min(max(0.0, seconds), self._duration() or seconds)
        self.player.seek(seconds)
        self._sync_position(seconds)

    def _on_slider_released(self) -> None:
        self._slider_held = False
        self.seek(self.pos_slider.value() / 1000.0)

    def _on_play_state(self, playing: bool) -> None:
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))
        if playing:
            self.ticker.start()
        else:
            self.ticker.stop()
            self._sync_position(self.player.position())
            self.player.mix.take_meters()
            self.mixer_panel.reset_meters()
            if hasattr(self, "tracks_view"):
                self.tracks_view.reset_meters()

    def _on_tick(self) -> None:
        self._sync_position(self.player.position())
        peaks, master = self.player.mix.take_meters()
        self._push_meters(peaks, master)

    def _push_meters(self, peaks: dict, master: float) -> None:
        self.mixer_panel.push_meters(peaks, master)
        if hasattr(self, "tracks_view"):
            self.tracks_view.push_meters(peaks)

    def _sync_position(self, t: float) -> None:
        self.roll.set_playhead(t)
        if self.view_stack.currentIndex() == 1:
            self.tracks_view.set_playhead(t)
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
        stem = self._project_title()
        start = Path(self.settings.value("export_dir", str(self.source_path.parent if self.source_path else Path.home())))
        path, _ = QFileDialog.getSaveFileName(self, tr(title), str(start / f"{stem}.{ext}"), tr(filter_text))
        if not path:
            return None
        path = Path(path)
        if path.suffix.lower() != f".{ext}":
            path = path.with_name(path.name + f".{ext}")
        self.settings.setValue("export_dir", str(path.parent))
        return path

    def _export(self, func, path: Path | None, done: str | None = None) -> bool:
        if path is None:
            return False
        try:
            func(path)
            self.status_text.setText(done.format(path=path) if done else tr("Saved {path}").format(path=path))
            return True
        except Exception as exc:
            show_message(self, QMessageBox.Critical, tr("Could not save the file."), informative=str(exc))
            return False

    def _export_text(self, kind: str) -> None:
        segs = self._export_segments()
        if kind == "lrc":
            segs = [dataclasses.replace(s, text=s.text.replace("\n", " / ")) for s in segs]
        title = self._project_title() if self.source_path else None
        makers = {"txt": (exporters.transcript_text, "Text files (*.txt)"),
                  "srt": (exporters.transcript_srt, "Subtitles (*.srt)"),
                  "lrc": (lambda s: exporters.transcript_lrc(s, title), "Timed lyrics (*.lrc)")}
        make, filt = makers[kind]
        path = self._save_path("Save transcript", kind, filt)
        self._export(lambda p: exporters.write_text(p, make(segs)), path)

    def _export_chords(self, kind: str) -> None:
        segs = self.result.segments        # chords line up with the original words and their timing
        lane = self._current_lane()
        title = self._project_title()
        if kind == "txt":
            path = self._save_path("Save lyrics with chords", "txt", "Text files (*.txt)")
            self._export(lambda p: exporters.write_text(p, exporters.lyrics_with_chords(segs, lane, title)), path)
        else:
            path = self._save_path("Save lyrics with chords (ChordPro)", "cho", "ChordPro files (*.cho *.chordpro)")
            self._export(lambda p: exporters.write_text(p, exporters.chordpro(segs, lane, title)), path)

    def _exportable_tracks(self):
        """The parts that are shown, with only the notes that pass the scale filter. What you
        see in the piano roll is what is saved."""
        out = []
        for t in self.result.tracks:
            if not t.visible:
                continue
            notes = [n for n in t.notes if self.note_filter.allows(n.pitch, t.name == "Drums")]
            if notes:
                out.append(dataclasses.replace(t, notes=notes))
        return out

    def _export_note(self, tracks) -> str:
        count = sum(len(t.notes) for t in tracks)
        text = tr_n(count, "Saved {n} note to {path}", "Saved {n} notes to {path}")
        if self.note_filter.active:
            text += f" ({self.note_filter.describe()})"
        return text

    def _export_midi(self) -> None:
        path = self._save_path("Save notes as MIDI", "mid", "MIDI files (*.mid)")
        tracks = self._exportable_tracks()
        if self._export(lambda p: exporters.save_midi(tracks, p, self.result.tempo, self.result.meter), path,
                        self._export_note(tracks)):
            self._saved_count = self._edit_count

    def _export_csv(self) -> None:
        path = self._save_path("Save notes as a spreadsheet", "csv", "CSV files (*.csv)")
        tracks = self._exportable_tracks()
        if self._export(lambda p: exporters.write_text(p, exporters.notes_csv(tracks)), path, self._export_note(tracks)):
            self._saved_count = self._edit_count

    def _export_stems(self) -> None:
        start = self.settings.value("export_dir", str(self.source_path.parent))
        folder = QFileDialog.getExistingDirectory(self, tr("Choose a folder for the stems"), start)
        if not folder:
            return
        stem = self._project_title()
        try:
            for name, src in self.result.audio_files.items():
                if name != "Original":
                    shutil.copyfile(src, Path(folder) / f"{stem} - {tr(name)}.wav")
            self.settings.setValue("export_dir", folder)
            self.status_text.setText(tr("Saved stems to {folder}").format(folder=folder))
        except Exception as exc:
            show_message(self, QMessageBox.Critical, tr("Could not save the stems."), informative=str(exc))

    def _bounce_spec(self):
        """Everything needed to mix the song again away from the player, as it sounds now."""
        from .bounce import BounceSpec, LaneSpec
        r = self.result
        mode = self.mode_box.currentData() or "recording"
        spec = BounceSpec(r.duration, mode, self.notes_level.value() / 100.0, self.volume.value() / 100.0,
                          click_level=self.click_level.value() / 100.0)
        for t in r.tracks:
            lane = LaneSpec(t.uid, tr(t.name), t.audio, t.offset, t.take, playback.db_to_gain(t.volume_db), t.pan,
                            t.visible)
            if mode != "recording":
                job = self._notes_job(t)
                if job.notes:
                    lane.notes_job = job
                    path = self._lane_files.get(job.signature())
                    lane.notes_file = path if path and Path(path).exists() else None
            spec.lanes.append(lane)
        if not r.tracks:
            spec.lanes.append(LaneSpec("original", tr("Recording"), r.audio_files["Original"], 0.0, False, 1.0, 0.0,
                                       True))
        if self.click_chk.isChecked() and r.beats:
            spec.click_job = mixer.ClickJob(list(r.beats), r.meter, r.duration)
            path = self._lane_files.get(spec.click_job.signature())
            spec.click_file = path if path and Path(path).exists() else None
        return spec

    def _export_sound(self) -> None:
        """Bounce: saves what the mixer plays, as one file or one per part."""
        from . import bounce
        r = self.result
        if not r or self._project_thread is not None:
            return
        dialog = bounce.BounceDialog(self, self.roll.span is not None, len(r.tracks))
        if dialog.exec() != bounce.QDialog.Accepted:
            return
        each, flac = dialog.each_part(), dialog.flac()
        span = self.roll.span if dialog.span_only() else None
        title = "".join(c for c in self._project_title() if c.isalnum() or c in " -_()").strip() or "audio"
        if each:
            start = self.settings.value("export_dir", str(self.source_path.parent if self.source_path else Path.home()))
            folder = QFileDialog.getExistingDirectory(self, tr("Choose a folder for the parts"), str(start))
            if not folder:
                return
            target = Path(folder)
            self.settings.setValue("export_dir", folder)
        else:
            ext = "flac" if flac else "wav"
            target = self._save_path("Save what you hear", ext, "FLAC files (*.flac)" if flac else "WAV files (*.wav)")
            if target is None:
                return
        spec = self._bounce_spec()
        work = r.work_dir

        def job(report, should_stop):
            return bounce.run(spec, target, each, span, flac, title, work, report, should_stop)

        def done(paths):
            if len(paths) == 1:
                self.status_text.setText(tr("Saved {path}").format(path=paths[0]))
            else:
                self.status_text.setText(tr_n(len(paths), "Saved {n} file to {folder}", "Saved {n} files to {folder}")
                                         .replace("{folder}", str(target)))

        self._run_project_job(job, tr("Bouncing..."), done, "Could not save the sound.")

    def _copy_transcript(self) -> None:
        if self.result and self.result.segments:
            QApplication.clipboard().setText(exporters.transcript_text(self._export_segments()))
            self.status_text.setText(tr("Transcript copied."))

    # Window events ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                if self._busy() or self._project_thread is not None:
                    self.status_text.setText(tr("Wait for the current analysis to finish first."))
                else:
                    self.open_file(url.toLocalFile())
                break

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.thread:
            answer = QMessageBox.question(self, APP_NAME, tr("An analysis is still running. Quit anyway?"))
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.thread.stop()
            self.hide()
            self.thread.wait()
        elif not self._confirm_discard_edits(tr("Closing the app discards them.")):
            event.ignore()
            return
        if self._take_rec is not None:
            self._take_rec.stop()
            self._take_rec = None
        self.lanes.stop()
        self.lanes.wait(10000)
        self.tracks_view.shutdown()
        for thread in list(self._decoders):
            thread.wait(30000)
        for job in (self._project_thread, self._translate_thread, self._take_job):
            if job is not None:
                job.stop()
                job.wait(20000)
        self._save_settings()
        self.player.shutdown()
        self.player.mix.set_lanes([])
        self._clips.clear()
        self._notes_clips.clear()
        if self.result:
            remove_dir(self.result.work_dir)
        remove_dir(self._work_dir)
        self._stop_update_job()
        self._apply_pending_update()
        event.accept()
