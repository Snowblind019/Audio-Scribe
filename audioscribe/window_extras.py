"""The newer features of the main window, kept apart so window.py stays readable:

  * interface language and note names
  * the Chords tab, the chord lane, and dropping chords onto the piano roll
  * the beat grid (tempo, first beat, beats per bar, tap tempo) and the click track
  * practice speed
  * clean up notes
  * projects (save and open)
  * recording from a microphone, downloading from YouTube
  * dragging MIDI out to a DAW
  * sheet music
  * translating the transcript

MainWindow mixes this class in, so `self` here is the main window.
"""

from __future__ import annotations

import logging
import tempfile
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, QThread, QTimer, QUrl, Signal, Qt
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QSlider, QToolButton)

from . import APP_NAME, cleanup, exporters, grid, i18n, notation, project, synth, theory
from .app import cache_dir
from .chords_view import ChordsPanel, chords_from_payload
from .i18n import tr, tr_n
from .music import NAME_STYLES, set_naming
from .theory import Chord

log = logging.getLogger(__name__)

SPEEDS = [50, 60, 70, 75, 80, 90, 100, 110, 125, 150]
METERS = [2, 3, 4, 5, 6, 7]


class MidiDragButton(QToolButton):
    """Drag this into a DAW (Reaper, Cubase ...) or a folder to drop the notes in as a MIDI file."""

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setText("Drag MIDI")
        self.setToolTip("Drag this into your DAW (Reaper, Cubase, FL Studio ...) or a folder to drop in the notes "
                        "as a MIDI file.\nWith a span selected, only that part is dragged.")
        self.setCursor(Qt.OpenHandCursor)
        self._press: QPoint | None = None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._press is not None and (event.position().toPoint() - self._press).manhattanLength() >= 6:
            self._press = None
            self.setDown(False)
            path = self.window._write_drag_midi()
            if path:
                mime = QMimeData()
                mime.setUrls([QUrl.fromLocalFile(str(path))])
                drag = QDrag(self)
                drag.setMimeData(mime)
                drag.exec(Qt.CopyAction)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._press is not None:
            self.window.status_text.setText(tr("Hold the mouse button on Drag MIDI and drag it into your DAW or a folder."))
        self._press = None
        super().mouseReleaseEvent(event)


class _Worker(QThread):
    """Runs a function on a thread and reports the result."""

    done = Signal(object)
    failed = Signal(str, str)
    progressed = Signal(float)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.done.emit(self.func(self.progressed.emit, lambda: self._stop))
        except (project.Stopped, KeyboardInterrupt):
            self.failed.emit("", "")
        except Exception as exc:
            if self._stop:
                self.failed.emit("", "")
                return
            log.exception("Background job failed")
            self.failed.emit(str(exc), traceback.format_exc())


class ExtrasMixin:
    # Building --------------------------------------------------------------------------------

    def _build_header_extras(self, layout) -> None:
        self.names_box = QComboBox()
        self.names_box.setToolTip("How notes are named everywhere in the app")
        for key, label in NAME_STYLES:
            self.names_box.addItem(label, key)
        self.names_box.currentIndexChanged.connect(lambda _i: self._set_naming_style(self.names_box.currentData()))
        self.lang_combo = QComboBox()
        self.lang_combo.setProperty("i18n_skip_items", True)
        self.lang_combo.setToolTip("Language of the app")
        for code, label in i18n.LANGUAGES:
            self.lang_combo.addItem(label, code)
        self.lang_combo.currentIndexChanged.connect(lambda _i: self._set_language(self.lang_combo.currentData()))
        self.midi_drag = MidiDragButton(self)
        for w in (QLabel("Notes"), self.names_box, self.lang_combo, self.midi_drag):
            if isinstance(w, QLabel):
                w.setObjectName("Meta")
            layout.addWidget(w)

    def _build_file_extras(self, layout) -> None:
        self.youtube_btn = QPushButton("Download from YouTube...")
        self.youtube_btn.clicked.connect(self._open_youtube)
        self.record_btn = QPushButton("Record from microphone...")
        self.record_btn.clicked.connect(self._open_recorder)
        self.save_project_btn = QPushButton("Save project...")
        self.save_project_btn.setToolTip("Save everything (words, notes, edits, chords, audio) in one file, "
                                         "so you never have to analyze this file again (Ctrl+S)")
        self.save_project_btn.clicked.connect(self._save_project)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self.youtube_btn)
        layout.addLayout(row)
        layout.addWidget(self.record_btn)
        layout.addWidget(self.save_project_btn)

    def _build_grid_section(self, play_lay) -> None:
        self.grid_box, l = self._section("Tempo and beats", "Set the beat grid by hand when the found beat is off. "
                                                             "Snapping, chords, sheet music and MIDI follow it.")
        form = self._form()
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(20.0, 300.0)
        self.bpm_spin.setDecimals(1)
        self.bpm_spin.setSuffix(" BPM")
        self.bpm_spin.setKeyboardTracking(False)
        self.tap_btn = QPushButton("Tap tempo")
        self.tap_btn.setToolTip("Tap along with the beat (T). While the song plays, the grid also lines up with your taps.")
        self.tap_btn.clicked.connect(self._tap)
        self.first_spin = QDoubleSpinBox()
        self.first_spin.setRange(0.0, 60.0)
        self.first_spin.setDecimals(2)
        self.first_spin.setSingleStep(0.01)
        self.first_spin.setSuffix(" s")
        self.first_spin.setKeyboardTracking(False)
        self.first_spin.setToolTip("Where the first beat is. Nudge it until the grid lines sit on the beats.")
        self.meter_box = QComboBox()
        for n in METERS:
            self.meter_box.addItem(f"{n}/4", n)
        self.meter_box.setProperty("i18n_skip_items", True)
        form.addRow("Tempo", self.bpm_spin)
        form.addRow("First beat", self.first_spin)
        form.addRow("Beats per bar", self.meter_box)
        l.addLayout(form)
        self.grid_reset_btn = QPushButton("Use the found beat")
        self.grid_reset_btn.clicked.connect(self._reset_grid)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.tap_btn, 1)
        buttons.addWidget(self.grid_reset_btn, 1)
        l.addLayout(buttons)
        self.grid_box.hide()
        play_lay.addWidget(self.grid_box)
        self._grid_timer = QTimer(self)
        self._grid_timer.setSingleShot(True)
        self._grid_timer.setInterval(250)
        self._grid_timer.timeout.connect(self._on_grid_edited)
        self.bpm_spin.valueChanged.connect(lambda _v: self._grid_loading or self._grid_timer.start())
        self.first_spin.valueChanged.connect(lambda _v: self._grid_loading or self._grid_timer.start())
        self.meter_box.currentIndexChanged.connect(lambda _i: self._grid_loading or self._grid_timer.start())
        self._grid_loading = False
        self._tapper = grid.TapTempo()

    def _build_click_controls(self, layout, form) -> None:
        self.click_chk = QCheckBox("Click track")
        self.click_chk.setToolTip("A metronome on the beat grid, mixed into what plays")
        self.click_chk.toggled.connect(lambda _on: self._queue_sound())
        self.click_level = QSlider(Qt.Horizontal)
        self.click_level.setRange(5, 100)
        self.click_level.setValue(60)
        self.click_level.setToolTip("Click volume")
        self.click_level.valueChanged.connect(lambda _v: self.click_chk.isChecked() and self._queue_sound())
        layout.addWidget(self.click_chk)
        form2 = self._form()
        form2.addRow("Click level", self.click_level)
        layout.addLayout(form2)

    def _build_speed_controls(self, layout) -> None:
        self.speed_box = QComboBox()
        for s in SPEEDS:
            self.speed_box.addItem(f"{s}%", s)
        self.speed_box.setProperty("i18n_skip_items", True)
        self.speed_box.setCurrentIndex(SPEEDS.index(100))
        self.speed_box.setToolTip("Practice speed. Slower playback keeps the pitch, so you can learn parts by ear.")
        self.speed_box.currentIndexChanged.connect(lambda _i: self._on_speed_changed())
        label = QLabel("Speed")
        label.setObjectName("Meta")
        layout.addWidget(label)
        layout.addWidget(self.speed_box)

    def _build_transcript_extras(self, bar) -> None:
        self.tr_target = QComboBox()
        for code, label in (("ro", "Romanian"), ("en", "English")):
            self.tr_target.addItem(label, code)
        self.tr_btn = QPushButton("Translate")
        self.tr_btn.setToolTip("Translate the words on this computer (the translation model downloads once, "
                               "about 640 MB)")
        self.tr_btn.clicked.connect(self._translate_transcript)
        self.tr_show = QComboBox()
        self.tr_show.addItem("Original", "original")
        self.tr_show.addItem("Translation", "translation")
        self.tr_show.addItem("Both", "both")
        self.tr_show.setToolTip("Which words are shown, copied and exported")
        self.tr_show.currentIndexChanged.connect(lambda _i: self._apply_transcript_view())
        self.tr_show.setEnabled(False)
        label = QLabel("into")
        label.setObjectName("Meta")
        show = QLabel("Show")
        show.setObjectName("Meta")
        for w in (self.tr_btn, label, self.tr_target, show, self.tr_show):
            bar.insertWidget(bar.count() - 1, w)

    def _build_chords_tab(self, tabs) -> None:
        self.chords = ChordsPanel(self.settings)
        self.chords.auditionRequested.connect(self._audition_chords)
        self.chords.insertRequested.connect(self._insert_chords_at_playhead)
        self.chords.keyChanged.connect(self._on_song_key_changed)
        self.chords.laneSourceChanged.connect(lambda _s: self._update_chord_lane())
        self.chords.chordSeek.connect(self.seek)
        tabs.addTab(self.chords, "Chords")

    def _wire_roll_extras(self) -> None:
        self.roll.chordsDropped.connect(self._on_chords_dropped)
        self.roll.dropRefused.connect(lambda: self.status_text.setText(
            tr("Turn on Edit notes (E) first, then drop chords onto the piano roll.")))
        self.roll.chordClicked.connect(self._on_chord_lane_clicked)
        self.roll.drop_preview = self._drop_preview

    # State ----------------------------------------------------------------------------------

    def _init_extras_state(self) -> None:
        self.song_key: tuple[int, str] | None = None
        self.translation: dict | None = None          # {"target": "ro", "lines": [...]}
        self._lane_cache: list = []
        self._project_thread: _Worker | None = None
        self._translate_thread: _Worker | None = None
        self._audition_n = 0
        self._original_from_project: dict | None = None
        self._take_rec = None            # TakeRecorder while a take is being recorded
        self._take_from = 0.0
        self._take_job: _Worker | None = None

    def _extras_clear(self) -> None:
        if self._take_rec is not None:          # the song is closing: the take has nowhere to go
            rec, self._take_rec = self._take_rec, None
            rec.stop()
            rec.path.unlink(missing_ok=True)
            self._take_timer.stop()
            self.rec_btn.setChecked(False)
            self.tracks_view.canvas.recording_from = None
        self.song_key = None
        self.translation = None
        self._lane_cache = []
        self.tr_show.blockSignals(True)
        self.tr_show.setCurrentIndex(0)
        self.tr_show.blockSignals(False)
        self.tr_show.setEnabled(False)
        self.chords.set_song_chords([])
        self.chords.set_song_key(None, None)
        self.roll.set_chords([])
        set_naming(tonic=None, mode="major")

    def _extras_show_result(self, r) -> None:
        self._load_grid_controls()
        if r.key:
            self.song_key = (r.key.tonic, r.key.mode)
            set_naming(tonic=r.key.tonic, mode=r.key.mode)
            self.chords.set_song_key(r.key.tonic, r.key.mode)
        else:
            self.chords.set_song_key(None, None)
        self._update_chord_lane()
        self._refresh_names()

    def _extras_update_controls(self) -> None:
        r = self.result
        busy = self._busy()
        has_notes = bool(r and any(t.notes for t in r.tracks))
        self.save_project_btn.setEnabled(bool(r) and not busy and self._project_thread is None)
        self.act_save_project.setEnabled(bool(r) and not busy)
        self.youtube_btn.setEnabled(not busy)
        self.record_btn.setEnabled(not busy)
        self.rec_btn.setEnabled(bool(r) and not busy and self._project_thread is None)
        self.midi_drag.setEnabled(has_notes)
        self.act_sheet.setEnabled(has_notes)
        lane = bool(r and self._current_lane())
        for act in (self.act_chord_txt, self.act_chordpro):
            act.setEnabled(bool(r and r.segments) and lane)
        self.grid_box.setVisible(bool(r))
        self.tr_btn.setEnabled(bool(r and r.segments) and self._translate_thread is None)
        self.cleanup_btn.setEnabled(has_notes)

    # Language and note names -------------------------------------------------------------------

    def _set_language(self, code: str) -> None:
        if code == i18n.language():
            return
        i18n.set_language(code)
        self.settings.setValue("ui/language", code)
        for w in QApplication.topLevelWidgets():
            i18n.retranslate(w, code)
        self._retranslate_dynamic()
        self.status_text.setText(tr("The app is now in English.") if code == "en" else tr("The app is now in Romanian."))

    def _retranslate_dynamic(self) -> None:
        """Text that is built in code (counts, names, messages) is built again."""
        r = self.result
        if self.source_path is None:
            self.file_label.setText(tr("No file open yet."))
        self._fit_export_button()
        self._retranslate_update()
        self._update_span_controls(self.roll.span)
        self.roll.refresh()
        self.scale_filter.refresh_names()
        self.chords.rebuild()
        if r:
            self._update_meta()
            self._refresh_edit_tracks()          # the "Add to" list shows part names
            self.summary.set_result(r)
            self._refresh_transcript_label()
            self._fill_notes_table()
            for row in getattr(self, "part_rows", []):
                row.refresh_name()
            if self.roll.span:
                self._update_selection_details()
        else:
            self.summary.clear()
        self._update_controls()

    def _set_naming_style(self, style: str) -> None:
        set_naming(style=style)
        self.settings.setValue("ui/note_names", style)
        self._refresh_names()

    def _on_song_key_changed(self, tonic: int, mode: str) -> None:
        self.song_key = (tonic, mode)
        set_naming(tonic=tonic, mode=mode)
        self._refresh_names()

    def _refresh_names(self) -> None:
        self.roll.refresh()
        self.scale_filter.refresh_names()
        self.chords.rebuild()
        r = self.result
        if r:
            self.summary.set_result(r)
            self._fill_notes_table()
            self._update_meta()
            if self.roll.span:
                self._update_selection_details()
            self.chords.set_song_chords(self._current_lane())

    # Chords ----------------------------------------------------------------------------------------

    def _beat_period(self) -> float:
        r = self.result
        p = grid.period_of(r.beats) if r else None
        return p or 0.5

    def _current_lane(self) -> list:
        r = self.result
        if not r:
            return []
        source = self.chords.lane_source()
        if source == "off":
            return []
        if source == "recording" and r.chords:
            return r.chords
        return self._lane_cache

    def _update_chord_lane(self) -> None:
        r = self.result
        if not r:
            self.roll.set_chords([])
            return
        if self.chords.lane_source() == "notes" or not r.chords:
            pitched = [n for t in r.tracks if t.visible and t.name != "Drums" for n in t.notes
                       if self.note_filter.allows(n.pitch)]
            self._lane_cache = theory.guess_chords(pitched, 0.0, r.duration, self._beat_period() * 2)
        lane = self._current_lane()
        self.roll.set_chords(lane)
        self.chords.set_song_chords(lane)
        self._update_controls_extras_only()

    def _update_controls_extras_only(self) -> None:
        if hasattr(self, "act_chord_txt"):
            self._extras_update_controls()

    def _part_for_chords(self) -> int | None:
        tracks = self._tracks()
        ti = self._edit_index()
        if 0 <= ti < len(tracks) and tracks[ti].visible and tracks[ti].name != "Drums":
            return ti
        for i, t in enumerate(tracks):
            if t.visible and t.name != "Drums":
                return i
        return None

    def _chord_sound(self) -> str:
        key = self.chords.sound()
        if key:
            return key
        ti = self._part_for_chords()
        tracks = self._tracks()
        return tracks[ti].instrument if ti is not None else synth.DEFAULT_INSTRUMENT

    def _audition_chords(self, chords: list[Chord]) -> None:
        if not chords:
            return
        rows = self.chords.render_notes(chords, 0.0, self._beat_period() if self.result else 0.5)
        rows = [r for r in rows if r[0] < 24.0]
        self.audition.rows(self._chord_sound(), rows)

    def _drop_preview(self, payload: dict, t: float) -> list:
        return self.chords.render_notes(chords_from_payload(payload), t, self._beat_period())

    def _insert_chords(self, chords: list[Chord], t: float) -> None:
        if not chords or not self.result:
            return
        if not self.roll.edit_mode:
            self.status_text.setText(tr("Turn on Edit notes (E) first, then add chords."))
            return
        ti = self._part_for_chords()
        if ti is None:
            self.status_text.setText(tr("Show a part that isn't Drums to add chords to it."))
            return
        rows = self.chords.render_notes(chords, t, self._beat_period())
        rows = [r for r in rows if r[0] < self.result.duration]
        if not rows:
            self.status_text.setText(tr("There is no room for chords after the end of the song."))
            return
        self.roll.insert_notes(ti, rows)
        self.status_text.setText(tr_n(len(chords), "Added {n} chord to {part}. Ctrl+Z undoes it.",
                                      "Added {n} chords to {part}. Ctrl+Z undoes it.").replace(
            "{part}", tr(self._tracks()[ti].name)))

    def _on_chords_dropped(self, payload: dict, t: float) -> None:
        chords = chords_from_payload(payload)
        if chords:
            self.chords.current = chords[-1]
            self._insert_chords(chords, t)

    def _insert_chords_at_playhead(self, chords: list[Chord]) -> None:
        t = self.player.position()
        snap = self.roll._drop_time(self.roll.x_of(t)) if self.result else t
        self._insert_chords(chords, snap)

    def _on_chord_lane_clicked(self, start: float, chord) -> None:
        self.seek(start)
        if chord is not None:
            self.chords.current = chord
            self._audition_chords([chord])
            self.status_text.setText(tr("{chord} at {time}").format(chord=chord.name(), time=self._fmt(start)))

    @staticmethod
    def _fmt(t: float) -> str:
        from .music import format_time
        return format_time(t, 1)

    # Beat grid --------------------------------------------------------------------------------------

    def _load_grid_controls(self) -> None:
        r = self.result
        self._grid_loading = True
        if r:
            bpm = r.tempo or grid.bpm_of(r.beats) or 120.0
            self.bpm_spin.setValue(round(bpm, 1))
            self.first_spin.setMaximum(max(1.0, min(60.0, r.duration)))
            self.first_spin.setValue(round(r.beats[0], 2) if r.beats else 0.0)
            i = self.meter_box.findData(r.meter)
            self.meter_box.setCurrentIndex(i if i >= 0 else METERS.index(4))
            self.grid_reset_btn.setEnabled(bool(r.detected_beats))
        self._grid_loading = False

    def _on_grid_edited(self) -> None:
        r = self.result
        if not r:
            return
        self._apply_grid(self.bpm_spin.value(), self.first_spin.value(), self.meter_box.currentData())

    def _apply_grid(self, bpm: float, first: float, meter: int) -> None:
        r = self.result
        r.tempo = float(bpm)
        r.beats = grid.make_beats(bpm, first, r.duration)
        r.meter = int(meter)
        self._after_grid_change()
        self.status_text.setText(tr("Beat grid set to {bpm} BPM, {meter} beats per bar.").format(
            bpm=f"{bpm:.1f}", meter=meter))

    def _after_grid_change(self) -> None:
        r = self.result
        self.roll.set_beats(r.beats, r.meter)
        self.tracks_view.set_beats(r.beats, r.meter)
        self._update_meta()
        self.summary.set_result(r)
        if self.roll.span:
            self._update_selection_details()
        if self.chords.lane_source() == "notes" or not r.chords:
            self._update_chord_lane()
        if self.click_chk.isChecked():
            self._queue_sound()

    def _reset_grid(self) -> None:
        r = self.result
        if not r:
            return
        r.beats = list(r.detected_beats)
        r.tempo = r.detected_tempo
        self._load_grid_controls()
        self._after_grid_change()
        self.status_text.setText(tr("Back to the beat the analysis found."))

    def _tap(self) -> None:
        r = self.result
        if not r:
            return
        song_t = self.player.position() if self.player.is_playing() else None
        bpm, first = self._tapper.tap(song_t)
        if bpm is None:
            self.status_text.setText(tr("Keep tapping along with the beat..."))
            return
        first = first if first is not None else (r.beats[0] if r.beats else 0.0)
        self._grid_loading = True
        self.bpm_spin.setValue(round(bpm, 1))
        self.first_spin.setValue(round(first, 2))
        self._grid_loading = False
        self._apply_grid(bpm, first, self.meter_box.currentData())

    # Speed --------------------------------------------------------------------------------------------

    def _on_speed_changed(self) -> None:
        rate = (self.speed_box.currentData() or 100) / 100.0
        self.player.set_rate(rate)
        self.status_text.setText(tr("Playing at {n}% speed.").format(n=int(rate * 100)))

    # Clean up --------------------------------------------------------------------------------------------

    def _open_cleanup(self) -> None:
        r = self.result
        if not r or not self.roll.edit_mode:
            return
        scopes = [("shown", tr("All notes that are shown"))]
        if self.roll.selected:
            scopes.insert(0, ("selected", tr("The selected notes")))
        if self.roll.span:
            scopes.insert(0, ("span", tr("Notes inside the selected span")))

        def groups(scope: str):
            out = []
            selected = self.roll.selected
            span = self.roll.span
            for t in r.tracks:
                if not t.visible:
                    continue
                drum = t.name == "Drums"
                notes = [n for n in t.notes if self.note_filter.allows(n.pitch, drum)]
                if scope == "selected":
                    notes = [n for n in notes if n in selected]
                elif scope == "span" and span:
                    notes = [n for n in notes if n.end > span[0] and n.start < span[1]]
                if notes:
                    out.append((notes, not drum))
            return out

        dialog = cleanup.CleanupDialog(self, groups, scopes, self.song_key, r.beats, r.duration)
        if dialog.exec() != QDialog.Accepted:
            return
        plan, scope = dialog.plan(), dialog.scope_key()
        self._on_edit_started()
        group_list = groups(scope)
        removals, counts = cleanup.apply(plan, group_list, r.beats, r.duration)
        gone = set().union(*removals) if removals else set()
        for t in r.tracks:
            t.notes[:] = [n for n in t.notes if n not in gone]
        self.roll.selected = set()
        self.roll.reindex()
        self.roll.selectionChanged.emit()
        self._after_notes_changed()
        self.status_text.setText(cleanup.done_text(counts))

    # Projects --------------------------------------------------------------------------------------------

    def _project_state(self) -> dict:
        state = {"song_key": list(self.song_key) if self.song_key else None,
                 "lane_source": self.chords.lane_source(),
                 "sketch": [[c.to_dict() for c in line] for line in self.chords.sketch],
                 "translation": self.translation}
        if self._original:
            names = {t.uid: t.name for t in self._tracks()}
            state["original"] = {names[uid]: project._notes_rows(rows) for uid, rows in self._original.items()
                                 if uid in names}
        return state

    def _save_project(self) -> None:
        r = self.result
        if not r or self._project_thread is not None:
            return
        stem = self._project_title()
        start = Path(str(self.settings.value("project_dir", str(Path.home())))) / f"{stem}{project.SUFFIX}"
        name = self.title_label.text()
        if not name or name == APP_NAME:
            name = self.source_path.name if self.source_path else stem
        path, _ = QFileDialog.getSaveFileName(self, tr("Save project"), str(start),
                                              tr("Audio Scribe projects (*.ascribe)"))
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() != project.SUFFIX:
            path = path.with_name(path.name + project.SUFFIX)
        self.settings.setValue("project_dir", str(path.parent))
        state = self._project_state()
        state["source_name"] = name
        edit_count = self._edit_count

        def job(report, should_stop):
            project.save(path, r, state, report, should_stop)
            return str(path)

        self._run_project_job(job, tr("Saving the project..."),
                              lambda p: self._project_saved(p, edit_count))

    def _project_title(self) -> str:
        """A name for saved files: the song's file name without its extension."""
        if self.source_path is None:
            return "audio"
        return self.source_path.stem

    def _project_saved(self, path: str, edit_count: int) -> None:
        self._saved_count = edit_count
        self.status_text.setText(tr("Project saved: {path}").format(path=path))

    def _run_project_job(self, job, message: str, on_done,
                         fail_text: str = "The project could not be saved or opened.") -> None:
        self._project_thread = _Worker(job, self)
        self._project_thread.progressed.connect(lambda f: self.progress.setValue(int(f * 1000)))
        self._project_thread.done.connect(on_done)
        self._project_thread.failed.connect(lambda m, d: self._on_project_failed(m, d, fail_text))
        self._project_thread.finished.connect(self._on_project_thread_finished)
        self.progress.setValue(0)
        self.progress.show()
        self.status_text.setText(message)
        self._project_thread.start()
        self._update_controls()

    def _on_project_failed(self, message: str, details: str,
                           fail_text: str = "The project could not be saved or opened.") -> None:
        if not message:
            self.status_text.setText(tr("Cancelled."))
            return
        self.status_text.setText(tr(fail_text))
        from .window import show_message
        show_message(self, QMessageBox.Warning, tr(message), details=details)

    def _on_project_thread_finished(self) -> None:
        if self._project_thread:
            self._project_thread.deleteLater()
        self._project_thread = None
        self.progress.hide()
        self._update_controls()

    def _open_project(self, path: Path) -> None:
        if self._project_thread is not None:
            return
        if not self._confirm_discard_edits():
            return
        from .window import _work_root
        work = Path(tempfile.mkdtemp(prefix="project-", dir=_work_root()))

        def job(report, should_stop):
            return project.load(path, work, report, should_stop)

        def done(loaded):
            result, state = loaded
            self._edit_count = self._saved_count = 0     # nothing unsaved yet
            self.source_path = Path(path)
            self.settings.setValue("last_dir", str(Path(path).parent))
            self._show_result(result)
            self._apply_project_state(state)
            name = state.get("source_name") or Path(path).stem
            self.title_label.setText(name)
            self.file_label.setText(Path(path).name)
            self.setWindowTitle(f"{name} - {APP_NAME}")
            self.status_text.setText(tr("Project opened. Nothing needs to be analyzed again."))

        self._run_project_job(job, tr("Opening the project..."), done)

    def _apply_project_state(self, state: dict) -> None:
        key = state.get("song_key")
        if isinstance(key, list) and len(key) == 2 and isinstance(key[0], int) and 0 <= key[0] < 12 \
                and key[1] in ("major", "minor"):
            self.chords.set_key(key[0], key[1])
        lane = state.get("lane_source")
        if lane in ("recording", "notes", "off"):
            i = self.chords.lane_box.findData(lane)
            self.chords.lane_box.setCurrentIndex(i)
        sketch = state.get("sketch")
        if isinstance(sketch, list):
            lines = []
            for line in sketch[:7]:
                if isinstance(line, list):
                    lines.append(chords_from_payload({"chords": line}))
            while len(lines) < 7:
                lines.append([])
            self.chords.sketch = lines
            self.chords._fill_sketch()
        original = state.get("original")
        if isinstance(original, dict) and original:
            self._original = {t.uid: [(n.start, n.end, n.pitch, n.velocity, n.edited) for n in original.get(t.name, [])]
                              for t in self._tracks()}
        tr_state = state.get("translation")
        if isinstance(tr_state, dict) and tr_state.get("target") in ("ro", "en") and isinstance(tr_state.get("lines"), list):
            lines = [x if isinstance(x, str) else "" for x in tr_state["lines"]][:len(self.result.segments)]
            if len(lines) == len(self.result.segments):
                self.translation = {"target": tr_state["target"], "lines": [x[:20000] for x in lines]}
                self._fill_transcript(self.result)
                self.tr_show.setEnabled(True)
                self.tr_show.setCurrentIndex(2)

    # Recording and YouTube -------------------------------------------------------------------------------

    def _open_recorder(self) -> None:
        from .recorder import RecordDialog, default_folder
        folder = Path(str(self.settings.value("record/folder", str(default_folder()))))
        dialog = RecordDialog(self, folder)
        dialog.recorded.connect(self._on_recorded)
        dialog.exec()
        self.settings.setValue("record/folder", str(dialog.folder))

    def _on_recorded(self, path: str, notes: bool, words: bool) -> None:
        self.open_file(path)
        if self.source_path and Path(path) == self.source_path and (notes or words):
            self.chk_notes.setChecked(notes)
            self.chk_words.setChecked(words)
            self.start_analysis()

    def _open_youtube(self) -> None:
        from .youtube import YouTubeDialog
        dialog = YouTubeDialog(self, self.settings)
        dialog.downloaded.connect(self._on_downloaded)
        dialog.exec()

    def _on_downloaded(self, path: str, open_it: bool, analyze: bool) -> None:
        if not open_it:
            self.status_text.setText(tr("Saved {path}").format(path=path))
            return
        self.open_file(path)
        if analyze and self.source_path and Path(path) == self.source_path:
            self.start_analysis()

    # Takes: recording along with the song onto a new track ---------------------------------------------

    TAKE_COLORS = ["#E0533B", "#D97BC8", "#7BC8D9", "#C8D97B", "#D9A87B", "#8F7BD9"]

    def _build_record_button(self, layout) -> None:
        self.rec_btn = QToolButton()
        self.rec_btn.setObjectName("RecordButton")
        self.rec_btn.setText("● Rec")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setToolTip("Record a take from your microphone while the song plays (R). It goes on its own "
                                "track, so you can hear it with the song, set its volume, and find its notes.")
        self.rec_btn.clicked.connect(self._toggle_take)
        layout.addWidget(self.rec_btn)
        self._take_timer = QTimer(self)
        self._take_timer.setInterval(250)
        self._take_timer.timeout.connect(self._take_tick)

    def _toggle_take(self) -> None:
        if self._take_rec is None:
            self._start_take()
        else:
            self._stop_take()

    def _start_take(self) -> None:
        from .recorder import TakeRecorder
        r = self.result
        self.rec_btn.setChecked(False)
        if not r:
            self.status_text.setText(tr("Analyze a song (or open a project) first, then record a take along with it."))
            return
        if self._busy() or self._project_thread is not None:
            return
        if (self.speed_box.currentData() or 100) != 100:
            self.status_text.setText(tr("Set Speed to 100% to record a take, so it lines up with the song."))
            return
        folder = Path(r.work_dir) / "takes"
        folder.mkdir(parents=True, exist_ok=True)
        rec = TakeRecorder(folder / f"raw-{int(time.time() * 1000)}.wav", self)
        error = rec.start()
        if error:
            from .window import show_message
            show_message(self, QMessageBox.Warning, error)
            return
        self._take_rec = rec
        self._take_from = self.player.position()
        self.rec_btn.setChecked(True)
        self.tracks_view.canvas.recording_from = self._take_from
        if not self.player.is_playing():
            self.player.play()
        self._take_timer.start()
        self.status_text.setText(tr("Recording a take. Press Rec again (or R) to stop."))

    def _take_tick(self) -> None:
        if self._take_rec is not None:
            self.status_text.setText(tr("Recording a take... {time}").format(time=self._fmt(self._take_rec.seconds())))
            self.tracks_view.refresh()

    def _stop_take(self) -> None:
        rec, self._take_rec = self._take_rec, None
        self._take_timer.stop()
        self.rec_btn.setChecked(False)
        self.tracks_view.canvas.recording_from = None
        if rec is None:
            return
        seconds = rec.stop()
        self.player.pause()
        r = self.result
        if not r or seconds < 0.3:
            rec.path.unlink(missing_ok=True)
            self.status_text.setText(tr("That was too short to use."))
            self.tracks_view.refresh()
            return
        from . import audio
        from .engine import SAMPLE_RATE, Track
        try:
            data = audio.decode(rec.path, SAMPLE_RATE, 2)
            out = rec.path.with_name(f"take-{int(time.time() * 1000)}.wav")
            audio.write_wav(out, data, SAMPLE_RATE)
        except Exception as exc:
            from .window import show_message
            show_message(self, QMessageBox.Warning, tr("Could not use the recording."), informative=str(exc))
            return
        finally:
            rec.path.unlink(missing_ok=True)
        names = {t.name for t in r.tracks}
        n = 1
        while tr("Take {n}").format(n=n) in names:
            n += 1
        takes = sum(1 for t in r.tracks if t.take)
        track = Track(tr("Take {n}").format(n=n), self.TAKE_COLORS[takes % len(self.TAKE_COLORS)], [], audio=str(out),
                      offset=self._take_from, take=True, instrument="piano")
        r.tracks.append(track)
        self._tracks_changed()
        self._edit_count += 1
        self.status_text.setText(tr("Added {part}. Right-click it in the Tracks view to find its notes, "
                                    "or hold Alt and drag it to line it up.").format(part=tr(track.name)))
        self._show_view(1)

    def _tracks_changed(self) -> None:
        """Parts were added or removed: every view and the mixer follow."""
        r = self.result
        self.roll.set_tracks(r.tracks)
        self._apply_visibility()
        self._fill_parts(r.tracks)
        self.tracks_view.set_data(r.duration, r.tracks, r.beats, r.meter)
        self._refresh_edit_tracks()
        self._rebuild_lanes()
        self._schedule_refresh()
        self._update_controls()

    def _take_menu(self, track, pos) -> None:
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        ti = self._tracks().index(track) if track in self._tracks() else -1
        menu.addAction(tr("Edit the notes in the piano roll"), lambda: self._edit_track_in_roll(ti))
        if track.take:
            menu.addSeparator()
            find = menu.addAction(tr("Find the notes in this take"), lambda: self._find_take_notes(track))
            find.setEnabled(self._take_job is None)
            menu.addAction(tr("Rename..."), lambda: self._rename_take(track))
            menu.addAction(tr("Remove this take"), lambda: self._remove_take(track))
        menu.exec(pos)

    def _rename_take(self, track) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, tr("Rename"), tr("Name"), text=tr(track.name))
        name = (name or "").strip()[:40]
        if not ok or not name or not name.isprintable():
            return
        if any(t is not track and t.name == name for t in self._tracks()) or name in ("Full mix", "Original",
                                                                                       *self._stem_names()):
            self.status_text.setText(tr("Another part already has that name."))
            return
        track.name = name
        self._tracks_changed()
        self._edit_count += 1

    @staticmethod
    def _stem_names() -> list[str]:
        from .engine import STEM_ORDER
        return list(STEM_ORDER)

    def _remove_take(self, track) -> None:
        r = self.result
        if not r or track not in r.tracks:
            return
        answer = QMessageBox.question(self, APP_NAME, tr("Remove {part}? This can't be undone.").format(
            part=tr(track.name)))
        if answer != QMessageBox.Yes:
            return
        r.tracks.remove(track)
        self._tracks_changed()
        self._edit_count += 1

    def _on_take_moved(self, track, old_offset: float) -> None:
        """A take was dragged: its notes move with it, and the mixer plays it from the new place."""
        delta = track.offset - old_offset
        if abs(delta) < 1e-6:
            return
        new_offset, track.offset = track.offset, old_offset
        self._on_edit_started()            # the undo step remembers where it was
        track.offset = new_offset
        end = self.result.duration
        for n in track.notes:
            n.start, n.end = n.start + delta, min(end, n.end + delta)
        # notes pushed past the end of the song are left out (undo brings them back)
        track.notes[:] = [n for n in track.notes if n.start < end - 0.01]
        self.roll.reindex()
        self._after_notes_changed()
        self._rebuild_lanes()
        self.status_text.setText(tr("{part} now starts at {time}.").format(part=tr(track.name),
                                                                           time=self._fmt(track.offset)))

    def _find_take_notes(self, track) -> None:
        r = self.result
        if not r or not track.audio or self._take_job is not None:
            return
        from .engine import Analyzer, Options
        opts = Options(sensitivity=self.sens.value(), min_note_ms=self.min_len.value())
        wav, offset, work = track.audio, track.offset, r.work_dir

        def job(report, should_stop):
            from .engine import Note
            notes = Analyzer(wav, opts, work)._notes(wav, track.name)
            return [Note(n.start + offset, n.end + offset, n.pitch, n.velocity) for n in notes
                    if n.start + offset >= 0]

        def done(notes):
            if self.result is not r or track not in r.tracks:
                return
            self._on_edit_started()
            track.notes[:] = notes
            self.roll.reindex()
            self._after_notes_changed()
            self.status_text.setText(tr_n(len(notes), "Found {n} note in {part}.", "Found {n} notes in {part}.")
                                     .replace("{part}", tr(track.name)))

        self._take_job = _Worker(job, self)
        self._take_job.done.connect(done)
        self._take_job.failed.connect(lambda m, d: m and self.status_text.setText(
            tr("Could not find the notes: {error}").format(error=m)))
        self._take_job.finished.connect(self._on_take_job_finished)
        self.status_text.setText(tr("Finding the notes in {part}...").format(part=tr(track.name)))
        self._take_job.start()

    def _on_take_job_finished(self) -> None:
        if self._take_job is not None:
            self._take_job.deleteLater()
        self._take_job = None

    # MIDI drag out ----------------------------------------------------------------------------------------

    def _write_drag_midi(self) -> Path | None:
        r = self.result
        if not r:
            return None
        tracks = self._exportable_tracks()
        span = self.roll.span
        if span:
            import dataclasses
            from .engine import Note
            clipped = []
            for t in tracks:
                notes = [Note(max(0.0, n.start - span[0]), min(n.end, span[1]) - span[0], n.pitch, n.velocity)
                         for n in t.notes if n.end > span[0] and n.start < span[1] and n.start >= span[0] - 0.01]
                if notes:
                    clipped.append(dataclasses.replace(t, notes=notes))
            tracks = clipped
        if not tracks:
            self.status_text.setText(tr("No notes to drag."))
            return None
        folder = cache_dir() / "drag"
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.mid"):
            if time.time() - old.stat().st_mtime > 3600:
                old.unlink(missing_ok=True)
        stem = "".join(c for c in self._project_title() if c.isalnum() or c in " -_()").strip() or "notes"
        name = f"{stem} ({tr('span')}).mid" if span else f"{stem}.mid"
        path = folder / name
        exporters.save_midi(tracks, path, r.tempo, r.meter)
        self.status_text.setText(tr("Dragging the notes as a MIDI file."))
        return path

    # Sheet music --------------------------------------------------------------------------------------------

    def _open_sheet(self) -> None:
        r = self.result
        if not r:
            return
        try:
            from .sheet_view import SheetDialog, verovio_available
        except ImportError:
            verovio_available = lambda: False  # noqa: E731
        if not verovio_available():
            from .window import show_message
            show_message(self, QMessageBox.Warning, tr("Sheet music needs Verovio, which is not installed."),
                         informative=tr("Run the installer again to add it."))
            return
        names = [t.name for t in r.tracks if t.visible and t.notes]

        def make_xml(opts, parts):
            tracks = [t for t in self._exportable_tracks() if t.name in parts]
            opts.key = self.song_key
            opts.meter = r.meter
            opts.tempo = r.tempo
            segments = r.segments if opts.lyrics else []   # word timings only exist for the original words
            return notation.build(tracks, r.beats, r.duration, opts, segments, self._current_lane() if opts.chords else [])

        dialog = SheetDialog(self, make_xml, names, self._project_title(),
                             str(self.settings.value("export_dir", str(Path.home()))))
        dialog.exec()

    # Translation ---------------------------------------------------------------------------------------------

    def _translate_transcript(self) -> None:
        r = self.result
        if not r or not r.segments or self._translate_thread is not None:
            return
        target = self.tr_target.currentData()
        source = "en" if r.translated else (r.language or "")
        if source == target:
            self.status_text.setText(tr("The words are already in that language."))
            return
        from . import translate as tl
        if source not in tl.NLLB:
            self.status_text.setText(tr("Translating from this language isn't supported."))
            return
        lines = [s.text for s in r.segments]
        device = self.device_box.currentData() or "cpu"

        def job(report, should_stop):
            folder = tl.model_dir(should_stop=should_stop)
            try:
                engine = tl.Translator(folder, device)
            except Exception:
                engine = tl.Translator(folder, "cpu")
            return target, engine.translate(lines, source, target, report, should_stop)

        self._translate_thread = _Worker(job, self)
        self._translate_thread.progressed.connect(lambda f: self.progress.setValue(int(f * 1000)))
        self._translate_thread.done.connect(self._on_translated)
        self._translate_thread.failed.connect(self._on_translate_failed)
        self._translate_thread.finished.connect(self._on_translate_finished)
        self.progress.setValue(0)
        self.progress.show()
        self.status_text.setText(tr("Translating. The first time, the translation model downloads (about 640 MB)..."))
        self._translate_thread.start()
        self._update_controls()

    def _on_translated(self, result) -> None:
        target, lines = result
        self.translation = {"target": target, "lines": lines}
        self._fill_transcript(self.result)
        self.tr_show.setEnabled(True)
        self.tr_show.setCurrentIndex(2)
        self.status_text.setText(tr("Translated. Use Show to pick which words are shown, copied and exported."))

    def _on_translate_failed(self, message: str, details: str) -> None:
        if not message:
            self.status_text.setText(tr("Translation cancelled."))
            return
        self.status_text.setText(tr("Translation failed."))
        from .engine import friendly_error
        from .window import show_message
        show_message(self, QMessageBox.Warning, tr(friendly_error(Exception(message))), details=details)

    def _on_translate_finished(self) -> None:
        if self._translate_thread:
            self._translate_thread.deleteLater()
        self._translate_thread = None
        self.progress.hide()
        self._update_controls()

    def _apply_transcript_view(self) -> None:
        view = self.tr_show.currentData()
        has = self.translation is not None
        self.transcript.setColumnHidden(1, has and view == "translation")
        self.transcript.setColumnHidden(2, not has or view == "original")

    def _export_segments(self):
        """The transcript lines as shown: original, translated, or both."""
        r = self.result
        if not r:
            return []
        view = self.tr_show.currentData() if self.translation else "original"
        if view == "original" or not self.translation:
            return r.segments
        import dataclasses
        out = []
        for seg, line in zip(r.segments, self.translation["lines"]):
            if view == "translation":
                out.append(dataclasses.replace(seg, text=line, words=[]))
            else:
                out.append(dataclasses.replace(seg, text=f"{seg.text}\n{line}" if line else seg.text, words=[]))
        return out
