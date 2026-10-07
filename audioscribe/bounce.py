"""Bounce: save what you hear as audio files, the way a DAW exports a mix.

Either the whole mix in one file, or each part in its own file (for another DAW), as WAV or
FLAC, for the whole song or only the selected span. Volume, pan, mute, solo and the Play
setting are applied, so the file sounds like the mixer. Parts' notes that were never played
on their instruments yet are rendered first.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

from . import mixer, playback
from .i18n import tr


@dataclass
class LaneSpec:
    uid: str
    name: str
    audio: str | None
    offset: float
    always_audio: bool
    gain: float
    pan: float
    audible: bool
    notes_job: mixer.NotesJob | None = None
    notes_file: str | None = None


@dataclass
class BounceSpec:
    duration: float
    mode: str
    notes_level: float
    master: float
    lanes: list[LaneSpec] = field(default_factory=list)
    click_job: mixer.ClickJob | None = None
    click_file: str | None = None
    click_level: float = 0.6


def build_mix(spec: BounceSpec, scratch: Path, report=None, should_stop=None) -> playback.Mix:
    """A mixer set up like the one playing, rendering any notes it still needs."""
    mix = playback.Mix()
    mix.frames = int(round(spec.duration * playback.SR))
    mix.mode, mix.notes_level, mix.master = spec.mode, spec.notes_level, spec.master
    need = [lane for lane in spec.lanes if lane.notes_job is not None and not lane.notes_file]
    lanes = []
    for lane in spec.lanes:
        notes = None
        if lane.notes_job is not None:
            if not lane.notes_file:
                lane.notes_file = str(scratch / f"notes-{lane.uid}.wav")
                mixer.render(lane.notes_job, lane.notes_file, should_stop=should_stop)
                if report:
                    report(0.5 * (need.index(lane) + 1) / max(1, len(need)))
            notes = playback.Clip.from_wav(lane.notes_file)
        audio = playback.Clip.from_wav(lane.audio, lane.offset) if lane.audio else None
        lanes.append(playback.Lane(lane.uid, audio=audio, notes=notes, gain=lane.gain, pan=lane.pan,
                                   audible=lane.audible, always_audio=lane.always_audio))
    mix.set_lanes(lanes)
    if spec.click_job is not None:
        if not spec.click_file:
            spec.click_file = str(scratch / "click.wav")
            mixer.render(spec.click_job, spec.click_file, should_stop=should_stop)
        mix.click = playback.Clip.from_wav(spec.click_file)
        mix.click_on = True
        mix.click_level = spec.click_level
    return mix


def run(spec: BounceSpec, target: Path, each_part: bool, span: tuple[float, float] | None, flac: bool,
        title: str, work_dir: str, report=None, should_stop=None) -> list[str]:
    """Writes the file(s) and returns their paths."""
    from .project import _encode_flac

    with tempfile.TemporaryDirectory(prefix="bounce-", dir=work_dir) as tmp:
        scratch = Path(tmp)
        mix = build_mix(spec, scratch, report, should_stop)
        start, end = span if span else (0.0, None)
        jobs = []
        if each_part:
            for lane in spec.lanes:
                safe = "".join(c for c in lane.name if c.isalnum() or c in " -_()").strip() or lane.uid
                jobs.append((lane.uid, target / f"{title} - {safe}"))
        else:
            jobs.append((None, target.with_suffix("")))
        written = []
        for i, (only, stem) in enumerate(jobs):
            out = stem.with_name(stem.name + (".flac" if flac else ".wav"))
            wav = scratch / f"part-{i}.wav" if flac else out

            def part_report(f, i=i):
                if report:
                    report(0.5 + 0.5 * (i + f) / len(jobs))

            playback.bounce(mix, wav, start, end, only=only, master=only is None, report=part_report,
                            should_stop=should_stop)
            if flac:
                _encode_flac(str(wav), out)
            written.append(str(out))
        return written


class BounceDialog(QDialog):
    def __init__(self, parent, has_span: bool, parts: int):
        super().__init__(parent)
        self.setWindowTitle(tr("Bounce (save what you hear)"))
        lay = QVBoxLayout(self)
        intro = QLabel(tr("Saves the sound with your volumes, pans, mutes and solos, and the Play setting "
                          "(recording, notes on instruments, or both)."))
        intro.setWordWrap(True)
        intro.setObjectName("Hint")
        lay.addWidget(intro)
        form = QFormLayout()
        self.what = QComboBox()
        self.what.addItem(tr("The whole mix, in one file"), "mix")
        self.what.addItem(tr("Each part in its own file"), "parts")
        self.what.model().item(1).setEnabled(parts > 0)
        self.range = QComboBox()
        self.range.addItem(tr("The whole song"), "all")
        self.range.addItem(tr("Only the selected span"), "span")
        self.range.model().item(1).setEnabled(has_span)
        if has_span:
            self.range.setCurrentIndex(1)
        self.format = QComboBox()
        self.format.addItem(tr("WAV (16-bit)"), "wav")
        self.format.addItem(tr("FLAC (lossless, smaller)"), "flac")
        for w in (self.what, self.range, self.format):
            w.setProperty("i18n_skip_items", True)
        form.addRow(tr("Save"), self.what)
        form.addRow(tr("Length"), self.range)
        form.addRow(tr("Format"), self.format)
        lay.addLayout(form)
        note = QLabel(tr("Each part on its own keeps its volume and pan, but not the master volume."))
        note.setWordWrap(True)
        note.setObjectName("Hint")
        lay.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Bounce..."))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def each_part(self) -> bool:
        return self.what.currentData() == "parts"

    def span_only(self) -> bool:
        return self.range.currentData() == "span"

    def flac(self) -> bool:
        return self.format.currentData() == "flac"
