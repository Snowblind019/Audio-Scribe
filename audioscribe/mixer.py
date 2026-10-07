"""Renders the sounds the live mixer (playback.py) plays that are not recordings:

  * one file per part with its notes played on the chosen instrument,
  * the click track.

Each is rendered once into a WAV file in the work folder and only again when the notes, the
instrument, the room sound or the beat grid change. Volume, pan, mute and solo are applied
live while playing, so they never need a new render.

Nothing here touches Qt, so it can run on a worker thread. It works through the song in
10 second blocks, so even a long file only needs a little memory.
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import synth

SR = synth.SR
BLOCK = 10 * SR

MODES = [("recording", "Recording"), ("notes", "Notes on instruments"), ("both", "Recording and notes")]


class Stopped(Exception):
    """Raised when a newer request makes the one in progress pointless."""


@dataclass
class NotesJob:
    """One part's notes, to be played on an instrument."""
    uid: str
    instrument: str
    notes: list                # (start, end, pitch, velocity) tuples
    duration: float
    room: bool

    def signature(self) -> str:
        """Identifies what the output would sound like, so it is only rendered again when it changes."""
        h = hashlib.sha1(usedforsecurity=False)   # only a cache key
        h.update(f"notes|{self.instrument}|{int(self.room)}|{self.duration:.3f}|".encode())
        h.update(np.asarray(self.notes, dtype=np.float64).tobytes())
        return h.hexdigest()


@dataclass
class ClickJob:
    beats: list[float]
    meter: int
    duration: float
    uid: str = "click"
    notes: list = field(default_factory=list)

    def signature(self) -> str:
        h = hashlib.sha1(usedforsecurity=False)
        h.update(f"click|{self.meter}|{self.duration:.3f}|".encode())
        h.update(np.asarray(self.beats, dtype=np.float64).tobytes())
        return h.hexdigest()


def write_stereo(path: str | Path, data: np.ndarray) -> None:
    """Write float audio shaped (frames, 2) as a 16-bit WAV file."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _auto_gain(ns: synth.NoteSet, duration: float) -> float:
    """Busy music has many notes sounding at once, which adds up. Turn it down a little
    when the average number of notes at once is high, so it doesn't clip."""
    if ns.inst.one_shot:
        return 1.0
    held = float(np.sum(ns.end - ns.start))
    average = held / max(duration, 1.0)
    return 1.0 / max(1.0, (average / 3.0) ** 0.5)


def render_notes(job: NotesJob, out_path: str | Path, report: Callable[[float], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> None:
    """One part's notes on its instrument, as a stereo WAV as long as the song."""
    total = int(round(job.duration * SR))
    ns = synth.NoteSet.from_notes(job.instrument, job.notes)
    room = synth.Room() if job.room else None
    level = np.float32(_auto_gain(ns, job.duration))
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SR)
        pos = 0
        while pos < total:
            if should_stop and should_stop():
                raise Stopped()
            n = min(BLOCK, total - pos)
            dry = synth.render_chunk([ns], pos / SR, n) if len(ns.start) else np.zeros(n, np.float32)
            wet = room.process(dry) if room else np.repeat(dry[:, None], 2, axis=1)
            buf = synth.soft_clip(wet * level, 0.9)
            out.writeframes((np.clip(buf, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
            pos += n
            if report:
                report(pos / max(total, 1))


def render_click(job: ClickJob, out_path: str | Path, report: Callable[[float], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> None:
    """A metronome on the beat grid. Its level is set live in the mixer."""
    total = int(round(job.duration * SR))
    data = np.zeros(total, np.float32)
    tick = {True: synth.click_sound(True), False: synth.click_sound(False)}
    for i, t in enumerate(job.beats):
        if should_stop and should_stop():
            raise Stopped()
        start = int(round(t * SR))
        if start < 0 or start >= total:
            continue
        sound = tick[i % max(1, job.meter) == 0]
        end = min(total, start + len(sound))
        data[start:end] += sound[:end - start]
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        for pos in range(0, total, BLOCK):
            out.writeframes((np.clip(data[pos:pos + BLOCK], -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    if report:
        report(1.0)


def render(job, out_path: str | Path, report: Callable[[float], None] | None = None,
           should_stop: Callable[[], bool] | None = None) -> None:
    if isinstance(job, ClickJob):
        render_click(job, out_path, report, should_stop)
    else:
        render_notes(job, out_path, report, should_stop)
