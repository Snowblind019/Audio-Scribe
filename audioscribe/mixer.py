"""Builds the audio file the player plays.

Depending on what is muted and which sound is chosen, that file is:
  * the recording (all of it, or only the stems that are not muted),
  * the notes played on instruments, or
  * both together.

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
class PartSpec:
    name: str
    audio: str | None          # the stem (or the original) for this part
    audible: bool
    instrument: str
    notes: list                # (start, end, pitch, velocity) tuples that should be played


@dataclass
class MixSpec:
    mode: str                  # "recording", "notes" or "both"
    duration: float
    notes_level: float         # 0..1, only used when mode is "both"
    room: bool
    parts: list[PartSpec] = field(default_factory=list)

    @property
    def uses_recording(self) -> bool:
        return self.mode in ("recording", "both")

    @property
    def uses_notes(self) -> bool:
        return self.mode in ("notes", "both")

    def is_plain(self) -> bool:
        """True when the player can just open the original file."""
        return self.mode == "recording" and all(p.audible for p in self.parts)

    def signature(self) -> str:
        """Identifies what the output would sound like, so it is only rebuilt when it changes."""
        h = hashlib.sha1()
        h.update(f"{self.mode}|{self.duration:.3f}".encode())
        if self.uses_recording:
            h.update(("|".join(sorted(p.audio or "" for p in self.parts if p.audible))).encode())
        if self.uses_notes:
            h.update(f"|{int(self.room)}".encode())
            if self.mode == "both":
                h.update(f"|{self.notes_level:.3f}".encode())
            for p in self.parts:
                if p.audible and p.notes:
                    h.update(f"|{p.name}|{p.instrument}".encode())
                    h.update(np.asarray(p.notes, dtype=np.float64).tobytes())
        return h.hexdigest()


class _Reader:
    """Reads a 16-bit WAV file from the start, a block at a time."""

    def __init__(self, path: str) -> None:
        self.wav = wave.open(path, "rb")
        if self.wav.getsampwidth() != 2 or self.wav.getframerate() != SR:
            raise ValueError("Unexpected audio format in a working file.")
        self.channels = self.wav.getnchannels()

    def add_to(self, buf: np.ndarray) -> None:
        raw = self.wav.readframes(len(buf))
        if not raw:
            return
        data = np.frombuffer(raw, dtype="<i2").reshape(-1, self.channels).astype(np.float32) / 32768.0
        n = min(len(buf), len(data))
        if self.channels == 1:
            buf[:n] += data[:n]
        else:
            buf[:n] += data[:n, :2]

    def close(self) -> None:
        self.wav.close()


def write_stereo(path: str | Path, data: np.ndarray) -> None:
    """Write float audio shaped (frames, 2) as a 16-bit WAV file."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def _auto_gain(sets: list, duration: float) -> float:
    """Busy music has many notes sounding at once, which adds up. Turn it down a little
    when the average number of notes at once is high, so it doesn't clip."""
    held = sum(float(np.sum(ns.end - ns.start)) for ns in sets if not ns.inst.one_shot)
    average = held / max(duration, 1.0)
    return 1.0 / max(1.0, (average / 3.0) ** 0.5)


def render_mix(spec: MixSpec, out_path: str | Path, report: Callable[[float], None] | None = None,
               should_stop: Callable[[], bool] | None = None) -> None:
    total = int(round(spec.duration * SR))
    readers: list[_Reader] = []
    try:
        if spec.uses_recording:
            readers = [_Reader(p.audio) for p in spec.parts if p.audible and p.audio]
        sets = []
        if spec.uses_notes:
            sets = [synth.NoteSet.from_notes(p.instrument, p.notes) for p in spec.parts if p.audible and p.notes]
        room = synth.Room() if (spec.room and sets) else None
        level = spec.notes_level if spec.mode == "both" else 1.0
        level *= _auto_gain(sets, spec.duration)
        with wave.open(str(out_path), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(SR)
            pos = 0
            while pos < total:
                if should_stop and should_stop():
                    raise Stopped()
                n = min(BLOCK, total - pos)
                buf = np.zeros((n, 2), np.float32)
                for r in readers:
                    r.add_to(buf)
                if sets:
                    dry = synth.render_chunk(sets, pos / SR, n)
                    wet = room.process(dry) if room else np.repeat(dry[:, None], 2, axis=1)
                    buf += wet * np.float32(level)
                    buf = synth.soft_clip(buf, 0.9)
                out.writeframes((np.clip(buf, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
                pos += n
                if report:
                    report(pos / max(total, 1))
    finally:
        for r in readers:
            r.close()
