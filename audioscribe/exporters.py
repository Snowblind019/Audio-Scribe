"""Saving results: plain text, subtitles, timed lyrics, MIDI, and CSV."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from . import synth
from .music import note_name

# General MIDI programs (0-based) used when a track has no instrument chosen.
_PROGRAMS = {"Vocals": 53, "Bass": 33, "Other": 0, "Full mix": 0}


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcript_text(segments) -> str:
    return "\n".join(s.text for s in segments if s.text) + "\n"


def transcript_srt(segments) -> str:
    blocks = []
    for i, s in enumerate((s for s in segments if s.text), start=1):
        blocks.append(f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{s.text}\n")
    return "\n".join(blocks)


def transcript_lrc(segments, title: str | None = None) -> str:
    lines = [f"[ti:{title}]"] if title else []
    for s in segments:
        if not s.text:
            continue
        cs = int(round(s.start * 100))
        m, cs = divmod(cs, 6000)
        sec, cs = divmod(cs, 100)
        lines.append(f"[{m:02d}:{sec:02d}.{cs:02d}]{s.text}")
    return "\n".join(lines) + "\n"


def notes_csv(tracks) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["track", "note", "midi", "start_s", "end_s", "length_s", "strength"])
    rows = [(t.name, n) for t in tracks for n in t.notes]
    rows.sort(key=lambda r: (r[1].start, r[1].pitch))
    for name, n in rows:
        writer.writerow([name, note_name(n.pitch), n.pitch, f"{n.start:.3f}", f"{n.end:.3f}",
                         f"{n.end - n.start:.3f}", f"{n.velocity:.2f}"])
    return buf.getvalue()


def save_midi(tracks, path: str | Path, tempo: float | None) -> None:
    """One MIDI track per analysis track. The detected tempo is written in so
    the notes line up with the bar grid when opened in a DAW."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo) if tempo else 120.0)
    for track in tracks:
        is_drum = track.name == "Drums" or getattr(track, "instrument", None) == synth.DRUM_KEY
        chosen = synth.BY_KEY.get(getattr(track, "instrument", None))
        program = chosen.program if chosen is not None and not is_drum else _PROGRAMS.get(track.name, 0)
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=track.name)
        for n in track.notes:
            velocity = max(1, min(127, int(round(30 + 97 * n.velocity))))
            inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(n.pitch),
                                               start=float(n.start),
                                               end=float(max(n.end, n.start + 0.01))))
        pm.instruments.append(inst)
    pm.write(str(path))


def write_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
