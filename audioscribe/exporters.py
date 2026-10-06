"""Saving results: plain text, subtitles, timed lyrics, MIDI, and CSV."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from . import synth
from .i18n import tr
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
        writer.writerow([tr(name), note_name(n.pitch), n.pitch, f"{n.start:.3f}", f"{n.end:.3f}",
                         f"{n.end - n.start:.3f}", f"{n.velocity:.2f}"])
    return buf.getvalue()


def save_midi(tracks, path: str | Path, tempo: float | None, meter: int = 4) -> None:
    """One MIDI track per analysis track. The tempo and beats per bar are written in so
    the notes line up with the bar grid when opened in a DAW."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo) if tempo else 120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(int(meter or 4), 4, 0.0))
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


# Lyrics with chords -----------------------------------------------------------------------

def _chord_changes(chords) -> list[tuple[float, object]]:
    out = []
    for start, _end, chord in chords:
        if chord is None:
            continue
        if out and out[-1][1] == chord:
            continue
        out.append((start, chord))
    return out


def _line_words(seg) -> list[tuple[float, str]]:
    words = [(w.start, w.text.strip()) for w in (seg.words or []) if w.text.strip()]
    parts = seg.text.split()
    # Use the word timings when they cover the line. Otherwise spread the words of the line evenly.
    if words and len(words) >= len(parts):
        return words
    if not parts:
        return []
    step = (seg.end - seg.start) / len(parts)
    return [(seg.start + i * step, p) for i, p in enumerate(parts)]


def _chord_lines(segments, chords) -> list[tuple[str, list[tuple[int, object]] | None, list]]:
    """For each lyric line: (text, [(word index, chord)], instrumental chords before it)."""
    changes = _chord_changes(chords)
    segs = [s for s in segments if s.text.strip()]
    out = []
    used = 0
    current = None
    for i, seg in enumerate(segs):
        words = _line_words(seg)
        next_start = segs[i + 1].start if i + 1 < len(segs) else float("inf")
        line_end = min(seg.end + 0.2, next_start - 0.3)
        between = []
        while used < len(changes) and changes[used][0] < seg.start - 0.3:
            current = changes[used][1]
            between.append(changes[used])
            used += 1
        # Chords between lines are shown on their own line when they make a real gap.
        gap = (seg.start - between[0][0]) if between else 0.0
        instrumental = [c for _t, c in between] if (len(between) >= 2 or gap >= 2.0) else []
        placed: list[tuple[int, object]] = []
        starts = [w[0] for w in words]
        if current is not None and words:
            placed.append((0, current))
        while used < len(changes) and changes[used][0] < line_end - 0.3:
            t, chord = changes[used]
            used += 1
            current = chord
            idx = 0
            for k, s in enumerate(starts):
                if s <= t + 0.15:
                    idx = k
            placed = [p for p in placed if p[0] != idx]
            placed.append((idx, chord))
        out.append((" ".join(w for _, w in words), sorted(placed, key=lambda p: p[0]), instrumental, words))
    tail = [c for _t, c in changes[used:]]
    if tail:
        out.append(("", [], tail, []))
    return out


def lyrics_with_chords(segments, chords, title: str | None = None) -> str:
    """Plain text with the chord names above the words they fall on."""
    lines = [title, ""] if title else []
    for text, placed, instrumental, words in _chord_lines(segments, chords):
        if instrumental:
            lines.append("(" + "  ".join(c.name() for c in instrumental) + ")")
        if not words:
            continue
        offsets, pos = [], 0
        for _t, w in words:
            offsets.append(pos)
            pos += len(w) + 1
        chord_line = ""
        for idx, chord in placed:
            col = max(offsets[idx], len(chord_line) + (1 if chord_line else 0))
            chord_line = chord_line.ljust(col) + chord.name()
        if chord_line:
            lines.append(chord_line.rstrip())
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def chordpro(segments, chords, title: str | None = None) -> str:
    """ChordPro format: [C]words with the chords in brackets, read by most chord sheet apps."""
    lines = [f"{{title: {title}}}", ""] if title else []
    for _text, placed, instrumental, words in _chord_lines(segments, chords):
        if instrumental:
            lines.append("{comment: " + " ".join(c.name() for c in instrumental) + "}")
        if not words:
            continue
        at = dict(placed)
        parts = []
        for k, (_t, w) in enumerate(words):
            parts.append((f"[{at[k].name()}]" if k in at else "") + w)
        lines.append(" ".join(parts))
    return "\n".join(lines).rstrip() + "\n"
