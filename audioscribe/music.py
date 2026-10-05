"""Small music helpers: note names, time formatting, and key estimation."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BLACK_KEYS = {1, 3, 6, 8, 10}
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)

# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def note_name(pitch: int) -> str:
    """MIDI pitch to a name like C4 (middle C is 60)."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def is_black(pitch: int) -> bool:
    return pitch % 12 in BLACK_KEYS


def format_time(seconds: float, decimals: int = 1) -> str:
    """Seconds to m:ss.d (or h:mm:ss.d for long files)."""
    seconds = max(0.0, float(seconds))
    scale = 10 ** decimals
    total = int(round(seconds * scale))
    whole, frac = divmod(total, scale)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    if decimals:
        text += f".{frac:0{decimals}d}"
    return text


@dataclass
class KeyEstimate:
    tonic: int
    mode: str  # "major" or "minor"
    confidence: str  # "high", "medium" or "low"
    alternative: str | None = None

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {self.mode}"

    @property
    def scale(self) -> set[int]:
        steps = MAJOR_STEPS if self.mode == "major" else MINOR_STEPS
        return {(self.tonic + s) % 12 for s in steps}


def pitch_class_weights(notes: Iterable) -> list[float]:
    """Total sounding time per pitch class (C to B), weighted a bit by loudness."""
    weights = [0.0] * 12
    for n in notes:
        weights[n.pitch % 12] += max(0.0, n.end - n.start) * (0.5 + n.velocity)
    return weights


def estimate_key(weights: list[float]) -> KeyEstimate | None:
    """Best matching major or minor key for a pitch class histogram."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0 or np.count_nonzero(w) < 3:
        return None
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(w, np.roll(profile, tonic))[0, 1])
            scores.append((r, tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    margin = best[0] - second[0]
    if best[0] > 0.75 and margin > 0.08:
        confidence = "high"
    elif best[0] > 0.6 and margin > 0.03:
        confidence = "medium"
    else:
        confidence = "low"
    alternative = f"{NOTE_NAMES[second[1]]} {second[2]}" if margin < 0.05 else None
    return KeyEstimate(best[1], best[2], confidence, alternative)


def used_notes(notes: Iterable) -> dict[int, tuple[int, float]]:
    """Pitch -> (times played, total seconds)."""
    out: dict[int, list] = {}
    for n in notes:
        entry = out.setdefault(n.pitch, [0, 0.0])
        entry[0] += 1
        entry[1] += max(0.0, n.end - n.start)
    return {p: (c, t) for p, (c, t) in out.items()}


# Names for the drum hits that Audio Scribe finds (General MIDI drum notes).
DRUM_NAMES = {35: "Kick", 36: "Kick", 37: "Rim", 38: "Snare", 39: "Clap", 40: "Snare", 41: "Tom",
              42: "Hi-hat", 43: "Tom", 44: "Hi-hat", 45: "Tom", 46: "Open hat", 47: "Tom", 48: "Tom",
              49: "Crash", 50: "Tom", 51: "Ride", 52: "Crash", 53: "Ride", 54: "Shaker", 55: "Crash",
              56: "Cowbell", 57: "Crash", 59: "Ride", 69: "Shaker", 70: "Shaker"}
KICK_PITCH, SNARE_PITCH, HAT_PITCH = 36, 38, 42


def note_label(pitch: int, part: str | None = None) -> str:
    """Note name, or the drum name on the Drums part."""
    if part == "Drums" and pitch in DRUM_NAMES:
        return DRUM_NAMES[pitch]
    return note_name(pitch)


def pitch_hz(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


# Scales for the "show only" filter. Steps are semitones above the root.
SCALES: list[tuple[str, tuple[int, ...] | None]] = [
    ("All notes", None),
    ("Major", MAJOR_STEPS),
    ("Natural minor", MINOR_STEPS),
    ("Harmonic minor", (0, 2, 3, 5, 7, 8, 11)),
    ("Melodic minor", (0, 2, 3, 5, 7, 9, 11)),
    ("Major pentatonic", (0, 2, 4, 7, 9)),
    ("Minor pentatonic", (0, 3, 5, 7, 10)),
    ("Blues", (0, 3, 5, 6, 7, 10)),
    ("Dorian", (0, 2, 3, 5, 7, 9, 10)),
    ("Phrygian", (0, 1, 3, 5, 7, 8, 10)),
    ("Lydian", (0, 2, 4, 6, 7, 9, 11)),
    ("Mixolydian", (0, 2, 4, 5, 7, 9, 10)),
    ("Locrian", (0, 1, 3, 5, 6, 8, 10)),
    ("Whole tone", (0, 2, 4, 6, 8, 10)),
    ("Major chord", (0, 4, 7)),
    ("Minor chord", (0, 3, 7)),
]


def scale_classes(root: int, steps: tuple[int, ...] | None) -> frozenset[int] | None:
    """Pitch classes (0 = C) of a scale, or None for 'every note'."""
    if steps is None:
        return None
    return frozenset((root + s) % 12 for s in steps)


def match_scale(classes: frozenset[int] | None) -> tuple[int, int] | None:
    """(root, index into SCALES) when the pitch classes are exactly one of the scales."""
    if classes is None:
        return 0, 0
    for i, (_, steps) in enumerate(SCALES[1:], start=1):
        for root in range(12):
            if scale_classes(root, steps) == classes:
                return root, i
    return None


@dataclass
class NoteFilter:
    """Which notes to show: only some pitch classes (a scale), and/or only a range of notes."""
    classes: frozenset[int] | None = None   # None means every note is allowed
    outside: bool = False                   # True: show the notes that are NOT in `classes`
    low: int = 0
    high: int = 127
    label: str = ""

    def allows(self, pitch: int, drum: bool = False) -> bool:
        """Drum hits are never filtered: their "pitch" only says which drum it is, so a
        scale or a range of notes means nothing for them. Mute the drums to hide them."""
        if drum:
            return True
        if pitch < self.low or pitch > self.high:
            return False
        if self.classes is None:
            return True
        return ((pitch % 12) in self.classes) != self.outside

    @property
    def active(self) -> bool:
        return self.classes is not None or self.low > 0 or self.high < 127

    def describe(self) -> str:
        parts = []
        if self.classes is not None:
            names = self.label or ", ".join(NOTE_NAMES[c] for c in sorted(self.classes))
            parts.append(f"not in {names}" if self.outside else f"only {names}")
        if self.low > 0 or self.high < 127:
            parts.append(f"{note_name(max(self.low, 0))} to {note_name(min(self.high, 127))}")
        return ", ".join(parts)


# Chords ------------------------------------------------------------------------------

CHORD_TYPES = [("", (0, 4, 7)), ("m", (0, 3, 7)), ("dim", (0, 3, 6)), ("aug", (0, 4, 8)),
               ("sus4", (0, 5, 7)), ("sus2", (0, 2, 7)), ("7", (0, 4, 7, 10)), ("maj7", (0, 4, 7, 11)),
               ("m7", (0, 3, 7, 10)), ("m7b5", (0, 3, 6, 10)), ("6", (0, 4, 7, 9)), ("m6", (0, 3, 7, 9))]


def guess_chord(weights: list[float], bass: int | None = None) -> tuple[str, float] | None:
    """Best matching chord name for a pitch class histogram, and a 0..1 score.

    This is a rough guess. It scores how much of the sound sits on the chord's notes, and
    penalizes chord notes that are missing and notes that don't belong."""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return None
    present = np.count_nonzero(w > 0.06 * w.max())
    if present < 2:
        pc = int(np.argmax(w))
        return NOTE_NAMES[pc], 0.3
    best = None
    for root in range(12):
        for suffix, steps in CHORD_TYPES:
            tones = {(root + s) % 12 for s in steps}
            inside = sum(w[t] for t in tones) / total
            missing = sum(1 for t in tones if w[t] < 0.06 * w.max())
            score = inside - 0.18 * missing - 0.015 * (len(steps) - 3)
            if bass is not None and bass == root:
                score += 0.06
            if best is None or score > best[0]:
                best = (score, root, suffix, tones)
    score, root, suffix, tones = best
    name = f"{NOTE_NAMES[root]}{suffix}"
    if bass is not None and bass in tones and bass != root:
        name += f"/{NOTE_NAMES[bass]}"
    return name, float(max(0.0, min(1.0, score)))


def guess_chords(notes: Iterable, t0: float, t1: float, step: float) -> list[tuple[float, float, str]]:
    """Walks through [t0, t1) in windows of `step` seconds and names the chord in each.
    Windows with the same chord next to each other are joined."""
    notes = sorted((n for n in notes if n.end > t0 and n.start < t1), key=lambda n: n.start)
    starts = [n.start for n in notes]
    longest = max((n.end - n.start for n in notes), default=0.0)
    out: list[tuple[float, float, str]] = []
    t = t0
    while t < t1 - 1e-6:
        e = min(t + step, t1)
        weights = [0.0] * 12
        low_pitch, low_weight = None, 0.0
        for n in notes[bisect.bisect_left(starts, t - longest):bisect.bisect_left(starts, e)]:
            overlap = min(n.end, e) - max(n.start, t)
            if overlap <= 0:
                continue
            w = overlap * (0.5 + n.velocity)
            weights[n.pitch % 12] += w
            if n.pitch < 55 and w > low_weight:
                low_pitch, low_weight = n.pitch, w
        guess = guess_chord(weights, None if low_pitch is None else low_pitch % 12)
        name = guess[0] if guess and guess[1] >= 0.45 else ("-" if guess is None else "?")
        if out and out[-1][2] == name:
            out[-1] = (out[-1][0], e, name)
        else:
            out.append((t, e, name))
        t = e
    return out
