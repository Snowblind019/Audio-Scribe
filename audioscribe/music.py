"""Small music helpers: note names, time formatting, and key estimation."""

from __future__ import annotations

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
