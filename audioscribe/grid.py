"""The beat grid: where the beats are, how many make a bar, and tap tempo.

The analysis finds the beats on its own. When it gets them wrong (half or double speed, or
shifted), the grid can be set by hand from a tempo and the time of the first beat.
"""

from __future__ import annotations

import bisect
import time

import numpy as np


def make_beats(bpm: float, first: float, duration: float) -> list[float]:
    """Evenly spaced beats from `first` (going back to the start too) to the end."""
    if bpm <= 0 or duration <= 0:
        return []
    period = 60.0 / bpm
    start = first - period * int(first // period) if first > 0 else first
    count = int((duration - start) / period) + 1
    return [round(start + i * period, 5) for i in range(max(0, count)) if start + i * period <= duration + 1e-6]


def period_of(beats: list[float]) -> float | None:
    if len(beats) < 2:
        return None
    return float(np.median(np.diff(beats)))


def bpm_of(beats: list[float]) -> float | None:
    p = period_of(beats)
    return 60.0 / p if p else None


def time_to_beat(t: float, beats: list[float]) -> float:
    """Position in beats (fractional) of time t, following the grid even where the tempo drifts."""
    if len(beats) < 2:
        return t / 0.5
    p0, p1 = beats[1] - beats[0], beats[-1] - beats[-2]
    if t <= beats[0]:
        return (t - beats[0]) / p0
    if t >= beats[-1]:
        return len(beats) - 1 + (t - beats[-1]) / p1
    i = bisect.bisect_right(beats, t) - 1
    return i + (t - beats[i]) / (beats[i + 1] - beats[i])


def beat_to_time(b: float, beats: list[float]) -> float:
    if len(beats) < 2:
        return b * 0.5
    p0, p1 = beats[1] - beats[0], beats[-1] - beats[-2]
    if b <= 0:
        return beats[0] + b * p0
    if b >= len(beats) - 1:
        return beats[-1] + (b - (len(beats) - 1)) * p1
    i = int(b)
    return beats[i] + (b - i) * (beats[i + 1] - beats[i])


class TapTempo:
    """Tap along with the song: the tempo is the average gap between recent taps. If a song
    position is given with each tap, the grid also lines up with the taps."""

    def __init__(self) -> None:
        self.taps: list[tuple[float, float | None]] = []

    def tap(self, song_time: float | None = None, now: float | None = None) -> tuple[float | None, float | None]:
        now = time.monotonic() if now is None else now
        if self.taps and now - self.taps[-1][0] > 2.5:
            self.taps = []
        self.taps.append((now, song_time))
        self.taps = self.taps[-12:]
        if len(self.taps) < 3:
            return None, None
        gaps = np.diff([t for t, _ in self.taps])
        period = float(np.median(gaps))
        if period <= 0.15:
            return None, None
        bpm = 60.0 / period
        song = [s for _, s in self.taps if s is not None]
        first = None
        if len(song) >= 3:
            # average phase of the taps against the period
            phases = np.array(song) % period
            angle = np.angle(np.mean(np.exp(2j * np.pi * phases / period)))
            first = float((angle / (2 * np.pi)) * period) % period
        return bpm, first
