"""Finding the chords in the sound itself (not in the notes that were found).

It measures how strong each of the 12 note names is over time (chroma), averages that
between beats, and lets theory.chords_from_chroma pick the most likely chord sequence.
When the song was split into stems, the drums are left out before measuring.
"""

from __future__ import annotations

import numpy as np

from .theory import Chord, chords_from_chroma

SR = 22050
HOP = 512


def detect(y: np.ndarray, sr: int, beats: list[float], duration: float) -> list[tuple[float, float, Chord | None]]:
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if len(y) < sr // 2 or float(np.max(np.abs(y))) < 1e-4:
        return []
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, bins_per_octave=36, n_octaves=6,
                                        fmin=librosa.note_to_hz("C1"))
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=SR, hop_length=HOP)

    # One slot per beat when there is a steady beat, otherwise half a second.
    if len(beats) >= 8:
        edges = [0.0] + [b for b in beats if 0.0 < b < duration] + [duration]
    else:
        edges = list(np.arange(0.0, duration, 0.5)) + [duration]
    edges = sorted(set(round(e, 4) for e in edges))
    slots, times = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 0.05:
            continue
        mask = (frame_t >= a) & (frame_t < b)
        slots.append(np.median(chroma[:, mask], axis=1) if mask.any() else np.zeros(12))
        times.append(a)
    if not slots:
        return []
    found = chords_from_chroma(np.array(slots).T, times, duration)
    # Very short chords (under a beat or so) are usually passing notes.
    out: list[tuple[float, float, Chord | None]] = []
    for a, b, c in found:
        if out and b - a < 0.35:
            pa, _pb, pc = out[-1]
            out[-1] = (pa, b, pc)
            continue
        if out and out[-1][2] == c:
            out[-1] = (out[-1][0], b, c)
        else:
            out.append((a, b, c))
    return out
