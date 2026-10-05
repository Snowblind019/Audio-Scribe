"""Finding drum hits in a drum recording.

Drums have no real pitch, so note finders like Basic Pitch mostly return noise for them.
Instead this watches how fast the sound jumps up in different parts of the spectrum:

  * the low end (30 to 150 Hz) jumping is a kick,
  * the mid and upper mid (150 Hz to 5 kHz) jumping together is a snare,
  * the very top (5 kHz and up) jumping, with no snare there, is a hi-hat or cymbal.

Each part is picked separately, so a kick and a hi-hat on the same beat give two hits.
"""

from __future__ import annotations

import numpy as np

from .music import HAT_PITCH, KICK_PITCH, SNARE_PITCH

HOP = 256
N_FFT = 1024
# Frequency bands in Hz.
BANDS = {"low": (30.0, 150.0), "mid": (150.0, 1500.0), "upper": (1500.0, 5000.0), "high": (5000.0, 11000.0)}
HIT_LENGTH = 0.12


def _jump(power: np.ndarray, lag: int = 2) -> np.ndarray:
    """How quickly a band gets louder, in decibels, scaled so a big jump is about 1."""
    db = 10.0 * np.log10(power + 1e-12)
    db = np.maximum(db, db.max() - 30.0)
    out = np.zeros_like(db)
    out[lag:] = np.maximum(0.0, db[lag:] - db[:-lag])
    scale = float(np.percentile(out, 99)) or 1.0
    return np.minimum(out / scale, 1.5)


def detect_hits(y: np.ndarray, sr: int, sensitivity: int = 50) -> list[tuple[float, int, float]]:
    """Returns (start_seconds, pitch, strength 0..1) for each hit found in the mono audio `y`.

    Pitches follow General MIDI: 36 kick, 38 snare, 42 closed hi-hat."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if len(y) < N_FFT * 2 or float(np.max(np.abs(y))) < 1e-4:
        return []
    spec = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = {name: spec[(freqs >= lo) & (freqs < min(hi, sr / 2))].sum(axis=0) for name, (lo, hi) in BANDS.items()}
    jump = {name: _jump(p) for name, p in power.items()}

    s = min(max(sensitivity, 0), 100) / 100.0
    delta = 0.5 - 0.4 * s
    frames_per_s = sr / HOP

    def pick(env: np.ndarray, gap: float) -> np.ndarray:
        return librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=6, post_avg=6, delta=delta,
                                      wait=max(1, int(gap * frames_per_s)))

    snare_env = np.minimum(jump["mid"], jump["upper"])
    kicks = pick(jump["low"], 0.05)
    snares = pick(snare_env, 0.06)
    hats = pick(jump["high"], 0.03)

    # The jump pickers only know "something got louder", so a kick's click can look like a
    # snare and a snare's body can look like a kick. Check how the new energy is split
    # between the bands: a kick is mostly low, a snare is mostly mid and upper mid.
    n_frames = spec.shape[1]

    def share(f: int) -> dict[str, float]:
        before = slice(max(0, f - 6), max(1, f - 1))
        after = slice(min(f + 1, n_frames - 1), min(f + 5, n_frames))
        flux = {name: max(0.0, float(p[after].mean() - p[before].mean())) for name, p in power.items()}
        total = sum(flux.values()) or 1.0
        return {name: v / total for name, v in flux.items()}

    kicks = [f for f in kicks if share(f)["low"] >= 0.2]
    snares = [f for f in snares if (lambda sh: sh["mid"] + sh["upper"] >= 0.3 and sh["mid"] >= 0.1)(share(f))]

    near = int(0.035 * frames_per_s) + 1
    snare_set = np.asarray(snares)
    hats = [f for f in hats if len(snare_set) == 0 or np.min(np.abs(snare_set - f)) > near]

    hits = []
    for pitch, frames, env in ((KICK_PITCH, kicks, jump["low"]), (SNARE_PITCH, snares, snare_env),
                               (HAT_PITCH, hats, jump["high"])):
        for f in frames:
            strength = float(min(1.0, max(0.3, 0.3 + 0.7 * env[f])))
            # Plain Python numbers only: numpy numbers can crash Qt when they are stored in its items.
            hits.append((float(max(0.0, f * HOP / sr)), int(pitch), float(strength)))
    hits.sort()
    return hits
