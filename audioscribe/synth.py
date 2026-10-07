"""Sound for playing notes back.

Every instrument here is made from math (numpy and scipy), so there are no sound
files, no sound fonts, and nothing extra to download. That keeps the install small and
fully pinned, and it works offline. The sounds are good for hearing what the notes are
and for checking an edit, but they are synthesized, not recordings of real instruments.

How it works:
  * Each instrument knows how to render one note (a pitch, a velocity layer, a length).
  * A Bank keeps every rendered note, so a pitch is only built once per session.
  * render_chunk() lays the notes of a time window into a buffer, and Room adds a little
    reverb so it doesn't sound dry and flat.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates

SR = 44100
TARGET_RMS = 0.075
TWO_PI = 2.0 * np.pi


# Helpers ---------------------------------------------------------------------------

def hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def _rng(*key) -> np.random.Generator:
    """The same note always gets the same noise, so renders are repeatable."""
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def _sine(freq: float, n: int, phase: float = 0.0, tau: float | None = None) -> np.ndarray:
    """exp(-t / tau) * sin(2 pi freq t + phase) as float32.

    Built from powers of one complex number (a block of small steps times a block of big
    steps), which is much faster than calling sin on every sample and just as accurate.
    """
    if n <= 0:
        return np.zeros(0, np.float32)
    step = complex(0.0 if tau is None else -1.0 / tau, TWO_PI * freq) / SR
    block = 256
    blocks = -(-n // block)
    inner = np.exp(step * np.arange(block))
    outer = np.exp(step * block * np.arange(blocks) + 1j * phase)
    z = np.multiply.outer(outer, inner).reshape(-1)[:n]
    return z.imag.astype(np.float32)


def _filt(x: np.ndarray, kind: str, freq, order: int = 2) -> np.ndarray:
    if kind == "bandpass":
        lo, hi = freq
        freq = (max(20.0, lo), min(hi, SR * 0.46))
    else:
        freq = min(max(freq, 20.0), SR * 0.46)
    sos = signal.butter(order, freq, kind, fs=SR, output="sos")
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float64)).astype(np.float32)


def _lp(x, fc, order=2):
    return _filt(x, "lowpass", fc, order)


def _hp(x, fc, order=2):
    return _filt(x, "highpass", fc, order)


def _bp(x, lo, hi, order=2):
    return _filt(x, "bandpass", (lo, hi), order)


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n).astype(np.float32)
    return x / (np.sqrt(np.mean(x * x)) + 1e-9)


def _gauss(f: float, centre: float, octaves: float) -> float:
    return float(np.exp(-0.5 * (np.log2(max(f, 1.0) / centre) / octaves) ** 2))


def _attack(n: int, seconds: float) -> np.ndarray:
    """Smooth rise from 0 to 1 that never has a corner (no click at the start)."""
    t = np.arange(n, dtype=np.float32) / SR
    return (1.0 - np.exp(-t / max(seconds / 3.0, 1e-4))).astype(np.float32)


def _warp(sig: np.ndarray, n: int, rate: float, depth: float, delay: float, ramp: float,
          rng: np.random.Generator) -> np.ndarray:
    """Vibrato: reads the signal at a gently wobbling speed. depth is a fraction (0.01 = 1%)."""
    t = np.arange(n) / SR
    d = depth * np.clip((t - delay) / max(ramp, 1e-3), 0.0, 1.0) * np.sin(TWO_PI * rate * t + rng.uniform(0, TWO_PI))
    pos = np.cumsum(1.0 + d)
    pos -= pos[0]
    return map_coordinates(sig, [pos], order=3, mode="nearest").astype(np.float32)


def _harmonics(f: float, n: int, partials, rng, fmax: float = 11000.0) -> np.ndarray:
    """A sum of sine partials [(multiple, amplitude), ...], each with its own random phase."""
    out = np.zeros(n, np.float32)
    for mult, amp in partials:
        freq = f * mult
        if freq >= fmax or amp < 1e-4:
            continue
        out += _sine(freq, n, rng.uniform(0, TWO_PI)) * np.float32(amp)
    return out


# Pianos ----------------------------------------------------------------------------

def _piano(pitch: int, layer: int, length: float, soft: bool) -> np.ndarray:
    """Additive piano: slightly stretched partials, two strings per note that beat a little,
    a quick decay for high partials, and a soft thump from the hammer."""
    n = int(length * SR)
    f0 = hz(pitch)
    stretch = 1e-4 * 2.0 ** ((pitch - 36) / 12.0 * 0.9)
    tau1 = float(np.clip(4.5 * 2.0 ** (-(pitch - 48) / 15.0), 0.45, 8.0))
    cutoff = ((900, 1500, 2400) if soft else (2400, 4200, 7000))[layer]
    rng = _rng("piano", soft, pitch, layer)
    out = np.zeros(n, np.float32)
    for k in range(1, 49):
        fk = k * f0 * np.sqrt(1.0 + stretch * k * k)
        if fk > 10500:
            break
        amp = k ** -0.85 * (0.2 + 0.8 * abs(np.sin(np.pi * k * 0.13)))
        amp /= np.sqrt(1.0 + (fk / cutoff) ** 4)
        tau = max(0.02, tau1 / (1.0 + 0.5 * (k - 1)))
        for share, slow, detune in ((0.62, 1.0, -0.0004), (0.38, 2.4, 0.0004)):
            m = min(n, int(tau * slow * 7 * SR))
            if m >= 16:
                out[:m] += _sine(fk * (1 + detune), m, rng.uniform(0, TWO_PI), tau * slow) * np.float32(amp * share)
    peak = float(np.max(np.abs(out))) or 1.0
    t = np.arange(min(n, int(0.12 * SR))) / SR
    thump = _lp(_noise(len(t), rng), cutoff * 0.5) * np.exp(-t / 0.014).astype(np.float32)
    out[:len(t)] += thump * np.float32(peak * (0.05 if soft else 0.08))
    ramp = min(n, int(0.003 * SR))
    out[:ramp] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp))).astype(np.float32)
    return out


def _epiano(pitch: int, layer: int, length: float) -> np.ndarray:
    """Electric piano: a bell-like two-operator FM tone whose brightness dies away."""
    n = int(length * SR)
    t = np.arange(n) / SR
    f0 = hz(pitch)
    tau = float(np.clip(2.6 * 2.0 ** (-(pitch - 60) / 20.0), 0.5, 5.0))
    index = (1.1, 2.0, 3.2)[layer] * np.exp(-t / 0.35) + 0.12
    ph = TWO_PI * f0 * t
    body = np.sin(ph + index * np.sin(ph)) * np.exp(-t / tau)
    tine = np.sin(TWO_PI * f0 * 7.0 * t + 1.5 * np.sin(TWO_PI * f0 * 7.0 * t)) * np.exp(-t / 0.045)
    trem = 1.0 + 0.07 * np.sin(TWO_PI * 5.0 * t)
    out = (body + 0.12 * (0.5 + layer * 0.4) * tine) * trem
    out = out.astype(np.float32)
    out *= _attack(n, 0.004)
    return out


# Bowed strings ---------------------------------------------------------------------

def _bowed(pitch: int, length: float, *, voices: int, spread: float, attack: float, rate: float,
           depth: float, delay: float, formants, base: float, rolloff: float, top: float,
           scratch: float) -> np.ndarray:
    n = int(length * SR)
    f0 = hz(pitch)
    pad = int(n * 0.04) + 32
    out = np.zeros(n, np.float32)
    for v in range(voices):
        rng = _rng("bowed", pitch, v, voices)
        cents = 0.0 if voices == 1 else spread * (2.0 * v / (voices - 1) - 1.0)
        f = f0 * 2.0 ** (cents / 1200.0)
        partials = []
        for k in range(1, 60):
            fk = k * f
            if fk > top:
                break
            body = base + sum(g * _gauss(fk, c, w) for c, w, g in formants)
            partials.append((k, k ** -rolloff * body))
        raw = _harmonics(f, n + pad, partials, rng)
        voice = _warp(raw, n, rate * rng.uniform(0.9, 1.1), depth, delay + rng.uniform(0, 0.15), 0.5, rng)
        out += voice * _attack(n, attack * rng.uniform(0.9, 1.15))
    if scratch > 0:
        rng = _rng("bow-noise", pitch)
        t = np.arange(n) / SR
        hiss = _bp(_noise(n, rng), 1500, 5500) * np.exp(-t / 0.4).astype(np.float32)
        out += hiss * np.float32(scratch * np.max(np.abs(out)) * 0.5)
    return out / np.float32(voices ** 0.5)


_VIOLIN_BODY = [(290, 0.30, 0.9), (560, 0.30, 0.9), (1100, 0.5, 0.4), (2700, 0.45, 1.3)]
_CELLO_BODY = [(110, 0.30, 1.2), (220, 0.30, 1.0), (620, 0.5, 0.6), (1500, 0.5, 0.3)]


def _violin(pitch, layer, length):
    return _bowed(pitch, length, voices=1, spread=0, attack=0.07, rate=5.6, depth=0.007, delay=0.25,
                  formants=_VIOLIN_BODY, base=0.22, rolloff=1.0 - 0.1 * layer, top=9000, scratch=0.05)


def _strings(pitch, layer, length):
    return _bowed(pitch, length, voices=4, spread=9, attack=0.16, rate=5.2, depth=0.004, delay=0.3,
                  formants=_VIOLIN_BODY, base=0.25, rolloff=1.15, top=7500, scratch=0.02)


def _cello(pitch, layer, length):
    return _bowed(pitch, length, voices=1, spread=0, attack=0.10, rate=5.0, depth=0.006, delay=0.3,
                  formants=_CELLO_BODY, base=0.2, rolloff=1.15, top=4500, scratch=0.05)


# Plucked strings (Karplus-Strong) ----------------------------------------------------

def _pluck(pitch: int, layer: int, length: float, *, damping: float, t60: float, position: float,
           pick_cutoff: float, post) -> np.ndarray:
    """A string model: a burst of noise travels around a loop that loses a little energy,
    and a little brightness, every trip. The loop length sets the pitch, and a fractional
    delay keeps even high notes in tune. Done one trip at a time so numpy can do each trip
    in one go."""
    n = int(length * SR)
    f0 = hz(pitch)
    delay = SR / f0 - damping
    block = max(3, int(np.floor(delay)))
    frac = delay - block
    fall = float(np.clip(t60 * (f0 / 110.0) ** -0.25, 0.5, 12.0))
    gain = 10.0 ** (-3.0 / (f0 * fall))
    c0 = (1 - damping) * (1 - frac)
    c1 = (1 - damping) * frac + damping * (1 - frac)
    c2 = damping * frac
    rng = _rng("pluck", pitch, layer, damping)
    burst = _lp(_noise(block, rng), pick_cutoff * (0.55, 0.8, 1.15)[layer], 1)
    comb = int(round(position * block))
    if comb > 0:
        burst = burst - np.roll(burst, comb)
    burst -= burst.mean()
    burst /= (np.sqrt(np.mean(burst ** 2)) + 1e-9)
    y = np.zeros(n + block + 4, np.float32)
    off = 2
    y[off:off + block] = burst
    for s in range(off + block, n + off, block):
        e = min(s + block, n + off)
        m = e - s
        y[s:e] = gain * (c0 * y[s - block:s - block + m] + c1 * y[s - block - 1:s - block - 1 + m]
                         + c2 * y[s - block - 2:s - block - 2 + m])
    out = y[off:off + n]
    return post(out, f0)


def _eq_acoustic(x, f0):
    x = _lp(x, 5500, 2)
    body = _bp(x, 80, 260, 1)
    return x + 0.6 * body


def _eq_electric(x, f0):
    x = _lp(x, 6500, 2)
    x = np.tanh(2.2 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
    return _bp(x, 120, 4500, 1) * 1.4 + 0.2 * x


def _eq_bass(x, f0):
    return _lp(x, 1400, 2)


def _eq_harp(x, f0):
    return _lp(x, 9000, 1)


def _guitar(pitch, layer, length):
    return _pluck(pitch, layer, length, damping=0.45, t60=3.8, position=0.17, pick_cutoff=3800, post=_eq_acoustic)


def _eguitar(pitch, layer, length):
    return _pluck(pitch, layer, length, damping=0.32, t60=5.5, position=0.12, pick_cutoff=7000, post=_eq_electric)


def _bass(pitch, layer, length):
    return _pluck(pitch, layer, length, damping=0.5, t60=4.5, position=0.3, pick_cutoff=1100, post=_eq_bass)


def _harp(pitch, layer, length):
    return _pluck(pitch, layer, length, damping=0.2, t60=7.0, position=0.25, pick_cutoff=7500, post=_eq_harp)


# Reeds, breath, brass, voices ---------------------------------------------------------

def _accordion(pitch: int, layer: int, length: float) -> np.ndarray:
    """Three slightly detuned reeds (the 'musette' sound) with a boxy resonance."""
    n = int(length * SR)
    f0 = hz(pitch)
    out = np.zeros(n, np.float32)
    for v, cents in enumerate((-7.0, 0.0, 7.0)):
        rng = _rng("accordion", pitch, v)
        f = f0 * 2.0 ** (cents / 1200.0)
        partials = [(k, k ** -0.8 * (0.45 + 0.9 * _gauss(k * f, 1000, 1.1))) for k in range(1, 40) if k * f < 9000]
        out += _harmonics(f, n, partials, rng)
    t = np.arange(n) / SR
    out *= _attack(n, 0.035) * (1.0 + 0.04 * np.sin(TWO_PI * 0.35 * t + 1.0)).astype(np.float32)
    rng = _rng("accordion-key", pitch)
    click = _bp(_noise(int(0.02 * SR), rng), 800, 4000) * np.exp(-np.arange(int(0.02 * SR)) / (0.004 * SR)).astype(np.float32)
    out[:len(click)] += click * np.float32(0.25 * np.max(np.abs(out)))
    return out


def _organ(pitch: int, layer: int, length: float) -> np.ndarray:
    n = int(length * SR)
    f0 = hz(pitch)
    drawbars = [(0.5, 0.45), (1, 1.0), (1.5, 0.3), (2, 0.7), (3, 0.35), (4, 0.45), (5, 0.12), (6, 0.18), (8, 0.1)]
    out = np.zeros(n, np.float32)
    for v, detune in enumerate((0.0, 0.0017)):
        rng = _rng("organ", pitch, v)
        out += _harmonics(f0 * (1 + detune), n, drawbars, rng, fmax=12000)
    t = np.arange(n) / SR
    out *= _attack(n, 0.008) * (1.0 + 0.05 * np.sin(TWO_PI * 6.0 * t)).astype(np.float32)
    rng = _rng("organ-click", pitch)
    click = _hp(_noise(int(0.012 * SR), rng), 1500) * np.exp(-np.arange(int(0.012 * SR)) / (0.002 * SR)).astype(np.float32)
    out[:len(click)] += click * np.float32(0.15 * np.max(np.abs(out)))
    return out


def _flute(pitch: int, layer: int, length: float) -> np.ndarray:
    n = int(length * SR)
    f0 = hz(pitch)
    rng = _rng("flute", pitch)
    pad = int(n * 0.04) + 32
    raw = _harmonics(f0, n + pad, [(1, 1.0), (2, 0.32 + 0.1 * layer), (3, 0.10), (4, 0.05), (5, 0.02)], rng)
    tone = _warp(raw, n, 5.0, 0.004, 0.35, 0.5, rng)
    breath = _bp(_noise(n, rng), 1800, 6500) * np.float32(0.06)
    out = (tone + breath) * _attack(n, 0.07)
    chiff = int(0.05 * SR)
    out[:chiff] += _bp(_noise(chiff, rng), 1500, 5000) * np.hanning(chiff * 2)[:chiff].astype(np.float32) * 0.15
    return out.astype(np.float32)


def _clarinet(pitch: int, layer: int, length: float) -> np.ndarray:
    n = int(length * SR)
    f0 = hz(pitch)
    rng = _rng("clarinet", pitch)
    pad = int(n * 0.04) + 32
    partials = [(k, (1.0 / k if k % 2 else 0.1 / k) * (1.0 / (1.0 + (k * f0 / 3200.0) ** 2))) for k in range(1, 40)]
    raw = _harmonics(f0, n + pad, partials, rng)
    tone = _warp(raw, n, 4.8, 0.003, 0.5, 0.6, rng)
    return (tone * _attack(n, 0.045)).astype(np.float32)


def _trumpet(pitch: int, layer: int, length: float) -> np.ndarray:
    """Brass gets brighter as the note opens up: high harmonics arrive a little later."""
    n = int(length * SR)
    f0 = hz(pitch)
    rng = _rng("trumpet", pitch)
    t = np.arange(n) / SR
    opening = 1.0 - np.exp(-t / 0.05)
    out = np.zeros(n, np.float32)
    for k in range(1, 20):
        fk = k * f0
        if fk > 9000:
            break
        amp = k ** -0.5 * (0.35 + _gauss(fk, 1300, 1.0)) / (1.0 + (fk / 6000.0) ** 2)
        env = (opening ** (0.5 * (k - 1))).astype(np.float32)
        out += _sine(fk, n, rng.uniform(0, TWO_PI)) * env * np.float32(amp)
    out *= _attack(n, 0.03)
    return _warp(np.concatenate([out, np.zeros(int(n * 0.04) + 32, np.float32)]), n, 5.2, 0.003, 0.4, 0.5, rng)


def _choir(pitch: int, layer: int, length: float) -> np.ndarray:
    """Voices singing 'ooh': harmonics shaped by fixed vowel resonances."""
    n = int(length * SR)
    f0 = hz(pitch)
    pad = int(n * 0.04) + 32
    out = np.zeros(n, np.float32)
    voices = 5
    for v in range(voices):
        rng = _rng("choir", pitch, v)
        cents = 14.0 * (2.0 * v / (voices - 1) - 1.0)
        f = f0 * 2.0 ** (cents / 1200.0)
        partials = []
        for k in range(1, 50):
            fk = k * f
            if fk > 5000:
                break
            shape = 1.0 * _gauss(fk, 300, 0.45) + 0.45 * _gauss(fk, 870, 0.3) + 0.06 * _gauss(fk, 2250, 0.4)
            partials.append((k, k ** -0.6 * shape + 0.01))
        raw = _harmonics(f, n + pad, partials, rng)
        voice = _warp(raw, n, rng.uniform(4.8, 5.8), 0.005, 0.3 + rng.uniform(0, 0.2), 0.5, rng)
        out += voice * _attack(n, 0.22 * rng.uniform(0.85, 1.2))
    rng = _rng("choir-air", pitch)
    out += _bp(_noise(n, rng), 800, 3500) * np.float32(0.03 * np.max(np.abs(out)))
    return out


def _pad(pitch: int, layer: int, length: float) -> np.ndarray:
    n = int(length * SR)
    f0 = hz(pitch)
    out = np.zeros(n, np.float32)
    for v, cents in enumerate((-12.0, -6.0, 0.0, 6.0, 12.0)):
        rng = _rng("pad", pitch, v)
        f = f0 * 2.0 ** (cents / 1200.0)
        partials = [(k, 1.0 / k / (1.0 + (k * f / 1800.0) ** 2)) for k in range(1, 50) if k * f < 9000]
        out += _harmonics(f, n, partials, rng)
    return out * _attack(n, 0.35)


def _bells(pitch: int, layer: int, length: float) -> np.ndarray:
    """Music box: a tuned metal bar, so the overtones are not whole multiples."""
    n = int(length * SR)
    f0 = hz(pitch)
    tau = float(np.clip(1.4 * 2.0 ** (-(pitch - 72) / 18.0), 0.3, 3.0))
    rng = _rng("bells", pitch)
    out = np.zeros(n, np.float32)
    for ratio, amp, speed in ((1.0, 1.0, 1.0), (2.76, 0.32, 2.0), (5.40, 0.12, 3.5), (8.93, 0.05, 6.0)):
        f = f0 * ratio
        if f < 12000:
            m = min(n, int(tau / speed * 7 * SR))
            if m > 16:
                out[:m] += _sine(f, m, rng.uniform(0, TWO_PI), tau / speed) * np.float32(amp)
    out *= _attack(n, 0.002)
    return out


# Drums ---------------------------------------------------------------------------------

# General MIDI drum notes that matter. Anything else is grouped by pitch range.
KICK, SNARE, RIM, CLAP, HAT, OPEN_HAT, CRASH, RIDE, COWBELL, SHAKER = (
    "kick", "snare", "rim", "clap", "hat", "open hat", "crash", "ride", "cowbell", "shaker")
_GM_DRUMS = {35: KICK, 36: KICK, 37: RIM, 38: SNARE, 39: CLAP, 40: SNARE, 42: HAT, 44: HAT, 46: OPEN_HAT,
             49: CRASH, 51: RIDE, 52: CRASH, 53: RIDE, 54: SHAKER, 55: CRASH, 56: COWBELL, 57: CRASH,
             59: RIDE, 69: SHAKER, 70: SHAKER, 82: SHAKER}
_TOMS = {41: 82.0, 43: 98.0, 45: 115.0, 47: 135.0, 48: 150.0, 50: 175.0}


def drum_for_pitch(pitch: int) -> str | float:
    """The drum a MIDI pitch plays: a name, or a tom frequency."""
    if pitch in _GM_DRUMS:
        return _GM_DRUMS[pitch]
    if pitch in _TOMS:
        return _TOMS[pitch]
    if pitch < 36:
        return KICK
    if pitch < 41:
        return SNARE
    if pitch < 51:
        return 60.0 + (pitch - 41) * 12.0
    if pitch < 60:
        return RIDE
    if pitch < 72:
        return HAT
    return OPEN_HAT


def _metal(n: int, base: float = 205.3) -> np.ndarray:
    """The cluster of square waves that makes cymbals and hats sound metallic."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for ratio in (1.0, 1.4829, 1.8, 2.5459, 2.6304, 3.8967):
        out += np.sign(np.sin(TWO_PI * base * ratio * t))
    return (out / 6.0).astype(np.float32)


def _drum(kind, layer: int) -> np.ndarray:
    rng = _rng("drum", str(kind))
    loud = (0.7, 0.85, 1.0)[layer]
    if kind == KICK:
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        f = 46.0 + 120.0 * np.exp(-t / 0.028)
        body = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-t / 0.15)
        click = _hp(_noise(n, rng), 1800) * np.exp(-t / 0.003).astype(np.float32)
        return _lp(body.astype(np.float32) + 0.35 * click * loud, 9000)
    if kind == SNARE:
        n = int(0.4 * SR)
        t = np.arange(n) / SR
        tone = 0.55 * np.sin(TWO_PI * 190 * t) * np.exp(-t / 0.06) + 0.3 * np.sin(TWO_PI * 330 * t) * np.exp(-t / 0.045)
        hiss = _bp(_noise(n, rng), 1400, 9500) * np.exp(-t / 0.1).astype(np.float32)
        return (tone.astype(np.float32) + 0.85 * hiss).astype(np.float32)
    if kind == RIM:
        n = int(0.15 * SR)
        t = np.arange(n) / SR
        out = np.sin(TWO_PI * 1750 * t) * np.exp(-t / 0.006) + 0.6 * np.sin(TWO_PI * 480 * t) * np.exp(-t / 0.02)
        return (out + 0.3 * _hp(_noise(n, rng), 2500) * np.exp(-t / 0.004)).astype(np.float32)
    if kind == CLAP:
        n = int(0.4 * SR)
        t = np.arange(n) / SR
        env = np.zeros(n)
        for delay in (0.0, 0.009, 0.019):
            idx = int(delay * SR)
            env[idx:] += np.exp(-np.arange(n - idx) / (0.008 * SR))
        env += 0.8 * np.exp(-t / 0.11) * (t > 0.02)
        return (_bp(_noise(n, rng), 900, 3000) * env.astype(np.float32)).astype(np.float32)
    if kind in (HAT, OPEN_HAT):
        decay = 0.022 if kind == HAT else 0.16
        n = int((0.18 if kind == HAT else 0.9) * SR)
        t = np.arange(n) / SR
        mix = 0.6 * _noise(n, rng) + 0.5 * _metal(n)
        return (_hp(mix, 6500, 3) * np.exp(-t / decay).astype(np.float32) * 0.9).astype(np.float32)
    if kind == CRASH:
        n = int(2.6 * SR)
        t = np.arange(n) / SR
        mix = 0.8 * _noise(n, rng) + 0.5 * _metal(n, 180.0)
        return (_hp(mix, 3500, 2) * np.exp(-t / 0.55).astype(np.float32) * 0.8).astype(np.float32)
    if kind == RIDE:
        n = int(1.6 * SR)
        t = np.arange(n) / SR
        ping = np.sin(TWO_PI * 3400 * t) * np.exp(-t / 0.05) * 0.4
        mix = 0.4 * _noise(n, rng) + 0.7 * _metal(n, 240.0)
        return (_hp(mix, 4500, 2) * np.exp(-t / 0.35).astype(np.float32) * 0.7 + ping).astype(np.float32)
    if kind == COWBELL:
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        sq = np.sign(np.sin(TWO_PI * 560 * t)) + np.sign(np.sin(TWO_PI * 845 * t))
        return (_bp(sq.astype(np.float32), 500, 2500, 2) * np.exp(-t / 0.12).astype(np.float32)).astype(np.float32)
    if kind == SHAKER:
        n = int(0.3 * SR)
        t = np.arange(n) / SR
        env = (1.0 - np.exp(-t / 0.012)) * np.exp(-t / 0.05)
        return (_hp(_noise(n, rng), 5500) * env.astype(np.float32) * 0.8).astype(np.float32)
    # a tom: the number is its base frequency
    freq = float(kind)
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    f = freq * (1.0 + 0.7 * np.exp(-t / 0.04))
    body = np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-t / 0.2)
    thud = _lp(_noise(n, rng), 1500) * np.exp(-t / 0.01).astype(np.float32)
    return (body.astype(np.float32) + 0.3 * thud).astype(np.float32)


def _drum_kit(pitch: int, layer: int, length: float) -> np.ndarray:
    return _drum(drum_for_pitch(pitch), layer)


# Instrument list ---------------------------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    key: str
    label: str
    render: Callable[[int, int, float], np.ndarray]
    release: float = 0.08      # fade out after the note ends, in seconds
    layers: int = 1            # velocity layers (1 means the strength only changes loudness)
    decays: bool = False       # dies away on its own, so a short sample is enough
    max_len: float = 12.0      # longest sample ever made, in seconds
    program: int = 0           # General MIDI program number, used when exporting MIDI
    one_shot: bool = False     # plays in full whatever the note length (drums)
    trim: float = 1.0          # loudness adjustment against the other instruments


INSTRUMENTS: list[Instrument] = [
    Instrument("piano", "Soft piano", lambda p, l, n: _piano(p, l, n, True), release=0.28, layers=3,
               decays=True, max_len=6.0, program=0),
    Instrument("grand", "Bright piano", lambda p, l, n: _piano(p, l, n, False), release=0.22, layers=3,
               decays=True, max_len=6.0, program=1),
    Instrument("epiano", "Electric piano", _epiano, release=0.22, layers=3, decays=True, max_len=6.0, program=4),
    Instrument("violin", "Violin", _violin, release=0.12, program=40),
    Instrument("strings", "Strings (section)", _strings, release=0.35, program=48),
    Instrument("cello", "Cello", _cello, release=0.15, program=42),
    Instrument("guitar", "Acoustic guitar", _guitar, release=0.12, layers=3, decays=True, max_len=5.0, program=24),
    Instrument("eguitar", "Electric guitar", _eguitar, release=0.12, layers=3, decays=True, max_len=6.0, program=27),
    Instrument("bass", "Bass guitar", _bass, release=0.1, layers=3, decays=True, max_len=6.0, program=33),
    Instrument("harp", "Harp", _harp, release=0.2, layers=3, decays=True, max_len=7.0, program=46),
    Instrument("accordion", "Accordion", _accordion, release=0.06, program=21),
    Instrument("organ", "Organ", _organ, release=0.05, program=16),
    Instrument("flute", "Flute", _flute, release=0.08, program=73),
    Instrument("clarinet", "Clarinet", _clarinet, release=0.07, program=71),
    Instrument("trumpet", "Trumpet", _trumpet, release=0.07, program=56),
    Instrument("choir", "Choir (ooh)", _choir, release=0.4, program=53),
    Instrument("pad", "Warm pad", _pad, release=0.7, program=89),
    Instrument("bells", "Music box", _bells, release=0.1, decays=True, max_len=4.0, program=10),
    Instrument("drums", "Drum kit", _drum_kit, release=0.0, layers=3, one_shot=True, decays=True, max_len=3.0, program=0,
               trim=0.55),
]
BY_KEY = {i.key: i for i in INSTRUMENTS}
DEFAULT_INSTRUMENT = "piano"
DRUM_KEY = "drums"


def instrument(key: str | None) -> Instrument:
    return BY_KEY.get(key or "", BY_KEY[DEFAULT_INSTRUMENT])


_PART_SOUNDS = {"Drums": DRUM_KEY, "Bass": "bass", "Guitar": "guitar", "Piano": "grand"}


def default_instrument_for(part_name: str) -> str:
    return _PART_SOUNDS.get(part_name, DEFAULT_INSTRUMENT)


# Rendering notes ---------------------------------------------------------------------

def _velocity_layer(inst: Instrument, velocity: float) -> int:
    if inst.layers <= 1:
        return 0
    return 0 if velocity < 0.4 else (1 if velocity < 0.72 else 2)


def _level(velocity: float) -> float:
    return 0.25 + 0.75 * min(max(velocity, 0.0), 1.0)


class Bank:
    """Renders each (instrument, pitch, velocity layer) once and keeps it."""

    LIMIT_BYTES = 450 * 1024 * 1024

    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[np.ndarray, float]] = {}
        self._bytes = 0

    def _sample(self, inst: Instrument, pitch: int, layer: int, need: float) -> np.ndarray:
        key = (inst.key, pitch, layer)
        hit = self._cache.get(key)
        if hit is not None and (hit[1] >= min(need, inst.max_len) or inst.one_shot):
            return hit[0]
        length = min(inst.max_len, max(need * 1.4, 1.0)) if not inst.one_shot else inst.max_len
        if hit is not None and not inst.one_shot:
            length = min(inst.max_len, max(length, hit[1] * 2.0))
        raw = np.nan_to_num(inst.render(int(pitch), layer, length)).astype(np.float32)
        window = raw[:int(0.6 * SR)]
        rms = float(np.sqrt(np.mean(window * window))) if len(window) else 0.0
        peak = float(np.max(np.abs(raw))) if len(raw) else 0.0
        scale = TARGET_RMS * inst.trim / (rms + 1e-9)
        scale = min(scale, 1.6 / (peak + 1e-9))
        raw *= np.float32(scale)
        if not inst.one_shot:
            taper = min(len(raw), int(0.03 * SR))
            if taper > 0:
                raw[-taper:] *= np.linspace(1.0, 0.0, taper, dtype=np.float32)
        if hit is not None:
            self._bytes -= hit[0].nbytes
        if self._bytes > self.LIMIT_BYTES:
            self._cache.clear()
            self._bytes = 0
        self._cache[key] = (raw, length)
        self._bytes += raw.nbytes
        return raw

    def note(self, inst: Instrument, pitch: int, velocity: float, duration: float) -> np.ndarray:
        """Mono float32 samples for one note, from its start, including the fade out."""
        layer = _velocity_layer(inst, velocity)
        level = np.float32(_level(velocity))
        if inst.one_shot:
            return self._sample(inst, pitch, layer, inst.max_len) * level
        duration = max(0.04, duration)
        sample = self._sample(inst, pitch, layer, duration + inst.release)
        held = int(duration * SR)
        total = min(len(sample), held + int(inst.release * SR))
        out = sample[:total].copy()
        if total > held:
            tail = total - held
            out[held:] *= (np.cos(np.linspace(0.0, np.pi / 2, tail)) ** 2).astype(np.float32)
        out *= level
        return out


_bank = Bank()


@dataclass
class NoteSet:
    """The notes of one part, as arrays, with the instrument they are played on."""
    inst: Instrument
    start: np.ndarray
    end: np.ndarray
    pitch: np.ndarray
    velocity: np.ndarray
    stop: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.inst.one_shot:
            self.stop = self.start + self.inst.max_len
        else:
            self.stop = self.start + np.minimum((self.end - self.start) + self.inst.release + 0.05, self.inst.max_len)

    @classmethod
    def from_notes(cls, key: str, notes) -> "NoteSet":
        """notes: iterable of (start, end, pitch, velocity)."""
        arr = np.array(list(notes), dtype=np.float64).reshape(-1, 4)
        order = np.argsort(arr[:, 0], kind="stable")
        arr = arr[order]
        return cls(instrument(key), arr[:, 0], arr[:, 1], arr[:, 2].astype(int), arr[:, 3])


def render_chunk(sets: list[NoteSet], t0: float, frames: int, bank: Bank | None = None) -> np.ndarray:
    """Mono float32 audio for the window [t0, t0 + frames / SR)."""
    bank = bank or _bank
    buf = np.zeros(frames, np.float32)
    t1 = t0 + frames / SR
    for ns in sets:
        if len(ns.start) == 0:
            continue
        hit = np.nonzero((ns.start < t1) & (ns.stop > t0))[0]
        for i in hit:
            audio = bank.note(ns.inst, int(ns.pitch[i]), float(ns.velocity[i]), float(ns.end[i] - ns.start[i]))
            s0 = int(round((ns.start[i] - t0) * SR))
            a = max(0, -s0)
            b = min(len(audio), frames - s0)
            if b > a:
                buf[s0 + a:s0 + b] += audio[a:b]
    return buf


_ir_cache: dict[float, list[np.ndarray]] = {}


def _room_irs(seconds: float) -> list[np.ndarray]:
    """Left and right impulse responses of the reverb (a decaying burst of noise)."""
    if seconds not in _ir_cache:
        n = int(seconds * SR)
        t = np.arange(n) / SR
        irs = []
        for seed in (11, 23):
            rng = np.random.default_rng(seed)
            ir = rng.standard_normal(n) * np.exp(-t * 6.9 / seconds)
            ir[:int(0.012 * SR)] = 0.0
            ir *= np.minimum(1.0, t / 0.004 + 1e-3)
            ir = signal.sosfilt(signal.butter(2, 5200, "low", fs=SR, output="sos"), ir)
            irs.append((ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32))
        _ir_cache[seconds] = irs
    return _ir_cache[seconds]


class Room:
    """A little reverb: it makes the notes sound like they are in a space, and gives the
    sound some width. Works on one block at a time and carries the tail into the next."""

    def __init__(self, seconds: float = 1.3, wet: float = 0.22) -> None:
        self.wet = wet
        self.irs = _room_irs(seconds)
        self.reset()

    def reset(self) -> None:
        self.tail = [np.zeros(len(ir) - 1, np.float32) for ir in self.irs]

    def process(self, dry: np.ndarray) -> np.ndarray:
        """Mono block in, stereo block (frames, 2) out."""
        n = len(dry)
        out = np.empty((n, 2), np.float32)
        for ch, ir in enumerate(self.irs):
            full = signal.fftconvolve(dry, ir).astype(np.float32)
            full[:len(self.tail[ch])] += self.tail[ch]
            self.tail[ch] = full[n:].copy()
            out[:, ch] = dry + self.wet * full[:n]
        return out


def soft_clip(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Leaves quiet audio alone and rounds off peaks above the knee instead of clipping."""
    a = np.abs(x)
    over = a > knee
    if not np.any(over):
        return x
    room = 1.0 - knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + room * np.tanh((a[over] - knee) / room))
    return y


# Previews ----------------------------------------------------------------------------

def preview_notes(key: str) -> list[tuple[float, float, int, float]]:
    """A short phrase that shows what an instrument sounds like."""
    if key == DRUM_KEY:
        hits = [(0.0, 36), (0.0, 42), (0.25, 42), (0.5, 38), (0.5, 42), (0.75, 42), (1.0, 36), (1.25, 42),
                (1.5, 38), (1.5, 49)]
        return [(t, t + 0.1, p, 0.85) for t, p in hits]
    base = 48 if key in ("bass", "cello") else 60
    return [(0.0, 0.42, base, 0.7), (0.38, 0.8, base + 4, 0.7), (0.76, 1.2, base + 7, 0.7),
            (1.14, 2.3, base + 12, 0.75)]


def render_preview(key: str, notes=None, bank: Bank | None = None) -> np.ndarray:
    """Stereo float32 (frames, 2) for the preview phrase (or the given notes)."""
    notes = notes or preview_notes(key)
    frames = int((max(n[1] for n in notes) + 1.6) * SR)
    dry = render_chunk([NoteSet.from_notes(key, notes)], 0.0, frames, bank)
    return soft_clip(Room().process(dry))


def render_single(key: str, pitch: int, velocity: float, seconds: float = 0.7, bank: Bank | None = None) -> np.ndarray:
    """One note, stereo, for the click-to-hear sound in Edit mode."""
    inst = instrument(key)
    length = 0.15 if inst.one_shot else seconds
    return render_preview(key, [(0.0, length, int(pitch), velocity)], bank)


# Click track ------------------------------------------------------------------------------

def click_sound(accent: bool) -> np.ndarray:
    """A short woodblock-like tick. The first beat of a bar is higher and louder."""
    n = int(0.045 * SR)
    t = np.arange(n) / SR
    freq = 1760.0 if accent else 1320.0
    env = np.exp(-t / 0.009)
    tone = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2.01 * t)
    return (tone * env * (0.55 if accent else 0.38)).astype(np.float32)
