"""Small music helpers: note names, time formatting, and key estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .i18n import tr

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]  # letters with sharps
BLACK_KEYS = {1, 3, 6, 8, 10}
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)

# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# Note names ---------------------------------------------------------------------------
#
# Every name shown anywhere in the app goes through pc_name() or note_name(), so one
# setting changes them all:
#   letters   C, C#, D ...            (flats instead of sharps in flat keys: Bb, Eb ...)
#   fixed     Do, Do#, Re ... Si      (Do is always C, as taught in Romania, Italy, Spain ...)
#   fixed_ti  the same with Ti instead of Si
#   movable   Do is the key note of the song, so a song in G calls G "Do"

NAME_STYLES = [("letters", "C D E"), ("fixed", "Do Re Mi"), ("fixed_ti", "Do Re Mi (Ti)"),
               ("movable", "Movable Do")]

_LETTERS_SHARP = NOTE_NAMES
_LETTERS_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_FIXED_SHARP = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
_FIXED_FLAT = ["Do", "Reb", "Re", "Mib", "Mi", "Fa", "Solb", "Sol", "Lab", "La", "Sib", "Si"]
# Movable Do counts up from the key note. The chromatic names are the usual mix:
# Ra (flat 2), Me (flat 3), Fi (sharp 4), Le (flat 6), Te (flat 7).
_MOVABLE = ["Do", "Ra", "Re", "Me", "Mi", "Fa", "Fi", "Sol", "Le", "La", "Te", "Ti"]

# Keys written with flats: F, Bb, Eb, Ab, Db, Gb major and D, G, C, F, Bb, Eb minor.
_FLAT_MAJOR = {5, 10, 3, 8, 1, 6}
_FLAT_MINOR = {2, 7, 0, 5, 10, 3}


class _Naming:
    style = "letters"
    tonic: int | None = None   # key note of the song, used by movable Do and for flats
    mode = "major"


_naming = _Naming()


def set_naming(style: str | None = None, tonic: int | None = -1, mode: str | None = None) -> None:
    """Change how notes are named. tonic=-1 leaves the key as it was, None means no key."""
    if style is not None and style in dict(NAME_STYLES):
        _naming.style = style
    if tonic != -1:
        _naming.tonic = tonic
    if mode is not None:
        _naming.mode = mode


def naming_style() -> str:
    return _naming.style


def uses_flats(tonic: int | None = None, mode: str | None = None) -> bool:
    tonic = _naming.tonic if tonic is None else tonic
    mode = mode or _naming.mode
    if tonic is None:
        return False
    return tonic in (_FLAT_MINOR if mode == "minor" else _FLAT_MAJOR)


def letter_name(pc: int, flats: bool | None = None) -> str:
    """Always a letter name (used where a file format needs one)."""
    flats = uses_flats() if flats is None else flats
    return (_LETTERS_FLAT if flats else _LETTERS_SHARP)[pc % 12]


def pc_name(pc: int, absolute: bool = False, flats: bool | None = None) -> str:
    """Name of a pitch class (0 = C) in the chosen style.

    absolute=True names it the same way for every key, which is what key names need:
    with movable Do a key is still called by its fixed name (a song "in Sol major")."""
    pc %= 12
    style = _naming.style
    flats = uses_flats() if flats is None else flats
    if style == "movable" and not absolute:
        if _naming.tonic is None:
            return (_FIXED_FLAT if flats else _FIXED_SHARP)[pc]
        return _MOVABLE[(pc - _naming.tonic) % 12]
    if style in ("fixed", "fixed_ti", "movable"):
        name = (_FIXED_FLAT if flats else _FIXED_SHARP)[pc]
        if style == "fixed_ti" and name.startswith("Si"):
            name = "Ti" + name[2:]
        return name
    return (_LETTERS_FLAT if flats else _LETTERS_SHARP)[pc]


def label_pc() -> int:
    """The note the piano keyboard writes its octave labels on: C, or Do in movable Do."""
    if _naming.style == "movable" and _naming.tonic is not None:
        return _naming.tonic
    return 0


def chord_root_name(pc: int) -> str:
    """A chord root, spelled the way chord charts do: the chords borrowed from minor (flat 2,
    flat 3, flat 6, flat 7 of the key) use flats even in a sharp key, so C major has Ab and Bb."""
    t = _naming.tonic
    flats = True if (t is not None and (pc - t) % 12 in (1, 3, 8, 10)) else None
    return pc_name(pc, flats=flats)


def note_name(pitch: int) -> str:
    """MIDI pitch to a name like C4 or Do4 (middle C is 60)."""
    return f"{pc_name(pitch)}{pitch // 12 - 1}"


def key_name(tonic: int, mode: str) -> str:
    """A key such as "G major" or "Sol major", in the current language."""
    return f"{pc_name(tonic, absolute=True)} {tr(mode)}"


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
    alternative: tuple[int, str] | None = None   # the runner up (tonic, mode) when it's close

    @property
    def name(self) -> str:
        return key_name(self.tonic, self.mode)

    @property
    def alternative_name(self) -> str | None:
        return key_name(*self.alternative) if self.alternative else None

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
    alternative = (second[1], second[2]) if margin < 0.05 else None
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
        return tr(DRUM_NAMES[pitch])
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
    scale_root: int | None = None           # set when `classes` is a named scale
    scale_name: str = ""                    # English scale name, for example "Major"

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
            if self.scale_name and self.scale_root is not None:
                names = f"{pc_name(self.scale_root, absolute=True)} {tr(self.scale_name).lower()}"
            else:
                names = ", ".join(pc_name(c) for c in sorted(self.classes))
            parts.append((tr("not in {names}") if self.outside else tr("only {names}")).format(names=names))
        if self.low > 0 or self.high < 127:
            parts.append(tr("{low} to {high}").format(low=note_name(max(self.low, 0)),
                                                       high=note_name(min(self.high, 127))))
        return ", ".join(parts)
