"""Chords and harmony, in plain Python (no Qt).

  * Chord: a root, a quality and an optional bass note, named in the current note style
  * the chords of a key, Roman numerals, and a progression library to browse
  * voicings (inversions, drop 2, open ...) with voice leading between chords
  * what could come next, substitutes, chords from outside the key, ways to change key
  * patterns that turn a chord into notes (arpeggios, strums, bass lines ...)
  * guessing chords from notes, or from the sound itself (chroma)
"""

from __future__ import annotations

import bisect
import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .i18n import tr
from .music import MAJOR_STEPS, MINOR_STEPS, chord_root_name, key_name

# Qualities --------------------------------------------------------------------------------
# key: (intervals in semitones above the root, suffix shown after the root, MusicXML kind)
QUALITIES: dict[str, tuple[tuple[int, ...], str, str]] = {
    "maj": ((0, 4, 7), "", "major"),
    "min": ((0, 3, 7), "m", "minor"),
    "dim": ((0, 3, 6), "dim", "diminished"),
    "aug": ((0, 4, 8), "aug", "augmented"),
    "sus2": ((0, 2, 7), "sus2", "suspended-second"),
    "sus4": ((0, 5, 7), "sus4", "suspended-fourth"),
    "5": ((0, 7), "5", "power"),
    "6": ((0, 4, 7, 9), "6", "major-sixth"),
    "m6": ((0, 3, 7, 9), "m6", "minor-sixth"),
    "7": ((0, 4, 7, 10), "7", "dominant"),
    "maj7": ((0, 4, 7, 11), "maj7", "major-seventh"),
    "m7": ((0, 3, 7, 10), "m7", "minor-seventh"),
    "m7b5": ((0, 3, 6, 10), "m7b5", "half-diminished"),
    "dim7": ((0, 3, 6, 9), "dim7", "diminished-seventh"),
    "mmaj7": ((0, 3, 7, 11), "m(maj7)", "major-minor"),
    "aug7": ((0, 4, 8, 10), "aug7", "augmented-seventh"),
    "7sus4": ((0, 5, 7, 10), "7sus4", "suspended-fourth"),
    "add9": ((0, 4, 7, 14), "add9", "major"),
    "madd9": ((0, 3, 7, 14), "m(add9)", "minor"),
    "9": ((0, 4, 7, 10, 14), "9", "dominant-ninth"),
    "maj9": ((0, 4, 7, 11, 14), "maj9", "major-ninth"),
    "m9": ((0, 3, 7, 10, 14), "m9", "minor-ninth"),
}
MINORISH = {"min", "dim", "m6", "m7", "m7b5", "dim7", "mmaj7", "madd9", "m9"}


@dataclass(frozen=True)
class Chord:
    root: int                 # pitch class, 0 = C
    quality: str = "maj"
    bass: int | None = None   # a different bass note (slash chord), or None

    @property
    def intervals(self) -> tuple[int, ...]:
        return QUALITIES[self.quality][0]

    def tones(self) -> list[int]:
        return [(self.root + i) % 12 for i in self.intervals]

    def name(self) -> str:
        text = chord_root_name(self.root) + QUALITIES[self.quality][1]
        if self.bass is not None and self.bass % 12 != self.root % 12:
            text += "/" + chord_root_name(self.bass)
        return text

    def to_dict(self) -> dict:
        return {"root": self.root, "quality": self.quality, "bass": self.bass}

    @staticmethod
    def from_dict(d: dict) -> "Chord":
        quality = d.get("quality", "maj")
        if quality not in QUALITIES:
            raise ValueError(f"Unknown chord quality {quality!r}")
        bass = d.get("bass")
        return Chord(int(d["root"]) % 12, quality, None if bass is None else int(bass) % 12)


def chord_name(chord: Chord | None) -> str:
    return chord.name() if chord else "?"


# Keys and scales ----------------------------------------------------------------------------

KEY_SCALES: list[tuple[str, tuple[int, ...]]] = [
    ("Major", MAJOR_STEPS),
    ("Natural minor", MINOR_STEPS),
    ("Harmonic minor", (0, 2, 3, 5, 7, 8, 11)),
    ("Melodic minor", (0, 2, 3, 5, 7, 9, 11)),
    ("Dorian", (0, 2, 3, 5, 7, 9, 10)),
    ("Phrygian", (0, 1, 3, 5, 7, 8, 10)),
    ("Lydian", (0, 2, 4, 6, 7, 9, 11)),
    ("Mixolydian", (0, 2, 4, 5, 7, 9, 10)),
    ("Locrian", (0, 1, 3, 5, 6, 8, 10)),
]
SCALE_STEPS = dict(KEY_SCALES)
MINOR_LIKE = {"Natural minor", "Harmonic minor", "Melodic minor", "Dorian", "Phrygian", "Locrian"}

CHORD_KINDS = [("triads", "Triads"), ("sevenths", "Sevenths"), ("ninths", "Ninths"), ("sus2", "Sus2"),
               ("sus4", "Sus4"), ("add9", "Add9"), ("sixths", "Sixths")]


def mode_of(scale: str) -> str:
    return "minor" if scale in MINOR_LIKE else "major"


def _quality_for(intervals: tuple[int, ...]) -> str | None:
    want = tuple(sorted(set(intervals)))
    for key, (ivs, _s, _k) in QUALITIES.items():
        if tuple(sorted(set(ivs))) == want:
            return key
    return None


def diatonic(tonic: int, scale: str = "Major", kind: str = "triads") -> list[Chord]:
    """The chord on each step of the scale, built from the scale's own notes."""
    steps = SCALE_STEPS[scale]
    n = len(steps)
    out = []
    for i in range(n):
        def deg(k: int) -> int:
            # semitones from this chord's root to the k-th scale step above it
            octave, idx = divmod(i + k, n)
            return steps[idx] + 12 * octave - steps[i]

        triad = (0, deg(2), deg(4))
        quality = _quality_for(triad) or "maj"
        if kind == "sevenths":
            quality = _quality_for(triad + (deg(6),)) or quality
        elif kind == "ninths":
            quality = _quality_for(triad + (deg(6), deg(8))) or _quality_for(triad + (deg(6),)) or quality
        elif kind == "sus2":
            quality = _quality_for((0, deg(1), deg(4))) or quality
        elif kind == "sus4":
            quality = _quality_for((0, deg(3), deg(4))) or quality
        elif kind == "add9":
            quality = _quality_for(triad + (deg(1) + 12,)) or quality
        elif kind == "sixths":
            quality = _quality_for(triad + (deg(5),)) or quality
        out.append(Chord((tonic + steps[i]) % 12, quality))
    return out


_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]
# Semitones above the key note -> numeral, counted against the major scale (so a minor key
# shows i, ii°, bIII, iv, v, bVI, bVII, the usual way in pop and worship charts).
_DEGREE = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI",
           10: "bVII", 11: "VII"}
_NUMERAL_SUFFIX = {"maj": "", "min": "", "dim": "°", "aug": "+", "sus2": "sus2", "sus4": "sus4", "5": "5",
                   "6": "6", "m6": "6", "7": "7", "maj7": "maj7", "m7": "7", "m7b5": "ø7", "dim7": "°7",
                   "mmaj7": "(maj7)", "aug7": "+7", "7sus4": "7sus4", "add9": "add9", "madd9": "add9",
                   "9": "9", "maj9": "maj9", "m9": "9"}


def numeral(chord: Chord, tonic: int) -> str:
    """Roman numeral of a chord in a key, for example V7, vi, bVII or ii°."""
    text = _DEGREE[(chord.root - tonic) % 12]
    acc = text[0] if text[0] in "b#" else ""
    roman = text[len(acc):]
    if chord.quality in MINORISH:
        roman = roman.lower()
    return acc + roman + _NUMERAL_SUFFIX.get(chord.quality, "")


_NUM_RE = re.compile(r"^([b#]?)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)(.*)$")
_MAJOR_DEGREE = {r: s for r, s in zip(_ROMAN, MAJOR_STEPS)}


def _parse_one(text: str, tonic: int) -> Chord:
    m = _NUM_RE.match(text)
    if not m:
        raise ValueError(f"Not a Roman numeral: {text!r}")
    acc, roman, suffix = m.groups()
    lower = roman.islower()
    root = (tonic + _MAJOR_DEGREE[roman.upper()] + (-1 if acc == "b" else 1 if acc == "#" else 0)) % 12
    upper_map = {"": "maj", "7": "7", "maj7": "maj7", "6": "6", "9": "9", "maj9": "maj9", "+": "aug",
                 "+7": "aug7", "sus2": "sus2", "sus4": "sus4", "7sus4": "7sus4", "add9": "add9", "5": "5",
                 "°": "dim", "°7": "dim7", "ø7": "m7b5", "ø": "m7b5"}
    lower_map = {"": "min", "7": "m7", "6": "m6", "9": "m9", "°": "dim", "°7": "dim7", "ø7": "m7b5",
                 "ø": "m7b5", "add9": "madd9", "(maj7)": "mmaj7", "maj7": "mmaj7", "sus2": "sus2", "sus4": "sus4"}
    table = lower_map if lower else upper_map
    if suffix not in table:
        raise ValueError(f"Unknown chord suffix in {text!r}")
    return Chord(root, table[suffix])


def parse_numerals(text: str, tonic: int) -> list[Chord]:
    """'I V vi IV' or 'ii7 V7/V V7 I' in a key -> chords. X/Y means X of Y (V7/V is the
    dominant of the dominant)."""
    out = []
    for token in text.split():
        if "/" in token:
            head, target = token.split("/", 1)
            target_root = _parse_one(target, tonic).root
            out.append(_parse_one(head, target_root))
        else:
            out.append(_parse_one(token, tonic))
    return out


# Progression library ------------------------------------------------------------------------
# (category, name, numerals, mood). Written for a major key unless they start on i.
PROGRESSIONS: list[tuple[str, str, str, str]] = [
    ("Pop", "Four chords", "I V vi IV", "Happy"),
    ("Pop", "Sensitive", "vi IV I V", "Sad"),
    ("Pop", "Fifties", "I vi IV V", "Happy"),
    ("Pop", "Lift", "I IV vi V", "Hopeful"),
    ("Pop", "Start on four", "IV I V vi", "Hopeful"),
    ("Pop", "Climb", "I iii IV V", "Hopeful"),
    ("Pop", "Minor pop", "i bVI bIII bVII", "Sad"),
    ("Pop", "Pop punk", "I V vi iii IV I IV V", "Happy"),
    ("Worship and gospel", "Anthem", "IV I V vi", "Hopeful"),
    ("Worship and gospel", "Open worship", "I Vsus4 V vi IV", "Hopeful"),
    ("Worship and gospel", "Gospel turnaround", "I I7 IV iv I", "Happy"),
    ("Worship and gospel", "Gospel walk", "IV7 iii7 vi7 ii7 V7 I", "Hopeful"),
    ("Worship and gospel", "Lead to six", "I V7/vi vi IV", "Hopeful"),
    ("Worship and gospel", "Hymn close", "I IV I V I", "Calm"),
    ("Rock", "Mixolydian", "I bVII IV I", "Bold"),
    ("Rock", "Three chords", "I IV V IV", "Happy"),
    ("Rock", "Heavy", "i bVII bVI bVII", "Dark"),
    ("Rock", "Power", "I bIII IV I", "Bold"),
    ("Jazz", "Two five one", "ii7 V7 Imaj7", "Smooth"),
    ("Jazz", "Minor two five one", "iiø7 V7 i7", "Dark"),
    ("Jazz", "Turnaround", "Imaj7 vi7 ii7 V7", "Smooth"),
    ("Jazz", "Long turnaround", "iii7 VI7 ii7 V7 Imaj7", "Smooth"),
    ("Jazz", "Tritone sub", "ii7 bII7 Imaj7", "Smooth"),
    ("Blues", "Twelve bar", "I7 I7 I7 I7 IV7 IV7 I7 I7 V7 IV7 I7 V7", "Bold"),
    ("Blues", "Quick four", "I7 IV7 I7 I7 IV7 IV7 I7 I7 V7 IV7 I7 V7", "Bold"),
    ("Blues", "Minor blues", "i7 iv7 i7 i7 iv7 iv7 i7 i7 bVI7 V7 i7 V7", "Dark"),
    ("Folk and country", "Home and back", "I IV V I", "Happy"),
    ("Folk and country", "Country", "I V IV I", "Happy"),
    ("Folk and country", "Circle", "I vi ii V", "Calm"),
    ("Cinematic", "Epic minor", "i bVI bIII bVII", "Dark"),
    ("Cinematic", "Heroic", "I bVI bVII I", "Bold"),
    ("Cinematic", "Dorian", "i IV i IV", "Dreamy"),
    ("Cinematic", "Lydian float", "I II I II", "Dreamy"),
    ("Cinematic", "Tension", "i bII i bII", "Tense"),
    ("Cinematic", "Rise", "i bVI bVII i", "Bold"),
    ("Classical", "Perfect cadence", "IV V I", "Calm"),
    ("Classical", "Amen (plagal)", "IV I", "Calm"),
    ("Classical", "Deceptive", "ii V vi", "Sad"),
    ("Classical", "Circle of fifths", "I IV vii° iii vi ii V I", "Calm"),
    ("Classical", "Pachelbel", "I V vi iii IV I IV V", "Calm"),
    ("Classical", "Minor cadence", "i iv V i", "Sad"),
    ("Latin", "Andalusian", "i bVII bVI V", "Tense"),
    ("Latin", "Bolero", "i iv V7 i", "Sad"),
    ("Latin", "Bossa", "Imaj7 II7 ii7 V7", "Smooth"),
]
MOODS = ["Happy", "Sad", "Hopeful", "Calm", "Dark", "Dreamy", "Bold", "Smooth", "Tense"]


# Voicing ----------------------------------------------------------------------------------

VOICINGS = [("close", "Close"), ("inv1", "1st inversion"), ("inv2", "2nd inversion"),
            ("inv3", "3rd inversion"), ("drop2", "Drop 2"), ("drop3", "Drop 3"),
            ("open", "Open (spread)"), ("shell", "Shell (root, 3rd, 7th)")]


def _close(chord: Chord, octave: int) -> list[int]:
    base = 12 * (octave + 1) + chord.root
    return sorted(base + i for i in chord.intervals)


def _invert(pitches: list[int], times: int) -> list[int]:
    p = sorted(pitches)
    for _ in range(times % max(1, len(p))):
        p = sorted(p[1:] + [p[0] + 12])
    return p


def _shape(chord: Chord, voicing: str, octave: int) -> list[int]:
    p = _close(chord, octave)
    if voicing == "inv1":
        return _invert(p, 1)
    if voicing == "inv2":
        return _invert(p, 2)
    if voicing == "inv3":
        return _invert(p, 3 if len(p) > 3 else 2)
    if voicing == "drop2" and len(p) >= 3:
        q = sorted(p)
        q[-2] -= 12
        return sorted(q)
    if voicing == "drop3" and len(p) >= 4:
        q = sorted(p)
        q[-3] -= 12
        return sorted(q)
    if voicing == "open":
        return sorted([v + (12 if i % 2 else 0) for i, v in enumerate(p)])
    if voicing == "shell":
        ivs = chord.intervals
        keep = [0] + [i for i in ivs if i in (3, 4)][:1] + [i for i in ivs if i in (9, 10, 11)][:1]
        if len(keep) < 3:
            keep = list(ivs[:3])
        base = 12 * (octave + 1) + chord.root
        return sorted(base + i for i in keep)
    return p


def voice(chord: Chord, voicing: str = "close", octave: int = 4, prev: list[int] | None = None,
          lead: bool = False, bass: bool = False) -> list[int]:
    """MIDI pitches for a chord.

    lead=True with `prev` picks the inversion and octave that moves the fewest steps from
    the previous chord (voice leading), keeping the chord near the chosen octave."""
    if lead and prev:
        center = 12 * (octave + 1) + 6
        best, best_cost = None, None
        for inv in range(len(chord.intervals)):
            for shift in (-12, 0, 12):
                cand = [x + shift for x in _invert(_close(chord, octave), inv)]
                cost = _movement(prev, cand) + 0.15 * abs(sum(cand) / len(cand) - center)
                if best_cost is None or cost < best_cost:
                    best, best_cost = cand, cost
        pitches = best
    else:
        pitches = _shape(chord, voicing, octave)
    if bass:
        root = chord.bass if chord.bass is not None else chord.root
        low = min(pitches)
        b = low - ((low - root) % 12)
        if b == low:
            b -= 12
        pitches = [b] + pitches
    return [min(108, max(21, p)) for p in pitches]


def _movement(a: list[int], b: list[int]) -> float:
    """How far the voices move from chord a to chord b (each note to its nearest partner)."""
    return sum(min(abs(x - y) for y in a) for x in b) + sum(min(abs(x - y) for y in b) for x in a)


# Patterns ("motions") -------------------------------------------------------------------------

PATTERNS = [("block", "Block chord"), ("pulse", "Pulse on every beat"), ("pulse8", "Pulse on eighths"),
            ("arp_up", "Arpeggio up"), ("arp_down", "Arpeggio down"), ("arp_updown", "Arpeggio up and down"),
            ("broken", "Broken chord (1 5 3 5)"), ("alberti", "Alberti bass"), ("strum", "Strum"),
            ("pop", "Bass and chord (pop)"), ("waltz", "Waltz (bass, chord, chord)"),
            ("bass_root", "Bass: root notes"), ("bass_fifth", "Bass: root and fifth"),
            ("bass_walk", "Bass: walking")]


def _bass_note(chord: Chord, low: int = 36) -> int:
    root = chord.bass if chord.bass is not None else chord.root
    return low + (root - low) % 12


def pattern_notes(chord: Chord, pitches: list[int], start: float, length: float, beat: float,
                  pattern: str = "block", velocity: float = 0.7,
                  next_chord: Chord | None = None) -> list[tuple[float, float, int, float]]:
    """Turn one chord into timed notes: (start, end, pitch, strength)."""
    beat = max(0.05, beat)
    end = start + length
    out: list[tuple[float, float, int, float]] = []
    p = sorted(pitches)

    def add(t: float, dur: float, pitch: int, vel: float) -> None:
        if t < end - 1e-6:
            out.append((t, min(end, t + dur), int(pitch), float(max(0.05, min(1.0, vel)))))

    def steps(every: float):
        t = start
        i = 0
        while t < end - 1e-6:
            yield i, t
            i += 1
            t = start + i * every

    if pattern == "block":
        for x in p:
            add(start, length, x, velocity)
    elif pattern in ("pulse", "pulse8"):
        every = beat if pattern == "pulse" else beat / 2
        for i, t in steps(every):
            for x in p:
                add(t, every * 0.9, x, velocity * (1.0 if i % 2 == 0 or pattern == "pulse" else 0.8))
    elif pattern in ("arp_up", "arp_down", "arp_updown"):
        seq = p if pattern == "arp_up" else p[::-1]
        if pattern == "arp_updown":
            seq = p + p[-2:0:-1] if len(p) > 2 else p
        every = beat / 2
        for i, t in steps(every):
            add(t, every * 0.95, seq[i % len(seq)], velocity * (1.0 if i % 2 == 0 else 0.85))
    elif pattern in ("broken", "alberti"):
        low, high = p[0], p[-1]
        mid = p[len(p) // 2] if len(p) > 2 else p[-1]
        fifth = next((x for x in p if (x - p[0]) % 12 == 7), mid)
        third = next((x for x in p if (x - p[0]) % 12 in (3, 4)), mid)
        seq = [low, fifth, third, fifth] if pattern == "broken" else [low, high, mid, high]
        every = beat / 2
        for i, t in steps(every):
            add(t, every * 0.95, seq[i % 4], velocity * (1.0 if i % 4 == 0 else 0.8))
    elif pattern == "strum":
        # Down on the beat, up on the "and", like a guitar: D . D U . U D U over two beats.
        hits = [(0.0, "d"), (1.0, "d"), (1.5, "u"), (2.5, "u"), (3.0, "d"), (3.5, "u")]
        bar = 4 * beat
        t0 = start
        while t0 < end - 1e-6:
            for k, (pos, way) in enumerate(hits):
                t = t0 + pos * beat
                if t >= end - 1e-6:
                    break
                nxt = t0 + (hits[k + 1][0] * beat if k + 1 < len(hits) else bar)
                notes = p if way == "d" else p[::-1]
                for j, x in enumerate(notes):
                    add(t + j * 0.014, max(0.05, nxt - t - j * 0.014), x,
                        velocity * (1.0 if way == "d" else 0.75))
            t0 += bar
    elif pattern in ("pop", "waltz"):
        bass = _bass_note(chord)
        per_bar = 4 if pattern == "pop" else 3
        for i, t in steps(beat):
            pos = i % per_bar
            if pos == 0 or (pattern == "pop" and pos == 2):
                add(t, beat * 0.95, bass, velocity)
            else:
                for x in p:
                    add(t, beat * 0.8, x, velocity * 0.8)
    elif pattern in ("bass_root", "bass_fifth", "bass_walk"):
        bass = _bass_note(chord)
        fifth = bass + 7
        third = bass + (3 if chord.quality in MINORISH else 4)
        n_beats = max(1, int(round(length / beat)))
        for i, t in steps(beat):
            if pattern == "bass_root":
                note = bass
            elif pattern == "bass_fifth":
                note = bass if i % 2 == 0 else fifth
            else:
                walk = [bass, third, fifth, None]
                note = walk[i % 4]
                if note is None:
                    # last beat steps toward the next chord's root
                    target = _bass_note(next_chord) if next_chord else bass
                    note = target - 1 if target > bass else target + 1
                if i == n_beats - 1 and next_chord and i % 4 != 3:
                    target = _bass_note(next_chord)
                    note = target - 1 if target > note else target + 1
            add(t, beat * 0.92, note, velocity)
    else:
        for x in p:
            add(start, length, x, velocity)
    return out


# What comes next, substitutes, outside chords, key changes ------------------------------------

# Common moves between steps of a major key: degree -> [(degree, weight, reason)]
_NEXT_MAJOR = {
    0: [(3, .25, "Steps away from home"), (4, .2, "Builds tension"), (5, .2, "Turns softer, to the relative minor"),
        (1, .15, "Starts a two five one"), (2, .07, "Gentle step up")],
    1: [(4, .5, "Two to five, the classic setup"), (6, .1, "Leans hard toward home"), (3, .1, "Softer pre-dominant")],
    2: [(5, .4, "Falls a fifth"), (3, .3, "Steps up to four"), (1, .1, "Keeps the circle going")],
    3: [(4, .35, "Four to five, ready to resolve"), (0, .25, "Plagal, the Amen sound"),
        (1, .1, "Pre-dominant color"), (5, .1, "Turns softer")],
    4: [(0, .5, "Resolves home, the strongest pull"), (5, .25, "Deceptive, surprises the ear"),
        (3, .1, "Backs off, rock style"), (2, .05, "Unusual, floating")],
    5: [(3, .3, "Lifts to four"), (1, .25, "Falls a fifth to two"), (4, .15, "Builds to five"),
        (2, .1, "Steps down"), (0, .1, "Back home")],
    6: [(0, .6, "Leading tone resolves up"), (2, .1, "Moves to three")],
}
_NEXT_MINOR = {
    0: [(3, .25, "Steps away to four"), (5, .2, "Falls to the six, epic sound"), (6, .2, "Down to the seven"),
        (2, .1, "Up to the relative major")],
    1: [(4, .5, "Two to five in minor")],
    2: [(5, .3, "Bright move to six"), (3, .25, "To four"), (6, .2, "To seven")],
    3: [(4, .3, "Four to five, ready to resolve"), (0, .25, "Back home, plagal"), (6, .2, "Rock style to seven"),
        (5, .1, "To six")],
    4: [(0, .3, "Home"), (5, .3, "Deceptive, falls to six"), (3, .2, "Back to four")],
    5: [(6, .3, "Climbs to seven"), (3, .2, "To four"), (2, .15, "To the relative major"), (0, .1, "Home")],
    6: [(0, .35, "Resolves home, rock and epic style"), (2, .3, "Up to the relative major"), (5, .1, "Back to six")],
}


def _key_chords(tonic: int, mode: str, kind: str = "triads") -> list[Chord]:
    return diatonic(tonic, "Major" if mode == "major" else "Natural minor", kind)


def _same_family(a: Chord, b: Chord) -> bool:
    return a.root == b.root and ((a.quality in MINORISH) == (b.quality in MINORISH))


def _degree_of(chord: Chord, chords: list[Chord]) -> int | None:
    for i, c in enumerate(chords):
        if _same_family(chord, c):
            return i
    return None


def _as_kind(chord: Chord, kind: str, tonic: int, mode: str) -> Chord:
    """The same step of the key, with sevenths or ninths when the palette is set to those."""
    if kind == "triads":
        return chord
    for c in _key_chords(tonic, mode, kind):
        if c.root == chord.root and _same_family(c, chord):
            return c
    return chord


def suggest_next(chord: Chord, tonic: int, mode: str, kind: str = "triads") -> list[tuple[Chord, str]]:
    """Chords that sound natural after `chord` in this key, best first, each with a reason."""
    triads = _key_chords(tonic, mode)
    table = _NEXT_MAJOR if mode == "major" else _NEXT_MINOR
    out: list[tuple[Chord, str, float]] = []
    deg = _degree_of(chord, triads)
    if deg is not None:
        for target, weight, reason in table.get(deg, []):
            out.append((_as_kind(triads[target], kind, tonic, mode), tr(reason), weight))
        # A secondary dominant that leads strongly into the best next chord.
        for target, weight, _reason in table.get(deg, [])[:2]:
            t = triads[target]
            if t.root != tonic and t.quality != "dim":
                out.append((Chord((t.root + 7) % 12, "7"), tr("Leads strongly into {name}").format(name=t.name()),
                            weight * 0.35))
        if mode == "major" and deg == 3:
            out.append((Chord((tonic + 5) % 12, "min"), tr("Borrowed minor four, bittersweet"), 0.09))
        if mode == "minor" and deg in (3, 5):
            out.append((Chord((tonic + 7) % 12, "7"), tr("Major five, strong pull home"), 0.12))
    else:
        # Not from the key: a dominant chord wants to fall a fifth, others go to their nearest key chord.
        if chord.quality in ("7", "9", "maj", "aug7"):
            target = (chord.root + 5) % 12
            match = next((c for c in triads if c.root == target), Chord(target, "maj"))
            out.append((match, tr("Resolves down a fifth"), 0.5))
        out.append((triads[0], tr("Back home"), 0.3))
        out.append((triads[4], tr("Toward five"), 0.15))
    seen = set()
    result = []
    for c, reason, w in sorted(out, key=lambda x: -x[2]):
        if c == chord or c in seen:
            continue
        seen.add(c)
        result.append((c, reason))
    return result[:8]


def substitutes(chord: Chord, tonic: int, mode: str) -> list[tuple[Chord, str]]:
    """Chords that can stand in for `chord`, each with a reason."""
    out: list[tuple[Chord, str]] = []
    triads = _key_chords(tonic, mode)
    tones = set(chord.tones())
    for c in triads:
        if c.root != chord.root and len(tones & set(c.tones())) >= 2:
            out.append((c, tr("Shares two notes, a softer swap")))
    if chord.quality in ("7", "9") or (chord.quality == "maj" and (chord.root - tonic) % 12 == 7):
        out.append((Chord((chord.root + 6) % 12, "7"), tr("Tritone substitute, a jazzy slide down")))
    out.append((Chord((chord.root + 7) % 12, "7"), tr("Play it before this chord to lead in")))
    parallel = _key_chords(tonic, "minor" if mode == "major" else "major")
    for c in parallel:
        if c.root == chord.root and c.quality != chord.quality and c not in triads:
            out.append((c, tr("Borrowed from {key}").format(
                key=key_name(tonic, "minor" if mode == "major" else "major"))))
    if chord.quality in ("maj", "min"):
        out.append((Chord(chord.root, "sus2"), tr("Open sound, no third")))
        out.append((Chord(chord.root, "sus4"), tr("Suspended, wants to resolve")))
        richer = {"maj": ("maj7", "add9"), "min": ("m7", "madd9")}[chord.quality]
        for q in richer:
            out.append((Chord(chord.root, q), tr("Same chord with more color")))
    out.append((Chord((chord.root - 1) % 12, "dim7"), tr("Passing chord that slides into it")))
    seen, result = set(), []
    for c, reason in out:
        if c != chord and c not in seen:
            seen.add(c)
            result.append((c, reason))
    return result


def explore(tonic: int, mode: str) -> list[tuple[str, list[tuple[Chord, str]]]]:
    """Chords from outside the key, grouped, each with a short note."""
    triads = _key_chords(tonic, mode)
    inside = set(triads)
    other = "minor" if mode == "major" else "major"
    groups: list[tuple[str, list[tuple[Chord, str]]]] = []

    borrowed = [(c, numeral(c, tonic)) for c in _key_chords(tonic, other) if c not in inside]
    groups.append((tr("Borrowed from {key}").format(key=key_name(tonic, other)), borrowed))

    secondary = []
    for c in triads[1:]:
        if c.quality == "dim":
            continue
        v = Chord((c.root + 7) % 12, "7")
        secondary.append((v, tr("Five of {name}").format(name=c.name())))
    groups.append((tr("Secondary dominants"), secondary))

    mediants = []
    for step in (3, 4, 8, 9):
        c = Chord((tonic + step) % 12, "maj")
        if c not in inside:
            mediants.append((c, numeral(c, tonic)))
    groups.append((tr("Chromatic mediants"), mediants))

    colors = [(Chord((tonic + 1) % 12, "maj"), tr("Neapolitan")),
              (Chord((tonic + 1) % 12, "7"), tr("Tritone sub")),
              (Chord((tonic + 6) % 12, "dim7"), tr("Passing")),
              (Chord(tonic, "aug"), tr("Restless")),
              (Chord((tonic + 7) % 12, "aug"), tr("Pushes home")),
              (Chord((tonic + 5) % 12, "m6"), tr("Film noir"))]
    groups.append((tr("Other colors"), [(c, n) for c, n in colors if c not in inside]))
    return groups


def _triad_moves(c: Chord) -> dict[str, Chord]:
    """The neo-Riemannian moves from a major or minor triad."""
    if c.quality == "maj":
        return {"P": Chord(c.root, "min"), "R": Chord((c.root + 9) % 12, "min"), "L": Chord((c.root + 4) % 12, "min")}
    return {"P": Chord(c.root, "maj"), "R": Chord((c.root + 3) % 12, "maj"), "L": Chord((c.root + 8) % 12, "maj")}


def neo_riemannian(a: Chord, b: Chord) -> list[tuple[str, Chord]]:
    """Shortest chain of P, L and R moves from triad a to triad b."""
    start = Chord(a.root, "maj" if a.quality not in MINORISH else "min")
    goal = Chord(b.root, "maj" if b.quality not in MINORISH else "min")
    prev: dict[Chord, tuple[Chord, str] | None] = {start: None}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        if c == goal:
            break
        for move, nxt in _triad_moves(c).items():
            if nxt not in prev:
                prev[nxt] = (c, move)
                queue.append(nxt)
    path: list[tuple[str, Chord]] = []
    c = goal
    while prev.get(c):
        parent, move = prev[c]
        path.append((move, c))
        c = parent
    return [("", start)] + path[::-1]


def modulations(from_tonic: int, from_mode: str, to_tonic: int, to_mode: str) -> list[tuple[str, str, list[Chord]]]:
    """Ways to move from one key to another: (method, how it works, chords)."""
    home = Chord(from_tonic, "maj" if from_mode == "major" else "min")
    goal = Chord(to_tonic, "maj" if to_mode == "major" else "min")
    old, new = _key_chords(from_tonic, from_mode), _key_chords(to_tonic, to_mode)
    new_v7 = Chord((to_tonic + 7) % 12, "7")
    out: list[tuple[str, str, list[Chord]]] = []

    # 1. Pivot chord: a chord both keys share, heard as part of the old key and then the new one.
    common = [c for c in old if c in new and c != home and c.quality != "dim"]
    if common:
        def pre_dominant_rank(c: Chord) -> int:
            deg = (c.root - to_tonic) % 12
            return {2: 0, 5: 1, 9: 2, 3: 2, 8: 2}.get(deg, 3)
        pivot = sorted(common, key=pre_dominant_rank)[0]
        out.append((tr("Pivot chord"), tr("{pivot} belongs to both keys, so the ear hardly notices the change.").format(
            pivot=pivot.name()), [home, pivot, new_v7, goal]))

    # 2. Dominant approach: go straight to the new key's two and five.
    two = Chord((to_tonic + 2) % 12, "m7" if to_mode == "major" else "m7b5")
    out.append((tr("Two five into the new key"), tr("The new key's two and five chords point right at it."),
                [home, two, new_v7, goal]))

    # 3. Modal interchange: borrow a chord from the parallel key that also fits the new key.
    parallel = _key_chords(from_tonic, "minor" if from_mode == "major" else "major")
    borrowed = [c for c in parallel if c in new and c not in old]
    if borrowed:
        out.append((tr("Modal interchange"), tr("{chord} is borrowed from {key} and already belongs to the new key.").format(
            chord=borrowed[0].name(), key=key_name(from_tonic, "minor" if from_mode == "major" else "major")),
            [home, borrowed[0], new_v7, goal]))
    else:
        flip = Chord(from_tonic, "min" if from_mode == "major" else "maj")
        out.append((tr("Modal interchange"), tr("Switch the home chord between major and minor first, then move."),
                    [home, flip, new_v7, goal]))

    # 4. Chromatic mediant: a bright jump by a third, then into the new key.
    gap = (to_tonic - from_tonic) % 12
    if gap in (3, 4, 8, 9):
        out.append((tr("Chromatic mediant"), tr("The new key is a third away, so you can jump straight there."),
                    [home, Chord(to_tonic, "maj"), goal] if goal.quality == "min" else [home, goal]))
    else:
        mediant = Chord((from_tonic + (8 if from_mode == "major" else 4)) % 12, "maj")
        out.append((tr("Chromatic mediant"), tr("A major chord a third away opens the door, then the new five leads in."),
                    [home, mediant, new_v7, goal]))

    # 5. Neo-Riemannian: change one note at a time.
    chain = [c for _move, c in neo_riemannian(home, goal)]
    out.append((tr("Neo-Riemannian (P, L, R)"), tr("Each step changes one note of the chord: smooth and cinematic."),
                chain if len(chain) > 1 else [home, goal]))
    return out


# Circle of fifths -------------------------------------------------------------------------------

CIRCLE = [(i * 7) % 12 for i in range(12)]   # C G D A E B F# C# G# D# A# F


# Guessing chords from notes ------------------------------------------------------------------------

_GUESS_TYPES = ["maj", "min", "dim", "aug", "sus4", "sus2", "7", "maj7", "m7", "m7b5", "6", "m6"]


def guess_chord(weights: list[float], bass: int | None = None) -> tuple[Chord, float] | None:
    """Best matching chord for a pitch class histogram, and a 0..1 score.

    A rough guess: it scores how much of the sound sits on the chord's notes, and penalizes
    chord notes that are missing and notes that don't belong."""
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        return None
    present = np.count_nonzero(w > 0.06 * w.max())
    if present < 2:
        return Chord(int(np.argmax(w)), "5"), 0.3
    best = None
    for root in range(12):
        for q in _GUESS_TYPES:
            steps = QUALITIES[q][0]
            tones = {(root + s) % 12 for s in steps}
            inside = sum(w[t] for t in tones) / total
            missing = sum(1 for t in tones if w[t] < 0.06 * w.max())
            score = inside - 0.18 * missing - 0.015 * (len(steps) - 3)
            if bass is not None and bass == root:
                score += 0.06
            if best is None or score > best[0]:
                best = (score, root, q, tones)
    score, root, q, tones = best
    slash = bass if (bass is not None and bass in tones and bass != root) else None
    return Chord(root, q, slash), float(max(0.0, min(1.0, score)))


def guess_chords(notes: Iterable, t0: float, t1: float, step: float) -> list[tuple[float, float, Chord | None]]:
    """Walks through [t0, t1) in windows of `step` seconds and names the chord in each
    (None where it can't tell). Windows with the same chord next to each other are joined."""
    notes = sorted((n for n in notes if n.end > t0 and n.start < t1), key=lambda n: n.start)
    starts = [n.start for n in notes]
    longest = max((n.end - n.start for n in notes), default=0.0)
    out: list[tuple[float, float, Chord | None]] = []
    t = t0
    step = max(step, 0.1)
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
        chord = guess[0] if guess and guess[1] >= 0.45 else None
        if sum(weights) <= 0:
            chord = None
        if out and out[-1][2] == chord:
            out[-1] = (out[-1][0], e, chord)
        else:
            out.append((t, e, chord))
        t = e
    return out


# Guessing chords from the sound (chroma) ------------------------------------------------------------

_AUDIO_TYPES = [("maj", 1.0), ("min", 1.0), ("7", 0.93), ("m7", 0.93), ("maj7", 0.92), ("dim", 0.9)]


def _templates() -> tuple[np.ndarray, list[Chord | None]]:
    rows, labels = [], []
    for q, _bias in _AUDIO_TYPES:
        for root in range(12):
            t = np.zeros(12)
            for k, s in enumerate(QUALITIES[q][0]):
                t[(root + s) % 12] = 1.0 if k < 3 else 0.75
            rows.append(t / np.linalg.norm(t))
            labels.append(Chord(root, q))
    rows.append(np.ones(12) / np.sqrt(12))
    labels.append(None)
    return np.array(rows), labels


def chords_from_chroma(chroma: np.ndarray, times: list[float], end: float,
                       stay: float = 0.85) -> list[tuple[float, float, Chord | None]]:
    """chroma: 12 x N, one column per time slot starting at times[i]. Returns joined segments.

    Each slot is matched against chord shapes, then a Viterbi pass picks the most likely
    sequence, which stops the chords from flickering on passing notes."""
    if chroma.shape[1] == 0:
        return []
    templates, labels = _templates()
    biases = np.array([b for _q, b in _AUDIO_TYPES for _r in range(12)] + [0.55])
    energy = chroma.sum(axis=0)
    norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    sim = templates @ norm * biases[:, None]                      # states x frames
    quiet = energy < 0.08 * (np.percentile(energy, 95) + 1e-9)
    sim[-1, quiet] = 2.0                                          # silence is "no chord"
    emit = np.log(np.exp(sim * 12.0) / np.exp(sim * 12.0).sum(axis=0, keepdims=True) + 1e-12)
    s = len(labels)
    log_stay, log_move = np.log(stay), np.log((1 - stay) / (s - 1))
    score = emit[:, 0].copy()
    back = np.zeros(sim.shape, dtype=np.int32)
    for i in range(1, sim.shape[1]):
        best_prev = int(np.argmax(score))
        moved = score[best_prev] + log_move
        stayed = score + log_stay
        take_stay = stayed >= moved
        back[:, i] = np.where(take_stay, np.arange(s), best_prev)
        score = np.where(take_stay, stayed, moved) + emit[:, i]
    path = [int(np.argmax(score))]
    for i in range(sim.shape[1] - 1, 0, -1):
        path.append(int(back[path[-1], i]))
    path = path[::-1]
    out: list[tuple[float, float, Chord | None]] = []
    for i, state in enumerate(path):
        t0 = times[i]
        t1 = times[i + 1] if i + 1 < len(times) else end
        chord = labels[state]
        if out and out[-1][2] == chord:
            out[-1] = (out[-1][0], t1, chord)
        else:
            out.append((t0, t1, chord))
    return out


def chord_at(chords: list[tuple[float, float, Chord | None]], t: float) -> Chord | None:
    for a, b, c in chords:
        if a <= t < b:
            return c
    return None
