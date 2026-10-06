"""Turning the notes into sheet music (MusicXML).

The notes are lined up with the beat grid (quantized), split into bars, and written as
MusicXML: one part per shown track, with a key signature, a time signature, ties across bar
lines, beams, accidentals, chord symbols, and the lyrics under the vocal line. MusicXML opens
in MuseScore, Dorico, Finale, Sibelius and most other notation programs; the app itself
engraves it with Verovio for the preview, PDF and printing.

Simplifications, because these are notes found in a recording, not written by a composer:
  * notes that overlap in one part are cut where the next note starts (one rhythm per staff),
  * the beat is a quarter note (2/4, 3/4, 4/4 ... time),
  * nothing is shorter than the chosen smallest note (an 8th or a 16th).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from . import grid
from .i18n import tr
from .music import _LETTERS_FLAT, _LETTERS_SHARP, pc_name, uses_flats
from .theory import QUALITIES, Chord

# Drum hits drawn on a percussion staff: (display step, octave, notehead)
DRUM_POSITIONS = {35: ("F", 4, None), 36: ("F", 4, None), 37: ("C", 5, "x"), 38: ("C", 5, None),
                  39: ("C", 5, "x"), 40: ("C", 5, None), 41: ("A", 4, None), 42: ("G", 5, "x"),
                  43: ("A", 4, None), 44: ("D", 4, "x"), 45: ("B", 4, None), 46: ("G", 5, "circle-x"),
                  47: ("B", 4, None), 48: ("E", 5, None), 49: ("A", 5, "x"), 50: ("E", 5, None),
                  51: ("F", 5, "x"), 52: ("A", 5, "x"), 53: ("F", 5, "diamond"), 57: ("A", 5, "x"),
                  59: ("F", 5, "x")}

_MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: 7, 8: -4, 3: -3, 10: -2, 5: -1}
_KEY_SHARPS = ["F", "C", "G", "D", "A", "E", "B"]
_KEY_FLATS = ["B", "E", "A", "D", "G", "C", "F"]


@dataclass
class SheetOptions:
    title: str = ""
    division: int = 4               # parts of a beat: 2 = eighth notes, 4 = sixteenth notes
    lyrics: bool = True
    note_names: bool = False        # write the note names under the notes
    chords: bool = True
    grand_staff: bool = True        # wide parts get a treble and a bass staff
    display: bool = False           # True for the app's own preview (chord names as plain text)
    key: tuple[int, str] | None = None
    meter: int = 4
    tempo: float | None = None
    subtitle: str = ""


@dataclass
class _Event:
    start: int
    dur: int
    pitches: list[int] | None       # None for a rest
    tie_start: bool = False
    tie_stop: bool = False
    lyric: str = ""
    names: str = ""
    beam1: str = ""
    beam2: str = ""
    full_rest: bool = False


@dataclass
class _Staff:
    clef: str                       # "G", "F" or "percussion"
    events: list[_Event] = field(default_factory=list)
    drums: bool = False


def key_fifths(tonic: int, mode: str) -> int:
    major = tonic if mode == "major" else (tonic + 3) % 12
    fifths = _MAJOR_FIFTHS[major]
    if fifths > 0 and uses_flats(tonic, mode):
        fifths -= 12
    if fifths < -6:
        fifths += 12
    return fifths


def _key_alters(fifths: int) -> dict[str, int]:
    if fifths > 0:
        return {s: 1 for s in _KEY_SHARPS[:fifths]}
    if fifths < 0:
        return {s: -1 for s in _KEY_FLATS[:-fifths]}
    return {}


def spell(pitch: int, flats: bool) -> tuple[str, int, int]:
    """(step letter, alter, octave) for a MIDI pitch."""
    name = (_LETTERS_FLAT if flats else _LETTERS_SHARP)[pitch % 12]
    step = name[0]
    alter = 1 if "#" in name else -1 if "b" in name else 0
    return step, alter, pitch // 12 - 1


# Rhythm ---------------------------------------------------------------------------------

def _note_values(div: int) -> dict[int, tuple[str, int]]:
    """Ticks -> (type, dots) for every writable length."""
    out = {}
    for name, quarters in (("whole", 4), ("half", 2), ("quarter", 1), ("eighth", 0.5), ("16th", 0.25)):
        ticks = quarters * div
        if ticks >= 1 and float(ticks).is_integer():
            out[int(ticks)] = (name, 0)
            dotted = ticks * 1.5
            if float(dotted).is_integer() and quarters < 4:
                out[int(dotted)] = (name, 1)
    return out


def _split(start: int, dur: int, div: int, values: dict[int, tuple[str, int]]) -> list[tuple[int, int]]:
    """Breaks a length into writable pieces, cutting at beats when a note starts off the beat."""
    pieces = []
    while dur > 0:
        if dur in values and (start % div == 0 or (start // div) == ((start + dur - 1) // div)):
            pieces.append((start, dur))
            break
        if start % div:
            to_beat = div - start % div
            if to_beat < dur:
                piece = to_beat
            else:
                piece = max(v for v in values if v <= dur)
        else:
            piece = max(v for v in values if v <= dur)
        pieces.append((start, piece))
        start += piece
        dur -= piece
    return pieces


# Building the model ---------------------------------------------------------------------

def _ticks(notes, beats, div, origin) -> list[tuple[int, int, int, float]]:
    out = []
    for n in notes:
        a = int(round((grid.time_to_beat(n.start, beats) - origin) * div))
        b = int(round((grid.time_to_beat(n.end, beats) - origin) * div))
        a = max(0, a)
        out.append((a, max(a + 1, b), n.pitch, n.velocity))
    return out


def _chordify(rows, measure: int) -> list[_Event]:
    """One rhythm per staff: notes that start together form a chord, a note is cut where
    the next one starts, gaps become rests."""
    by_start: dict[int, list] = {}
    for a, b, p, _v in rows:
        by_start.setdefault(a, []).append((b, p))
    onsets = sorted(by_start)
    events: list[_Event] = []
    cursor = 0
    for i, o in enumerate(onsets):
        if o > cursor:
            events.append(_Event(cursor, o - cursor, None))
        nxt = onsets[i + 1] if i + 1 < len(onsets) else None
        end = max(b for b, _p in by_start[o])
        if nxt is not None:
            end = min(end, nxt)
        pitches = sorted({p for _b, p in by_start[o]})
        events.append(_Event(o, max(1, end - o), pitches))
        cursor = o + max(1, end - o)
    total = int(math.ceil(max(cursor, 1) / measure) * measure)
    if total > cursor:
        events.append(_Event(cursor, total - cursor, None))
    return events


def _barred(events: list[_Event], measure: int, div: int) -> list[list[_Event]]:
    """Cut at bar lines (with ties), then into writable lengths."""
    values = _note_values(div)
    bars: list[list[_Event]] = []
    for ev in events:
        pos, left = ev.start, ev.dur
        pieces: list[tuple[int, int]] = []
        while left > 0:
            bar_start = (pos // measure) * measure
            take = min(left, bar_start + measure - pos)
            splitter = _split if ev.pitches else _rest_split
            for ps, pd in splitter(pos - bar_start, take, div, values):
                pieces.append((bar_start + ps, pd))
            pos += take
            left -= take
        for k, (s, d) in enumerate(pieces):
            e = _Event(s, d, ev.pitches,
                       tie_start=bool(ev.pitches) and k < len(pieces) - 1,
                       tie_stop=bool(ev.pitches) and k > 0,
                       lyric=ev.lyric if k == 0 else "", names=ev.names if k == 0 else "")
            index = s // measure
            while len(bars) <= index:
                bars.append([])
            bars[index].append(e)
    for b, bar in enumerate(bars):
        if not bar:
            bars[b] = [_Event(b * measure, measure, None, full_rest=True)]
        elif len(bar) == 1 and bar[0].pitches is None and bar[0].dur == measure:
            bar[0].full_rest = True
        elif all(e.pitches is None for e in bar):
            bars[b] = [_Event(b * measure, measure, None, full_rest=True)]
    return bars


def _rest_split(start: int, dur: int, div: int, values) -> list[tuple[int, int]]:
    if dur in values and start % div == 0:
        return [(start, dur)]
    return _split(start, dur, div, values)


def _beam(bar: list[_Event], div: int) -> None:
    """Beams join eighths and sixteenths inside one beat."""
    groups: dict[int, list[_Event]] = {}
    for e in bar:
        if e.pitches and e.dur < div and not e.full_rest:
            groups.setdefault(e.start // div, []).append(e)
    for evs in groups.values():
        evs = [e for e in evs if (e.start + e.dur - 1) // div == e.start // div]
        runs, run = [], []
        for e in sorted(evs, key=lambda x: x.start):
            if run and run[-1].start + run[-1].dur != e.start:
                runs.append(run)
                run = []
            run.append(e)
        if run:
            runs.append(run)
        for run in runs:
            if len(run) < 2:
                continue
            for k, e in enumerate(run):
                e.beam1 = "begin" if k == 0 else "end" if k == len(run) - 1 else "continue"
            sixteenth = div // 4 if div >= 4 else 0
            if sixteenth:
                for k, e in enumerate(run):
                    if e.dur <= sixteenth:
                        prev16 = k > 0 and run[k - 1].dur <= sixteenth
                        next16 = k + 1 < len(run) and run[k + 1].dur <= sixteenth
                        if prev16 and next16:
                            e.beam2 = "continue"
                        elif next16:
                            e.beam2 = "begin"
                        elif prev16:
                            e.beam2 = "end"
                        else:
                            e.beam2 = "forward hook" if k == 0 else "backward hook"


# Writing MusicXML -----------------------------------------------------------------------

def _type_of(dur: int, div: int) -> tuple[str, int]:
    return _note_values(div).get(dur, ("16th", 0))


def _harmony_xml(chord: Chord, display: bool, offset: int = 0) -> str:
    """A chord symbol. offset (in divisions) places it later than the note or rest it sits on."""
    off = f"<offset>{offset}</offset>" if offset > 0 else ""
    if display:
        return ('<direction placement="above"><direction-type><words font-weight="bold">'
                f'{escape(chord.name())}</words></direction-type>{off}</direction>')
    root_step, root_alter, _ = spell(chord.root + 60, uses_flats())
    kind = QUALITIES[chord.quality][2]
    text = QUALITIES[chord.quality][1]
    xml = (f'<harmony><root><root-step text="{escape(pc_name(chord.root))}">{root_step}</root-step>'
           + (f"<root-alter>{root_alter}</root-alter>" if root_alter else "")
           + f'</root><kind text="{escape(text)}">{kind}</kind>')
    if chord.bass is not None and chord.bass != chord.root:
        bs, ba, _ = spell(chord.bass + 60, uses_flats())
        xml += f"<bass><bass-step>{bs}</bass-step>" + (f"<bass-alter>{ba}</bass-alter>" if ba else "") + "</bass>"
    return xml + off + "</harmony>"


def build(tracks, beats: list[float], duration: float, opts: SheetOptions, segments=(), chords=()) -> str:
    """MusicXML for the given tracks (already filtered to what should be shown)."""
    div = max(1, int(opts.division))
    meter = max(1, int(opts.meter))
    measure = meter * div
    if len(beats) < 2:
        beats = grid.make_beats(opts.tempo or 120.0, 0.0, duration + 2)
    first_beat = min([grid.time_to_beat(n.start, beats) for t in tracks for n in t.notes] or [0.0])
    # a note a hair before the first beat lands on it once quantized, so it needs no pickup bar
    first_beat = round(first_beat * div) / div
    origin = min(0.0, math.floor(first_beat / meter) * meter)
    flats = uses_flats(*(opts.key or (None, None))) if opts.key else False
    fifths = key_fifths(*opts.key) if opts.key else 0

    parts_xml = []
    part_list = []
    total_bars = 0
    built = []
    for pi, track in enumerate(tracks):
        rows = _ticks(track.notes, beats, div, origin)
        drums = track.name == "Drums"
        staves: list[_Staff] = []
        pitches = [r[2] for r in rows]
        if drums:
            staves.append(_Staff("percussion", _chordify(rows, measure), drums=True))
        elif opts.grand_staff and pitches and min(pitches) < 55 and max(pitches) > 64:
            staves.append(_Staff("G", _chordify([r for r in rows if r[2] >= 60], measure)))
            staves.append(_Staff("F", _chordify([r for r in rows if r[2] < 60], measure)))
        else:
            median = sorted(pitches)[len(pitches) // 2] if pitches else 60
            staves.append(_Staff("G" if median >= 57 else "F", _chordify(rows, measure)))
        # words under the vocal line, and note names
        if staves and not drums:
            top = staves[0]
            if opts.lyrics and track.name in ("Vocals", "Full mix") and segments:
                for seg in segments:
                    for w in (seg.words or []):
                        tick = (grid.time_to_beat(w.start, beats) - origin) * div
                        best = min((e for e in top.events if e.pitches), key=lambda e: abs(e.start - tick),
                                   default=None)
                        if best is not None and abs(best.start - tick) <= div:
                            best.lyric = (best.lyric + " " + w.text).strip()
            if opts.note_names:
                for st in staves:
                    for e in st.events:
                        if e.pitches:
                            e.names = pc_name(max(e.pitches))
        bars_per_staff = [_barred(st.events, measure, div) for st in staves]
        for st_bars in bars_per_staff:
            for bar in st_bars:
                _beam(bar, div)
        total_bars = max([total_bars] + [len(b) for b in bars_per_staff])
        built.append((pi, track, staves, bars_per_staff))

    chord_ticks = []
    if opts.chords:
        last = None
        for a, _b, c in chords:
            if c is None or c == last:
                continue
            last = c
            # chords change on beats, so each one goes on the nearest beat (one that starts just
            # before the first beat lands on it)
            beat = max(0.0, round(grid.time_to_beat(a, beats)) - origin)
            chord_ticks.append((int(beat * div), c))

    if chord_ticks and any(track.name != "Drums" for _pi, track, _s, _b in built):
        total_bars = max(total_bars, chord_ticks[-1][0] // measure + 1)   # chords after the last note too

    chord_part = next((pi for pi, track, _s, _b in built if track.name != "Drums"), 0)
    for pi, track, staves, bars_per_staff in built:
        pid = f"P{pi + 1}"
        part_list.append(f'<score-part id="{pid}"><part-name>{escape(tr(track.name))}</part-name></score-part>')
        out = [f'<part id="{pid}">']
        for b in range(total_bars):
            out.append(f'<measure number="{b + 1}">')
            if b == 0:
                out.append(f"<attributes><divisions>{div}</divisions>")
                if not staves[0].drums:
                    out.append(f"<key><fifths>{fifths}</fifths></key>")
                out.append(f"<time><beats>{meter}</beats><beat-type>4</beat-type></time>")
                if len(staves) > 1:
                    out.append(f"<staves>{len(staves)}</staves>")
                for si, st in enumerate(staves):
                    num = f' number="{si + 1}"' if len(staves) > 1 else ""
                    if st.clef == "G":
                        out.append(f"<clef{num}><sign>G</sign><line>2</line></clef>")
                    elif st.clef == "F":
                        out.append(f"<clef{num}><sign>F</sign><line>4</line></clef>")
                    else:
                        out.append(f"<clef{num}><sign>percussion</sign></clef>")
                out.append("</attributes>")
                if pi == 0 and opts.tempo:
                    if opts.display:
                        out.append('<direction placement="above"><direction-type><words>'
                                   + escape(tr("Tempo {bpm} BPM").format(bpm=f"{opts.tempo:.0f}"))
                                   + f'</words></direction-type><sound tempo="{opts.tempo:.1f}"/></direction>')
                    else:
                        out.append('<direction placement="above"><direction-type><metronome><beat-unit>quarter'
                                   f'</beat-unit><per-minute>{opts.tempo:.0f}</per-minute></metronome></direction-type>'
                                   f'<sound tempo="{opts.tempo:.1f}"/></direction>')
            for si, st in enumerate(staves):
                bars = bars_per_staff[si]
                bar = bars[b] if b < len(bars) else [_Event(b * measure, measure, None, full_rest=True)]
                if si > 0:
                    out.append(f"<backup><duration>{measure}</duration></backup>")
                alters = dict(_key_alters(fifths)) if not st.drums else {}
                bar_chords = [(t, c) for t, c in chord_ticks if b * measure <= t < (b + 1) * measure] \
                    if (pi == chord_part and si == 0) else []
                for k, e in enumerate(bar):
                    last_event = k == len(bar) - 1
                    # every chord that starts during this note or rest goes here, moved along
                    # by an offset when it starts after the note does
                    while bar_chords and (bar_chords[0][0] < e.start + e.dur or last_event):
                        t, chord = bar_chords.pop(0)
                        out.append(_harmony_xml(chord, opts.display, max(0, min(t, e.start + e.dur - 1) - e.start)))
                    out.append(_event_xml(e, st, si, len(staves), div, flats, alters))
            out.append("</measure>")
        out.append("</part>")
        parts_xml.append("".join(out))

    title = escape(opts.title or tr("Untitled"))
    credit = escape(opts.subtitle or tr("Transcribed with Audio Scribe"))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n'
            '<score-partwise version="4.0">'
            f"<work><work-title>{title}</work-title></work>"
            f'<identification><creator type="composer">{credit}</creator>'
            "<encoding><software>Audio Scribe</software></encoding></identification>"
            f"<part-list>{''.join(part_list)}</part-list>"
            + "".join(parts_xml) + "</score-partwise>\n")


def _event_xml(e: _Event, st: _Staff, si: int, n_staves: int, div: int, flats: bool, alters: dict) -> str:
    staff = f"<staff>{si + 1}</staff>" if n_staves > 1 else ""
    voice = f"<voice>{1 if si == 0 else 5}</voice>"
    if e.pitches is None:
        if e.full_rest:
            return f'<note><rest measure="yes"/><duration>{e.dur}</duration>{voice}{staff}</note>'
        kind, dots = _type_of(e.dur, div)
        return (f"<note><rest/><duration>{e.dur}</duration>{voice}<type>{kind}</type>"
                + "<dot/>" * dots + staff + "</note>")
    kind, dots = _type_of(e.dur, div)
    out = []
    for k, pitch in enumerate(sorted(e.pitches)):
        x = ["<note>"]
        if k:
            x.append("<chord/>")
        head = None
        if st.drums:
            step, octave, head = DRUM_POSITIONS.get(pitch, ("C", 5, "x"))
            x.append(f"<unpitched><display-step>{step}</display-step><display-octave>{octave}</display-octave></unpitched>")
        else:
            step, alter, octave = spell(pitch, flats)
            x.append(f"<pitch><step>{step}</step>" + (f"<alter>{alter}</alter>" if alter else "")
                     + f"<octave>{octave}</octave></pitch>")
        x.append(f"<duration>{e.dur}</duration>")
        if e.tie_stop:
            x.append('<tie type="stop"/>')
        if e.tie_start:
            x.append('<tie type="start"/>')
        x.append(voice)
        x.append(f"<type>{kind}</type>" + "<dot/>" * dots)
        if not st.drums:
            mark = f"{step}{octave}"
            current = alters.get(mark, alters.get(step, 0))
            if alter != current and not e.tie_stop:
                x.append("<accidental>" + {1: "sharp", -1: "flat", 0: "natural"}[alter] + "</accidental>")
            alters[mark] = alter
        if head:
            x.append(f"<notehead>{head}</notehead>")
        x.append(staff)
        if k == 0:
            if e.beam1:
                x.append(f'<beam number="1">{e.beam1}</beam>')
            if e.beam2:
                x.append(f'<beam number="2">{e.beam2}</beam>')
        ties = []
        if e.tie_stop:
            ties.append('<tied type="stop"/>')
        if e.tie_start:
            ties.append('<tied type="start"/>')
        if ties:
            x.append("<notations>" + "".join(ties) + "</notations>")
        if k == 0:
            number = 1
            if e.lyric:
                x.append(f'<lyric number="1"><syllabic>single</syllabic><text>{escape(e.lyric)}</text></lyric>')
                number = 2
            if e.names:
                x.append(f'<lyric number="{max(number, 2)}"><syllabic>single</syllabic>'
                         f"<text>{escape(e.names)}</text></lyric>")
        x.append("</note>")
        out.append("".join(x))
    return "".join(out)
