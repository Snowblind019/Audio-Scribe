"""Saving and opening projects (.ascribe files).

A project keeps everything from an analysis, so it never has to be run again: the words,
the notes (edited and as they were found), the chords, the beat grid, the choices made in
the app, and the audio (the recording and any stems, stored losslessly as FLAC).

The file is a zip with a fixed layout:
    project.json          the data (plain JSON, no code)
    audio/<name>.flac     one per stem: original, vocals, piano, guitar, bass, other, drums
    audio/take-<n>.flac   one per take recorded in the app (version 2)

Opening a project only ever reads those names, checks every value before using it, and
refuses files that are oversized, so a damaged or hostile file can't do anything beyond
failing to open.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable

from . import __version__, audio
from .engine import SAMPLE_RATE, Note, Result, Segment, Track, Word
from .music import KeyEstimate
from .theory import Chord

FORMAT = "audio-scribe-project"
VERSION = 2          # 2: volume, pan, piano and guitar stems, recorded takes
SUFFIX = ".ascribe"
STEMS = ["Original", "Vocals", "Piano", "Guitar", "Bass", "Other", "Drums"]
_MEMBER_RE = re.compile(r"^audio/(original|vocals|piano|guitar|bass|other|drums|take-\d{1,3})\.flac$")
MAX_TRACKS = 64
MAX_JSON = 64 * 1024 * 1024
MAX_MEMBER = 3 * 1024 * 1024 * 1024
MAX_TOTAL = 8 * 1024 * 1024 * 1024
MAX_NOTES = 400_000
MAX_SEGMENTS = 50_000
MAX_BEATS = 200_000


class Stopped(Exception):
    pass


class BadProject(ValueError):
    """The file is not a project this version can open."""


# Writing ------------------------------------------------------------------------------------

def _encode_flac(src: str, dst: Path) -> None:
    import av

    with av.open(src) as inp:
        in_stream = inp.streams.audio[0]
        with av.open(str(dst), "w", format="flac") as out:
            stream = out.add_stream("flac", rate=in_stream.rate)
            stream.codec_context.layout = in_stream.layout.name
            stream.codec_context.format = "s16"
            for frame in inp.decode(in_stream):
                frame.pts = None
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)


def _notes(notes) -> list:
    return [[round(n.start, 5), round(n.end, 5), int(n.pitch), round(n.velocity, 4), bool(n.edited)] for n in notes]


def _notes_rows(rows) -> list:
    """Snapshot tuples (start, end, pitch, strength, edited) as stored in a project."""
    return [[round(a, 5), round(b, 5), int(p), round(v, 4), bool(e)] for a, b, p, v, e in rows]


def _chords(lane) -> list:
    return [[round(a, 4), round(b, 4), c.to_dict() if c else None] for a, b, c in lane]


def _take_member(index: int) -> str:
    return f"audio/take-{index}.flac"


def _track_json(t: Track, index: int) -> dict:
    out = {"name": t.name, "color": t.color, "instrument": t.instrument, "muted": t.muted, "solo": t.solo,
           "notes": _notes(t.notes), "volume_db": round(t.volume_db, 2), "pan": round(t.pan, 3)}
    if t.take:
        out.update(take=True, offset=round(t.offset, 5), audio=_take_member(index))
    return out


def to_json(r: Result, state: dict) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "app_version": __version__,
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_name": str(state.get("source_name") or Path(r.source).name)[:1000],
        "duration": r.duration,
        "language": r.language,
        "language_probability": r.language_probability,
        "translated": r.translated,
        "words_requested": r.words_requested,
        "notes_requested": r.notes_requested,
        "tempo": r.tempo,
        "beats": [round(b, 5) for b in r.beats],
        "meter": r.meter,
        "detected_tempo": r.detected_tempo,
        "detected_beats": [round(b, 5) for b in r.detected_beats],
        "key": None if r.key is None else {"tonic": r.key.tonic, "mode": r.key.mode,
                                            "confidence": r.key.confidence,
                                            "alternative": list(r.key.alternative) if r.key.alternative else None},
        "segments": [{"start": s.start, "end": s.end, "text": s.text,
                      "words": [[w.start, w.end, w.text] for w in s.words]} for s in r.segments],
        "tracks": [_track_json(t, i) for i, t in enumerate(r.tracks)],
        "chords": _chords(r.chords),
        "audio": [name for name in STEMS if name in r.audio_files],
        "state": state,
    }


def save(path: str | Path, r: Result, state: dict, report: Callable[[float], None] | None = None,
         should_stop: Callable[[], bool] | None = None) -> None:
    """Writes the project. The file only appears once it is complete."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    tmp_dir = Path(r.work_dir) / f"save-{int(time.time() * 1000)}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in STEMS if n in r.audio_files]
    # every piece of audio: (where it is now, its name in the project)
    pieces = [(r.audio_files[n], f"audio/{n.lower()}.flac") for n in names]
    pieces += [(t.audio, _take_member(i)) for i, t in enumerate(r.tracks) if t.take and t.audio]
    try:
        with zipfile.ZipFile(part, "w") as z:
            z.writestr("project.json", json.dumps(to_json(r, state), ensure_ascii=False),
                       compress_type=zipfile.ZIP_DEFLATED)
            for i, (src, member) in enumerate(pieces):
                if should_stop and should_stop():
                    raise Stopped()
                flac = tmp_dir / f"piece-{i}.flac"
                _encode_flac(src, flac)
                z.write(flac, member, compress_type=zipfile.ZIP_STORED)
                flac.unlink(missing_ok=True)
                if report:
                    report((i + 1) / max(1, len(pieces)))
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# Reading ------------------------------------------------------------------------------------

def _num(x, lo: float = -1e9, hi: float = 1e9, name: str = "value") -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= hi:
        raise BadProject(f"Bad {name} in the project file.")
    return float(x)


def _int(x, lo: int, hi: int, name: str = "value") -> int:
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise BadProject(f"Bad {name} in the project file.")
    return int(x)


def _str(x, max_len: int = 10_000, name: str = "text", allow_none: bool = False) -> str | None:
    if x is None and allow_none:
        return None
    if not isinstance(x, str) or len(x) > max_len:
        raise BadProject(f"Bad {name} in the project file.")
    return x


def _bool(x, name: str = "flag") -> bool:
    if not isinstance(x, bool):
        raise BadProject(f"Bad {name} in the project file.")
    return x


def _list(x, max_len: int, name: str) -> list:
    if not isinstance(x, list) or len(x) > max_len:
        raise BadProject(f"Bad {name} in the project file.")
    return x


def _parse_notes(rows, duration: float) -> list[Note]:
    out = []
    for row in _list(rows, MAX_NOTES, "notes"):
        row = _list(row, 5, "note")
        if len(row) != 5:
            raise BadProject("Bad note in the project file.")
        start = _num(row[0], 0, duration + 1, "note start")
        end = _num(row[1], 0, duration + 1, "note end")
        out.append(Note(start, max(end, start + 0.001), _int(row[2], 0, 127, "pitch"),
                        _num(row[3], 0, 1, "strength"), _bool(row[4], "edited flag")))
    return out


def _parse_chords(rows, duration: float) -> list:
    out = []
    for row in _list(rows, MAX_BEATS, "chords"):
        row = _list(row, 3, "chord")
        if len(row) != 3:
            raise BadProject("Bad chord in the project file.")
        a, b = _num(row[0], 0, duration + 1, "chord start"), _num(row[1], 0, duration + 1, "chord end")
        c = None
        if row[2] is not None:
            if not isinstance(row[2], dict):
                raise BadProject("Bad chord in the project file.")
            try:
                c = Chord.from_dict({"root": _int(row[2].get("root"), 0, 11, "chord root"),
                                     "quality": _str(row[2].get("quality"), 16, "chord quality"),
                                     "bass": None if row[2].get("bass") is None
                                     else _int(row[2].get("bass"), 0, 11, "chord bass")})
            except ValueError as exc:
                raise BadProject(str(exc)) from None
        out.append((a, b, c))
    return out


def read_json(z: zipfile.ZipFile) -> dict:
    info = z.getinfo("project.json")
    if info.file_size > MAX_JSON:
        raise BadProject("The project file is too large.")
    with z.open(info) as f:
        raw = f.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise BadProject("The project file is too large.")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BadProject("The project data is damaged.") from None
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise BadProject("This is not an Audio Scribe project.")
    if not isinstance(data.get("version"), int) or data["version"] > VERSION:
        raise BadProject("This project was saved by a newer version of Audio Scribe.")
    return data


def check_zip(z: zipfile.ZipFile) -> list[str]:
    """Only the expected file names, and sane sizes. Returns the audio member names."""
    names = z.namelist()
    if "project.json" not in names:
        raise BadProject("This is not an Audio Scribe project.")
    total = 0
    audio_members = []
    for info in z.infolist():
        if info.filename == "project.json":
            pass
        elif _MEMBER_RE.match(info.filename):
            audio_members.append(info.filename)
        else:
            raise BadProject(f"Unexpected file inside the project: {info.filename[:80]}")
        if info.file_size > MAX_MEMBER:
            raise BadProject("A file inside the project is too large.")
        total += info.file_size
    if total > MAX_TOTAL or len(names) != len(set(names)):
        raise BadProject("The project is too large or damaged.")
    return audio_members


def load(path: str | Path, work_dir: str | Path, report: Callable[[float], None] | None = None,
         should_stop: Callable[[], bool] | None = None) -> tuple[Result, dict]:
    path, work_dir = Path(path), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise BadProject("This file is not an Audio Scribe project, or it is damaged.") from None
    with z:
        members = check_zip(z)
        data = read_json(z)
        duration = _num(data.get("duration"), 0.05, 24 * 3600, "duration")

        segments = []
        for s in _list(data.get("segments", []), MAX_SEGMENTS, "lines"):
            if not isinstance(s, dict):
                raise BadProject("Bad line in the project file.")
            words = []
            for w in _list(s.get("words", []), 10_000, "words"):
                w = _list(w, 3, "word")
                if len(w) != 3:
                    raise BadProject("Bad word in the project file.")
                words.append(Word(_num(w[0], 0, duration + 1), _num(w[1], 0, duration + 1), _str(w[2], 500)))
            segments.append(Segment(_num(s.get("start"), 0, duration + 1), _num(s.get("end"), 0, duration + 1),
                                    _str(s.get("text"), 20_000), words))

        wanted = {m.split("/")[1].split(".")[0] for m in members if not m.startswith("audio/take-")}
        files: dict[str, str] = {}
        for i, name in enumerate(n for n in STEMS if n.lower() in wanted):
            if should_stop and should_stop():
                raise Stopped()
            member = f"audio/{name.lower()}.flac"
            flac = work_dir / f"{name.lower()}.flac"
            with z.open(member) as src, open(flac, "wb") as dst:
                copied = 0
                while chunk := src.read(1 << 20):
                    copied += len(chunk)
                    if copied > MAX_MEMBER:
                        raise BadProject("A file inside the project is too large.")
                    dst.write(chunk)
            wav = work_dir / ("original.wav" if name == "Original" else f"{name.lower()}.wav")
            audio.write_wav(wav, audio.decode(flac, SAMPLE_RATE, 2, max_seconds=duration + 5.0), SAMPLE_RATE)
            flac.unlink(missing_ok=True)
            files[name] = str(wav)
            if report:
                report((i + 1) / max(1, len(wanted)))
        if "Original" not in files:
            raise BadProject("The project has no audio.")
        take_files: dict[str, str] = {}
        for member in sorted(m for m in members if m.startswith("audio/take-")):
            if should_stop and should_stop():
                raise Stopped()
            stem = member.split("/")[1].split(".")[0]
            flac = work_dir / f"{stem}.flac"
            with z.open(member) as src, open(flac, "wb") as dst:
                copied = 0
                while chunk := src.read(1 << 20):
                    copied += len(chunk)
                    if copied > MAX_MEMBER:
                        raise BadProject("A file inside the project is too large.")
                    dst.write(chunk)
            wav = work_dir / f"{stem}.wav"
            audio.write_wav(wav, audio.decode(flac, SAMPLE_RATE, 2, max_seconds=duration + 5.0), SAMPLE_RATE)
            flac.unlink(missing_ok=True)
            take_files[member] = str(wav)

        tracks = []
        names_seen = set()
        for t in _list(data.get("tracks", []), MAX_TRACKS, "parts"):
            if not isinstance(t, dict):
                raise BadProject("Bad part in the project file.")
            name = _str(t.get("name"), 40, "part name")
            take = _bool(t.get("take", False), "take flag")
            if name in names_seen or not name.strip() or (not take and name not in ("Full mix", *STEMS[1:])) \
                    or (take and (name in ("Full mix", *STEMS) or not name.isprintable())):
                raise BadProject("Bad part in the project file.")
            names_seen.add(name)
            color = _str(t.get("color"), 9, "color")
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
                raise BadProject("Bad color in the project file.")
            from .synth import BY_KEY
            instrument = _str(t.get("instrument"), 32, "instrument")
            if instrument not in BY_KEY:
                instrument = "piano"
            offset = 0.0
            if take:
                member = _str(t.get("audio"), 40, "take audio")
                if member not in members or not member.startswith("audio/take-"):
                    raise BadProject("Bad take in the project file.")
                audio_path = take_files.get(member)
                offset = _num(t.get("offset", 0.0), -duration, duration, "take start")
            else:
                audio_path = files.get(name, files["Original"] if name == "Full mix" else None)
            tracks.append(Track(name, color, _parse_notes(t.get("notes", []), duration),
                                muted=_bool(t.get("muted", False)), solo=_bool(t.get("solo", False)),
                                instrument=instrument, audio=audio_path,
                                volume_db=_num(t.get("volume_db", 0.0), -60, 12, "volume"),
                                pan=_num(t.get("pan", 0.0), -1, 1, "pan"), offset=offset, take=take))

        key = None
        if data.get("key") is not None:
            k = data["key"]
            if not isinstance(k, dict):
                raise BadProject("Bad key in the project file.")
            mode = _str(k.get("mode"), 8)
            conf = _str(k.get("confidence"), 8)
            if mode not in ("major", "minor") or conf not in ("high", "medium", "low"):
                raise BadProject("Bad key in the project file.")
            alt = k.get("alternative")
            if alt is not None:
                alt = _list(alt, 2, "key")
                if len(alt) != 2 or alt[1] not in ("major", "minor"):
                    raise BadProject("Bad key in the project file.")
                alt = (_int(alt[0], 0, 11), alt[1])
            key = KeyEstimate(_int(k.get("tonic"), 0, 11), mode, conf, alt)

        def floats(x, name):
            return [_num(b, -60, duration + 60, name) for b in _list(x or [], MAX_BEATS, name)]

        tempo = data.get("tempo")
        detected_tempo = data.get("detected_tempo")
        result = Result(
            source=str(path), duration=duration, work_dir=str(work_dir), audio_files=files, segments=segments,
            language=_str(data.get("language"), 16, "language", allow_none=True),
            language_probability=_num(data.get("language_probability", 0.0), 0, 1),
            tracks=tracks, tempo=None if tempo is None else _num(tempo, 1, 1000, "tempo"),
            beats=floats(data.get("beats"), "beats"), key=key,
            notes_requested=_bool(data.get("notes_requested", True)),
            words_requested=_bool(data.get("words_requested", True)),
            translated=_bool(data.get("translated", False)),
            chords=_parse_chords(data.get("chords", []), duration),
            meter=_int(data.get("meter", 4), 1, 16, "beats per bar"),
            detected_tempo=None if detected_tempo is None else _num(detected_tempo, 1, 1000, "tempo"),
            detected_beats=floats(data.get("detected_beats"), "beats"),
        )
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        state["source_name"] = _str(data.get("source_name", path.name), 1000)
        if "original" in state:
            state["original"] = {name: _parse_notes(rows, duration) for name, rows in
                                 (state["original"].items() if isinstance(state["original"], dict) else [])
                                 if isinstance(name, str) and name in names_seen}
        return result, state
