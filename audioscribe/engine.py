"""The analysis pipeline.

Steps, in order:
  1. Decode the file to a WAV (any format FFmpeg understands).
  2. Estimate the tempo and beat positions (librosa).
  3. Optionally split the song into stems (Demucs).
  4. Transcribe the words with timestamps (faster-whisper).
  5. Find the notes in the full mix or in each chosen stem (Basic Pitch).
  6. Find the chords in the sound (librosa chroma).
  7. Estimate the key from the notes.

Everything here runs on a worker thread. Progress is reported through a
callback and the work can be stopped between chunks.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio, audiochords, drums
from .i18n import tr
from .music import KeyEstimate, estimate_key, format_time, pitch_class_weights
from .synth import default_instrument_for

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100

TRACK_COLORS = {
    "Full mix": "#69A7E0",
    "Vocals": "#E58AA7",
    "Bass": "#8CCB72",
    "Other": "#B39DEB",
    "Drums": "#5CC6C0",
}
STEM_ORDER = ["Vocals", "Bass", "Other", "Drums"]

# Frequency limits (Hz) per stem. Keeping each stem to its normal range
# removes a lot of stray notes from leftover bleed.
NOTE_RANGES = {"Vocals": (65.0, 1400.0), "Bass": (28.0, 420.0)}

WHISPER_MODELS = [
    ("tiny", "Tiny (fastest)"),
    ("base", "Base (fast)"),
    ("small", "Small (balanced)"),
    ("medium", "Medium (accurate)"),
    ("large-v3-turbo", "Large turbo (accurate, quicker)"),
    ("large-v3", "Large (most accurate)"),
]
_WHISPER_WEIGHT = {"tiny": 2, "base": 3, "small": 5, "medium": 9, "large-v3-turbo": 7, "large-v3": 14}

LANGUAGES = [
    ("", "Auto detect"), ("en", "English"), ("ar", "Arabic"), ("zh", "Chinese"),
    ("cs", "Czech"), ("nl", "Dutch"), ("fr", "French"), ("de", "German"),
    ("el", "Greek"), ("hi", "Hindi"), ("hu", "Hungarian"), ("it", "Italian"),
    ("ja", "Japanese"), ("ko", "Korean"), ("pl", "Polish"), ("pt", "Portuguese"),
    ("ro", "Romanian"), ("ru", "Russian"), ("es", "Spanish"), ("sv", "Swedish"),
    ("tr", "Turkish"), ("uk", "Ukrainian"),
]
LANGUAGE_NAMES = {code: name for code, name in LANGUAGES if code}


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass(eq=False)  # compared by identity, so a Note can be kept in a set while it is edited
class Note:
    start: float
    end: float
    pitch: int
    velocity: float  # 0..1
    edited: bool = False  # added or changed by hand in Edit mode

    def __post_init__(self) -> None:
        # Always plain Python numbers. A numpy number stored here can crash Qt later, when the
        # value is handed to one of its widgets.
        self.start, self.end = float(self.start), float(self.end)
        self.pitch, self.velocity = int(self.pitch), float(self.velocity)


@dataclass
class Track:
    """One part of the music (the full mix, or one stem) with its notes."""
    name: str
    color: str
    notes: list[Note]
    visible: bool = True        # shown and heard (false when muted, or when another part is solo)
    muted: bool = False
    solo: bool = False
    instrument: str = "piano"   # the sound used when the notes are played back
    audio: str | None = None    # the recording for this part (a stem, or the original file)


@dataclass
class Options:
    transcribe: bool = True
    model: str = "small"
    language: str | None = None
    translate: bool = False
    vad: bool = True
    notes: bool = True
    sensitivity: int = 50  # 0..100, higher finds more notes
    min_note_ms: int = 120
    separate: bool = False
    note_stems: list[str] = field(default_factory=lambda: ["Vocals", "Bass", "Other", "Drums"])
    chords: bool = True    # find the chords in the sound
    device: str = "cpu"  # "cpu" or "cuda"


@dataclass
class Result:
    source: str
    duration: float
    work_dir: str
    audio_files: dict[str, str]
    segments: list[Segment]
    language: str | None
    language_probability: float
    tracks: list[Track]
    tempo: float | None
    beats: list[float]
    key: KeyEstimate | None
    notes_requested: bool
    words_requested: bool
    translated: bool = False
    elapsed: float = 0.0
    chords: list = field(default_factory=list)       # (start, end, Chord or None) heard in the recording
    meter: int = 4                                   # beats per bar
    detected_tempo: float | None = None              # what the analysis found, so the grid can be reset
    detected_beats: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.detected_tempo is None:
            self.detected_tempo = self.tempo
        if not self.detected_beats:
            self.detected_beats = list(self.beats)


class Cancelled(Exception):
    """Raised inside the pipeline when the user presses Cancel."""


def stems_available() -> bool:
    return (importlib.util.find_spec("demucs") is not None
            and importlib.util.find_spec("torch") is not None)


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def friendly_error(exc: BaseException) -> str:
    """A short message for the error dialog. Full details go in the expandable part."""
    name = type(exc).__name__
    text = str(exc) or name
    lowered = text.lower()
    network_hints = ("connection", "resolve", "timed out", "huggingface", "localentrynotfound",
                     "offline", "max retries", "ssl")
    if any(h in lowered for h in network_hints) or "LocalEntryNotFound" in name:
        return ("Could not download a model. Models download the first time you use them, "
                "so check your internet connection and try again.")
    if "out of memory" in lowered or isinstance(exc, MemoryError):
        return ("Ran out of memory. Try a smaller Whisper model, turn off stem splitting, "
                "or use a shorter file.")
    return text


# Loaded models are kept between runs so the second analysis starts faster.
_whisper_cache: dict[tuple[str, str], object] = {}
_pitch_model = None
_demucs_model = None


def _load_whisper(size: str, device: str):
    key = (size, device)
    if key not in _whisper_cache:
        from faster_whisper import WhisperModel
        compute = "int8" if device == "cpu" else "float16"
        threads = max(4, (os.cpu_count() or 8) // 2)
        _whisper_cache[key] = WhisperModel(size, device=device, compute_type=compute,
                                           cpu_threads=threads)
    return _whisper_cache[key]


def _load_pitch_model():
    """Returns (model, predict). Basic Pitch runs through onnxruntime here."""
    global _pitch_model
    logging.disable(logging.WARNING)  # basic_pitch warns about every backend it can't find
    try:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model, predict
    finally:
        logging.disable(logging.NOTSET)
    if _pitch_model is None:
        _pitch_model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    return _pitch_model, predict


def _load_demucs():
    global _demucs_model
    if _demucs_model is None:
        from demucs.pretrained import get_model
        # "hf://" loads the safetensors release from the Demucs author's Hugging Face
        # account. Plain "htdemucs" would fall back to older pickle files if that
        # download failed, and pickle files can run code when loaded.
        model = get_model("hf://htdemucs")
        model.eval()
        _demucs_model = model
    return _demucs_model


class _Progress:
    """Turns per-step progress into one overall 0..1 value."""

    def __init__(self, steps: list[tuple[str, float]], report: Callable[[float, str], None]):
        self.weights = dict(steps)
        self.total = sum(self.weights.values()) or 1.0
        self.report = report
        self.base = 0.0
        self.key: str | None = None
        self.text = ""

    def start(self, key: str, text: str) -> None:
        if self.key is not None:
            self.base += self.weights[self.key]
        self.key, self.text = key, text
        self.report(self.base / self.total, text)

    def update(self, fraction: float, text: str | None = None) -> None:
        if text:
            self.text = text
        fraction = min(max(fraction, 0.0), 1.0)
        self.report((self.base + self.weights[self.key] * fraction) / self.total, self.text)


class Analyzer:
    def __init__(self, path: str | Path, options: Options, work_dir: str | Path,
                 report: Callable[[float, str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None):
        self.path = Path(path)
        self.options = options
        self.work_dir = Path(work_dir)
        self.report = report or (lambda f, t: None)
        self.should_stop = should_stop or (lambda: False)

    def _check(self) -> None:
        if self.should_stop():
            raise Cancelled()

    def _note_targets(self) -> list[str]:
        o = self.options
        if not o.notes:
            return []
        if o.separate:
            return [s for s in STEM_ORDER if s in o.note_stems]
        return ["Full mix"]

    def run(self) -> Result:
        o = self.options
        began = time.monotonic()
        targets = self._note_targets()

        steps = [("read", 1.0), ("tempo", 1.0)]
        if o.separate:
            steps.append(("separate", 10.0))
        if o.transcribe:
            steps.append(("transcribe", float(_WHISPER_WEIGHT.get(o.model, 6))))
        steps += [(f"notes:{name}", 3.0) for name in targets]
        if o.chords:
            steps.append(("chords", 1.0))
        progress = _Progress(steps, self.report)

        # 1. Decode
        progress.start("read", tr("Reading the audio"))
        stereo = audio.decode(self.path, SAMPLE_RATE, 2)
        duration = stereo.shape[1] / SAMPLE_RATE
        if duration < 0.2:
            raise ValueError("The audio is too short to analyze.")
        original = self.work_dir / "original.wav"
        audio.write_wav(original, stereo, SAMPLE_RATE)
        files = {"Original": str(original)}
        self._check()

        # 2. Tempo and beats
        progress.start("tempo", tr("Finding the tempo"))
        tempo, beats = self._tempo(original)
        self._check()

        # 3. Stems
        if o.separate:
            progress.start("separate", tr("Splitting into stems (first run downloads the model)"))
            files.update(self._separate(stereo, progress))
        del stereo
        self._check()

        # 4. Words
        segments: list[Segment] = []
        language, language_prob = None, 0.0
        if o.transcribe:
            source = files.get("Vocals", files["Original"])
            progress.start("transcribe", tr("Loading Whisper (first run downloads the model)"))
            segments, language, language_prob = self._transcribe(source, progress)

        # 5. Notes
        tracks: list[Track] = []
        for name in targets:
            self._check()
            path = files["Original"] if name == "Full mix" else files.get(name)
            if not path:
                continue
            text = tr("Finding notes in the full mix") if name == "Full mix" else \
                tr("Finding notes in the part: {part}").format(part=tr(name))
            progress.start(f"notes:{name}", text)
            notes = self._notes(path, name)
            tracks.append(Track(name, TRACK_COLORS.get(name, "#69A7E0"), notes,
                                instrument=default_instrument_for(name), audio=path))
            progress.update(1.0)

        # Every stem is a part you can mute, even when no notes were looked for in it.
        have = {t.name for t in tracks}
        for name in STEM_ORDER:
            if name in files and name not in have:
                tracks.append(Track(name, TRACK_COLORS.get(name, "#69A7E0"), [],
                                    instrument=default_instrument_for(name), audio=files[name]))
        tracks.sort(key=lambda t: (STEM_ORDER.index(t.name) if t.name in STEM_ORDER else -1))

        # 6. Chords
        chords = []
        if o.chords:
            self._check()
            progress.start("chords", tr("Finding the chords"))
            chords = self._chords(files, beats, duration)

        # 7. Key
        pitched = [n for t in tracks if t.name != "Drums" for n in t.notes]
        key = estimate_key(pitch_class_weights(pitched)) if pitched else None

        self.report(1.0, tr("Done"))
        return Result(
            source=str(self.path), duration=duration, work_dir=str(self.work_dir),
            audio_files=files, segments=segments, language=language,
            language_probability=language_prob, tracks=tracks, tempo=tempo, beats=beats,
            key=key, notes_requested=o.notes, words_requested=o.transcribe,
            translated=o.transcribe and o.translate, elapsed=time.monotonic() - began,
            chords=chords,
        )

    # Individual steps ------------------------------------------------------

    def _chords(self, files: dict[str, str], beats: list[float], duration: float) -> list:
        """Chords heard in the recording. With stems, the drums are left out first."""
        try:
            parts = [files[n] for n in ("Vocals", "Bass", "Other") if n in files]
            if parts:
                arrays = [audio.decode(p, audiochords.SR, 1)[0] for p in parts]
                mono = np.zeros(max(len(a) for a in arrays), dtype=np.float32)
                for a in arrays:
                    mono[:len(a)] += a
            else:
                mono = audio.decode(files["Original"], audiochords.SR, 1)[0]
            return audiochords.detect(mono, audiochords.SR, beats, duration)
        except Cancelled:
            raise
        except Exception as exc:  # chords are a nice extra, never a reason to fail
            log.warning("Chord detection failed: %s", exc)
            return []

    def _tempo(self, wav: Path) -> tuple[float | None, list[float]]:
        try:
            import librosa
            mono = audio.decode(wav, 22050, 1)[0]
            bpm, frames = librosa.beat.beat_track(y=mono, sr=22050)
            bpm = float(np.atleast_1d(bpm)[0])
            beats = librosa.frames_to_time(frames, sr=22050).astype(float).tolist()
            if bpm <= 0 or len(beats) < 8:
                return None, []
            return bpm, beats
        except Exception as exc:  # tempo is a nice extra, never a reason to fail
            log.warning("Tempo detection failed: %s", exc)
            return None, []

    def _separate(self, stereo: np.ndarray, progress: _Progress) -> dict[str, str]:
        import torch
        from demucs.apply import apply_model

        model = _load_demucs()
        self._check()
        use_cuda = self.options.device == "cuda" and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        progress.update(0.0, tr("Splitting into stems on the GPU") if use_cuda else tr("Splitting into stems on the CPU"))

        wav = torch.from_numpy(np.ascontiguousarray(stereo))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        length = wav.shape[1]

        def on_chunk(info: dict) -> None:
            if self.should_stop():
                raise Cancelled()
            if info.get("state") == "end":
                models = max(1, info.get("models", 1))
                done = (info.get("model_idx_in_bag", 0) + info.get("segment_offset", 0) / length) / models
                progress.update(done)

        with torch.no_grad():
            out = apply_model(model, ((wav - mean) / std)[None], device=device, shifts=1,
                              split=True, overlap=0.25, progress=False, num_workers=0,
                              callback=on_chunk)[0]
        out = out * std + mean

        files = {}
        for name, source in zip(model.sources, out):
            label = name.capitalize()
            path = self.work_dir / f"{name}.wav"
            audio.write_wav(path, source.cpu().numpy(), model.samplerate)
            files[label] = str(path)
        progress.update(1.0)
        return files

    def _transcribe(self, wav: str, progress: _Progress):
        o = self.options
        samples = audio.decode(wav, 16000, 1)[0]
        total = len(samples) / 16000

        def attempt(device: str):
            model = _load_whisper(o.model, device)
            self._check()
            progress.update(0.02, tr("Transcribing words"))
            seg_iter, info = model.transcribe(
                samples, language=o.language or None,
                task="translate" if o.translate else "transcribe",
                vad_filter=o.vad, word_timestamps=True, beam_size=5,
                condition_on_previous_text=False,
            )
            found = []
            for seg in seg_iter:
                self._check()
                words = [Word(float(w.start), float(w.end), w.word.strip())
                         for w in (seg.words or []) if w.word.strip()]
                found.append(Segment(float(seg.start), float(seg.end), seg.text.strip(), words))
                progress.update(seg.end / max(total, 1e-6),
                                tr("Transcribing words ({done} of {total})").format(
                                    done=format_time(seg.end, 0), total=format_time(total, 0)))
            return found, info.language, float(info.language_probability or 0.0)

        if o.device == "cuda":
            try:
                return attempt("cuda")
            except Cancelled:
                raise
            except Exception as exc:
                log.warning("GPU transcription failed, falling back to CPU: %s", exc)
                progress.update(0.0, tr("GPU not usable for Whisper, using the CPU instead"))
        return attempt("cpu")

    def _drum_hits(self, wav: str) -> list[Note]:
        samples = audio.decode(wav, 22050, 1)[0]
        hits = drums.detect_hits(samples, 22050, self.options.sensitivity)
        return [Note(t, t + drums.HIT_LENGTH, pitch, strength) for t, pitch, strength in hits]

    def _notes(self, wav: str, name: str) -> list[Note]:
        if name == "Drums":
            return self._drum_hits(wav)
        model, predict = _load_pitch_model()
        o = self.options
        s = min(max(o.sensitivity, 0), 100) / 100.0
        low, high = NOTE_RANGES.get(name, (None, None))
        _, _, events = predict(
            wav, model,
            onset_threshold=0.7 - 0.4 * s,
            frame_threshold=0.45 - 0.25 * s,
            minimum_note_length=float(o.min_note_ms),
            minimum_frequency=low, maximum_frequency=high,
        )
        notes = [Note(float(st), float(en), int(p), float(min(max(a, 0.0), 1.0)))
                 for st, en, p, a, *_ in events]
        notes.sort(key=lambda n: (n.start, n.pitch))
        return notes


def remove_dir(path: str | Path | None) -> None:
    if path:
        shutil.rmtree(path, ignore_errors=True)
