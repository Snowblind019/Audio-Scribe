"""The live audio engine: plays every part at once, mixed while it plays.

Each part (a stem, the full mix, or a take you recorded) is a lane with its own volume, pan,
mute and solo. The lane can sound like the recording, like its notes played on an
instrument, or both. Moving a fader is heard about a tenth of a second later, while the song
keeps playing, because nothing is rendered ahead of time: every block of sound is mixed just
before it goes to the speakers.

    lanes (recording and notes, per part) --+
    click track ----------------------------+--> mix --> speed (WSOLA) --> master --> speakers
    a note you click to hear ------------------------------------------------^

Recordings and rendered notes are 16-bit WAV files in the work folder, read through numpy
memory maps, so a long song with six stems needs almost no memory.

Slower or faster practice speed keeps the pitch. That is WSOLA: the song is cut into
overlapping 46 ms pieces, which are laid down closer together or further apart, each one
moved a little so its waveform lines up with the one before.

Threads: the sound card asks for audio on the engine's own thread (_Engine lives there), so a
busy window doesn't make the sound stutter. The window talks to it through Player, which only
sets values under a lock; the audio thread reads them once per block.
"""

from __future__ import annotations

import collections
import logging
import struct
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QIODevice, QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot

log = logging.getLogger(__name__)

SR = 44100
BUFFER_SECONDS = 0.12          # how far ahead the sound card is fed: the delay before a change is heard
IDLE_CLOSE_SECONDS = 6.0       # let go of the sound card when nothing has played for this long
MIN_LOOP = 0.05


# Reading audio --------------------------------------------------------------------------------

def open_wav(path: str | Path) -> np.ndarray:
    """A 16-bit WAV file as a read-only array shaped (frames, channels), without loading it."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"Not a WAV file: {path.name}")
        channels = bits = rate = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                raise ValueError(f"No audio data in {path.name}")
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                fmt = f.read(size)
                _tag, channels, rate = struct.unpack("<HHI", fmt[:8])
                bits = struct.unpack("<H", fmt[14:16])[0]
                f.seek(size % 2, 1)
            elif cid == b"data":
                offset = f.tell()
                break
            else:
                f.seek(size + size % 2, 1)
    if bits != 16 or rate != SR or not channels:
        raise ValueError(f"Unexpected audio format in {path.name}")
    file_size = path.stat().st_size
    frames = max(0, min(size, file_size - offset)) // (2 * channels)
    if frames == 0:
        return np.zeros((0, channels), "<i2")
    return np.memmap(path, dtype="<i2", mode="r", offset=offset, shape=(frames, channels))


class Clip:
    """Audio placed on the song's timeline, starting at `offset` frames."""

    def __init__(self, data: np.ndarray, offset: int = 0):
        self.data = data
        self.offset = int(offset)
        self.frames = len(data)
        self.scale = np.float32(1 / 32768.0) if data.dtype.kind == "i" else np.float32(1.0)

    @classmethod
    def from_wav(cls, path: str | Path, offset_seconds: float = 0.0) -> "Clip":
        return cls(open_wav(path), int(round(offset_seconds * SR)))

    def add_to(self, out: np.ndarray, start: int, gl: float, gr: float) -> float:
        """Adds frames [start, start + len(out)) of the song to out, scaled, and returns the peak."""
        n = len(out)
        a = start - self.offset
        lo, hi = max(0, a), min(self.frames, a + n)
        if hi <= lo or (gl == 0.0 and gr == 0.0):
            return 0.0
        seg = self.data[lo:hi]
        dst = out[lo - a:hi - a]
        if seg.shape[1] == 1:
            mono = seg[:, 0].astype(np.float32) * self.scale
            left, right = mono * np.float32(gl), mono * np.float32(gr)
        else:
            left = seg[:, 0].astype(np.float32) * (self.scale * np.float32(gl))
            right = seg[:, 1].astype(np.float32) * (self.scale * np.float32(gr))
        dst[:, 0] += left
        dst[:, 1] += right
        return float(max(np.abs(left).max(initial=0.0), np.abs(right).max(initial=0.0)))


# The mix ---------------------------------------------------------------------------------------

def db_to_gain(db: float) -> float:
    return 0.0 if db <= -60.0 else float(10.0 ** (db / 20.0))


def pan_gains(gain: float, pan: float) -> tuple[float, float]:
    """Balance: the centre leaves both sides alone, turning one way turns the other side down."""
    pan = min(1.0, max(-1.0, pan))
    return gain * min(1.0, 1.0 - pan), gain * min(1.0, 1.0 + pan)


@dataclass
class Lane:
    uid: str
    audio: Clip | None = None       # the recording of this part
    notes: Clip | None = None       # its notes played on an instrument
    gain: float = 1.0
    pan: float = 0.0
    audible: bool = True            # false when muted, or another part is soloed
    always_audio: bool = False      # a take you recorded: heard in every Play setting


class Mix:
    """Everything that gets mixed. The window changes it; the audio thread reads it."""

    def __init__(self):
        self.lock = threading.RLock()
        self.lanes: dict[str, Lane] = {}
        self.mode = "recording"          # "recording", "notes" or "both"
        self.notes_level = 0.7           # instruments' level in "both"
        self.click: Clip | None = None
        self.click_on = False
        self.click_level = 0.6
        self.master = 0.85
        self.frames = 0                  # length of the song
        self.peaks: dict[str, float] = {}
        self.master_peak = 0.0

    def set_lanes(self, lanes: list[Lane]) -> None:
        with self.lock:
            self.lanes = {lane.uid: lane for lane in lanes}
            self.peaks = {uid: 0.0 for uid in self.lanes}

    def update_lane(self, uid: str, **values) -> None:
        with self.lock:
            lane = self.lanes.get(uid)
            if lane is not None:
                for k, v in values.items():
                    setattr(lane, k, v)

    def take_meters(self) -> tuple[dict[str, float], float]:
        """The loudest level per lane (and of the whole mix) since the last call."""
        with self.lock:
            peaks, master = dict(self.peaks), self.master_peak
            for uid in self.peaks:
                self.peaks[uid] = 0.0
            self.master_peak = 0.0
        return peaks, master

    def render(self, start: int, n: int, meters: bool = True, only: str | None = None) -> np.ndarray:
        """Frames [start, start + n) of the mix, before the master volume. With `only`, just that
        one lane (with its own volume and pan, even when it is muted)."""
        out = np.zeros((n, 2), np.float32)
        with self.lock:
            use_audio = self.mode in ("recording", "both")
            use_notes = self.mode in ("notes", "both")
            notes_scale = self.notes_level if self.mode == "both" else 1.0
            for lane in self.lanes.values():
                if only is not None:
                    if lane.uid != only:
                        continue
                elif not lane.audible:
                    continue
                gl, gr = pan_gains(lane.gain, lane.pan)
                peak = 0.0
                if lane.audio is not None and (use_audio or lane.always_audio):
                    peak = max(peak, lane.audio.add_to(out, start, gl, gr))
                if lane.notes is not None and use_notes:
                    peak = max(peak, lane.notes.add_to(out, start, gl * notes_scale, gr * notes_scale))
                if meters and peak > self.peaks.get(lane.uid, 0.0):
                    self.peaks[lane.uid] = peak
            if only is None and self.click_on and self.click is not None:
                self.click.add_to(out, start, self.click_level, self.click_level)
        return out


def soft_clip(x: np.ndarray, knee: float = 0.9) -> np.ndarray:
    a = np.abs(x)
    if a.max(initial=0.0) <= knee:
        return x
    over = a > knee
    room = 1.0 - knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + room * np.tanh((a[over] - knee) / room))
    return y


# Speed without changing the pitch ----------------------------------------------------------------

class Wsola:
    """Time stretching by waveform similarity overlap-add. `read(start, n)` gives song frames
    (start may be negative or past the end; those frames are silent)."""

    N = 2048          # piece length (46 ms)
    HS = 1024         # output step: pieces overlap by half
    TOL = 512         # how far a piece may move to line up with the one before
    DEC = 4           # the lining up is worked out on every 4th sample, which is plenty

    def __init__(self, read: Callable[[int, int], np.ndarray]):
        self.read = read
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.N) / self.N)).astype(np.float32)[:, None]
        self.reset(0.0)

    def reset(self, pos: float) -> None:
        self.pos = float(pos)                 # where the next piece should ideally start
        self.prev: int | None = None          # where the last piece really started
        self.tail = np.zeros((self.HS, 2), np.float32)
        self.ready = np.zeros((0, 2), np.float32)
        self.ready_pos: collections.deque = collections.deque()   # (frames, source start, rate)

    def _hop(self, rate: float) -> None:
        ideal = int(round(self.pos))
        if self.prev is None:
            start = ideal
            piece = self.read(start, self.N)
        else:
            natural = self.read(self.prev + self.HS, self.N)
            lo = ideal - self.TOL
            region = self.read(lo, self.N + 2 * self.TOL)
            r = region[::self.DEC].sum(axis=1)
            t = natural[::self.DEC].sum(axis=1)
            corr = np.correlate(r, t, mode="valid")
            energy = np.convolve(r * r, np.ones(len(t), np.float32), mode="valid")
            k = int(np.argmax(corr / np.sqrt(energy + 1e-9)))
            start = lo + k * self.DEC
            piece = region[k * self.DEC:k * self.DEC + self.N]
        piece = piece * self.window
        out = self.tail + piece[:self.HS]
        self.tail = piece[self.HS:].copy()
        self.prev = start
        self.ready = np.concatenate([self.ready, out]) if len(self.ready) else out
        self.ready_pos.append((self.HS, self.pos, rate))
        self.pos += self.HS * rate

    def produce(self, n: int, rate: float) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
        """n output frames, and where in the song they came from: [(frames, start, rate)]."""
        while len(self.ready) < n:
            self._hop(rate)
        out, self.ready = self.ready[:n], self.ready[n:]
        spans, left = [], n
        while left > 0:
            frames, start, r = self.ready_pos[0]
            take = min(frames, left)
            spans.append((take, start, r))
            if take == frames:
                self.ready_pos.popleft()
            else:
                self.ready_pos[0] = (frames - take, start + take * r, r)
            left -= take
        return out, spans


# The engine on its own thread ----------------------------------------------------------------------

class _Transport:
    """Play state shared by the window and the audio thread (always used under Mix.lock)."""

    def __init__(self):
        self.playing = False
        self.vpos = 0.0                  # position in "virtual" frames (a loop counts on past its end)
        self.rate = 1.0
        self.loop: tuple[int, int] | None = None
        self.oneshot: tuple[np.ndarray, int] | None = None
        self.out_total = 0               # frames handed to the sound card so far
        self.spans: collections.deque = collections.deque(maxlen=512)   # (out start, frames, vstart, rate, loop)
        self.clock = (0.0, 0)            # (time, frames heard by then)
        self.reset_stretch = True
        self.ended = False
        self.last_sound = 0.0
        self.is_open = False             # the sound card is open (or being opened)
        self.mark = 0                    # output frame of the latest jump: nothing queued before it counts

    def heard_out(self) -> int:
        """The output frame coming out of the speakers now."""
        when, heard = self.clock
        if self.playing and self.is_open:
            heard = min(self.out_total, heard + int((time.monotonic() - when) * SR))
        return max(heard, self.mark)

    def heard(self) -> float:
        """Where in the song the sound coming out of the speakers is, following the loop it was
        played with (not the loop now, which may have just changed)."""
        out = self.heard_out()
        for out_start, frames, vstart, rate, loop in reversed(self.spans):
            if out_start <= out:
                return real_frame(vstart + min(out - out_start, frames) * rate, loop)
        return real_frame(self.vpos, self.loop)

    def jump(self, frame: float) -> None:
        """Play on from this song frame. Audio queued before now no longer counts as heard."""
        self.vpos = float(frame)
        self.reset_stretch = True
        self.mark = self.out_total
        self.clock = (time.monotonic(), self.out_total)
        self.spans.append((self.out_total, 0, self.vpos, 0.0, self.loop))


def real_frame(v: float, loop: tuple[int, int] | None) -> float:
    if loop is not None and v >= loop[1]:
        a, b = loop
        return a + (v - a) % (b - a)
    return v


class _Puller(QIODevice):
    def __init__(self, engine: "_Engine"):
        super().__init__()
        self.engine = engine

    def readData(self, maxlen: int):  # noqa: N802 (Qt name)
        try:
            return self.engine.pull(maxlen)
        except Exception:   # never let an error reach the sound card's thread
            log.exception("Audio engine error")
            return bytes(maxlen - maxlen % 8)

    def writeData(self, data) -> int:  # noqa: N802
        return -1

    def bytesAvailable(self) -> int:  # noqa: N802
        return 1 << 24

    def isSequential(self) -> bool:  # noqa: N802
        return True


class _Engine(QObject):
    """Lives on the audio thread and feeds the sound card."""

    ended = Signal()
    failed = Signal(str)

    def __init__(self, mix: Mix, t: _Transport):
        super().__init__()
        self.mix, self.t = mix, t
        self.sink = None
        self.puller: _Puller | None = None
        self.format_float = True
        self.out_rate = SR
        self.bpf = 8
        self.stretch = Wsola(self._read)
        self._carry = np.zeros((0, 2), np.float32)    # song frames made but not yet resampled
        self._carry_pos = 0.0                          # where in the carry the next output frame falls

    # Thread entry points (queued from Player) -----------------------------------------------------

    @Slot()
    def open(self) -> None:
        if self.sink is not None:
            return
        try:
            self._open()
        except Exception as exc:  # report it; the window carries on without sound
            log.exception("Could not open the sound output")
            self.sink = None
            self.failed.emit(str(exc) or "The sound output could not be opened.")

    def _open(self) -> None:
        from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.failed.emit("No sound output was found.")
            return
        fmt = QAudioFormat()
        fmt.setChannelCount(2)
        fmt.setSampleRate(SR)
        fmt.setSampleFormat(QAudioFormat.Float)
        if not device.isFormatSupported(fmt):
            fmt.setSampleFormat(QAudioFormat.Int16)
            if not device.isFormatSupported(fmt):
                fmt.setSampleRate(device.preferredFormat().sampleRate() or SR)
        self.format_float = fmt.sampleFormat() == QAudioFormat.Float
        self.out_rate = fmt.sampleRate()
        self.bpf = 8 if self.format_float else 4
        self._carry = np.zeros((0, 2), np.float32)
        self._carry_pos = 0.0
        self.sink = QAudioSink(device, fmt)
        self.sink.setBufferSize(int(self.out_rate * BUFFER_SECONDS) * self.bpf)
        self.puller = _Puller(self)
        self.puller.open(QIODevice.ReadOnly)
        self.sink.start(self.puller)
        log.info("Sound output: %s, %d Hz, %s", device.description(), self.out_rate,
                 "float" if self.format_float else "16-bit")

    @Slot()
    def close(self) -> None:
        if self.sink is None:
            return
        self.sink.stop()
        self.sink.deleteLater()
        self.sink = None
        if self.puller is not None:
            self.puller.close()
            self.puller = None
        with self.mix.lock:
            self.t.clock = (time.monotonic(), self.t.out_total)

    @Slot()
    def flush(self) -> None:
        """Drop what is already queued for the sound card, so a jump is heard right away.
        Playback goes on from what was really heard, so nothing is skipped."""
        if self.sink is not None:
            with self.mix.lock:
                if self.t.playing:
                    self.t.jump(self.t.heard())
            self.close()
            self.open()

    # Making sound ------------------------------------------------------------------------------------

    def _read(self, start: int, n: int) -> np.ndarray:
        """Song frames [start, start + n) of the mix, following the loop."""
        t, mix = self.t, self.mix
        loop = t.loop
        if loop is None or start + n <= loop[1]:
            return mix.render(start, n)
        out = np.empty((n, 2), np.float32)
        done = 0
        v = start
        while done < n:
            r = int(real_frame(v, loop)) if v >= loop[1] else v
            take = min(n - done, loop[1] - r) if r < loop[1] else n - done
            take = max(1, take)
            out[done:done + take] = mix.render(r, take)
            done += take
            v += take
        return out

    def _song_block(self, n: int) -> tuple[np.ndarray, list]:
        t = self.t
        if t.rate == 1.0:
            block = self._read(int(round(t.vpos)), n)
            spans = [(n, t.vpos, 1.0)]
            t.vpos += n
            return block, spans
        if t.reset_stretch:
            self.stretch.reset(t.vpos)
            t.reset_stretch = False
        block, spans = self.stretch.produce(n, t.rate)
        t.vpos = self.stretch.pos - len(self.stretch.ready) * t.rate
        return block, spans

    def _make(self, n: int) -> np.ndarray:
        t, mix = self.t, self.mix
        with mix.lock:
            if t.playing:
                block, spans = self._song_block(n)
                if t.loop is None and real_frame(t.vpos, None) >= mix.frames > 0:
                    # the end of the song: finish this block, then stop
                    over = int((t.vpos - mix.frames) / t.rate)   # song frames past the end, in output frames
                    if over > 0:
                        block[max(0, n - over):] = 0.0
                    t.playing = False
                    t.vpos = float(mix.frames)
                    t.ended = True
                for frames, start, rate in spans:
                    t.spans.append((t.out_total, frames, start, rate, t.loop))
                    t.out_total += frames
            else:
                block = np.zeros((n, 2), np.float32)
                t.spans.append((t.out_total, n, t.vpos, 0.0, t.loop))
                t.out_total += n
            block *= np.float32(mix.master)
            if t.oneshot is not None:
                sound, pos = t.oneshot
                take = min(n, len(sound) - pos)
                block[:take] += sound[pos:pos + take] * np.float32(mix.master)
                t.oneshot = (sound, pos + take) if pos + take < len(sound) else None
            peak = float(np.abs(block).max(initial=0.0))
            mix.master_peak = max(mix.master_peak, peak)
            if peak > 1e-4:
                t.last_sound = time.monotonic()
        return soft_clip(block)

    def _resampled(self, n_out: int) -> np.ndarray:
        """n_out frames for a sound card that won't take 44.1 kHz (linear interpolation).
        Song frames not used up yet stay in the carry for the next call."""
        step = SR / self.out_rate
        last = self._carry_pos + (n_out - 1) * step
        need = int(np.floor(last)) + 2 - len(self._carry)
        if need > 0:
            self._carry = np.concatenate([self._carry, self._make(need)])
        x = self._carry_pos + np.arange(n_out) * step
        i = np.floor(x).astype(int)
        frac = (x - i)[:, None].astype(np.float32)
        out = self._carry[i] * (1 - frac) + self._carry[i + 1] * frac
        end = self._carry_pos + n_out * step
        used = int(np.floor(end))
        self._carry = self._carry[used:]
        self._carry_pos = end - used
        return out

    def pull(self, maxlen: int) -> bytes:
        n_out = maxlen // self.bpf
        if n_out <= 0:
            return b""
        ratio = SR / self.out_rate          # song frames per output frame
        block = self._make(n_out) if self.out_rate == SR else self._resampled(n_out)
        with self.mix.lock:
            queued = 0
            if self.sink is not None:
                queued = max(0, (self.sink.bufferSize() - self.sink.bytesFree()) // self.bpf)
            heard = self.t.out_total - len(self._carry) - (n_out + queued) * ratio
            self.t.clock = (time.monotonic(), max(0, int(heard)))
            ended, self.t.ended = self.t.ended, False
        if ended:
            self.ended.emit()
        if self.format_float:
            return block.astype("<f4").tobytes()
        return (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Player(QObject):
    """What the window uses: play, pause, seek, speed, loop, and the live mix."""

    stateChanged = Signal(bool)      # playing or not
    finished = Signal()              # reached the end of the song
    failed = Signal(str)
    _open = Signal()
    _close = Signal()
    _flush = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mix = Mix()
        self.t = _Transport()
        self.thread = QThread()
        self.thread.setObjectName("audio")
        self.engine = _Engine(self.mix, self.t)
        self.engine.moveToThread(self.thread)
        self._open.connect(self.engine.open)
        self._close.connect(self.engine.close)
        self._flush.connect(self.engine.flush)
        self.engine.ended.connect(self._on_ended)
        self.engine.failed.connect(self._on_failed)
        self.thread.start(QThread.TimeCriticalPriority)
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        self._idle = QTimer(self)
        self._idle.setInterval(1000)
        self._idle.timeout.connect(self._check_idle)
        self._idle.start()
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(25)
        self._flush_timer.timeout.connect(self._flush.emit)

    # Output ------------------------------------------------------------------------------------------

    def _ensure_open(self) -> bool:
        """Opens the sound card if needed. True when it was already open."""
        if self.t.is_open:
            return True
        with self.mix.lock:
            self.t.is_open = True
            self.t.last_sound = time.monotonic()
            self.t.clock = (time.monotonic(), self.t.out_total)
        self._open.emit()
        return False

    def _check_idle(self) -> None:
        if not self.t.is_open:
            return
        with self.mix.lock:
            busy = self.t.playing or self.t.oneshot is not None
            quiet_for = time.monotonic() - self.t.last_sound
            if not busy and quiet_for > IDLE_CLOSE_SECONDS:
                self.t.is_open = False
            else:
                return
        self._close.emit()

    def _on_failed(self, message: str) -> None:
        """The sound card could not be opened: stop, so the next Play tries again."""
        with self.mix.lock:
            self.t.is_open = False
            was_playing, self.t.playing = self.t.playing, False
            self.t.oneshot = None
        if was_playing:
            self.stateChanged.emit(False)
        self.failed.emit(message)

    def shutdown(self) -> None:
        if not self.thread.isRunning():
            return
        self._idle.stop()
        self._flush_timer.stop()
        with self.mix.lock:
            self.t.playing = False
            self.t.oneshot = None
            was_open, self.t.is_open = self.t.is_open, False
        if was_open:
            # wait for the audio thread to let go of the sound card before it stops
            QMetaObject.invokeMethod(self.engine, "close", Qt.BlockingQueuedConnection)
        self.thread.quit()
        self.thread.wait(3000)

    # Transport ------------------------------------------------------------------------------------------

    @property
    def duration(self) -> float:
        return self.mix.frames / SR

    def set_duration(self, seconds: float) -> None:
        with self.mix.lock:
            self.mix.frames = int(round(max(0.0, seconds) * SR))
            if self.t.vpos > self.mix.frames and self.t.loop is None:
                self.t.vpos = float(self.mix.frames)

    def is_playing(self) -> bool:
        return self.t.playing

    def play(self) -> None:
        with self.mix.lock:
            if self.t.playing or self.mix.frames <= 0:
                return
            start = self.t.vpos
            if self.t.loop is None and start >= self.mix.frames - 1:
                start = 0.0
            self.t.jump(start)
            self.t.playing = True
        if self._ensure_open():
            self._flush_if_open()
        self.stateChanged.emit(True)

    def pause(self) -> None:
        with self.mix.lock:
            if not self.t.playing:
                return
            self.t.jump(self.t.heard())
            self.t.playing = False
        self._flush_if_open()
        self.stateChanged.emit(False)

    def toggle(self) -> None:
        self.pause() if self.t.playing else self.play()

    def seek(self, seconds: float) -> None:
        with self.mix.lock:
            # jumping out of a loop is allowed; it loops again once it gets back there
            self.t.jump(min(max(0.0, seconds * SR), float(self.mix.frames)))
            playing = self.t.playing
        if playing:
            self._flush_if_open()

    def _flush_if_open(self) -> None:
        # Many jumps in a row (dragging the position slider) only drop the queued sound once.
        if self.t.is_open and not self._flush_timer.isActive():
            self._flush_timer.start()

    def position(self) -> float:
        with self.mix.lock:
            if not self.t.playing:
                return real_frame(self.t.vpos, self.t.loop) / SR
            return self.t.heard() / SR

    def set_rate(self, rate: float) -> None:
        with self.mix.lock:
            rate = min(2.0, max(0.25, float(rate)))
            if abs(rate - self.t.rate) < 1e-6:
                return
            here = self.t.heard() if self.t.playing else self.t.vpos
            self.t.rate = rate
            self.t.jump(here)
        self._flush_if_open()

    def set_loop(self, span: tuple[float, float] | None) -> None:
        with self.mix.lock:
            here = self.t.heard() if self.t.playing else real_frame(self.t.vpos, self.t.loop)
            if span and span[1] - span[0] >= MIN_LOOP:
                a, b = int(round(span[0] * SR)), int(round(span[1] * SR))
                self.t.loop = (a, b)
                if here >= b:
                    here = float(a)
            else:
                self.t.loop = None
            self.t.jump(here)
        self._flush_if_open()

    def set_volume(self, value: float) -> None:
        with self.mix.lock:
            self.mix.master = float(max(0.0, value))

    # Hearing one sound now (a note, a chord, an instrument preview) ---------------------------------------

    def audition(self, sound: np.ndarray) -> None:
        """Stereo float (frames, 2) at 44.1 kHz, played over whatever else is playing."""
        if sound is None or not len(sound):
            return
        sound = np.asarray(sound, np.float32).reshape(-1, 2)
        with self.mix.lock:
            self.t.oneshot = (sound, 0)
        self._ensure_open()

    def _on_ended(self) -> None:
        self.stateChanged.emit(False)
        self.finished.emit()


# Saving what you hear -------------------------------------------------------------------------------------

def bounce(mix: Mix, out_path: str | Path, start: float = 0.0, end: float | None = None, only: str | None = None,
           master: bool = True, report: Callable[[float], None] | None = None,
           should_stop: Callable[[], bool] | None = None) -> None:
    """Writes the mix (or one lane) to a 16-bit stereo WAV file, as it sounds with the
    current volumes, pans, mutes and solos."""
    with mix.lock:
        total = mix.frames
        gain = mix.master if master else 1.0
    a = int(round(max(0.0, start) * SR))
    b = total if end is None else min(total, int(round(end * SR)))
    block = 10 * SR
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(SR)
        pos = a
        while pos < b:
            if should_stop and should_stop():
                raise InterruptedError()
            n = min(block, b - pos)
            buf = soft_clip(mix.render(pos, n, meters=False, only=only) * np.float32(gain))
            out.writeframes((np.clip(buf, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
            pos += n
            if report:
                report((pos - a) / max(1, b - a))
