"""Reading any audio or video file, and writing WAV files.

Decoding goes through PyAV, which ships its own copy of FFmpeg, so nothing
extra has to be installed on the system to open mp3, flac, m4a, mp4, mkv
and so on.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def decode(path: str | Path, rate: int, channels: int, max_seconds: float | None = None) -> np.ndarray:
    """Decode the first audio track of a file.

    Returns float32 samples shaped (channels, samples) at the given rate. With max_seconds,
    decoding stops there, so a file can't decode into more audio than expected.
    """
    import av

    layout = "mono" if channels == 1 else "stereo"
    chunks: list[np.ndarray] = []

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("This file has no audio track.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        limit = None if max_seconds is None else int(max_seconds * rate)
        total = 0
        try:
            for frame in container.decode(stream):
                frame.pts = None
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
                    total += chunks[-1].shape[1]
                if limit is not None and total >= limit:
                    break
        except av.error.InvalidDataError:
            # Damaged data near the end of a file is common; keep what was read.
            pass
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())

    if not chunks:
        raise ValueError("Could not read any audio from this file.")
    data = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if limit is not None and data.shape[1] > limit:
        data = data[:, :limit]
    return data


def write_wav(path: str | Path, data: np.ndarray, rate: int) -> None:
    """Write (channels, samples) float audio as a 16-bit WAV file."""
    if data.ndim == 1:
        data = data[None, :]
    block = 1 << 20
    with wave.open(str(path), "wb") as out:
        out.setnchannels(int(data.shape[0]))
        out.setsampwidth(2)
        out.setframerate(int(rate))
        for start in range(0, data.shape[1], block):
            part = np.clip(data[:, start:start + block], -1.0, 1.0)
            out.writeframes((part.T * 32767.0).astype("<i2").tobytes())
