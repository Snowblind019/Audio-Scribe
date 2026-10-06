"""Downloading the audio of a YouTube video, then saving it in the chosen format.

yt-dlp does the downloading. Since late 2025 YouTube needs a small JavaScript program to be
solved before it hands out the audio, so yt-dlp runs Deno (a JavaScript runtime) for that.
Both are optional and installed by the installer from hash-checked pins
(requirements-youtube.txt).

What this module allows, on purpose:
  * only youtube.com, youtu.be and music.youtube.com links,
  * one video at a time (playlists are not followed),
  * no cookies, no logins, no yt-dlp config files, no extra components downloaded at run time.

Converting to the chosen format is done here with PyAV (which the app already has), so no
FFmpeg program is needed.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from PySide6.QtCore import Qt, QThread, Signal

ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be",
                 "www.youtube-nocookie.com", "youtube-nocookie.com"}

# key, label, file extension, codec, options
FORMATS = [
    ("mp3", "MP3", "mp3", "libmp3lame"),
    ("m4a", "M4A (AAC)", "m4a", "aac"),
    ("opus", "Opus", "opus", "libopus"),
    ("flac", "FLAC (lossless)", "flac", "flac"),
    ("wav", "WAV (uncompressed)", "wav", "pcm_s16le"),
    ("original", "Original (no conversion)", "", ""),
]
QUALITIES = [("320", "320 kbps"), ("256", "256 kbps"), ("192", "192 kbps"), ("128", "128 kbps")]
MAX_SECONDS = 4 * 3600
MAX_BYTES = 2 * 1024 * 1024 * 1024


class Stopped(Exception):
    pass


def available() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def deno_path() -> str | None:
    try:
        from deno import find_deno_bin
        return find_deno_bin()
    except Exception:
        return shutil.which("deno")


def versions() -> str:
    parts = []
    try:
        from yt_dlp.version import __version__ as v
        parts.append(f"yt-dlp {v}")
    except Exception:
        pass
    try:
        from importlib.metadata import version
        parts.append(f"Deno {version('deno')}")
    except Exception:
        pass
    return ", ".join(parts)


def check_url(text: str) -> str:
    """Returns a clean link to one video, or raises ValueError with a reason."""
    text = text.strip()
    if not text:
        raise ValueError("Paste a YouTube link first.")
    if not re.match(r"^https?://", text, re.I):
        text = "https://" + text
    u = urlparse(text)
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or host not in ALLOWED_HOSTS:
        raise ValueError("Only YouTube links are supported (youtube.com, youtu.be, music.youtube.com).")
    if u.username or u.password or (u.port not in (None, 443, 80)):
        raise ValueError("That link has parts YouTube links never have.")
    video = None
    if host == "youtu.be":
        video = u.path.strip("/").split("/")[0]
    elif u.path == "/watch":
        video = parse_qs(u.query).get("v", [None])[0]
    else:
        m = re.match(r"^/(?:shorts|live|embed|v)/([^/?#]+)", u.path)
        if m:
            video = m.group(1)
    if not video or not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video):
        raise ValueError("That doesn't look like a link to one YouTube video.")
    return f"https://www.youtube.com/watch?v={video}"


_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

# Extensions a YouTube audio download can have, for "Original (no conversion)".
ORIGINAL_SUFFIXES = {".m4a", ".webm", ".opus", ".ogg", ".mp3", ".mp4", ".aac", ".flac", ".wav", ".mka"}


def safe_name(title: str) -> str:
    """A file name made from a video title that is safe on Windows and Linux."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f\x7f]', " ", title)
    name = re.sub(r"\s+", " ", name).strip(" .")[:120].strip(" .")
    if name.split(".")[0].upper() in _WINDOWS_RESERVED:
        name = "_" + name
    return name or "YouTube audio"


def unique(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError("Too many files with that name.")


def convert(src: Path, dst: Path, codec: str, bitrate_kbps: int | None,
            should_stop: Callable[[], bool] | None = None, title: str = "", artist: str = "") -> None:
    """Re-encode the audio of `src` into `dst` with PyAV."""
    import av

    with av.open(str(src)) as inp:
        if not inp.streams.audio:
            raise ValueError("The download has no audio.")
        in_stream = inp.streams.audio[0]
        with av.open(str(dst), "w") as out:
            rate = in_stream.rate or 48000
            if codec in ("libmp3lame", "aac", "flac", "pcm_s16le") and rate not in (44100, 48000):
                rate = 48000
            if codec == "libopus":
                rate = 48000
            stream = out.add_stream(codec, rate=rate)
            stream.codec_context.layout = "stereo" if (in_stream.layout.nb_channels or 2) >= 2 else "mono"
            if bitrate_kbps and codec in ("libmp3lame", "aac", "libopus"):
                stream.codec_context.bit_rate = bitrate_kbps * 1000
            if title:
                out.metadata["title"] = title
            if artist:
                out.metadata["artist"] = artist
            fmt = stream.codec_context.codec.audio_formats[0].name if stream.codec_context.codec.audio_formats else "s16"
            resampler = av.AudioResampler(format=fmt, layout=stream.codec_context.layout.name, rate=rate)
            for frame in inp.decode(in_stream):
                if should_stop and should_stop():
                    raise Stopped()
                frame.pts = None
                for f in resampler.resample(frame):
                    for packet in stream.encode(f):
                        out.mux(packet)
            for f in resampler.resample(None):
                for packet in stream.encode(f):
                    out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)


def download(url: str, folder: Path, fmt_key: str, quality: str, work: Path,
             progress: Callable[[float, str], None], should_stop: Callable[[], bool]) -> Path:
    """Downloads one video's audio and saves it in `folder`. Returns the saved file."""
    import yt_dlp

    url = check_url(url)
    work.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if should_stop():
            raise Stopped()
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if (total and total > MAX_BYTES) or done > MAX_BYTES:
                raise ValueError("The download is too large.")
            progress(0.9 * done / total if total else 0.0, "downloading")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": {"default": str(work / "download.%(ext)s")},
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "restrictfilenames": True,
        "windowsfilenames": True,
        "overwrites": True,
        "cachedir": False,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": MAX_BYTES,
        "remote_components": [],
    }
    deno = deno_path()
    if deno:
        opts["js_runtimes"] = {"deno": {"path": deno}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        progress(0.0, "looking")
        info = ydl.extract_info(url, download=False)
        if not info or info.get("_type") in ("playlist", "multi_video"):
            raise ValueError("That link is not a single video.")
        if (info.get("duration") or 0) > MAX_SECONDS:
            raise ValueError("This video is too long (the limit is 4 hours).")
        if should_stop():
            raise Stopped()
        progress(0.02, "downloading")
        info = ydl.process_ie_result(info, download=True)
    if info is None:
        raise ValueError("Nothing was downloaded.")
    files = [p for p in work.iterdir() if p.name.startswith("download.") and p.is_file()]
    if not files:
        raise ValueError("Nothing was downloaded. The video may be private or blocked in your country.")
    src = max(files, key=lambda p: p.stat().st_size)
    title = safe_name(str(info.get("title") or "YouTube audio"))
    artist = str(info.get("artist") or info.get("uploader") or "")[:200]
    folder.mkdir(parents=True, exist_ok=True)
    spec = next(f for f in FORMATS if f[0] == fmt_key)
    progress(0.92, "converting")
    if fmt_key == "original":
        if src.suffix.lower() not in ORIGINAL_SUFFIXES:
            raise ValueError("The download came in an unexpected format.")
        dst = unique(folder / f"{title}{src.suffix.lower()}")
        shutil.copyfile(src, dst)
    else:
        dst = unique(folder / f"{title}.{spec[2]}")
        bitrate = int(quality) if fmt_key in ("mp3", "m4a", "opus") else None
        try:
            convert(src, dst, spec[3], bitrate, should_stop, title=str(info.get("title") or "")[:300], artist=artist)
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
    progress(1.0, "done")
    return dst


def friendly(exc: BaseException) -> str:
    text = str(exc)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)       # yt-dlp colors its messages
    text = text.replace("ERROR: ", "")
    low = text.lower()
    if any(k in low for k in ("sign in", "private video", "members-only", "age")):
        return "YouTube won't give out this video without signing in, so it can't be downloaded here."
    if any(k in low for k in ("unable to download", "http error 403", "nsig", "signature", "challenge",
                              "javascript", "player")):
        return ("YouTube refused the download. This usually means YouTube changed something and the "
                "downloader needs an update: run the installer again with --update-youtube "
                "(Windows: install.bat -UpdateYouTube).")
    if any(k in low for k in ("getaddrinfo", "connection", "timed out", "network")):
        return "Could not reach YouTube. Check your internet connection and try again."
    return text[:500] or "The download failed."


class DownloadThread(QThread):
    progressed = Signal(float, str)
    succeeded = Signal(str)
    failed = Signal(str, str)

    def __init__(self, url: str, folder: Path, fmt_key: str, quality: str, parent=None):
        super().__init__(parent)
        self.url, self.folder, self.fmt_key, self.quality = url, folder, fmt_key, quality
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        work = Path(tempfile.mkdtemp(prefix="yt-"))
        try:
            path = download(self.url, self.folder, self.fmt_key, self.quality, work,
                            lambda f, s: self.progressed.emit(f, s), lambda: self._stop)
            self.succeeded.emit(str(path))
        except Stopped:
            self.failed.emit("", "")
        except Exception as exc:
            if self._stop:
                self.failed.emit("", "")
                return
            import traceback
            self.failed.emit(friendly(exc), traceback.format_exc())
        finally:
            shutil.rmtree(work, ignore_errors=True)


# The window ------------------------------------------------------------------------------

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,  # noqa: E402
                               QLineEdit, QProgressBar, QPushButton, QVBoxLayout)

from .i18n import tr  # noqa: E402


def default_folder() -> Path:
    music = Path.home() / "Music"
    return (music if music.is_dir() else Path.home()) / "Audio Scribe"


class YouTubeDialog(QDialog):
    downloaded = Signal(str, bool, bool)   # path, open it, analyze it

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle(tr("Download from YouTube"))
        self.setMinimumWidth(560)
        self.thread: DownloadThread | None = None
        lay = QVBoxLayout(self)

        self.url = QLineEdit()
        self.url.setPlaceholderText(tr("Paste a YouTube link, for example https://www.youtube.com/watch?v=..."))
        self.url.setClearButtonEnabled(True)
        lay.addWidget(self.url)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Save as")))
        self.fmt = QComboBox()
        for key, label, _ext, _codec in FORMATS:
            self.fmt.addItem(tr(label), key)
        self.quality = QComboBox()
        for key, label in QUALITIES:
            self.quality.addItem(label, key)
        self.quality.setProperty("i18n_skip_items", True)
        row.addWidget(self.fmt, 1)
        row.addWidget(self.quality)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.folder = Path(str(settings.value("youtube/folder", str(default_folder()))))
        self.folder_label = QLabel()
        self.folder_label.setTextFormat(Qt.PlainText)
        self.folder_label.setObjectName("Hint")
        self.folder_label.setWordWrap(True)
        change = QPushButton(tr("Change folder..."))
        change.clicked.connect(self._pick_folder)
        row.addWidget(self.folder_label, 1)
        row.addWidget(change)
        lay.addLayout(row)
        self._show_folder()

        self.open_chk = QCheckBox(tr("Open it in Audio Scribe when it's done"))
        self.open_chk.setChecked(True)
        self.analyze_chk = QCheckBox(tr("Analyze it right away"))
        self.open_chk.toggled.connect(self.analyze_chk.setEnabled)
        lay.addWidget(self.open_chk)
        lay.addWidget(self.analyze_chk)

        note = QLabel(tr("Only download videos you have the right to keep, such as your own uploads or "
                         "freely licensed music. YouTube's terms don't allow downloading other videos."))
        note.setObjectName("Hint")
        note.setWordWrap(True)
        lay.addWidget(note)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        lay.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        row = QHBoxLayout()
        self.versions = QLabel(versions())
        self.versions.setObjectName("Hint")
        self.versions.setProperty("i18n_skip", True)
        row.addWidget(self.versions, 1)
        self.close_btn = QPushButton(tr("Close"))
        self.close_btn.clicked.connect(self.reject)
        self.go_btn = QPushButton(tr("Download"))
        self.go_btn.setObjectName("Primary")
        self.go_btn.clicked.connect(self._go)
        row.addWidget(self.close_btn)
        row.addWidget(self.go_btn)
        lay.addLayout(row)

        i = self.fmt.findData(settings.value("youtube/format", "mp3"))
        self.fmt.setCurrentIndex(max(0, i))
        i = self.quality.findData(str(settings.value("youtube/quality", "256")))
        self.quality.setCurrentIndex(max(0, i))
        self.fmt.currentIndexChanged.connect(self._sync)
        self.url.returnPressed.connect(self._go)
        self._sync()
        if not available():
            self.go_btn.setEnabled(False)
            self.url.setEnabled(False)
            self.status.setText(tr("The YouTube downloader is not installed. Run the installer again and say yes "
                                   "to YouTube downloads (or use --with-youtube, on Windows -WithYouTube)."))

    def _show_folder(self) -> None:
        self.folder_label.setText(tr("Saved in {folder}").format(folder=str(self.folder)))

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Where should downloads go?"), str(self.folder))
        if folder:
            self.folder = Path(folder)
            self.settings.setValue("youtube/folder", folder)
            self._show_folder()

    def _sync(self) -> None:
        self.quality.setEnabled(self.fmt.currentData() in ("mp3", "m4a", "opus"))

    def _go(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.status.setText(tr("Stopping..."))
            return
        try:
            url = check_url(self.url.text())
        except ValueError as exc:
            self.status.setText(tr(str(exc)))
            return
        self.settings.setValue("youtube/format", self.fmt.currentData())
        self.settings.setValue("youtube/quality", self.quality.currentData())
        self.thread = DownloadThread(url, self.folder, self.fmt.currentData(), self.quality.currentData(), self)
        self.thread.progressed.connect(self._on_progress)
        self.thread.succeeded.connect(self._on_done)
        self.thread.failed.connect(self._on_failed)
        self.thread.finished.connect(self._on_finished)
        self.progress.setValue(0)
        self.progress.show()
        self.go_btn.setText(tr("Cancel"))
        self.url.setEnabled(False)
        self.status.setText(tr("Looking up the video..."))
        self.thread.start()

    def _on_progress(self, fraction: float, stage: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        text = {"looking": "Looking up the video...", "downloading": "Downloading...",
                "converting": "Converting...", "done": "Done."}.get(stage, "")
        self.status.setText(tr(text))

    def _on_done(self, path: str) -> None:
        self.status.setText(tr("Saved {path}").format(path=path))
        self.downloaded.emit(path, self.open_chk.isChecked(), self.analyze_chk.isChecked() and self.open_chk.isChecked())
        if self.open_chk.isChecked():
            self.accept()

    def _on_failed(self, message: str, details: str) -> None:
        self.status.setText(tr(message) if message else tr("Download cancelled."))

    def _on_finished(self) -> None:
        self.thread.deleteLater()
        self.thread = None
        self.progress.hide()
        self.go_btn.setText(tr("Download"))
        self.url.setEnabled(True)

    def reject(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread.wait(15000)
        super().reject()
