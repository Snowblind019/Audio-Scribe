"""Checking for a new version of Audio Scribe and installing it.

New versions come from the main branch of the GitHub repo. To stay safe:

* Only two addresses are ever contacted, over HTTPS with certificate checks:
  raw.githubusercontent.com for the signed file list, and codeload.github.com for the code.
  Redirects to anywhere else are refused.
* The file list (update/manifest.json) is signed by the maintainer's Ed25519 key. The public
  half lives in update_key.txt next to this file. A list that fails the signature is ignored.
* Every file in the downloaded zip must match the SHA-256 written in the signed list, and only
  the files in the list are installed.
* Only a newer version is ever offered, so an old signed list can't roll you back.

The files are put in place by update_helper.py, which runs after the app has closed.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from . import __version__

REPO = "Snowblind019/Audio-Scribe"
BRANCH = "main"
MANIFEST_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/update/manifest.json"
SIGNATURE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/update/manifest.sig"
ZIP_URL = f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}"
ALLOWED_HOSTS = {"raw.githubusercontent.com", "codeload.github.com"}

FORMAT = "audio-scribe-update"
MAX_MANIFEST = 2 * 1024 * 1024
MAX_SIGNATURE = 4096
MAX_ZIP = 300 * 1024 * 1024
MAX_FILE = 100 * 1024 * 1024
MAX_FILES = 5000
TIMEOUT = 30

KEY_FILE = Path(__file__).with_name("update_key.txt")
# Never written to by an update, whatever the list says.
PROTECTED = {".venv", ".tools", ".update", ".git"}
_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_TOP_DOTFILES = {".gitignore", ".gitattributes"}


class UpdateError(Exception):
    """Something went wrong. The message is meant for the user."""


class NotConfigured(UpdateError):
    pass


class Stopped(Exception):
    pass


@dataclass
class UpdateInfo:
    version: str
    notes: str
    date: str
    files: dict[str, str]          # path -> sha256
    manifest: bytes
    signature: bytes


# Versions --------------------------------------------------------------------------------------

def parse_version(text: str) -> tuple[int, ...]:
    """"2.0" and "2.0.0" are the same version, and "2.0.10" is newer than "2.0.9"."""
    text = text.strip()
    if not re.fullmatch(r"\d{1,6}(\.\d{1,6}){0,3}", text):
        raise ValueError(f"not a version: {text!r}")
    parts = [int(x) for x in text.split(".")]
    return tuple(parts + [0] * (4 - len(parts)))


def is_newer(version: str, than: str = __version__) -> bool:
    return parse_version(version) > parse_version(than)


# Paths -----------------------------------------------------------------------------------------

def valid_path(path: str) -> bool:
    """A file path from the signed list: relative, plain, and never inside the app's own folders."""
    if not isinstance(path, str) or not path or len(path) > 200 or "\\" in path:
        return False
    parts = path.split("/")
    if parts[0] in PROTECTED:
        return False
    if len(parts) == 1 and parts[0] in _TOP_DOTFILES:
        return True
    return all(_PART.match(p) and p not in (".", "..") for p in parts)


def app_dir() -> Path:
    """The folder the app runs from (the one with install.sh in it)."""
    return Path(__file__).resolve().parent.parent


def cache_root() -> Path:
    from .app import cache_dir
    path = cache_dir() / "update"
    path.mkdir(parents=True, exist_ok=True)
    return path


def can_update() -> str | None:
    """None when this copy can update itself, otherwise the reason it can't (for the user)."""
    folder = app_dir()
    if (folder / ".git").exists():
        return "This copy is a git checkout. Update it with git pull instead."
    if not os.access(folder, os.W_OK):
        return "The app folder can't be written to, so it can't update itself."
    return None


# Signature -------------------------------------------------------------------------------------

def public_key(path: Path | None = None) -> bytes | None:
    """The maintainer's public key (32 bytes), or None when updates aren't set up."""
    try:
        lines = (path or KEY_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    text = "".join(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))
    if not text:
        return None
    try:
        key = base64.b64decode(text, validate=True)
    except ValueError:
        return None
    return key if len(key) == 32 else None


def verify_signature(data: bytes, signature: bytes, key: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        sig = base64.b64decode(signature.strip(), validate=True)
        Ed25519PublicKey.from_public_bytes(key).verify(sig, data)
        return True
    except (InvalidSignature, ValueError):
        return False


def parse_manifest(data: bytes) -> dict:
    try:
        m = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UpdateError("The update information is damaged.") from None
    if not isinstance(m, dict) or m.get("format") != FORMAT:
        raise UpdateError("The update information is not for Audio Scribe.")
    version = m.get("version")
    files = m.get("files")
    if not isinstance(version, str) or not isinstance(files, dict) or not files or len(files) > MAX_FILES:
        raise UpdateError("The update information is damaged.")
    try:
        parse_version(version)
    except ValueError:
        raise UpdateError("The update information is damaged.") from None
    for path, digest in files.items():
        if not valid_path(path) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise UpdateError("The update information lists a file it shouldn't.")
    if "audioscribe/__init__.py" not in files:
        raise UpdateError("The update information is damaged.")
    notes = m.get("notes", "")
    date = m.get("date", "")
    m["notes"] = notes[:4000] if isinstance(notes, str) else ""
    m["date"] = date[:40] if isinstance(date, str) else ""
    return m


# Network ---------------------------------------------------------------------------------------

class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        host = (urlparse(newurl).hostname or "").lower()
        if urlparse(newurl).scheme != "https" or host not in ALLOWED_HOSTS:
            raise urllib.error.HTTPError(newurl, code, "redirect to another site refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener():
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), _NoRedirects())


def fetch(url: str, limit: int, progress: Callable[[float], None] | None = None,
          should_stop: Callable[[], bool] | None = None) -> bytes:
    """Download a whole file into memory, refusing anything too big or from another site."""
    u = urlparse(url)
    if u.scheme != "https" or (u.hostname or "").lower() not in ALLOWED_HOSTS:
        raise UpdateError("Refused to download from an unexpected address.")
    req = urllib.request.Request(url, headers={"User-Agent": f"AudioScribe/{__version__}",
                                               "Cache-Control": "no-cache"})
    try:
        with _opener().open(req, timeout=TIMEOUT) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if total > limit:
                raise UpdateError("The download is larger than expected.")
            chunks, got = [], 0
            while True:
                if should_stop and should_stop():
                    raise Stopped()
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                got += len(chunk)
                if got > limit:
                    raise UpdateError("The download is larger than expected.")
                chunks.append(chunk)
                if progress and total:
                    progress(got / total)
            return b"".join(chunks)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("No update information was found on GitHub yet.") from None
        raise UpdateError("GitHub answered with an error. Try again later.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if isinstance(exc, UpdateError):
            raise
        raise UpdateError("Could not reach GitHub. Check your internet connection and try again.") from None


# Checking and downloading -------------------------------------------------------------------------

def check(fetcher: Callable[..., bytes] = fetch) -> UpdateInfo | None:
    """The newer version on GitHub, or None when this one is the newest."""
    key = public_key()
    if key is None:
        raise NotConfigured("Updates aren't set up in this copy of Audio Scribe.")
    manifest = fetcher(MANIFEST_URL, MAX_MANIFEST)
    signature = fetcher(SIGNATURE_URL, MAX_SIGNATURE)
    if not verify_signature(manifest, signature, key):
        raise UpdateError("The update information on GitHub is not signed by the Audio Scribe key, "
                          "so it was ignored.")
    m = parse_manifest(manifest)
    if not is_newer(m["version"]):
        return None
    return UpdateInfo(m["version"], m["notes"], m["date"], dict(m["files"]), manifest, signature)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(info: UpdateInfo, progress: Callable[[float], None] | None = None,
             should_stop: Callable[[], bool] | None = None,
             fetcher: Callable[..., bytes] = fetch) -> Path:
    """Downloads the code, checks every file against the signed list, and puts the files in a
    staging folder. Returns that folder."""
    data = fetcher(ZIP_URL, MAX_ZIP, progress, should_stop)
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UpdateError("The download is damaged.") from None
    entries = {}
    with z:
        for item in z.infolist():
            if item.is_dir():
                continue
            parts = item.filename.split("/", 1)       # GitHub puts everything in "<repo>-<branch>/"
            if len(parts) == 2 and parts[1]:
                entries[parts[1]] = item
        stage = cache_root() / f"staging-{info.version}"
        shutil.rmtree(stage, ignore_errors=True)
        files_dir = stage / "files"
        for path, digest in info.files.items():
            if should_stop and should_stop():
                raise Stopped()
            item = entries.get(path)
            if item is None or item.file_size > MAX_FILE:
                shutil.rmtree(stage, ignore_errors=True)
                raise UpdateError("The newest code on GitHub doesn't match its signed file list yet. "
                                  "Try again later.")
            with z.open(item) as fh:
                content = fh.read(MAX_FILE + 1)
            if len(content) > MAX_FILE or _sha256(content) != digest:
                shutil.rmtree(stage, ignore_errors=True)
                raise UpdateError("The newest code on GitHub doesn't match its signed file list yet. "
                                  "Try again later.")
            target = files_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    init = (files_dir / "audioscribe" / "__init__.py").read_text(encoding="utf-8", errors="replace")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not m or m.group(1) != info.version:
        shutil.rmtree(stage, ignore_errors=True)
        raise UpdateError("The downloaded code has a different version than its signed file list.")
    (stage / "manifest.json").write_bytes(info.manifest)
    (stage / "manifest.sig").write_bytes(info.signature)
    return stage


# Installing --------------------------------------------------------------------------------------

def start_install(stage: Path, relaunch: bool = True) -> subprocess.Popen:
    """Starts the helper that waits for this app to close, puts the new files in place, updates the
    packages if needed, and opens the app again. Call it right before quitting."""
    helper_src = Path(__file__).with_name("update_helper.py")
    run_dir = Path(tempfile.mkdtemp(prefix="audio-scribe-update-"))
    helper = run_dir / "update_helper.py"
    shutil.copyfile(helper_src, helper)          # run from a copy, since the original gets replaced
    args = [sys.executable, str(helper), "--staging", str(stage), "--app", str(app_dir()),
            "--pid", str(os.getpid()), "--python", sys.executable,
            "--result", str(cache_root() / "result.json")]
    if relaunch:
        args.append("--relaunch")
    log = open(cache_root() / "update.log", "a", encoding="utf-8")
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": log, "cwd": str(run_dir)}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(args, **kwargs)  # noqa: S603 (fixed program and arguments)


def take_result() -> dict | None:
    """What the last update did, once, so the app can say so after it opens again."""
    path = cache_root() / "result.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        path.unlink()
    except OSError:
        pass
    return data if isinstance(data, dict) else None


def last_check_age() -> float:
    """Seconds since the last automatic check (a large number if never)."""
    path = cache_root() / "last-check"
    try:
        return time.time() - float(path.read_text().strip())
    except (OSError, ValueError):
        return 1e12


def mark_checked() -> None:
    try:
        (cache_root() / "last-check").write_text(str(time.time()))
    except OSError:
        pass
