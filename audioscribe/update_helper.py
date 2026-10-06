"""Puts a downloaded and checked update in place. Started by the app right before it closes.

It runs from a temporary copy (the original gets replaced) and uses only the standard library:

1. waits for the app to close
2. backs up every file it is about to replace or remove
3. copies the new files in (each one written whole, then swapped in)
4. removes files the old version had and the new one doesn't
5. if the package lists or installers changed, runs the installer in update mode
6. writes what happened for the app to show, and opens the app again

If copying fails halfway, everything is put back from the backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROTECTED = {".venv", ".tools", ".update", ".git"}
_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
_TOP_DOTFILES = {".gitignore", ".gitattributes"}
DEPENDENCY_FILES = re.compile(r"^(requirements[\w.-]*\.txt|install\.(sh|ps1|bat))$")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def valid_path(path: str) -> bool:
    if not isinstance(path, str) or not path or len(path) > 200 or "\\" in path:
        return False
    parts = path.split("/")
    if parts[0] in PROTECTED:
        return False
    if len(parts) == 1 and parts[0] in _TOP_DOTFILES:
        return True
    return all(_PART.match(p) and p not in (".", "..") for p in parts)


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def wait_for_exit(pid: int, timeout: float = 120.0) -> bool:
    end = time.time() + timeout
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x00100000, False, pid)        # SYNCHRONIZE
        if not handle:
            return True
        try:
            return kernel32.WaitForSingleObject(handle, int(timeout * 1000)) == 0
        finally:
            kernel32.CloseHandle(handle)
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.2)
    return False


def load_files(manifest: Path) -> dict[str, str]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    files = data.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("bad file list")
    return {p: d for p, d in files.items() if valid_path(p)}


def write_result(path: Path, **data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:
        log(f"could not write the result: {exc}")


def install_files(stage: Path, app: Path, new: dict[str, str], old: dict[str, str], backup: Path) -> None:
    """Copies the new files in and removes the dropped ones. Restores everything on failure."""
    files_dir = stage / "files"
    removed = [p for p in old if p not in new]
    touched = list(new) + removed
    saved = []
    for rel in touched:
        src = app / rel
        if src.is_file():
            dst = backup / "files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            saved.append(rel)
    (backup / "saved.json").write_text(json.dumps({"saved": saved, "new": list(new)}), encoding="utf-8")
    done = []
    try:
        for rel in new:
            target = app / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".update-tmp")
            shutil.copyfile(files_dir / rel, tmp)
            if rel.endswith(".sh"):
                os.chmod(tmp, 0o755)
            os.replace(tmp, target)
            done.append(rel)
        for rel in removed:
            target = app / rel
            if target.is_file():
                target.unlink()
                done.append(rel)
    except Exception:
        log("copying failed, putting the old files back")
        for rel in done:
            original = backup / "files" / rel
            target = app / rel
            try:
                if original.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, target)
                elif target.is_file():
                    target.unlink()
            except OSError as exc:
                log(f"could not restore {rel}: {exc}")
        raise


def run_installer(app: Path) -> bool:
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(app / "install.ps1"), "-Update"]
    else:
        cmd = ["bash", str(app / "install.sh"), "--update"]
    log("updating packages: " + " ".join(cmd))
    extra = {"creationflags": 0x08000000} if os.name == "nt" else {}      # CREATE_NO_WINDOW: output goes to the log
    try:
        return subprocess.run(cmd, cwd=str(app), stdin=subprocess.DEVNULL, timeout=3600,  # noqa: S603
                              **extra).returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"the installer could not run: {exc}")
        return False


def relaunch(python: str, app: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(app) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    kwargs: dict = {"cwd": str(app), "env": env, "stdin": subprocess.DEVNULL,
                    "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([python, "-m", "audioscribe"], **kwargs)  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--app", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--relaunch", action="store_true")
    a = ap.parse_args(argv)
    stage, app, result = Path(a.staging), Path(a.app), Path(a.result)

    log(f"update from {stage} into {app}")
    if not wait_for_exit(a.pid):
        log("the app did not close, giving up")
        write_result(result, ok=False, message="The update was not installed because Audio Scribe did not close.")
        return 1
    try:
        version = json.loads((stage / "manifest.json").read_text(encoding="utf-8")).get("version", "")
        new = load_files(stage / "manifest.json")
        old_manifest = app / "update" / "manifest.json"
        old = load_files(old_manifest) if old_manifest.is_file() else {}
        deps_before = {p: sha256(app / p) for p in new if DEPENDENCY_FILES.match(p)}
        backup = stage.parent / f"backup-{time.strftime('%Y%m%d-%H%M%S')}"
        install_files(stage, app, new, old, backup)
        log(f"files updated to {version}")
    except Exception as exc:  # noqa: BLE001
        log(f"update failed: {exc!r}")
        write_result(result, ok=False, message="The update could not be installed, so nothing was changed.")
        if a.relaunch:
            relaunch(a.python, app)
        return 1

    deps_changed = any(sha256(app / p) != h for p, h in deps_before.items())
    ok, message = True, ""
    if deps_changed and not run_installer(app):
        ok = False
        message = ("Audio Scribe was updated, but updating its packages failed. Run the installer "
                   "again to finish (install.sh, or install.bat on Windows).")
    write_result(result, ok=ok, version=version, message=message)
    shutil.rmtree(stage, ignore_errors=True)
    for older in sorted(stage.parent.glob("backup-*"))[:-3]:        # keep the last three backups
        shutil.rmtree(older, ignore_errors=True)
    if a.relaunch:
        relaunch(a.python, app)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
