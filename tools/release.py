"""Signing new versions for the in-app updater. For the maintainer only.

    python tools/release.py keygen      once: make your signing key, put the public half in the app
    python tools/release.py sign        before each push: sign the list of files for this version
    python tools/release.py verify      check that update/ matches what is staged

Run it with the app's own Python (.venv/bin/python, or .venv\\Scripts\\python.exe on Windows),
which has the cryptography package.

The private key stays on your computer, outside the repo, in
~/.config/audio-scribe-release/signing-key.pem (or the path in AUDIO_SCRIBE_SIGNING_KEY).
Back it up somewhere safe. Without it you can't publish updates the app will accept, and anyone
who gets it can.

"sign" signs exactly what git has staged, so line endings and ignored files can't make the list
differ from what GitHub serves. Stage everything first (git add -A), then sign, then add the
update folder and commit:

    git add -A
    python tools/release.py sign --notes "What changed, in a sentence or two"
    git add update
    git commit -m "Audio Scribe 2.0.2"
    git push
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audioscribe.updater import FORMAT, KEY_FILE, public_key, valid_path, verify_signature  # noqa: E402

UPDATE_DIR = ROOT / "update"
MANIFEST = UPDATE_DIR / "manifest.json"
SIGNATURE = UPDATE_DIR / "manifest.sig"
SKIP = {"update/manifest.json", "update/manifest.sig"}


def key_path() -> Path:
    env = os.environ.get("AUDIO_SCRIBE_SIGNING_KEY")
    if env:
        return Path(env).expanduser()
    base = Path(os.environ.get("APPDATA") or Path.home() / ".config") if os.name == "nt" else \
        Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "audio-scribe-release" / "signing-key.pem"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def staged_files() -> dict[str, str]:
    """Every staged file and the SHA-256 of its staged content."""
    out = {}
    for line in git_bytes("ls-files", "-s", "-z").split(b"\0"):
        if not line:
            continue
        meta, path = line.split(b"\t", 1)
        mode, blob, _stage = meta.decode().split()
        path = path.decode("utf-8")
        if path in SKIP:
            continue
        if mode == "120000":
            sys.exit(f"{path} is a symbolic link. The updater only installs plain files.")
        if not valid_path(path):
            sys.exit(f"{path} has a name the updater won't install. Rename it or remove it from git.")
        out[path] = hashlib.sha256(git_bytes("cat-file", "blob", blob)).hexdigest()
    return out


def version() -> str:
    text = (ROOT / "audioscribe" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not m:
        sys.exit("Could not find __version__ in audioscribe/__init__.py.")
    return m.group(1)


def load_private_key():
    from cryptography.hazmat.primitives import serialization
    path = key_path()
    if not path.exists():
        sys.exit(f"No signing key at {path}. Run: python tools/release.py keygen")
    data = path.read_bytes()
    try:
        return serialization.load_pem_private_key(data, password=None)
    except TypeError:
        pw = getpass.getpass("Signing key passphrase: ").encode()
        return serialization.load_pem_private_key(data, password=pw)


def cmd_keygen(_args) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = key_path()
    if path.exists():
        sys.exit(f"A signing key already exists at {path}. Delete it first if you really want a new one "
                 "(copies of the app that trust the old key won't accept updates signed by a new one).")
    pw = getpass.getpass("Passphrase for the key (recommended, Enter for none): ")
    if pw and getpass.getpass("Again: ") != pw:
        sys.exit("The passphrases don't match.")
    key = Ed25519PrivateKey.generate()
    enc = serialization.BestAvailableEncryption(pw.encode()) if pw else serialization.NoEncryption()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    KEY_FILE.write_text("# Audio Scribe update signing key (public half). Updates must be signed by its\n"
                        "# private half, which only the maintainer has.\n"
                        + base64.b64encode(pub).decode() + "\n", encoding="utf-8")
    print(f"Private key saved to {path}  (back it up somewhere safe, never commit it)")
    shown = KEY_FILE.relative_to(ROOT) if KEY_FILE.is_relative_to(ROOT) else KEY_FILE
    print(f"Public key written to {shown}  (commit this)")


def cmd_sign(args) -> None:
    if public_key() is None:
        sys.exit("audioscribe/update_key.txt has no key yet. Run: python tools/release.py keygen")
    status = [line for line in git("status", "--porcelain").splitlines()
              if line[3:] not in SKIP and not line[3:].startswith("update/")]
    unstaged = [line for line in status if line[1] != " " or line.startswith("??")]
    if unstaged:
        sys.exit("Some changes aren't staged, so they wouldn't be in the signed list:\n  "
                 + "\n  ".join(unstaged[:20]) + "\nStage everything first: git add -A")
    files = staged_files()
    if "audioscribe/update_key.txt" not in files:
        sys.exit("audioscribe/update_key.txt isn't staged. git add it first.")
    ver = version()
    manifest = {"format": FORMAT, "version": ver, "date": time.strftime("%Y-%m-%d"),
                "notes": args.notes or "", "files": dict(sorted(files.items()))}
    data = (json.dumps(manifest, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    key = load_private_key()
    pub = key.public_key().public_bytes(*_raw_public())
    if pub != public_key():
        sys.exit("Your signing key doesn't match audioscribe/update_key.txt.")
    sig = base64.b64encode(key.sign(data)) + b"\n"
    if not verify_signature(data, sig, pub):
        sys.exit("The new signature doesn't check out. Nothing was written.")
    UPDATE_DIR.mkdir(exist_ok=True)
    MANIFEST.write_bytes(data)
    SIGNATURE.write_bytes(sig)
    print(f"Signed version {ver}: {len(files)} files.")
    print("Now: git add update && git commit && git push")


def _raw_public():
    from cryptography.hazmat.primitives import serialization
    return serialization.Encoding.Raw, serialization.PublicFormat.Raw


def cmd_verify(_args) -> None:
    key = public_key()
    if key is None:
        sys.exit("No public key in audioscribe/update_key.txt.")
    data, sig = MANIFEST.read_bytes(), SIGNATURE.read_bytes()
    if not verify_signature(data, sig, key):
        sys.exit("The signature does NOT match.")
    listed = json.loads(data)["files"]
    staged = staged_files()
    diff = sorted(set(listed.items()) ^ set(staged.items()))
    if diff:
        sys.exit("The signed list differs from what is staged:\n  " + "\n  ".join(p for p, _ in diff[:20]))
    print(f"OK: version {json.loads(data)['version']}, {len(listed)} files, signature good.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sign Audio Scribe versions for the in-app updater.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen", help="make your signing key (once)")
    s = sub.add_parser("sign", help="sign the staged files as this version")
    s.add_argument("--notes", help="a short note about what changed, shown in the update prompt")
    sub.add_parser("verify", help="check update/ against the staged files")
    args = ap.parse_args()
    {"keygen": cmd_keygen, "sign": cmd_sign, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
