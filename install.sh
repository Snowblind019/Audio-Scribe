#!/usr/bin/env bash
# Audio Scribe installer for Linux.
#
#   ./install.sh                 install, asks about the optional parts
#   ./install.sh --with-stems    also install stem separation (Demucs, about 1 GB)
#   ./install.sh --no-stems      skip stem separation without asking
#   ./install.sh --cuda          use an NVIDIA GPU build of PyTorch for stems
#   ./install.sh --with-youtube  also install YouTube download (yt-dlp and Deno)
#   ./install.sh --no-youtube    skip YouTube download without asking
#   ./install.sh --update-youtube  get the newest yt-dlp and Deno (when YouTube changed)
#   ./install.sh --update        what the in-app updater runs: no questions, keep the optional parts as they are
#   ./install.sh --yes           accept the default answer for every question
#   ./install.sh --uninstall     remove the launcher, menu entry, and .venv
#
# The app stays in this folder. A private Python 3.12 environment is created
# in .venv here, so nothing touches your system Python.

set -euo pipefail

APP_NAME="Audio Scribe"
APP_ID="audio-scribe"
PYTHON_VERSION="3.12"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$APP_DIR/.venv"
PY="$VENV/bin/python"
BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/$APP_ID"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
DESKTOP_FILE="$DATA_HOME/applications/$APP_ID.desktop"
ICON_FILE="$DATA_HOME/icons/hicolor/256x256/apps/$APP_ID.png"

# uv (the tool that sets up Python and installs packages) is downloaded at a
# fixed version and checked against these SHA-256 hashes before it is used.
# The hashes match the ones published with the uv 0.12.17 release on GitHub.
UV_VERSION="0.12.17"
UV_SHA256_X86_64="fa82fd8dde8e8eefdecada6aa0889666556cfceb690d06e0c3bca49eb3070a63"
UV_SHA256_AARCH64="d636d1b678e9e7f367ecb22b46bd1cabbed234d6bc3b4d96365d2b507f72f86c"
TOOLS_DIR="$APP_DIR/.tools"
TORCH_VERSION="2.14.0"

STEMS="ask"
YOUTUBE="ask"
GPU="cpu"
ASSUME_YES=0
UPDATE=0
UNINSTALL=0
UV=""

say()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mWarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }

confirm() {
    # confirm "Question" y|n   (second argument is the default)
    local question="$1" default="${2:-n}" hint reply
    if [ "$ASSUME_YES" -eq 1 ]; then [ "$default" = "y" ]; return; fi
    if [ "$default" = "y" ]; then hint="[Y/n]"; else hint="[y/N]"; fi
    if [ -r /dev/tty ]; then
        read -r -p "$question $hint " reply < /dev/tty || reply=""
    else
        reply=""
    fi
    reply="${reply:-$default}"
    case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

run_root() {
    # Run a command as root, using sudo only when needed and available.
    if [ "$(id -u)" -eq 0 ]; then "$@"
    elif command -v sudo >/dev/null 2>&1; then sudo "$@"
    else return 1
    fi
}

for arg in "$@"; do
    case "$arg" in
        --with-stems) STEMS="yes" ;;
        --no-stems)   STEMS="no" ;;
        --cuda)       GPU="cuda" ;;
        --with-youtube)   YOUTUBE="yes" ;;
        --no-youtube)     YOUTUBE="no" ;;
        --update-youtube) YOUTUBE="update" ;;
        -y|--yes)     ASSUME_YES=1 ;;
        --update)     UPDATE=1; ASSUME_YES=1 ;;
        --uninstall)  UNINSTALL=1 ;;
        -h|--help)    usage; exit 0 ;;
        *) usage; die "Unknown option: $arg" ;;
    esac
done

uninstall() {
    say "Removing $APP_NAME"
    rm -f "$LAUNCHER" "$DESKTOP_FILE" "$ICON_FILE"
    rm -rf "$VENV" "$TOOLS_DIR"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA_HOME/applications" >/dev/null 2>&1 || true
    note "Removed the launcher, the menu entry, .venv, and .tools."
    note "The source folder was left in place: $APP_DIR"
    note "Downloaded models are shared and stay in ~/.cache/huggingface (delete it to free the space)."
    note "Settings live in ~/.config/AudioScribe."
}

if [ "$UNINSTALL" -eq 1 ]; then uninstall; exit 0; fi

[ -f "$APP_DIR/audioscribe/__init__.py" ] || die "Run this script from inside the Audio Scribe folder."

# 1. System library that Qt needs under X11 -----------------------------------------
check_qt_libs() {
    local ldconfig_bin
    ldconfig_bin="$(command -v ldconfig || echo /sbin/ldconfig)"
    if "$ldconfig_bin" -p 2>/dev/null | grep -q 'libxcb-cursor\.so\.0'; then return; fi

    local cmd=()
    if command -v dnf >/dev/null 2>&1; then cmd=(dnf install -y xcb-util-cursor)
    elif command -v apt-get >/dev/null 2>&1; then cmd=(apt-get install -y libxcb-cursor0)
    elif command -v pacman >/dev/null 2>&1; then cmd=(pacman -S --needed --noconfirm xcb-util-cursor)
    elif command -v zypper >/dev/null 2>&1; then cmd=(zypper install -y libxcb-cursor0)
    fi

    say "Qt needs the xcb-cursor library to open windows under X11"
    note "Wayland sessions work without it, but it is small and makes X11 work too."
    if [ "${#cmd[@]}" -eq 0 ]; then
        warn "Could not tell which package manager you use. Install xcb-util-cursor (or libxcb-cursor0) yourself if the app fails to open under X11."
        return
    fi
    if confirm "Install it now with: ${cmd[*]}?" y; then
        run_root "${cmd[@]}" || warn "Could not install it. Run this yourself if needed: sudo ${cmd[*]}"
    fi
}

# 2. uv, which manages Python and the packages ----------------------------------------
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

find_uv() {
    # A uv you installed yourself (for example from your distro) is used as is.
    if command -v uv >/dev/null 2>&1; then UV="$(command -v uv)"; return; fi
    if [ -x "$TOOLS_DIR/uv" ]; then UV="$TOOLS_DIR/uv"; return; fi

    local target expected
    case "$(uname -m)" in
        x86_64|amd64)  target="x86_64-unknown-linux-gnu";  expected="$UV_SHA256_X86_64" ;;
        aarch64|arm64) target="aarch64-unknown-linux-gnu"; expected="$UV_SHA256_AARCH64" ;;
        *) die "No pinned uv build for $(uname -m). Install uv yourself (your distro may package it), then run this again." ;;
    esac
    command -v curl >/dev/null 2>&1 || die "Need curl to download uv."

    say "Downloading uv $UV_VERSION and checking its SHA-256 hash"
    local tmp archive actual
    tmp="$(mktemp -d)"
    archive="$tmp/uv.tar.gz"
    curl --proto '=https' --tlsv1.2 -fsSL -o "$archive" \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" \
        || { rm -rf "$tmp"; die "Could not download uv."; }
    actual="$(sha256_of "$archive")"
    if [ "$actual" != "$expected" ]; then
        rm -rf "$tmp"
        die "uv download failed the hash check (got $actual). Nothing was installed."
    fi
    note "Hash matches."
    tar -xzf "$archive" -C "$tmp"
    mkdir -p "$TOOLS_DIR"
    install -m 755 "$tmp/uv-$target/uv" "$TOOLS_DIR/uv"
    rm -rf "$tmp"
    UV="$TOOLS_DIR/uv"
}

uv_install() { "$UV" pip install --python "$PY" "$@"; }

[ "$UPDATE" -eq 1 ] || check_qt_libs
find_uv
note "Using uv at $UV"

# 3. Python environment --------------------------------------------------------------
if [ -x "$PY" ] && "$PY" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>/dev/null; then
    say "Reusing the existing environment in .venv"
else
    rm -rf "$VENV"
    say "Creating a Python $PYTHON_VERSION environment in .venv (uv downloads Python if needed)"
    "$UV" venv --python "$PYTHON_VERSION" "$VENV"
fi

say "Installing the app's packages (first time is a few hundred MB)"
# --require-hashes: every package must match the SHA-256 in the file or nothing installs.
uv_install --require-hashes -r "$APP_DIR/requirements.txt"
uv_install --require-hashes --no-deps -r "$APP_DIR/requirements-nodeps.txt"

has_module() { "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" 2>/dev/null; }

if [ "$UPDATE" -eq 1 ]; then
    # Keep the optional parts the way they are: update the ones that are installed, skip the rest.
    if has_module demucs; then STEMS="yes"; else STEMS="no"; fi
    if has_module yt_dlp; then YOUTUBE="yes"; else YOUTUBE="no"; fi
fi

# 4. Optional stem separation ----------------------------------------------------------
if [ "$STEMS" = "ask" ]; then
    say "Optional: stem separation"
    note "Splits a song into vocals, bass, drums, and everything else before analyzing."
    note "Lyrics and notes come out much cleaner on full songs. Adds about 1 GB of downloads."
    if confirm "Install stem separation?" n; then STEMS="yes"; else STEMS="no"; fi
fi

if [ "$STEMS" = "yes" ]; then
    if [ "$GPU" = "cpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
        if confirm "An NVIDIA GPU was found. Use it for stem separation? (about 3 GB more)" n; then GPU="cuda"; fi
    fi
    if [ "$GPU" = "cuda" ]; then
        say "Installing PyTorch $TORCH_VERSION with NVIDIA GPU support"
        uv_install "torch==$TORCH_VERSION"
    else
        say "Installing PyTorch $TORCH_VERSION (CPU build, from the official PyTorch index)"
        uv_install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
    fi
    say "Installing Demucs"
    uv_install --require-hashes --no-deps -r "$APP_DIR/requirements-stems.txt"
fi

# 5. Optional YouTube download -----------------------------------------------------------
if [ "$YOUTUBE" = "ask" ]; then
    say "Optional: YouTube download"
    note "Paste a YouTube link in the app and get the audio as MP3, M4A, Opus, FLAC or WAV."
    note "Uses yt-dlp and Deno (Deno runs the small checks YouTube needs). About 50 MB."
    if confirm "Install YouTube download?" y; then YOUTUBE="yes"; else YOUTUBE="no"; fi
fi

if [ "$YOUTUBE" = "yes" ]; then
    say "Installing yt-dlp and Deno"
    uv_install --require-hashes -r "$APP_DIR/requirements-youtube.txt"
elif [ "$YOUTUBE" = "update" ]; then
    say "Updating yt-dlp and Deno to the newest versions"
    note "YouTube changes often, and an older yt-dlp can stop working. This gets the newest"
    note "release from PyPI over HTTPS. Unlike everything else here it is not hash-pinned,"
    note "because the pinned hashes are for one fixed version."
    uv_install --upgrade-package yt-dlp --upgrade-package yt-dlp-ejs --upgrade-package deno "yt-dlp[default]" deno
fi

# 6. Check that everything imports --------------------------------------------------------
say "Checking the install"
PYTHONPATH="$APP_DIR" "$PY" -m audioscribe --check || die "The check failed. See the list above for what is missing."

# 7. Launcher, icon, and menu entry ---------------------------------------------------------
say "Adding the launcher and menu entry"
mkdir -p "$BIN_DIR" "$(dirname "$DESKTOP_FILE")" "$(dirname "$ICON_FILE")"
QT_QPA_PLATFORM=offscreen PYTHONPATH="$APP_DIR" "$PY" -m audioscribe --write-icons "$APP_DIR/audioscribe/assets" >/dev/null
cp "$APP_DIR/audioscribe/assets/$APP_ID.png" "$ICON_FILE"

# printf %q quotes the paths so unusual folder names can't break the script.
{
    printf '#!/usr/bin/env bash\n'
    printf '# Launcher for %s, created by install.sh\n' "$APP_NAME"
    printf 'export PYTHONPATH=%q"${PYTHONPATH:+:$PYTHONPATH}"\n' "$APP_DIR"
    printf 'exec %q -m audioscribe "$@"\n' "$PY"
} > "$LAUNCHER"
chmod +x "$LAUNCHER"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
GenericName=Audio transcriber
Comment=Transcribe the words and find the notes in any audio or video file
Exec="$LAUNCHER" %f
Icon=$APP_ID
Terminal=false
Categories=AudioVideo;Audio;Music;
Keywords=transcribe;lyrics;notes;midi;piano roll;whisper;
MimeType=audio/mpeg;audio/flac;audio/x-flac;audio/wav;audio/x-wav;audio/ogg;audio/opus;audio/mp4;audio/aac;audio/x-m4a;video/mp4;video/x-matroska;video/webm;video/quicktime;
StartupWMClass=$APP_ID
EOF

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA_HOME/applications" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t "$DATA_HOME/icons/hicolor" >/dev/null 2>&1 || true

say "Done"
note "Start it from your app menu, or run: $APP_ID"
note "You can also pass a file: $APP_ID song.mp3"
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not in your PATH, so the '$APP_ID' command won't be found until you add it. The menu entry works either way." ;;
esac
note "Whisper and Demucs models download the first time you use them."
