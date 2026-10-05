# Audio Scribe

Open any audio or video file and Audio Scribe will:

- Write out the words with a timestamp for every word (speech or song lyrics)
- Find the notes and show them on a piano roll timeline, like the Key Editor in Cubase
- Find the drum hits (kick, snare, hi-hat) when you split a song into stems
- Estimate the key and tempo, and show which notes are used the most
- Optionally split a song into vocals, bass, drums, and other first, for cleaner results
- Let you show only one scale, mute or solo any part, and hear the result
- Let you drag a selection over any stretch of the song to see it in detail and loop it
- Let you fix, move, and add notes (after you press Edit), then export the changed notes
- Play the notes back on a piano, strings, guitars, accordion, drums, and more

It runs completely on your own computer. Nothing is uploaded anywhere. The only time it goes online is to download a model the first time you use it.

Works on Linux and Windows.

## Install

You need an internet connection for the install. You do not need Python installed, the installer sets up its own copy inside the app folder.

### Linux

1. Extract the zip somewhere you want to keep it, for example `~/Documents/Audio Scribe`
2. Open a terminal in that folder and run:

   ```bash
   ./install.sh
   ```

3. Answer the question about stem separation (see below)
4. Start it from your app menu, or run `audio-scribe` in a terminal

The installer adds a launcher at `~/.local/bin/audio-scribe` and a menu entry with an icon. If you don't already have uv (the tool that sets up Python and installs packages), it downloads a fixed version into a `.tools` folder here and checks its SHA-256 hash first. If Qt's X11 library (`xcb-util-cursor`) is missing it offers to install it with your package manager. It knows dnf, apt, pacman, and zypper.

### Windows

1. Extract the zip somewhere you want to keep it, for example `Documents\Audio Scribe`
2. Double-click `install.bat`
3. Answer the question about stem separation (see below)
4. Start it from the Start menu (or the desktop shortcut if you added one)

### Installer options

Both installers take the same options. On Windows add them after `install.bat`, for example `install.bat -WithStems`.

| Linux | Windows | What it does |
|---|---|---|
| `--with-stems` | `-WithStems` | Install stem separation without asking |
| `--no-stems` | `-NoStems` | Skip stem separation without asking |
| `--cuda` | `-Cuda` | Use the NVIDIA GPU build of PyTorch for stems |
| `--yes` | `-Yes` | Take the default answer for every question |
| `--uninstall` | `-Uninstall` | Remove the shortcuts and the `.venv` and `.tools` folders |

Running the installer again is safe. It reuses what is already there and only adds what is missing, so you can run it later with `--with-stems` to add stem separation.

### Should I install stem separation?

It adds about 1 GB of downloads. It is worth it if you mostly use full songs, because:

- Whisper only hears the vocals, so the lyrics come out much better
- Notes are found per instrument instead of all mixed together, and each one gets its own color
- You can play back each stem on its own

For speech, voice memos, or solo instruments you don't need it.

## Using it

1. **Open a file** with the button, Ctrl+O, or by dropping it on the window. Almost any format works: mp3, wav, flac, ogg, opus, m4a, aac, and video files like mp4, mkv, webm, and mov. You can play it right away.
2. **Pick your options** in the panel on the left.
3. **Press Analyze** (or Ctrl+Enter).

The first run of each Whisper model downloads it, so that run takes longer. After that it starts right away.

### Options

**Words**
- **Model:** bigger is more accurate but slower. Small is a good starting point. For songs, try Large turbo if Small misses words.
- **Language:** auto detect works well on speech. For songs and short clips, picking the language helps a lot.
- **Translate to English:** writes the English translation instead of the original language.
- **Skip silent parts:** faster, and stops Whisper from making up text in quiet parts. Turn it off if quiet singing gets skipped.

**Notes**
- **Sensitivity:** higher finds quieter notes but also more stray ones. Lower it if you see lots of short junk notes.
- **Shortest note:** anything shorter is ignored.

**Stems**
- **Split into stems first:** separates the song before analyzing.
- **Find notes in:** which stems to look for notes in. For Drums it finds drum hits (kick, snare, and hi-hat) instead of pitches. If you used an earlier version, your saved choice may have Drums off, so tick it once.

**Run on:** CPU or NVIDIA GPU. The GPU choice is greyed out if no NVIDIA GPU with CUDA is found. If the GPU fails for Whisper, it falls back to the CPU on its own.

### The piano roll

- The keyboard on the left lights up with the notes playing at the playhead.
- The lyrics lane above the notes shows each word where it is sung, and the current word turns amber while playing.
- Hover a note to see its name, start time, and length.
- Click a note to jump there. Click the ruler or empty space to move the playhead. Drag on the ruler to scrub.
- Click a key on the keyboard to highlight every time that note is played. Click it again or press Esc to clear.
- The grid lines follow the detected beat, with a stronger line every 4 beats.
- Drum hits show as short blocks on the rows for kick (C2), snare (D2), and hi-hat (F#2).

| Action | How |
|---|---|
| Play or pause | Space |
| Go to start | Home |
| Zoom time | Ctrl + scroll, or Ctrl + plus / minus |
| Zoom rows taller or shorter | Ctrl + Shift + scroll |
| Scroll sideways | Shift + scroll |
| Fit the whole file | Ctrl+0, or the Fit button |
| Open a file | Ctrl+O |
| Analyze | Ctrl+Enter or F5 |
| Zoom to the selected span, and back | Z |
| Loop the selected span on or off | L |
| Edit mode on or off | E |
| Clear the span or selection | Esc |
| Undo and redo (Edit mode) | Ctrl+Z and Ctrl+Y |

**Follow** keeps the playhead in view while playing. Turn it off to look around freely.

### Parts and sound

After an analysis, the **Parts and sound** tab at the top of the left panel has three sections.

**Parts.** Each part (Vocals, Bass, Other, Drums, or Full mix if you did not split into stems) has an **M** button and an **S** button.

- **M** (mute) hides the part's notes and silences it. Mute the vocals and you hear only the music. Mute the drums and you hear everything else. Mute Drums and Vocals and you hear only the bass and the rest.
- **S** (solo) shows and plays only the soloed parts. This also replaces the old stem menu in the bottom bar, since soloing a stem plays just that stem.
- **Sound** is the instrument that part's notes use when they are played back: Soft piano (the default), Grand piano, Electric piano, Violin, Strings (section), Cello, Acoustic guitar, Electric guitar, Bass guitar, Harp, Accordion, Organ, Flute, Clarinet, Trumpet, Choir, Warm pad, Music box, and Drum kit. Drums always use the drum kit. Your choices are remembered for next time. Picking one plays a short sample so you can hear it.

What you hide is hidden everywhere: the piano roll, the Notes tab, the Summary and Selection tabs, playback, and the exports.

**Show only.** Pick a scale to see only its notes, for example C major, A minor, or a pentatonic scale. Choose the key note and the scale, or click the little keys to pick any set of notes you like. **Show** can flip it to the notes outside the scale, which is a quick way to find wrong notes. **Use the detected key** fills it in from the key the app found. **From** and **to** limit the range of notes, and **Show every note** puts everything back. Drum hits ignore this filter because they have no pitch. Mute the drums to hide them.

**Sound.** The **Play** menu in the bottom bar picks what you hear:

- **Recording** plays the original audio, minus any part you muted.
- **Notes on instruments** plays the notes the app found, on the instruments you chose.
- **Recording and notes** plays both together. **Notes level** sets how loud the instruments are, and **Add room sound** adds a little reverb so they don't sound dry.

The first time you change what is muted or which instrument is used, the app builds the new sound in the background (a few seconds for a full song) while the old one keeps playing, then switches over at the same spot. Sounds it has already built come back instantly.

### Selecting a span, zooming, and looping

Drag across the piano roll to select a span of time. It is shaded, and its length shows at the top. Drag either edge to adjust it. The toolbar above the piano roll then lets you:

- **Zoom to span** (or press Z, or double-click inside the span) to fill the piano roll with just that stretch so you can see it in more detail. Press it again (it says **Zoom back**) to return.
- **Loop** (or press L) to play the span over and over. It can be turned on and off at any time, and the span stays selected when it is off. Playback starts at the start of the span.
- **Clear** (or press Esc) to remove the span.
- **Snap** (to the beat, half beat, or quarter beat) makes the span edges, and moved notes, land on the beat grid.

The **Selection** tab shows details for the span: its length in seconds, beats, and bars, the key and range in it, how many notes each part plays, which note is played most, a chart of how long each note sounds, a list of every note used, and a rough guess at the chords as they change. The **Notes** tab follows the span too, unless you uncheck **Only the selected span**.

### Editing notes

Notes cannot be changed until you press **Edit notes** (or E), so nothing moves by accident. The piano roll gets an amber border and a small EDIT badge so you can tell it is on.

| Action | How |
|---|---|
| Move a note in time or pitch | Drag it |
| Make it longer or shorter | Drag its right edge |
| Add a note | Double-click empty space. It goes into the part chosen in **Add to** |
| Delete notes | Select them, then press Delete |
| Select several | Drag a box on empty space, or Ctrl or Shift + click. Ctrl+A selects all |
| Nudge selected notes | Arrow keys (up and down are a semitone, left and right are a small step, or one snap step). Hold Shift for an octave or a bigger step |
| Change how loud | Select notes, then change the **%** box |
| Select a span while editing | Hold Shift and drag on empty space |
| Undo and redo | Ctrl+Z and Ctrl+Y, or the Undo and Redo buttons |
| Put everything back | **Revert all** |

Moving a note plays it, so you can hear where it lands. Edited and added notes are marked "Changed by hand" when you hover them. Turn Edit off again and the notes are locked.

Edits change the notes the app found, not the recording. To hear them, set **Play** to **Notes on instruments** or **Recording and notes**. Edits are kept in memory only, so export before you open another file. The app asks first if you try to close with unexported changes.

### Tabs

- **Transcript:** each line with its start time. Click a line to jump there. The current line highlights while playing. Copy text copies the whole thing.
- **Notes:** every note shown, with part, start, length, and strength. Click a column header to sort. Click a row to jump to that note.
- **Summary:** key, tempo, range, and counts, a chart of how much time each note gets (notes in the key are brighter, the root note is amber), and a list of every note used. Clicking a note in that list highlights it in the piano roll.
- **Selection:** the details for the selected span, described above.

### Export

The Export button at the top right has:

- **Transcript as text (.txt)**
- **Transcript as subtitles (.srt)** for video players and editors
- **Lyrics with timing (.lrc)** for music players that show synced lyrics
- **Notes as MIDI (.mid)** with one track per part and the detected tempo written in, so the notes line up with the bar grid in Cubase or any other DAW. Each track carries the General MIDI instrument that matches the Sound you picked (flute, cello, and so on), and the drums go on the drum channel.
- **Notes as spreadsheet (.csv)**
- **Stems as WAV files**
- **Recording or notes as WAV (what you hear)** saves exactly what the Play menu is playing right now, for example the song without the vocals, or just the notes on a piano.

What you export matches what you see. Muted parts are left out, the scale filter applies, and any notes you edited are the edited ones.

## How accurate is it?

**Words:** very good on clear speech. Songs are harder, since singing, effects, and loud music all get in the way. Stem separation plus picking the language plus a bigger model gives the best results.

**Notes:** very good on a single instrument or a clean voice. Full mixes are much harder for any program, Cubase's own audio to MIDI included. Expect some extra notes, especially octave copies of loud notes. Stem separation helps the most. After that, lower the sensitivity and raise the shortest note length.

**Drums:** hits are found by watching how fast the low, middle, and high parts of the sound jump up, which separates kicks, snares, and hi-hats well in clean drum stems. It has only been checked on generated drum audio so far, so on real recordings expect some missed ghost notes and the odd wrong hit, especially with cymbal crashes or heavy bleed. Stem separation helps a lot. Sensitivity applies to drums too.

**Instrument sounds** are made by the app itself from math (additive and physical modelling), not from recordings, so there is nothing extra to download. They are meant for checking notes and hearing a part on a different instrument, not for finished music. The piano is soft and mellow. Real sample libraries sound richer.

**Chords** in the Selection tab are a rough guess from the notes in each beat or two. Treat them as a starting point.

**Key and tempo** are estimates. The key is based on the notes found, so a song can show as its relative major or minor (C major and A minor share the same notes). The summary says how sure it is and names the runner up when it's close.

## Security

Everything the installers download is pinned to an exact version and checked against a SHA-256 hash, so a file that was swapped or tampered with after these lists were made gets refused.

**How the install is protected**

- **uv is pinned and hash-checked.** The installers don't pipe a script from the internet into a shell. They download uv 0.12.17 from its GitHub release and compare its SHA-256 with the hash written into the installer. If it doesn't match, they stop before running anything. If you already have uv installed (for example from your distro), that one is used instead.
- **Every Python package is pinned and hash-checked.** `requirements.txt` lists every package, including the ones pulled in indirectly, with an exact version and its hashes. The installers use `--require-hashes`, so any file that doesn't match won't install.
- **Nothing brand new.** The package lists only include releases from before Sept 21, 2026. Hijacked releases are usually caught within days, so a short wait avoids most of them.
- **Only what's needed.** Demucs is installed without `sphn` and `lameenc`, which are only used by its own file reading and command-line tool. Audio Scribe never calls either, and `sphn` publishes no author or source information to check against.
- **No code in model files.** Demucs is only loaded from its safetensors release, a format that can only hold data. Its fallback to older pickle files, which can run code when loaded, is turned off. Whisper models are weight files plus text settings. Basic Pitch's model is an ONNX file that ships inside its package.
- **sudo is only used for one optional system package** (xcb-cursor on Linux), and it asks first.

**What the app itself does**

- The only network traffic is downloading models from Hugging Face the first time you use them. Your audio and results never leave your computer. The optional usage reporting in the Hugging Face and ONNX Runtime libraries is turned off.
- It never runs other programs or shell commands.
- It only deletes its own temporary files in its cache folder.
- File names and error messages are always shown as plain text.

**What you're still trusting**

- The projects themselves: Qt, faster-whisper (SYSTRAN), Basic Pitch (Spotify), Demucs (Alexandre Defossez), PyTorch, NumPy, and the rest. Pinning protects against a release being swapped later, not against a problem that was already in the pinned version.
- On Linux, PyTorch comes from the official PyTorch index at a pinned version (2.14.0), but its hash isn't written into these files. The same goes for the optional NVIDIA builds on either system. The Windows CPU build is fully hash-checked.
- Model files from Hugging Face aren't pinned to a specific revision. Whisper models come from the Systran account (the faster-whisper team), except Large turbo, which faster-whisper itself gets from the mobiuslabsgmbh account. Demucs comes from the adefossez account.
- Opening a media file runs it through FFmpeg, the same as any media player, so a file made to attack an FFmpeg bug is a risk with any player. Only open files from sources you trust, and update the pins now and then.

To move to newer versions later, regenerate the lists from the `.in` files with `uv pip compile --universal --python-version 3.12 --generate-hashes`.

## Where things are stored

| What | Linux | Windows |
|---|---|---|
| The app and its Python | the folder you extracted, in `.venv` | same |
| Downloaded models | `~/.cache/huggingface` | `%USERPROFILE%\.cache\huggingface` |
| Temporary audio while open | `~/.cache/audio-scribe/work` | `%LOCALAPPDATA%\audio-scribe\work` |
| Log file | `~/.cache/audio-scribe/audio-scribe.log` | `%LOCALAPPDATA%\audio-scribe\audio-scribe.log` |
| Settings | `~/.config/AudioScribe` | Registry, `HKCU\Software\AudioScribe` |

Temporary audio (including the sounds built for playback) is deleted when you close the app or analyze another file. Your instrument choices are saved in the settings.

Rough Whisper model sizes: Small is about 500 MB, Large turbo about 1.6 GB, and Large about 3 GB.

## Uninstall

Run `./install.sh --uninstall` on Linux, or `install.bat -Uninstall` on Windows, then delete the folder. To free the model space too, delete the `huggingface` folder listed above, but check first that no other program uses it.

## Troubleshooting

**Check that everything is installed.** From the app folder:

```bash
# Linux
.venv/bin/python -m audioscribe --check
```

```bat
:: Windows
.venv\Scripts\python.exe -m audioscribe --check
```

It lists every part and says what is missing.

**The window doesn't open on Linux under X11.** Install `xcb-util-cursor` (Fedora, Arch) or `libxcb-cursor0` (Debian, Ubuntu).

**"Could not download a model".** The first use of each model needs internet. Check your connection or proxy and try again.

**It's slow.** Use a smaller Whisper model, turn off stem splitting, or try a shorter clip first. On the CPU, a 4 minute song with the Small model and no stems usually takes a minute or two. Stem splitting adds a few minutes.

**Something else went wrong.** The error box has a Details button with the full error, and the log file (see the table above) has more.

## What's inside

```
audio-scribe/
  install.sh                 Linux installer
  install.bat, install.ps1   Windows installer
  requirements.in            top-level packages (source for requirements.txt)
  requirements.txt           every package, pinned with SHA-256 hashes
  requirements-nodeps.txt    Basic Pitch, pinned with its hash
  requirements-stems.in/.txt optional stem separation, pinned with hashes
  requirements-torch-windows.txt  PyTorch CPU build for Windows, pinned with hashes
  audioscribe/
    app.py                   startup, theme, --check
    window.py                main window
    piano_roll.py            the piano roll timeline, span, loop, and editing
    widgets.py               parts list, scale filter, summary and selection tabs
    engine.py                the analysis steps
    drums.py                 finding drum hits
    audio.py                 reading audio and video files
    music.py                 note names, scales, chords, key estimation
    synth.py                 the instrument sounds
    mixer.py                 builds the audio that plays (recording, notes, or both)
    exporters.py             txt, srt, lrc, MIDI, CSV
    icon.py                  the app icon
```

Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for the words, [Basic Pitch](https://github.com/spotify/basic-pitch) for the notes, [Demucs](https://pypi.org/project/demucs/) for stems, [librosa](https://librosa.org) for tempo, and [PySide6](https://doc.qt.io/qtforpython-6/) for the interface.

You can run it without the installer too: create a Python 3.12 environment, then `pip install --require-hashes -r requirements.txt`, `pip install --require-hashes --no-deps -r requirements-nodeps.txt`, and from this folder run `python -m audioscribe`.
