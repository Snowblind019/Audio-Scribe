# Audio Scribe

https://github.com/user-attachments/assets/bb6e6690-f1bb-4221-bea3-9f0397f0036e

Open any audio or video file and Audio Scribe will:

- Write out the words with a timestamp for every word (speech or song lyrics)
- Find the notes and show them on a piano roll timeline, like the Key Editor in Cubase
- Find the chords, the key, and the tempo, and show the chords in a lane above the notes
- Find the drum hits (kick, snare, hi-hat) when you split a song into stems
- Optionally split a song into vocals, bass, drums, and other first, for cleaner results
- Let you show only one scale, mute or solo any part, and hear the result
- Let you drag a selection over any stretch of the song to see it in detail and loop it
- Let you fix, move, and add notes (after you press Edit), then export the changed notes
- Play the notes back on a piano, strings, guitars, accordion, drums, and more

New in 1.2:

- **Chords tab** with ready-made chords and progressions you can drag onto the piano roll, voicings, voice leading, patterns, next-chord ideas, substitutes, key changes, a circle of fifths, and a sketch pad
- **Download from YouTube** as MP3, M4A, Opus, FLAC, or WAV
- **Do Re Mi note names** (fixed Do, with Si or Ti, or movable Do) everywhere in the app
- **Romanian or English interface**, switchable any time, and **lyrics translation** between Romanian and English, offline
- **Sheet music** you can save as PDF, print, or open in MuseScore (MusicXML)
- **Projects:** save everything and open it again later without analyzing again
- **Slow down** without changing the pitch, a **click track**, **tap tempo**, and a beat grid you can fix by hand
- **Cleanup tools** for stray notes, **record from your microphone**, **drag MIDI** straight into Reaper or any DAW, and **lyrics with chords** export

It runs on your own computer. Your audio is never uploaded anywhere. It only goes online to download a model the first time you use it, and to download from YouTube when you ask it to.

Works on Linux and Windows.

## Install

You need an internet connection for the install. You do not need Python installed, the installer sets up its own copy inside the app folder.

### Linux

1. Extract the zip somewhere you want to keep it, for example `~/Documents/Audio Scribe`
2. Open a terminal in that folder and run:

   ```bash
   ./install.sh
   ```

3. Answer the questions about stem separation and YouTube download (see below)
4. Start it from your app menu, or run `audio-scribe` in a terminal

The installer adds a launcher at `~/.local/bin/audio-scribe` and a menu entry with an icon. If you don't already have uv (the tool that sets up Python and installs packages), it downloads a fixed version into a `.tools` folder here and checks its SHA-256 hash first. If Qt's X11 library (`xcb-util-cursor`) is missing it offers to install it with your package manager. It knows dnf, apt, pacman, and zypper.

### Windows

1. Extract the zip somewhere you want to keep it, for example `Documents\Audio Scribe`
2. Double-click `install.bat`
3. Answer the questions about stem separation and YouTube download (see below)
4. Start it from the Start menu (or the desktop shortcut if you added one)

### Updating from an older version

Extract the new zip over the old folder (or into a new folder) and run the installer again. Your settings are kept.

### Installer options

Both installers take the same options. On Windows add them after `install.bat`, for example `install.bat -WithStems`.

| Linux | Windows | What it does |
|---|---|---|
| `--with-stems` | `-WithStems` | Install stem separation without asking |
| `--no-stems` | `-NoStems` | Skip stem separation without asking |
| `--cuda` | `-Cuda` | Use the NVIDIA GPU build of PyTorch for stems |
| `--with-youtube` | `-WithYouTube` | Install YouTube download without asking |
| `--no-youtube` | `-NoYouTube` | Skip YouTube download without asking |
| `--update-youtube` | `-UpdateYouTube` | Get the newest yt-dlp and Deno, for when YouTube downloads stop working |
| `--yes` | `-Yes` | Take the default answer for every question |
| `--uninstall` | `-Uninstall` | Remove the shortcuts and the `.venv` and `.tools` folders |

Running the installer again is safe. It reuses what is already there and only adds what is missing, so you can run it later with `--with-stems` to add stem separation.

### Should I install stem separation?

It adds about 1 GB of downloads. It is worth it if you mostly use full songs, because:

- Whisper only hears the vocals, so the lyrics come out much better
- Notes are found per instrument instead of all mixed together, and each one gets its own color
- You can play back each stem on its own

For speech, voice memos, or solo instruments you don't need it.

### Should I install YouTube download?

It adds about 50 MB: [yt-dlp](https://github.com/yt-dlp/yt-dlp), the downloader, and [Deno](https://deno.com), which runs the small JavaScript checks YouTube asks for before it hands out audio. Say yes if you want to paste YouTube links into the app. You can add it later with `--with-youtube`.

YouTube changes things now and then, and an older yt-dlp can stop working. If downloads start failing, run the installer again with `--update-youtube` (Windows: `install.bat -UpdateYouTube`).

## Using it

1. **Open a file** with the button, Ctrl+O, or by dropping it on the window. Almost any format works: mp3, wav, flac, ogg, opus, m4a, aac, and video files like mp4, mkv, webm, and mov. You can play it right away. You can also **download from YouTube**, **record from your microphone**, or open a saved **project** (see below).
2. **Pick your options** in the panel on the left.
3. **Press Analyze** (or Ctrl+Enter).

The first run of each Whisper model downloads it, so that run takes longer. After that it starts right away.

### Options

**Words**
- **Model:** bigger is more accurate but slower. Small is a good starting point. For songs, try Large turbo if Small misses words.
- **Language:** auto detect works well on speech. For songs and short clips, picking the language helps a lot.
- **Translate to English:** has Whisper write English instead of the original language. To translate into Romanian, or to keep both, use **Translate** in the Transcript tab instead (see below).
- **Skip silent parts:** faster, and stops Whisper from making up text in quiet parts. Turn it off if quiet singing gets skipped.

**Notes**
- **Find notes:** turn off if you only want the words.
- **Sensitivity:** higher finds quieter notes but also more stray ones. Lower it if you see lots of short junk notes.
- **Shortest note:** anything shorter is ignored.
- **Find chords:** listens for the chords in the recording, beat by beat, and puts them in the chord lane.

**Stems**
- **Split into stems first:** separates the song before analyzing.
- **Find notes in:** which stems to look for notes in. For Drums it finds drum hits (kick, snare, and hi-hat) instead of pitches.

**Run on:** CPU or NVIDIA GPU. The GPU choice is greyed out if no NVIDIA GPU with CUDA is found. If the GPU fails for Whisper, it falls back to the CPU on its own.

### Language and note names

At the top right:

- **Notes** picks how note names are written everywhere in the app (piano roll, keyboard, tabs, chords, exports, and sheet music):
  - **C D E** (letters)
  - **Do Re Mi** (fixed Do, with Si for the 7th note, as taught in Romania, Italy, France, and Spain)
  - **Do Re Mi (Ti)** (the same with Ti)
  - **Movable Do**, where Do is always the key note, so the syllables show each note's job in the key. In A minor, A is Do and C is Me. Notes outside the scale get their own syllables (Ra, Me, Fi, Le, Te).
- **English / Română** switches the whole interface between English and Romanian right away, with no restart. Your choice is remembered.

Flats are used where they belong, so F major shows Bb (or Sib) rather than A#.

### The piano roll

- The keyboard on the left lights up with the notes playing at the playhead.
- The lyrics lane above the notes shows each word where it is sung, and the current word turns amber while playing.
- The chord lane under it shows the chords. Click a chord to jump there and see it in the Chords tab.
- Hover a note to see its name, start time, and length.
- Click a note to jump there. Click the ruler or empty space to move the playhead. Drag on the ruler to scrub.
- Click a key on the keyboard to highlight every time that note is played. Click it again or press Esc to clear.
- The grid lines follow the beat, with a stronger line at the start of every bar.
- Drum hits show as short blocks on the rows for kick (C2), snare (D2), and hi-hat (F#2).

| Action | How |
|---|---|
| Play or pause | Space |
| Go to start | Home |
| Zoom time | Ctrl + scroll, or Ctrl + plus / minus |
| Zoom rows taller or shorter | Ctrl + Shift + scroll |
| Scroll sideways | Shift + scroll |
| Fit the whole file | Ctrl+0, or the Fit button |
| Open a file or project | Ctrl+O |
| Save the project | Ctrl+S |
| Analyze | Ctrl+Enter or F5 |
| Zoom to the selected span, and back | Z |
| Loop the selected span on or off | L |
| Edit mode on or off | E |
| Tap tempo | T |
| Clear the span or selection | Esc |
| Undo and redo (Edit mode) | Ctrl+Z and Ctrl+Y |

**Follow** keeps the playhead in view while playing. Turn it off to look around freely.

**Speed** in the bottom bar plays slower or faster (50% to 150%) without changing the pitch, so you can learn a part by ear. It works for the recording and the notes.

### Parts and sound

After an analysis, the **Parts and sound** tab at the top of the left panel has these sections.

**Parts.** Each part (Vocals, Bass, Other, Drums, or Full mix if you did not split into stems) has an **M** button and an **S** button.

- **M** (mute) hides the part's notes and silences it. Mute the vocals and you hear only the music.
- **S** (solo) shows and plays only the soloed parts.
- **Sound** is the instrument that part's notes use when they are played back: Soft piano (the default), Bright piano, Electric piano, Violin, Strings (section), Cello, Acoustic guitar, Electric guitar, Bass guitar, Harp, Accordion, Organ, Flute, Clarinet, Trumpet, Choir (ooh), Warm pad, Music box, and Drum kit. Drums always use the drum kit. Your choices are remembered for next time.

What you hide is hidden everywhere: the piano roll, the Notes tab, the Summary and Selection tabs, playback, and the exports.

**Show only.** Pick a scale to see only its notes, for example C major, A minor, or a pentatonic scale. Choose the key note and the scale, or click the little keys to pick any set of notes you like. **Show** can flip it to the notes outside the scale, which is a quick way to find wrong notes. **Use the detected key** fills it in from the key the app found. **From** and **to** limit the range of notes, and **Show every note** puts everything back.

**Tempo and beats.** If the beat grid is off, fix it here, and snapping, chords, sheet music, and MIDI all follow:

- **Tempo** in BPM, **First beat** (nudge it until the grid lines sit on the beats), and **Beats per bar** (2/4 up to 7/4).
- **Tap tempo** (or press T) along with the music. Three or more taps set the tempo. While the song plays, the grid also lines up with your taps.
- **Use the found beat** puts back the beat the app found.

**Sound.** The **Play** menu in the bottom bar picks what you hear:

- **Recording** plays the original audio, minus any part you muted.
- **Notes on instruments** plays the notes the app found, on the instruments you chose.
- **Recording and notes** plays both together. **Notes level** sets how loud the instruments are, and **Add room sound** adds a little reverb.
- **Click track** adds a metronome on the beat grid, louder on the first beat of each bar. **Click level** sets its volume.

The first time you change what is muted or which instrument is used, the app builds the new sound in the background while the old one keeps playing, then switches over at the same spot.

### Selecting a span, zooming, and looping

Drag across the piano roll to select a span of time. It is shaded, and its length shows at the top. Drag either edge to adjust it. The toolbar above the piano roll then lets you:

- **Zoom to span** (or press Z, or double-click inside the span) to fill the piano roll with just that stretch. Press it again (it says **Zoom back**) to return.
- **Loop** (or press L) to play the span over and over.
- **Clear** (or press Esc) to remove the span.
- **Snap** (to the beat, half beat, or quarter beat) makes the span edges, moved notes, and dropped chords land on the beat grid.

The **Selection** tab shows details for the span: its length in seconds, beats, and bars, the key and range in it, how many notes each part plays, which note is played most, and the chords in it.

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

**Clean up...** (next to Undo and Redo in Edit mode) tidies up the notes in one step. Pick what to clean (all notes shown, the selected span, or the selected notes) and what to do:

- Remove notes shorter than a length you choose
- Remove notes quieter than a level you choose
- Remove octave ghosts (a quieter copy of a note an octave away, a common mistake in found notes)
- Join broken notes that have a short gap in the middle
- Remove notes outside the key
- Line notes up with the beat grid

It tells you how many notes each step will change before you press OK, and the whole cleanup is one step to undo.

Edits change the notes the app found, not the recording. To hear them, set **Play** to **Notes on instruments** or **Recording and notes**. Save a project (Ctrl+S) to keep your edits. The app asks first if you try to close with unsaved changes.

### Chords

The **Chords** tab at the bottom is a chord workshop, inspired by tools like Scaler.

**On the left** is the chord palette for a key:

- **Key:** pick the key note and scale (major, the three minors, and the modes: Dorian, Phrygian, Lydian, Mixolydian, Locrian). **Song key** sets it to the key the app found.
- **Chords:** Triads, Sevenths, Ninths, Sus2, Sus4, Add9, or Sixths. Each button shows the chord name and its number in the key (I, ii, iii, and so on).
- **Voicing:** Close, 1st, 2nd, and 3rd inversion, Drop 2, Drop 3, Open (spread), or Shell (root, 3rd, 7th), plus the octave.
- **Voice leading:** each chord moves to the nearest notes of the one before, so changes sound smooth. **Bass note** adds the root an octave below.
- **Pattern:** how the chord is played: Block chord, pulses, arpeggios up, down, or both, broken chord, Alberti bass, strum, bass and chord, waltz, or bass lines (root notes, root and fifth, or walking).
- **Length:** 1 beat up to 4 bars. **Hear with:** the sound of the part, or any instrument.

Click a chord to hear it. To put chords into the song, turn on **Edit notes**, then **drag a chord onto the piano roll**. A ghost shows where it will land, and it snaps to the grid. **Insert at playhead** does the same at the playhead. Chords go into the part chosen in **Add to**.

**On the right** are these pages:

- **Song:** the chords found in the song, with their numbers in the key. **Chord lane** shows the chords from the recording, chords worked out from the notes, or nothing. Click a row to jump there.
- **Progressions:** 44 ready-made progressions in nine styles (Pop, Rock, Jazz, Blues, Worship and gospel, Folk and country, Cinematic, Classical, Latin), filtered by mood. They play in the key you picked. **Play**, **Insert at playhead**, drag the row in, or send it **To sketch**.
- **Next:** chords that sound good after the picked one, with a short reason for each (for example "Builds tension").
- **Substitutes:** chords that can stand in for the picked one, such as the relative minor or a tritone substitution.
- **Outside key:** chords borrowed from outside the key that still fit: borrowed chords from the parallel key, secondary dominants, chromatic mediants, and more.
- **Key change:** pick a new key and get chord paths into it, five ways: pivot chord, two five into the new key, modal interchange, chromatic mediant, and neo-Riemannian moves.
- **Circle:** the circle of fifths. Click a key to switch the palette to it.
- **Sketch:** seven lines to collect chord ideas on. Pick a line, then use **Add to sketch** under the palette, or **To sketch** in Progressions. Each line can be played, dragged onto the piano roll, shortened, or cleared, and it is saved with the project.

Chord names follow your note name choice, so in Do Re Mi mode a G7 shows as Sol7.

### Recording from the microphone

**Record from microphone...** in the File section opens a small recorder. Pick the microphone, press **Record**, sing, hum, whistle, or play, and press **Stop**. The take opens right away and its notes are found, like humming a melody into a MIDI keyboard. Tick **Also transcribe the words** to get the lyrics too. Recordings are saved as WAV files in `Music/Audio Scribe` (you can pick another folder).

### Downloading from YouTube

**Download from YouTube...** in the File section opens a box where you paste a link (youtube.com, youtu.be, or music.youtube.com). Pick the format (MP3, M4A, Opus, FLAC, WAV, or the original file with no conversion) and the quality, and press **Download**. Files go to `Music/Audio Scribe` unless you pick another folder, and can open in the app and be analyzed right away.

Only one video is downloaded at a time (never a whole playlist), up to 4 hours long. Only download videos you have the right to keep, such as your own uploads or videos the owner lets you download.

### Translating the lyrics

In the **Transcript** tab, press **Translate** and pick **into Romanian** or **into English**. It works from any of about 50 languages Whisper can recognize, and runs on your own computer with Meta's NLLB-200 translation model (the distilled 600M version). The model downloads the first time (about 640 MB) and is then reused.

**Show** picks what the Transcript tab shows: the **Original**, the **Translation**, or **Both** side by side (in copied text and exports, each line is followed by its translation). Copy text and the text, subtitle, and LRC exports follow the same choice, so you can save Romanian subtitles for an English song. Lyrics with chords and sheet music always use the original words, because the chords line up with them.

Machine translation of lyrics is a good starting point but not poetry, especially for slang and wordplay. The NLLB model is licensed for non-commercial use (CC-BY-NC 4.0), so it is fine for your own use but not for selling translations.

### Projects

**Save project...** (Ctrl+S) saves everything in one `.ascribe` file: the audio (and the stems), the words and any translation, the notes with your edits, the chords, your beat grid fixes, the sketch pad, and the original notes the app found (so Revert all still works). Open it again with Ctrl+O or by dropping it on the window, and everything is back with nothing to analyze again.

A project is a zip file with FLAC audio inside, about half the size of the WAV files. When opening one, the app checks every part of it and refuses anything unexpected, so a damaged or tampered file can't write anywhere or run anything.

### Drag MIDI into your DAW

**Drag MIDI** at the top right: hold the mouse button on it and drag it into Reaper (or Cubase, Ableton, FL Studio, a folder, and so on). It drops a MIDI file of what you see, with one track per part, the tempo, and the bar grid. If a span is selected, only that stretch goes, starting at the beginning of the file. In Reaper, drop it on a track at the spot where it should start.

### Sheet music

**Export > Sheet music** opens the sheet music window. Pick the parts and whether you want:

- **Smallest note:** eighth (simpler, easier to read) or sixteenth (more detail)
- **Two staves for wide parts** (piano-style treble and bass)
- **Lyrics under the vocal line**
- **Note names under the notes** (in your note name style, so Do Re Mi works here too)
- **Chord names above**

The preview updates as you change things. **Save as PDF**, **Print**, or **Save as MusicXML** to open and edit it in MuseScore, Dorico, Sibelius, or Finale. Drums are written on a drum staff.

Notes found in a recording are never as tidy as written music, so the result is a good draft, not a finished score. Running **Clean up** first (especially lining notes up with the beat grid) and checking the tempo and first beat gives much cleaner sheets. The engraving is done by [Verovio](https://www.verovio.org), on your computer.

### Tabs

- **Transcript:** each line with its start time. Click a line to jump there. The current line highlights while playing. Translate and Show are described above. Copy text copies the whole thing.
- **Notes:** every note shown, with part, start, length, and strength. Click a column header to sort. Click a row to jump to that note.
- **Summary:** key, tempo, range, and counts, a chart of how much time each note gets, and a list of every note used.
- **Selection:** the details for the selected span.
- **Chords:** described above.

### Export

The Export button at the top right has:

- **Transcript as text (.txt)**
- **Transcript as subtitles (.srt)** for video players and editors
- **Lyrics with timing (.lrc)** for music players that show synced lyrics
- **Lyrics with chords (.txt)**, chord names above the words they fall on, ready to print for a band
- **Lyrics with chords (ChordPro)** for chord sheet apps like OnSong and SongbookPro
- **Notes as MIDI (.mid)** with one track per part, the tempo, and the bar grid, so the notes line up in Reaper, Cubase, or any other DAW. Each track carries the General MIDI instrument that matches the Sound you picked, and the drums go on the drum channel.
- **Notes as spreadsheet (.csv)**
- **Sheet music (PDF, print, MusicXML)**
- **Stems as WAV files**
- **Recording or notes as WAV (what you hear)** saves exactly what the Play menu is playing right now
- **Project** (everything, to open again later)

What you export matches what you see. Muted parts are left out, the scale filter applies, and any notes you edited are the edited ones.

## How accurate is it?

**Words:** very good on clear speech. Songs are harder, since singing, effects, and loud music all get in the way. Stem separation plus picking the language plus a bigger model gives the best results.

**Notes:** very good on a single instrument or a clean voice. Full mixes are much harder for any program, Cubase's own audio to MIDI included. Expect some extra notes, especially octave copies of loud notes. Stem separation helps the most. After that, lower the sensitivity, raise the shortest note length, or use Clean up.

**Chords** from the recording are found beat by beat from the harmony you hear, and work best on songs with clear chords (guitar, piano, pads). They are checked on generated audio so far. Fast jazz changes, heavy distortion, or very busy mixes will confuse it. Chords from the notes are only as good as the notes.

**Drums:** hits are found by watching how fast the low, middle, and high parts of the sound jump up, which separates kicks, snares, and hi-hats well in clean drum stems. It has only been checked on generated drum audio so far, so on real recordings expect some missed ghost notes and the odd wrong hit.

**Instrument sounds** are made by the app itself from math (additive and physical modelling), not from recordings, so there is nothing extra to download. They are meant for checking notes and hearing a part on a different instrument, not for finished music.

**Key and tempo** are estimates. The key is based on the notes found, so a song can show as its relative major or minor (C major and A minor share the same notes). The summary says how sure it is and names the runner up when it's close. The tempo can be off by double or half. Fix it in Tempo and beats.

## Security

Everything the installers download is pinned to an exact version and checked against a SHA-256 hash, so a file that was swapped or tampered with after these lists were made gets refused.

**How the install is protected**

- **uv is pinned and hash-checked.** The installers don't pipe a script from the internet into a shell. They download uv 0.12.17 from its GitHub release and compare its SHA-256 with the hash written into the installer. If it doesn't match, they stop before running anything. If you already have uv installed (for example from your distro), that one is used instead.
- **Every Python package is pinned and hash-checked.** `requirements.txt` lists every package, including the ones pulled in indirectly, with an exact version and its hashes. The installers use `--require-hashes`, so any file that doesn't match won't install. The optional YouTube packages (`requirements-youtube.txt`) and stem packages are pinned and hash-checked the same way.
- **Nothing brand new.** The package lists only include releases from before Sept 21, 2026. Hijacked releases are usually caught within days, so a short wait avoids most of them.
- **Only what's needed.** Sheet music added one package (Verovio). Translation uses CTranslate2 and the tokenizer library that Whisper already needs, so it added none. YouTube download is optional. Demucs is installed without `sphn` and `lameenc`, which Audio Scribe never calls.
- **No code in model files.** Demucs is only loaded from its safetensors release, a format that can only hold data. Its fallback to older pickle files, which can run code when loaded, is turned off. Whisper and translation models are weight files plus text settings. Basic Pitch's model is an ONNX file that ships inside its package.
- **The translation model is pinned and hash-checked.** It is downloaded from one exact revision on Hugging Face, and every file is compared with the hash written into the app before it is used. A file that doesn't match is deleted and not used.
- **sudo is only used for one optional system package** (xcb-cursor on Linux), and it asks first.

**What the app itself does**

- The only network traffic is downloading models from Hugging Face the first time you use them, and YouTube downloads when you start one. Your audio and results never leave your computer. The optional usage reporting in the Hugging Face and ONNX Runtime libraries is turned off.
- YouTube links are checked before anything is fetched: only youtube.com, youtu.be, and music.youtube.com links are accepted, and the app always fetches the plain video address over https, one video at a time, no playlists, at most 4 hours and 2 GB. Files are saved under a safe file name in the folder you picked and never overwrite an existing file.
- The only other program it ever starts is Deno, and only during a YouTube download. yt-dlp uses it to run YouTube's JavaScript check inside Deno's sandbox, with no access to your files, the network, or other programs, and with the checking code coming from the installed (hash-checked) yt-dlp-ejs package rather than from the internet.
- Project files are checked before anything is read from them: only the expected files, with size limits, and every value checked. Nothing from a project is ever run, and file names inside it are never used as paths.
- It only deletes its own temporary files in its cache folder.
- File names and error messages are always shown as plain text.

**What you're still trusting**

- The projects themselves: Qt, faster-whisper (SYSTRAN), Basic Pitch (Spotify), Demucs (Alexandre Defossez), Verovio, yt-dlp, Deno, PyTorch, NumPy, and the rest. Pinning protects against a release being swapped later, not against a problem that was already in the pinned version.
- Deno 2.9.7 has one published advisory (CVE-2026-103473, Sept 30, 2026): on Windows, a Deno program that starts other programs through `node:child_process` with a shell can be tricked into running commands. It doesn't apply here, because yt-dlp runs Deno with no permissions at all, so the code inside it can't start any program. No fixed version was on PyPI when this was written. When one is, `--update-youtube` picks it up.
- `--update-youtube` gets the newest yt-dlp and Deno from PyPI over HTTPS without hash pins, because the pins are for one fixed version. That trade is there because an old yt-dlp eventually stops working with YouTube. It only updates yt-dlp, yt-dlp-ejs, and Deno.
- On Linux, PyTorch comes from the official PyTorch index at a pinned version (2.14.0), but its hash isn't written into these files. The same goes for the optional NVIDIA builds on either system. The Windows CPU build is fully hash-checked.
- Whisper and Demucs model files from Hugging Face aren't pinned to a specific revision. Whisper models come from the Systran account (the faster-whisper team), except Large turbo, which faster-whisper itself gets from the mobiuslabsgmbh account. Demucs comes from the adefossez account. The translation model is pinned, and is a conversion of Meta's NLLB-200 published by the JustFrederik account.
- Opening a media file runs it through FFmpeg, the same as any media player, so a file made to attack an FFmpeg bug is a risk with any player. Only open files from sources you trust, and update the pins now and then.

To move to newer versions later, regenerate the lists from the `.in` files with `uv pip compile --universal --python-version 3.12 --generate-hashes` (for the YouTube list, add `-c requirements.txt`).

## Where things are stored

| What | Linux | Windows |
|---|---|---|
| The app and its Python | the folder you extracted, in `.venv` | same |
| Downloaded models | `~/.cache/huggingface` | `%USERPROFILE%\.cache\huggingface` |
| Recordings and YouTube downloads | `~/Music/Audio Scribe` (you can change it) | `Music\Audio Scribe` |
| Temporary audio while open | `~/.cache/audio-scribe/work` | `%LOCALAPPDATA%\audio-scribe\work` |
| Log file | `~/.cache/audio-scribe/audio-scribe.log` | `%LOCALAPPDATA%\audio-scribe\audio-scribe.log` |
| Settings | `~/.config/AudioScribe` | Registry, `HKCU\Software\AudioScribe` |

Temporary audio (including the sounds built for playback) is deleted when you close the app or analyze another file. Your instrument, language, and note name choices are saved in the settings.

Rough model sizes: Whisper Small is about 500 MB, Large turbo about 1.6 GB, and Large about 3 GB. The translation model is about 640 MB.

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

**YouTube downloads fail with "YouTube refused the download".** YouTube changed something. Run the installer again with `--update-youtube` (Windows: `install.bat -UpdateYouTube`). Videos that need signing in, or are private or blocked in your country, can't be downloaded.

**The microphone isn't found.** Plug it in before opening the Record window. On Linux, check that it works in your sound settings first. On Windows, check that Settings > Privacy (& security) > Microphone lets desktop apps use it.

**It's slow.** Use a smaller Whisper model, turn off stem splitting, or try a shorter clip first. On the CPU, a 4 minute song with the Small model and no stems usually takes a minute or two. Stem splitting adds a few minutes. Translation of a whole song takes a few seconds once the model is downloaded.

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
  requirements-youtube.in/.txt  optional YouTube download, pinned with hashes
  requirements-torch-windows.txt  PyTorch CPU build for Windows, pinned with hashes
  audioscribe/
    app.py                   startup, theme, --check
    window.py                main window
    window_extras.py         projects, recording, YouTube, translation, beat grid, cleanup, MIDI drag
    piano_roll.py            the piano roll timeline, lanes, span, loop, editing, and chord drops
    widgets.py               parts list, scale filter, summary and selection tabs
    chords_view.py           the Chords tab
    theory.py                chords, keys, progressions, voicings, patterns, suggestions, key changes
    audiochords.py           finding chords in the recording
    grid.py                  the beat grid and tap tempo
    cleanup.py               the Clean up tools
    engine.py                the analysis steps
    drums.py                 finding drum hits
    audio.py                 reading audio and video files
    music.py                 note names (letters and Do Re Mi), scales, key estimation
    synth.py                 the instrument sounds
    mixer.py                 builds the audio that plays (recording, notes, click, or both)
    exporters.py             txt, srt, lrc, chord sheets, ChordPro, MIDI, CSV
    notation.py              turns notes into MusicXML for sheet music
    sheet_view.py            the sheet music window (preview, PDF, print)
    project.py               saving and opening .ascribe projects
    recorder.py              recording from the microphone
    youtube.py               YouTube download
    translate.py             lyrics translation
    i18n.py, i18n_ro.py      the interface languages and the Romanian text
    icon.py                  the app icon
```

Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for the words, [Basic Pitch](https://github.com/spotify/basic-pitch) for the notes, [Demucs](https://pypi.org/project/demucs/) for stems, [librosa](https://librosa.org) for tempo and chords, [Verovio](https://www.verovio.org) for sheet music, [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) with [CTranslate2](https://github.com/OpenNMT/CTranslate2) for translation, [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube, and [PySide6](https://doc.qt.io/qtforpython-6/) for the interface.

You can run it without the installer too: create a Python 3.12 environment, then `pip install --require-hashes -r requirements.txt`, `pip install --require-hashes --no-deps -r requirements-nodeps.txt`, optionally `pip install --require-hashes -r requirements-youtube.txt`, and from this folder run `python -m audioscribe`.
