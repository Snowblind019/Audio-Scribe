"""Clean up found notes in one go: remove stray short or quiet notes, octave ghosts, notes
outside the key, join notes that were broken in two, and line notes up with the beat grid.

The functions work on lists of Note and are used by the Clean up dialog (Edit mode), which
applies them as one undo step.
"""

from __future__ import annotations

import bisect
import copy
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QLabel,
                               QSpinBox, QVBoxLayout)

from .i18n import tr, tr_n
from .music import key_name


@dataclass
class Plan:
    short_ms: int | None = 80          # remove notes shorter than this
    quiet_pct: int | None = None       # remove notes weaker than this
    ghosts: bool = True                # remove quieter copies an octave away
    join_ms: int | None = 60           # join same notes with a gap shorter than this
    key_classes: frozenset | None = None   # remove notes outside these pitch classes
    quantize_div: int = 0              # 0 off, 1 beat, 2 half beat, 4 quarter beat
    quantize_strength: float = 1.0


def short_notes(notes, min_seconds: float) -> set:
    return {n for n in notes if n.end - n.start < min_seconds}


def quiet_notes(notes, min_velocity: float) -> set:
    return {n for n in notes if n.velocity < min_velocity}


def octave_ghosts(notes) -> set:
    """A note is a ghost when a louder note one or two octaves away starts at nearly the same
    time and covers most of it. Note finders often add these "harmonic" copies."""
    ordered = sorted(notes, key=lambda n: n.start)
    starts = [n.start for n in ordered]
    gone = set()
    for n in ordered:
        lo = bisect.bisect_left(starts, n.start - 0.06)
        hi = bisect.bisect_right(starts, n.start + 0.06)
        for m in ordered[lo:hi]:
            if m is n or abs(m.pitch - n.pitch) not in (12, 19, 24):
                continue
            overlap = min(n.end, m.end) - max(n.start, m.start)
            if overlap >= 0.6 * (n.end - n.start) and m.velocity >= n.velocity * 1.1:
                gone.add(n)
                break
    return gone


def outside(notes, classes: frozenset) -> set:
    return {n for n in notes if n.pitch % 12 not in classes}


def join_broken(notes, max_gap: float) -> tuple[set, int]:
    """Joins a note with the next one of the same pitch when the gap between them is tiny.
    Changes the first note in place. Returns (notes to remove, how many joins)."""
    by_pitch: dict[int, list] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    gone = set()
    for same in by_pitch.values():
        same.sort(key=lambda n: n.start)
        cur = same[0]
        for nxt in same[1:]:
            if nxt.start - cur.end <= max_gap:
                cur.end = max(cur.end, nxt.end)
                cur.velocity = max(cur.velocity, nxt.velocity)
                cur.edited = True
                gone.add(nxt)
            else:
                cur = nxt
    return gone, len(gone)


def grid_lines(beats: list[float], div: int, duration: float) -> list[float]:
    if len(beats) < 2:
        step = 0.5 / max(1, div)
        return [i * step for i in range(int(duration / step) + 2)]
    lines = []
    for a, b in zip(beats[:-1], beats[1:]):
        for k in range(div):
            lines.append(a + (b - a) * k / div)
    lines.append(beats[-1])
    period = beats[-1] - beats[-2]
    t = beats[-1]
    while t < duration + period:
        t += period / div
        lines.append(t)
    first = beats[0]
    while first > 0:
        first -= (beats[1] - beats[0]) / div
        lines.insert(0, first)
    return lines


def quantize(notes, beats: list[float], div: int, strength: float, duration: float) -> int:
    """Moves note starts and ends toward the nearest grid line. Returns how many changed."""
    lines = grid_lines(beats, div, duration)
    if len(lines) < 2:
        return 0

    def nearest(t: float) -> float:
        i = bisect.bisect_left(lines, t)
        cands = lines[max(0, i - 1):i + 1]
        return min(cands, key=lambda x: abs(x - t)) if cands else t

    changed = 0
    for n in notes:
        s, e = nearest(n.start), nearest(n.end)
        if e <= s:
            i = bisect.bisect_right(lines, s)
            e = lines[i] if i < len(lines) else s + 0.1
        new_s = n.start + (s - n.start) * strength
        new_e = n.end + (e - n.end) * strength
        new_s = max(0.0, new_s)
        new_e = max(new_s + 0.03, min(duration, new_e))
        if abs(new_s - n.start) > 1e-4 or abs(new_e - n.end) > 1e-4:
            n.start, n.end = new_s, new_e
            n.edited = True
            changed += 1
    return changed


def apply(plan: Plan, groups: list[list], beats: list[float], duration: float) -> tuple[list[set], dict]:
    """Runs the plan on each group of notes (one group per part, drums excluded by the caller
    from pitch rules). Changes notes in place and returns the notes to remove per group,
    plus counts of what happened."""
    counts = {"short": 0, "quiet": 0, "ghosts": 0, "outside": 0, "joined": 0, "moved": 0}
    removals = []
    for notes, pitched in groups:
        gone = set()
        if plan.join_ms:
            joined_away, joins = join_broken(notes, plan.join_ms / 1000.0)
            gone |= joined_away
            counts["joined"] += joins
        rest = [n for n in notes if n not in gone]
        if plan.short_ms:
            s = short_notes(rest, plan.short_ms / 1000.0)
            counts["short"] += len(s)
            gone |= s
        if plan.quiet_pct:
            q = quiet_notes([n for n in rest if n not in gone], plan.quiet_pct / 100.0)
            counts["quiet"] += len(q)
            gone |= q
        if pitched and plan.ghosts:
            g = octave_ghosts([n for n in rest if n not in gone])
            counts["ghosts"] += len(g)
            gone |= g
        if pitched and plan.key_classes is not None:
            o = outside([n for n in rest if n not in gone], plan.key_classes)
            counts["outside"] += len(o)
            gone |= o
        if plan.quantize_div:
            counts["moved"] += quantize([n for n in rest if n not in gone], beats, plan.quantize_div,
                                        plan.quantize_strength, duration)
        removals.append(gone)
    return removals, counts


def done_text(counts: dict) -> str:
    parts = []
    removed = counts["short"] + counts["quiet"] + counts["ghosts"] + counts["outside"]
    if removed:
        parts.append(tr_n(removed, "removed {n} note", "removed {n} notes"))
    if counts["joined"]:
        parts.append(tr_n(counts["joined"], "joined {n} broken note", "joined {n} broken notes"))
    if counts["moved"]:
        parts.append(tr_n(counts["moved"], "moved {n} note onto the grid", "moved {n} notes onto the grid"))
    if not parts:
        return tr("Nothing needed cleaning up.")
    return tr("Cleaned up: {what}. Ctrl+Z undoes it.").format(what=", ".join(parts))


def describe(counts: dict) -> str:
    parts = []
    removed = counts["short"] + counts["quiet"] + counts["ghosts"] + counts["outside"]
    if removed:
        parts.append(tr_n(removed, "removes {n} note", "removes {n} notes"))
    if counts["joined"]:
        parts.append(tr_n(counts["joined"], "joins {n} broken note", "joins {n} broken notes"))
    if counts["moved"]:
        parts.append(tr_n(counts["moved"], "moves {n} note onto the grid", "moves {n} notes onto the grid"))
    if not parts:
        return tr("Nothing to change with these settings.")
    return tr("This {what}.").format(what=", ".join(parts))


class CleanupDialog(QDialog):
    """Pick what to clean up. Shows how many notes each choice would change before applying."""

    def __init__(self, parent, groups_for_scope, scopes: list[tuple[str, str]], key, beats, duration):
        super().__init__(parent)
        self.setWindowTitle(tr("Clean up notes"))
        self.groups_for_scope = groups_for_scope   # callable(scope) -> [(notes, pitched)]
        self.key = key
        self.beats = beats
        self.duration = duration
        lay = QVBoxLayout(self)
        intro = QLabel(tr("Tidy up the found notes in one go. You can undo it afterwards with Ctrl+Z."))
        intro.setWordWrap(True)
        intro.setObjectName("Hint")
        lay.addWidget(intro)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.short_chk = QCheckBox(tr("Remove notes shorter than"))
        self.short_ms = QSpinBox()
        self.short_ms.setRange(10, 1000)
        self.short_ms.setValue(80)
        self.short_ms.setSuffix(" ms")
        self.quiet_chk = QCheckBox(tr("Remove notes quieter than"))
        self.quiet_pct = QSpinBox()
        self.quiet_pct.setRange(1, 90)
        self.quiet_pct.setValue(20)
        self.quiet_pct.setSuffix("%")
        self.ghost_chk = QCheckBox(tr("Remove octave ghosts (a quieter copy an octave away)"))
        self.join_chk = QCheckBox(tr("Join broken notes with a gap shorter than"))
        self.join_ms = QSpinBox()
        self.join_ms.setRange(5, 500)
        self.join_ms.setValue(60)
        self.join_ms.setSuffix(" ms")
        self.key_chk = QCheckBox(tr("Remove notes outside the key ({key})").format(
            key=key_name(*key) if key else tr("no key found")))
        self.key_chk.setEnabled(key is not None)
        self.quant_chk = QCheckBox(tr("Line notes up with the beat grid"))
        self.quant_div = QComboBox()
        for label, div in ((tr("beat"), 1), (tr("1/2 beat"), 2), (tr("1/4 beat"), 4)):
            self.quant_div.addItem(label, div)
        self.quant_div.setCurrentIndex(2)
        self.quant_strength = QDoubleSpinBox()
        self.quant_strength.setRange(10, 100)
        self.quant_strength.setValue(100)
        self.quant_strength.setDecimals(0)
        self.quant_strength.setSuffix("%")
        self.quant_strength.setToolTip(tr("100% moves notes all the way onto the grid, 50% halfway"))
        self.scope = QComboBox()
        for key_, label in scopes:
            self.scope.addItem(label, key_)

        self.short_chk.setChecked(True)
        self.ghost_chk.setChecked(True)
        self.join_chk.setChecked(True)
        grid.addWidget(self.short_chk, 0, 0)
        grid.addWidget(self.short_ms, 0, 1)
        grid.addWidget(self.quiet_chk, 1, 0)
        grid.addWidget(self.quiet_pct, 1, 1)
        grid.addWidget(self.ghost_chk, 2, 0, 1, 2)
        grid.addWidget(self.join_chk, 3, 0)
        grid.addWidget(self.join_ms, 3, 1)
        grid.addWidget(self.key_chk, 4, 0, 1, 2)
        grid.addWidget(self.quant_chk, 5, 0)
        grid.addWidget(self.quant_div, 5, 1)
        grid.addWidget(self.quant_strength, 5, 2)
        apply_label = QLabel(tr("Apply to"))
        grid.addWidget(apply_label, 6, 0, Qt.AlignRight)
        grid.addWidget(self.scope, 6, 1, 1, 2)
        lay.addLayout(grid)
        drums = QLabel(tr("Drum hits are only affected by the length, strength and grid options."))
        drums.setObjectName("Hint")
        drums.setWordWrap(True)
        lay.addWidget(drums)
        self.preview = QLabel("")
        self.preview.setObjectName("FieldName")
        self.preview.setWordWrap(True)
        lay.addWidget(self.preview)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Clean up"))
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._update_preview)
        for w in (self.short_chk, self.quiet_chk, self.ghost_chk, self.join_chk, self.key_chk, self.quant_chk):
            w.toggled.connect(self._timer.start)
        for w in (self.short_ms, self.quiet_pct, self.join_ms, self.quant_strength):
            w.valueChanged.connect(self._timer.start)
        for w in (self.quant_div, self.scope):
            w.currentIndexChanged.connect(self._timer.start)
        self._update_preview()

    def plan(self) -> Plan:
        classes = None
        if self.key_chk.isChecked() and self.key:
            from .music import MAJOR_STEPS, MINOR_STEPS
            tonic, mode = self.key
            steps = MAJOR_STEPS if mode == "major" else MINOR_STEPS
            classes = frozenset((tonic + s) % 12 for s in steps)
        return Plan(
            short_ms=self.short_ms.value() if self.short_chk.isChecked() else None,
            quiet_pct=self.quiet_pct.value() if self.quiet_chk.isChecked() else None,
            ghosts=self.ghost_chk.isChecked(),
            join_ms=self.join_ms.value() if self.join_chk.isChecked() else None,
            key_classes=classes,
            quantize_div=self.quant_div.currentData() if self.quant_chk.isChecked() else 0,
            quantize_strength=self.quant_strength.value() / 100.0,
        )

    def scope_key(self) -> str:
        return self.scope.currentData()

    def _update_preview(self) -> None:
        groups = [(copy.deepcopy(notes), pitched) for notes, pitched in self.groups_for_scope(self.scope_key())]
        _removals, counts = apply(self.plan(), groups, self.beats, self.duration)
        self.preview.setText(describe(counts))
