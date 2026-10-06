"""The interface language: English or Romanian, and it can be changed while the app runs.

How it works:
  * Text that is built while the app runs (status messages, counts, labels with numbers)
    goes through tr(), which looks the English text up in the Romanian table.
  * Fixed text on buttons, labels, menus, tabs and tooltips is written in English when the
    window is built. retranslate() walks every widget and swaps each known text for the
    other language, so switching needs no restart.

The Romanian table lives in i18n_ro.py. Text that is missing from it simply stays English.
"""

from __future__ import annotations

from typing import Callable

LANGUAGES = [("en", "English"), ("ro", "Română")]

_lang = "en"
_reverse: dict[str, str] | None = None


def _table() -> dict[str, str]:
    from .i18n_ro import RO
    return RO


def _reverse_table() -> dict[str, str]:
    global _reverse
    if _reverse is None:
        _reverse = {ro: en for en, ro in _table().items()}
    return _reverse


def language() -> str:
    return _lang


def set_language(code: str) -> None:
    global _lang
    _lang = code if code in dict(LANGUAGES) else "en"


def tr(text: str) -> str:
    """The text in the current language (English text is the key)."""
    if _lang == "en" or not text:
        return text
    return _table().get(text, text)


def tr_n(n: int, one: str, many: str) -> str:
    """A count with the right word form, for example "1 note" or "5 notes".

    Both texts contain {n}. Romanian also needs "de" before the noun from 20 up
    ("20 de note", but "19 note" and "101 note"), which the Romanian text marks with {de}."""
    text = tr(one if n == 1 else many)
    de = "de " if _lang == "ro" and n != 1 and (n % 100 >= 20 or (n and n % 100 == 0)) else ""
    return text.replace("{de}", de).replace("{n}", str(n))


def english(text: str) -> str:
    """The English original of a text in either language."""
    return _reverse_table().get(text, text)


def _convert(text: str, target: str) -> str:
    """One text into the target language. Text already in that language stays as it is,
    so converting twice is harmless."""
    if not text:
        return text
    if target == "en":
        return _reverse_table().get(text, text)
    if text in _reverse_table():   # already Romanian
        return text
    return _table().get(text, text)


def retranslate(root, target: str) -> None:
    """Swap every known fixed text under the widget `root` into the `target` language.

    A widget with the property i18n_skip set keeps its texts (used for things like
    file names). A combo box with i18n_skip_items keeps its item texts."""
    from PySide6.QtWidgets import (QAbstractButton, QComboBox, QDialog, QGroupBox, QLabel, QLineEdit,
                                   QMainWindow, QMenu, QTabWidget, QTreeWidget, QWidget)

    def conv(text: str) -> str:
        return _convert(text, target)

    widgets = [root] + root.findChildren(QWidget)
    seen_actions = set()
    for w in widgets:
        if w.property("i18n_skip"):
            continue
        tip = w.toolTip()
        if tip:
            w.setToolTip(conv(tip))
        if isinstance(w, QLabel):
            w.setText(conv(w.text()))
        elif isinstance(w, QAbstractButton):
            w.setText(conv(w.text()))
        elif isinstance(w, QGroupBox):
            w.setTitle(conv(w.title()))
        elif isinstance(w, QLineEdit):
            if w.placeholderText():
                w.setPlaceholderText(conv(w.placeholderText()))
        elif isinstance(w, QComboBox):
            if not w.property("i18n_skip_items"):
                for i in range(w.count()):
                    w.setItemText(i, conv(w.itemText(i)))
                # the width was worked out for the old texts, so let it follow the new ones
                if w.sizeAdjustPolicy() == QComboBox.AdjustToContentsOnFirstShow:
                    w.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        elif isinstance(w, QTabWidget):
            for i in range(w.count()):
                w.setTabText(i, conv(w.tabText(i)))
                if w.tabToolTip(i):
                    w.setTabToolTip(i, conv(w.tabToolTip(i)))
        elif isinstance(w, QTreeWidget):
            header = w.headerItem()
            for c in range(w.columnCount()):
                header.setText(c, conv(header.text(c)))
        if isinstance(w, QMenu):
            w.setTitle(conv(w.title()))
        if isinstance(w, (QMainWindow, QDialog)) and w.windowTitle():
            w.setWindowTitle(conv(w.windowTitle()))
        for action in w.actions():
            if id(action) in seen_actions:
                continue
            seen_actions.add(id(action))
            if action.text():
                action.setText(conv(action.text()))
            if action.toolTip() and action.toolTip() != action.text():
                action.setToolTip(conv(action.toolTip()))


# Things to call after the language changes, so text built in code gets rebuilt.
_listeners: list[Callable[[], None]] = []


def on_change(callback: Callable[[], None]) -> None:
    _listeners.append(callback)


def notify() -> None:
    for callback in list(_listeners):
        try:
            callback()
        except RuntimeError:  # the widget behind it is gone
            _listeners.remove(callback)
