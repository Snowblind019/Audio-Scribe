"""The Sheet music window: preview, save as PDF, print, or save as MusicXML.

Verovio engraves the MusicXML from notation.py into SVG pages, in a separate process
(engrave.py). Qt's SVG renderer only understands a simpler kind of SVG, so engrave.svg_for_qt()
rewrites Verovio's output a little (no nested <svg>, plain <text> elements) before drawing it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QMarginsF, QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPageSize, QPainter, QPdfWriter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
                               QWidget)

from . import engrave, notation
from .i18n import tr, tr_n

PAPER = [("A4", QPageSize.A4, 2100, 2970), ("Letter", QPageSize.Letter, 2159, 2794)]


def verovio_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("verovio") is not None


def render_pages(xml: str, paper: tuple = PAPER[0], scale: int = 40) -> list[str]:
    """Engrave MusicXML into SVG pages (already made Qt friendly). Verovio runs in its own
    process, see engrave.py for why."""
    return engrave.render_pages(xml, paper[2], paper[3], scale)


def paint_page(painter: QPainter, svg: str, target: QRectF) -> None:
    from PySide6.QtSvg import QSvgRenderer

    r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    painter.fillRect(target, QColor("white"))
    r.render(painter, target)


# Engraving waits for the engraving process on this one background thread, so the window
# stays responsive and only one engraving runs at a time.
_engraver = None


def _engrave_executor():
    global _engraver
    if _engraver is None:
        from concurrent.futures import ThreadPoolExecutor
        _engraver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engrave")
    return _engraver


class RenderThread(QObject):
    """One engraving job on the engraving thread. Signals arrive in the window's thread."""
    done = Signal(object, str)     # pages, error
    finished = Signal()

    def __init__(self, xml: str, paper, parent=None):
        super().__init__(parent)
        self.xml, self.paper = xml, paper
        self._future = None

    def start(self) -> None:
        self._future = _engrave_executor().submit(self._run)

    def wait(self, timeout: float = 30.0) -> None:
        if self._future is not None:
            try:
                self._future.result(timeout)
            except Exception:  # noqa: BLE001  (the error was already reported through done)
                pass

    def _run(self) -> None:
        try:
            result = (render_pages(self.xml, self.paper), "")
        except Exception as exc:  # shown in the window, never fatal
            result = ([], str(exc))
        try:
            self.done.emit(*result)
            self.finished.emit()
        except RuntimeError:       # the window was closed meanwhile
            pass


class PageView(QWidget):
    """The pages stacked vertically, white on the dark background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pages: list[QPixmap] = []
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_pages(self, svgs: list[str], width: int) -> None:
        self.pages = []
        for svg in svgs:
            from PySide6.QtSvg import QSvgRenderer
            r = QSvgRenderer(QByteArray(svg.encode("utf-8")))
            size = r.defaultSize()
            h = int(width * size.height() / max(1, size.width()))
            img = QImage(width, h, QImage.Format_ARGB32_Premultiplied)
            img.fill(QColor("white"))
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            p.setRenderHint(QPainter.TextAntialiasing)
            r.render(p, QRectF(0, 0, width, h))
            p.end()
            self.pages.append(QPixmap.fromImage(img))
        total = sum(pm.height() + 16 for pm in self.pages) + 16
        self.setMinimumSize(width + 32, max(200, total))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        y = 16
        for pm in self.pages:
            x = (self.width() - pm.width()) // 2
            p.fillRect(x + 3, y + 3, pm.width(), pm.height(), QColor(0, 0, 0, 90))
            p.drawPixmap(x, y, pm)
            y += pm.height() + 16


class SheetDialog(QDialog):
    """Options on the left, the engraved pages on the right."""

    def __init__(self, parent, make_xml, part_names: list[str], default_title: str, save_dir: str):
        super().__init__(parent)
        self.setWindowTitle(tr("Sheet music"))
        self.resize(1180, 820)
        self.make_xml = make_xml            # callable(SheetOptions, parts) -> MusicXML
        self.save_dir = save_dir
        self.pages: list[str] = []
        self.thread: RenderThread | None = None
        self._dirty = False

        lay = QHBoxLayout(self)
        side = QWidget()
        side.setFixedWidth(280)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        title = QLabel(tr("Parts"))
        title.setObjectName("SectionTitle")
        sl.addWidget(title)
        self.part_list = QListWidget()
        self.part_list.setMaximumHeight(130)
        for name in part_names:
            item = QListWidgetItem(tr(name))
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name != "Drums" else Qt.Unchecked)
            self.part_list.addItem(item)
        sl.addWidget(self.part_list)

        self.division = QComboBox()
        self.division.addItem(tr("Smallest note: eighth"), 2)
        self.division.addItem(tr("Smallest note: sixteenth"), 4)
        self.division.setCurrentIndex(0)
        self.division.setToolTip(tr("Eighths give simpler, easier to read music. Sixteenths keep more detail."))
        self.grand = QCheckBox(tr("Two staves for wide parts"))
        self.grand.setChecked(True)
        self.lyrics = QCheckBox(tr("Lyrics under the vocal line"))
        self.lyrics.setChecked(True)
        self.names = QCheckBox(tr("Note names under the notes"))
        self.chords = QCheckBox(tr("Chord names above"))
        self.chords.setChecked(True)
        self.paper = QComboBox()
        for name, *_rest in PAPER:
            self.paper.addItem(name)
        self.paper.setProperty("i18n_skip_items", True)
        for w in (self.division, self.grand, self.lyrics, self.names, self.chords, self.paper):
            sl.addWidget(w)
        hint = QLabel(tr("Notes found in a recording are not as tidy as written music. Clean up notes "
                         "(Edit mode) first, especially lining notes up with the beat grid, for the best result."))
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        sl.addWidget(hint)
        self.status = QLabel("")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        sl.addWidget(self.status)
        sl.addStretch(1)
        self.pdf_btn = QPushButton(tr("Save as PDF..."))
        self.pdf_btn.setObjectName("Primary")
        self.print_btn = QPushButton(tr("Print..."))
        self.xml_btn = QPushButton(tr("Save as MusicXML..."))
        self.xml_btn.setToolTip(tr("Opens in MuseScore, Dorico, Finale, Sibelius and most notation programs"))
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.reject)
        for b in (self.pdf_btn, self.print_btn, self.xml_btn, close):
            sl.addWidget(b)
        lay.addWidget(side)

        self.view = PageView()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.view)
        scroll.setStyleSheet("QScrollArea { background: #1C2026; border: none; }")
        lay.addWidget(scroll, 1)
        self.scroll = scroll

        self.title_text = default_title
        self.pdf_btn.clicked.connect(self.save_pdf)
        self.print_btn.clicked.connect(self.print_pages)
        self.xml_btn.clicked.connect(self.save_xml)
        for w in (self.grand, self.lyrics, self.names, self.chords):
            w.toggled.connect(self.refresh)
        for w in (self.division, self.paper):
            w.currentIndexChanged.connect(self.refresh)
        self.part_list.itemChanged.connect(self.refresh)
        self.refresh()

    def chosen_parts(self) -> list[str]:
        out = []
        for i in range(self.part_list.count()):
            item = self.part_list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        return out

    def options(self, display: bool) -> notation.SheetOptions:
        return notation.SheetOptions(title=self.title_text, division=self.division.currentData(),
                                     lyrics=self.lyrics.isChecked(), note_names=self.names.isChecked(),
                                     chords=self.chords.isChecked(), grand_staff=self.grand.isChecked(),
                                     display=display)

    def xml(self, display: bool) -> str:
        return self.make_xml(self.options(display), self.chosen_parts())

    def refresh(self) -> None:
        if self.thread is not None:
            self._dirty = True
            return
        if not self.chosen_parts():
            self.pages = []
            self.view.set_pages([], 800)
            self.status.setText(tr("Pick at least one part."))
            return
        self.status.setText(tr("Engraving..."))
        for b in (self.pdf_btn, self.print_btn):
            b.setEnabled(False)
        self.thread = RenderThread(self.xml(display=True), PAPER[self.paper.currentIndex()])
        self.thread.done.connect(self._on_rendered)
        self.thread.finished.connect(self._on_thread_done)
        self.thread.start()

    def _on_rendered(self, pages: list, error: str) -> None:
        self.pages = pages
        if error:
            self.status.setText(tr("Could not engrave the music: {error}").format(error=error))
        else:
            self.status.setText(tr_n(len(pages), "{n} page.", "{n} pages."))
        width = max(400, min(900, self.scroll.viewport().width() - 40))
        self.view.set_pages(pages, width)
        for b in (self.pdf_btn, self.print_btn):
            b.setEnabled(bool(pages))

    def _on_thread_done(self) -> None:
        self.thread.deleteLater()
        self.thread = None
        if self._dirty:
            self._dirty = False
            self.refresh()

    def _ask_path(self, title: str, ext: str, filt: str) -> Path | None:
        start = str(Path(self.save_dir) / f"{self.title_text}.{ext}")
        path, _ = QFileDialog.getSaveFileName(self, title, start, filt)
        if not path:
            return None
        path = Path(path)
        if path.suffix.lower() != f".{ext}":
            path = path.with_name(path.name + f".{ext}")
        return path

    def _paint_all(self, painter, page_rect_fn, new_page) -> None:
        for i, svg in enumerate(self.pages):
            if i:
                new_page()
            paint_page(painter, svg, page_rect_fn())

    def write_pdf(self, path: Path) -> None:
        paper = PAPER[self.paper.currentIndex()]
        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(paper[1]))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        writer.setResolution(300)
        writer.setTitle(self.title_text)
        writer.setCreator("Audio Scribe")
        painter = QPainter(writer)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = lambda: QRectF(0, 0, writer.width(), writer.height())  # noqa: E731
        self._paint_all(painter, rect, writer.newPage)
        painter.end()

    def save_pdf(self) -> None:
        path = self._ask_path(tr("Save sheet music as PDF"), "pdf", tr("PDF files (*.pdf)"))
        if path:
            try:
                self.write_pdf(path)
                self.status.setText(tr("Saved {path}").format(path=str(path)))
            except Exception as exc:
                QMessageBox.warning(self, tr("Sheet music"), tr("Could not save the file.") + "\n" + str(exc))

    def print_pages(self) -> None:
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter

        printer = QPrinter(QPrinter.HighResolution)
        printer.setPageSize(QPageSize(PAPER[self.paper.currentIndex()][1]))
        printer.setDocName(self.title_text)
        dialog = QPrintDialog(printer, self)
        if dialog.exec() != QDialog.Accepted:
            return
        painter = QPainter(printer)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = lambda: QRectF(printer.pageLayout().paintRectPixels(printer.resolution()))  # noqa: E731
        rect_fn = lambda: QRectF(0, 0, rect().width(), rect().height())  # noqa: E731
        self._paint_all(painter, rect_fn, printer.newPage)
        painter.end()
        self.status.setText(tr("Sent to the printer."))

    def save_xml(self) -> None:
        path = self._ask_path(tr("Save sheet music as MusicXML"), "musicxml", tr("MusicXML files (*.musicxml *.xml)"))
        if path:
            try:
                path.write_text(self.xml(display=False), encoding="utf-8")
                self.status.setText(tr("Saved {path}").format(path=str(path)))
            except Exception as exc:
                QMessageBox.warning(self, tr("Sheet music"), tr("Could not save the file.") + "\n" + str(exc))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.thread is not None:
            self.thread.wait()
        super().closeEvent(event)

    def reject(self) -> None:
        if self.thread is not None:
            self.thread.wait()
        super().reject()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(1180, 820)
