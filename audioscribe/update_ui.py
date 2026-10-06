"""The update check in the main window: the Updates section, the question when a new version is out,
and handing over to the update helper when the window closes. The checking itself is in updater.py.

MainWindow mixes this class in, so `self` here is the main window.
"""

from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QMessageBox, QPushButton

from . import APP_NAME, __version__, updater
from .i18n import tr

log = logging.getLogger(__name__)

AUTO_CHECK_EVERY = 20 * 3600        # seconds between automatic checks


class _UpdateJob(QThread):
    done = Signal(object, str)          # result, error text for the user
    progressed = Signal(float)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.done.emit(self.func(self.progressed.emit, lambda: self._stop), "")
        except updater.Stopped:
            self.done.emit(None, "")
        except updater.UpdateError as exc:
            self.done.emit(None, str(exc))
        except Exception:  # noqa: BLE001  (never let a check crash the app)
            log.error("Update job failed:\n%s", traceback.format_exc())
            self.done.emit(None, "Something went wrong while checking for updates. The log file has details.")


class UpdateMixin:
    def _build_update_section(self, layout) -> None:
        box, l = self._section("Updates")
        self.version_label = QLabel(tr("You have version {version}.").format(version=__version__))
        self.version_label.setObjectName("Hint")
        self.update_auto_chk = QCheckBox("Check for updates when Audio Scribe opens")
        self.update_auto_chk.setToolTip("Looks at most once a day. Only the version number and the signed list of "
                                        "files are fetched from GitHub. Nothing about you or your files is sent.")
        self.update_auto_chk.setChecked(self.settings.value("update/auto", True, type=bool))
        self.update_auto_chk.toggled.connect(lambda on: self.settings.setValue("update/auto", on))
        self.update_btn = QPushButton("Check for updates")
        self.update_btn.clicked.connect(lambda: self._check_for_updates(manual=True))
        l.addWidget(self.version_label)
        l.addWidget(self.update_auto_chk)
        l.addWidget(self.update_btn)
        layout.addWidget(box)
        self._update_job = None
        self._pending_update = None

    def _retranslate_update(self) -> None:
        self.version_label.setText(tr("You have version {version}.").format(version=__version__))

    def _start_update_checks(self) -> None:
        result = updater.take_result()
        if result:
            QTimer.singleShot(600, lambda: self._show_update_result(result))
        if (self.update_auto_chk.isChecked() and updater.public_key() is not None
                and updater.can_update() is None and updater.last_check_age() > AUTO_CHECK_EVERY):
            QTimer.singleShot(4000, lambda: self._check_for_updates(manual=False))

    def _show_update_result(self, result: dict) -> None:
        if result.get("ok"):
            self.status_text.setText(tr("Updated to Audio Scribe {version}.").format(version=result.get("version", "")))
        else:
            from .window import show_message
            show_message(self, QMessageBox.Warning, tr(str(result.get("message") or "The update didn't finish.")))

    # Checking ------------------------------------------------------------------------------------

    def _check_for_updates(self, manual: bool) -> None:
        if self._update_job is not None:
            return
        reason = updater.can_update()
        if reason and manual:
            from .window import show_message
            show_message(self, QMessageBox.Information, tr(reason))
            return
        if reason:
            return
        if manual:
            self.status_text.setText(tr("Checking for updates..."))
        self.update_btn.setEnabled(False)
        job = _UpdateJob(lambda _p, _s: updater.check(), self)
        job.done.connect(lambda info, err: self._update_on_checked(info, err, manual))
        job.finished.connect(job.deleteLater)
        self._update_job = job
        job.start()

    def _update_on_checked(self, info, error: str, manual: bool) -> None:
        self._update_job = None
        self.update_btn.setEnabled(True)
        if not manual:
            updater.mark_checked()
        if error:
            log.info("Update check: %s", error)
            if manual:
                from .window import show_message
                show_message(self, QMessageBox.Warning, tr(error))
            return
        if info is None:
            if manual:
                self.status_text.setText(tr("You have the newest version ({version}).").format(version=__version__))
            return
        self._offer_update(info)

    def _offer_update(self, info) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(APP_NAME)
        box.setTextFormat(Qt.PlainText)
        box.setText(tr("Audio Scribe {new} is available. You have {old}.").format(new=info.version, old=__version__))
        notes = info.notes.strip()
        more = tr("Update now? It downloads the new version, checks its signature, then closes, "
                  "updates itself and opens again.")
        box.setInformativeText((notes + "\n\n" if notes else "") + more)
        yes = box.addButton(tr("Update now"), QMessageBox.YesRole)
        box.addButton(tr("Not now"), QMessageBox.NoRole)
        box.setDefaultButton(yes)
        box.exec()
        if box.clickedButton() is yes:
            self._download_update(info)

    # Downloading and installing --------------------------------------------------------------------

    def _download_update(self, info) -> None:
        self.status_text.setText(tr("Downloading Audio Scribe {version}...").format(version=info.version))
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.show()
        self.update_btn.setEnabled(False)
        job = _UpdateJob(lambda progress, stop: updater.download(info, progress, stop), self)
        job.progressed.connect(lambda f: self.progress.setValue(int(f * 1000)))
        job.done.connect(lambda stage, err: self._update_on_downloaded(info, stage, err))
        job.finished.connect(job.deleteLater)
        self._update_job = job
        job.start()

    def _update_on_downloaded(self, info, stage, error: str) -> None:
        self._update_job = None
        self.update_btn.setEnabled(True)
        self.progress.hide()
        if error or stage is None:
            from .window import show_message
            if error:
                show_message(self, QMessageBox.Warning, tr(error))
            self.status_text.setText(tr("The update was not installed."))
            return
        self._pending_update = stage
        self.status_text.setText(tr("Closing to install the update..."))
        if not self.close():
            self._pending_update = None
            self.status_text.setText(tr("The update was not installed because the app stayed open. "
                                        "Check for updates again when you're ready."))

    def _apply_pending_update(self) -> None:
        """Called as the window closes. Starts the helper that installs the update and reopens the app."""
        stage, self._pending_update = self._pending_update, None
        if stage is None:
            return
        try:
            updater.start_install(stage, relaunch=True)
        except Exception:  # noqa: BLE001
            log.error("Could not start the update helper:\n%s", traceback.format_exc())

    def _stop_update_job(self) -> None:
        job = self._update_job
        if job is not None:
            job.stop()
            job.wait(5000)
