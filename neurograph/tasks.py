"""Background task execution for the desktop application.

Why not ``QRunnable`` + a per-task ``QObject`` of signals? Qt auto-deletes the
runnable when it finishes, and the signals object it owns can be torn down while
the worker thread is still emitting. That produces errors such as
``'WorkerSignals' does not have a signal with the signature succeeded(...)``
followed by ``wrapped C/C++ object ... has been deleted``.

``TaskRunner`` avoids the problem: a single long-lived ``QObject`` owned by the
window carries the only signal, and worker threads are plain daemon threads that
never own any Qt object.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)

SuccessCallback = Callable[[Any], None]
ErrorCallback = Callable[[str], None]


class TaskRunner(QObject):
    """Run callables off the GUI thread and deliver results on the GUI thread.

    Create it, and call :meth:`submit`, only from the GUI thread.
    """

    _completed = pyqtSignal(int, object, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        #: Tasks whose callbacks have not run yet, keyed by task ID.
        self.pending: dict[int, tuple[SuccessCallback | None, ErrorCallback | None]] = {}
        self._next_id = 1
        self._closed = False
        self._completed.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)

    def submit(
        self,
        function: Callable[[], Any],
        on_success: SuccessCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> int:
        """Start ``function`` on a worker thread and return its task ID."""
        task_id = self._next_id
        self._next_id += 1
        self.pending[task_id] = (on_success, on_error)
        threading.Thread(
            target=self._execute,
            args=(task_id, function),
            name=f"neurograph-task-{task_id}",
            daemon=True,
        ).start()
        return task_id

    def close(self) -> None:
        """Stop delivering results (call when the window closes)."""
        self._closed = True
        self.pending.clear()

    def _execute(self, task_id: int, function: Callable[[], Any]) -> None:
        result: Any = None
        error: str | None = None
        try:
            result = function()
        except Exception as exc:
            logger.exception("Background task failed")
            error = str(exc) or type(exc).__name__
        if self._closed:
            return
        try:
            self._completed.emit(task_id, result, error)
        except RuntimeError:
            logger.debug("Task runner was deleted before task %s finished", task_id)

    @pyqtSlot(int, object, object)
    def _dispatch(self, task_id: int, result: Any, error: str | None) -> None:
        callbacks = self.pending.get(task_id)
        if callbacks is None:
            return
        on_success, on_error = callbacks
        try:
            if error is None:
                if on_success is not None:
                    on_success(result)
            elif on_error is not None:
                on_error(error)
            else:
                logger.error("Unhandled background task error: %s", error)
        except Exception:
            logger.exception("Background task callback failed")
        finally:
            # Removed only after the callback so tasks it starts are registered first.
            self.pending.pop(task_id, None)
