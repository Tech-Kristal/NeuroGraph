"""Regression tests for the background task runner (replaces the QRunnable workers)."""

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEventLoop
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from neurograph.tasks import TaskRunner


def _wait_until(application, predicate, timeout_ms=5000):
    elapsed = 0
    while elapsed < timeout_ms and not predicate():
        application.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        QTest.qWait(10)
        elapsed += 10
    assert predicate()


def test_results_and_errors_are_delivered_on_the_gui_thread():
    application = QApplication.instance() or QApplication([])
    runner = TaskRunner()
    main_thread = threading.get_ident()
    seen = []
    runner.submit(
        lambda: 21 * 2,
        lambda value: seen.append(("ok", value, threading.get_ident() == main_thread)),
        lambda message: seen.append(("error", message)),
    )
    runner.submit(
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        lambda value: seen.append(("ok", value)),
        lambda message: seen.append(("error", message, threading.get_ident() == main_thread)),
    )
    _wait_until(application, lambda: not runner.pending)
    assert ("ok", 42, True) in seen
    assert ("error", "boom", True) in seen


def test_many_short_tasks_do_not_lose_or_crash_signals():
    application = QApplication.instance() or QApplication([])
    runner = TaskRunner()
    results = []
    for index in range(300):
        runner.submit(lambda index=index: index, results.append, lambda message: results.append(message))
    _wait_until(application, lambda: not runner.pending, timeout_ms=15000)
    assert sorted(results) == list(range(300))


def test_a_failing_callback_does_not_break_later_tasks():
    application = QApplication.instance() or QApplication([])
    runner = TaskRunner()
    done = []

    def bad(_value):
        raise ValueError("callback bug")

    runner.submit(lambda: 1, bad)
    runner.submit(lambda: 2, done.append)
    _wait_until(application, lambda: not runner.pending)
    assert done == [2]
