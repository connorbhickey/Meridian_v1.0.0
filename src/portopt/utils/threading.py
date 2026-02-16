"""QThread worker wrappers for running long tasks off the GUI thread."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class WorkerSignals(QObject):
    """Signals emitted by a Worker runnable."""
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    progress = Signal(str)


class Worker(QRunnable):
    """Generic worker that runs a callable on the thread pool."""

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


def run_in_thread(
    fn: Callable,
    *args,
    on_result: Callable | None = None,
    on_error: Callable | None = None,
    on_finished: Callable | None = None,
    on_progress: Callable | None = None,
    **kwargs,
) -> Worker:
    """Run a function in a background thread.

    Args:
        fn: The function to run
        on_result: Callback for the result (called on main thread)
        on_error: Callback for errors (called on main thread)
        on_finished: Callback when done (called on main thread)

    Returns:
        The Worker instance (keep a reference to prevent GC)
    """
    worker = Worker(fn, *args, **kwargs)
    if on_result:
        worker.signals.result.connect(on_result)
    if on_error:
        worker.signals.error.connect(on_error)
    if on_finished:
        worker.signals.finished.connect(on_finished)
    if on_progress:
        worker.signals.progress.connect(on_progress)
    QThreadPool.globalInstance().start(worker)
    return worker
