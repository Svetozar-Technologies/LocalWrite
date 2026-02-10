"""
Workers - Background thread workers for non-blocking UI operations.
Handles LLM operations in separate threads.
"""

from PyQt6.QtCore import QThread, pyqtSignal, QObject

from .paraphraser import Paraphraser


class WorkerSignals(QObject):
    """Signals for worker communication."""
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(bool, str)  # success, message
    error = pyqtSignal(str)  # error message
    status = pyqtSignal(str)  # status message


class ModelLoaderWorker(QThread):
    """Worker for loading LLM model in background."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, paraphraser: Paraphraser, model_path: str):
        super().__init__()
        self.paraphraser = paraphraser
        self.model_path = model_path

    def run(self):
        try:
            success = self.paraphraser.load_model(
                self.model_path,
                progress_callback=self._on_progress
            )

            if success:
                self.finished.emit(True, "Model loaded successfully")
            else:
                self.finished.emit(False, "Failed to load model")

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def _on_progress(self, message: str):
        self.progress.emit(message)
