"""
Main Window - PyQt6 main application window for PDF Paraphraser.
Features drag & drop, progress tracking, and preview panes.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QComboBox, QGroupBox,
    QSplitter, QFrame, QStatusBar, QToolBar, QSpinBox
)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QAction, QIcon

from src.pdf_processor import PDFProcessor, extract_text_for_preview
from src.paraphraser import Paraphraser, ParaphraserConfig
from src.workers import ModelLoaderWorker, FullProcessWorker


class DropZone(QFrame):
    """Drag & drop zone for PDF files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f5f5f5;
            }
            DropZone:hover {
                border-color: #666;
                background-color: #eee;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Drop PDF here\nor click to browse")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; color: #666;")
        layout.addWidget(self.label)

        self.file_label = QLabel("")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("font-size: 12px; color: #333; font-weight: bold;")
        layout.addWidget(self.file_label)

        self.parent_window = parent

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith('.pdf'):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    DropZone {
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        background-color: #e8f5e9;
                    }
                """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f5f5f5;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f5f5f5;
            }
        """)

        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith('.pdf'):
                if self.parent_window:
                    self.parent_window.load_pdf(file_path)

    def mousePressEvent(self, event):
        if self.parent_window:
            self.parent_window.browse_pdf()

    def set_file(self, filename: str):
        """Update the display with the loaded filename."""
        self.file_label.setText(filename)
        if filename:
            self.label.setText("PDF Loaded")
        else:
            self.label.setText("Drop PDF here\nor click to browse")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("KaiBot - AI PDF Paraphraser")
        self.setMinimumSize(900, 700)

        # Initialize components
        self.paraphraser = Paraphraser()
        self.pdf_processor = PDFProcessor()
        self.current_pdf_path = ""
        self.current_worker = None
        self.model_worker = None
        self.model_loaded = False

        # Setup UI
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()

        # Load settings
        self._load_settings()

    def _setup_ui(self):
        """Setup the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Top section - Model and settings
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)

        # Model path
        settings_layout.addWidget(QLabel("Model:"))
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #666;")
        settings_layout.addWidget(self.model_label)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.browse_model)
        settings_layout.addWidget(self.load_model_btn)

        settings_layout.addStretch()

        # Style selector
        settings_layout.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["default", "academic", "casual", "technical"])
        self.style_combo.setCurrentText("default")
        self.style_combo.setMinimumWidth(120)
        settings_layout.addWidget(self.style_combo)

        # Min words filter
        settings_layout.addWidget(QLabel("Min words:"))
        self.min_words_spin = QSpinBox()
        self.min_words_spin.setRange(1, 50)
        self.min_words_spin.setValue(3)
        self.min_words_spin.setToolTip("Skip text blocks with fewer words than this")
        settings_layout.addWidget(self.min_words_spin)

        main_layout.addWidget(settings_group)

        # Middle section - Drop zone and previews
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Drop zone and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.drop_zone = DropZone(self)
        left_layout.addWidget(self.drop_zone)

        # Process button
        self.process_btn = QPushButton("Process PDF")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.hide()
        left_layout.addWidget(self.cancel_btn)

        left_layout.addStretch()

        content_splitter.addWidget(left_widget)

        # Right side - Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_label = QLabel("Preview (first 2000 chars)")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("PDF preview will appear here...")
        preview_layout.addWidget(self.preview_text)

        content_splitter.addWidget(preview_widget)
        content_splitter.setSizes([300, 600])

        main_layout.addWidget(content_splitter, 1)

        # Bottom section - Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.stage_label = QLabel("Ready")
        self.stage_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)

        self.progress_detail = QLabel("")
        self.progress_detail.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.progress_detail)

        main_layout.addWidget(progress_group)

    def _setup_toolbar(self):
        """Setup the toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open PDF", self)
        open_action.triggered.connect(self.browse_pdf)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready - Load a model and PDF to begin")

    def _load_settings(self):
        """Load saved settings."""
        # Could load from QSettings here
        pass

    def browse_pdf(self):
        """Open file dialog to select a PDF."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF File",
            "",
            "PDF Files (*.pdf)"
        )
        if file_path:
            self.load_pdf(file_path)

    def load_pdf(self, file_path: str):
        """Load a PDF file."""
        self.current_pdf_path = file_path
        filename = os.path.basename(file_path)
        self.drop_zone.set_file(filename)

        # Show preview
        preview = extract_text_for_preview(file_path)
        self.preview_text.setPlainText(preview)

        # Enable process button if model is loaded
        self._update_process_button()

        self.statusbar.showMessage(f"Loaded: {filename}")

    def browse_model(self):
        """Open file dialog to select a GGUF model."""
        # Start in the models directory if it exists
        import os
        start_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        if not os.path.exists(start_dir):
            start_dir = ""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model",
            start_dir,
            "All Files (*);;GGUF Models (*.gguf)"
        )
        if file_path:
            self.load_model(file_path)

    def load_model(self, model_path: str):
        """Load the LLM model."""
        self.load_model_btn.setEnabled(False)
        self.model_label.setText("Loading...")
        self.statusbar.showMessage("Loading model... This may take a moment.")

        self.model_worker = ModelLoaderWorker(self.paraphraser, model_path)
        self.model_worker.progress.connect(self._on_model_progress)
        self.model_worker.finished.connect(self._on_model_loaded)
        self.model_worker.start()

    def _on_model_progress(self, message: str):
        """Handle model loading progress."""
        self.statusbar.showMessage(message)

    def _on_model_loaded(self, success: bool, message: str):
        """Handle model loading completion."""
        self.load_model_btn.setEnabled(True)

        if success:
            model_name = os.path.basename(self.paraphraser.config.model_path)
            self.model_label.setText(model_name)
            self.model_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.model_loaded = True
            self._update_process_button()
        else:
            self.model_label.setText("Failed to load")
            self.model_label.setStyleSheet("color: #f44336;")
            QMessageBox.critical(self, "Error", message)

        self.statusbar.showMessage(message)

    def _update_process_button(self):
        """Update the process button state."""
        can_process = bool(self.model_loaded and self.current_pdf_path)
        self.process_btn.setEnabled(can_process)

    def start_processing(self):
        """Start the paraphrasing process."""
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return

        if not self.current_pdf_path:
            QMessageBox.warning(self, "Warning", "Please select a PDF first.")
            return

        # Get output path
        input_path = Path(self.current_pdf_path)
        default_output = input_path.parent / f"{input_path.stem}_paraphrased.pdf"

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Paraphrased PDF",
            str(default_output),
            "PDF Files (*.pdf)"
        )

        if not output_path:
            return

        # Update UI for processing
        self.process_btn.hide()
        self.cancel_btn.show()
        self.cancel_btn.setEnabled(True)
        self.load_model_btn.setEnabled(False)
        self.drop_zone.setEnabled(False)

        # Start the worker
        self.current_worker = FullProcessWorker(
            paraphraser=self.paraphraser,
            input_path=self.current_pdf_path,
            output_path=output_path,
            style=self.style_combo.currentText(),
            min_words=self.min_words_spin.value()
        )

        self.current_worker.progress.connect(self._on_progress)
        self.current_worker.stage_changed.connect(self._on_stage_changed)
        self.current_worker.finished.connect(self._on_processing_finished)
        self.current_worker.start()

    def cancel_processing(self):
        """Cancel the current processing."""
        if self.current_worker:
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            self.current_worker.cancel()

    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress updates."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            percentage = int((current / total) * 100)
            self.progress_bar.setFormat(f"{percentage}%")

        self.progress_detail.setText(message)

    def _on_stage_changed(self, stage: str):
        """Handle stage changes."""
        self.stage_label.setText(f"Stage: {stage}")

    def _on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
        # Reset UI
        self.cancel_btn.hide()
        self.cancel_btn.setText("Cancel")
        self.cancel_btn.setEnabled(True)
        self.process_btn.show()
        self.load_model_btn.setEnabled(True)
        self.drop_zone.setEnabled(True)

        self.current_worker = None

        if success:
            self.stage_label.setText("Complete!")
            self.progress_bar.setValue(self.progress_bar.maximum())
            QMessageBox.information(self, "Success", message)
        else:
            self.stage_label.setText("Failed")
            QMessageBox.critical(self, "Error", message)

        self.statusbar.showMessage(message)

    def show_settings(self):
        """Show settings dialog."""
        from ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self.paraphraser.config, self)
        if dialog.exec():
            # Settings were saved
            pass

    def closeEvent(self, event):
        """Handle window close."""
        # Check if any worker is running
        workers_running = (
            (self.current_worker and self.current_worker.isRunning()) or
            (self.model_worker and self.model_worker.isRunning())
        )

        if workers_running:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            # Cancel and wait for workers
            if self.current_worker and self.current_worker.isRunning():
                self.current_worker.cancel()
                self.current_worker.wait(5000)

            if self.model_worker and self.model_worker.isRunning():
                self.model_worker.wait(5000)

        # Cleanup
        self.paraphraser.unload_model()
        self.pdf_processor.close()

        event.accept()
