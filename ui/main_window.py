"""
Main Window - PyQt6 main application window for KaiBot.
Features text paraphrasing, PDF processing, drag & drop, and progress tracking.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit, QPlainTextEdit,
    QFileDialog, QMessageBox, QComboBox, QGroupBox,
    QSplitter, QFrame, QStatusBar, QToolBar, QSpinBox,
    QTabWidget, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QAction, QFont

from src.pdf_processor import PDFProcessor, extract_text_for_preview
from src.paraphraser import Paraphraser, ParaphraserConfig
from src.workers import ModelLoaderWorker, FullProcessWorker


class TextParaphraseWorker(QThread):
    """Worker for paraphrasing text in background."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, paraphraser: Paraphraser, text: str, style: str):
        super().__init__()
        self.paraphraser = paraphraser
        self.text = text
        self.style = style
        self._cancelled = False

    def run(self):
        try:
            if self._cancelled:
                self.finished.emit(False, "")
                return

            self.progress.emit("Paraphrasing text...")
            result = self.paraphraser.paraphrase(self.text, self.style)

            if self._cancelled:
                self.finished.emit(False, "")
            else:
                self.finished.emit(True, result)

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def cancel(self):
        self._cancelled = True


class DropZone(QFrame):
    """Drag & drop zone for PDF files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
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

        self.label = QLabel("Drop PDF here or click to browse")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 14px; color: #666;")
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
            self.label.setText("Drop PDF here or click to browse")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("KaiBot - AI Humanizer & PDF Paraphraser")
        self.setMinimumSize(1000, 750)

        # Initialize components
        self.paraphraser = Paraphraser()
        self.pdf_processor = PDFProcessor()
        self.current_pdf_path = ""
        self.current_worker = None
        self.model_worker = None
        self.text_worker = None
        self.model_loaded = False

        # Setup UI
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()

    def _setup_ui(self):
        """Setup the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Top section - Model and settings
        settings_group = QGroupBox("Model Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Model:"))
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #666;")
        settings_layout.addWidget(self.model_label)

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.browse_model)
        settings_layout.addWidget(self.load_model_btn)

        settings_layout.addStretch()

        settings_layout.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["default", "academic", "casual", "technical"])
        self.style_combo.setCurrentText("default")
        self.style_combo.setMinimumWidth(120)
        settings_layout.addWidget(self.style_combo)

        main_layout.addWidget(settings_group)

        # Tab widget for Text and PDF modes
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
            QTabBar::tab {
                padding: 10px 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)

        # Text Paraphraser Tab
        text_tab = self._create_text_tab()
        self.tab_widget.addTab(text_tab, "Text Humanizer")

        # PDF Paraphraser Tab
        pdf_tab = self._create_pdf_tab()
        self.tab_widget.addTab(pdf_tab, "PDF Paraphraser")

        main_layout.addWidget(self.tab_widget, 1)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.stage_label = QLabel("Ready")
        self.stage_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addWidget(progress_group)

    def _create_text_tab(self) -> QWidget:
        """Create the text paraphraser tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Input/Output splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Input section
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 5, 0)

        input_header = QHBoxLayout()
        input_label = QLabel("Input Text")
        input_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        input_header.addWidget(input_label)
        input_header.addStretch()

        self.input_word_count = QLabel("0 words")
        self.input_word_count.setStyleSheet("color: #666;")
        input_header.addWidget(self.input_word_count)

        self.clear_input_btn = QPushButton("Clear")
        self.clear_input_btn.setMaximumWidth(60)
        self.clear_input_btn.clicked.connect(self._clear_input)
        input_header.addWidget(self.clear_input_btn)

        input_layout.addLayout(input_header)

        self.input_text = QPlainTextEdit()
        self.input_text.setPlaceholderText("Paste or type your AI-generated text here...")
        self.input_text.setStyleSheet("""
            QPlainTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        self.input_text.textChanged.connect(self._update_input_word_count)
        input_layout.addWidget(self.input_text)

        splitter.addWidget(input_widget)

        # Output section
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(5, 0, 0, 0)

        output_header = QHBoxLayout()
        output_label = QLabel("Humanized Output")
        output_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        output_header.addWidget(output_label)
        output_header.addStretch()

        self.output_word_count = QLabel("0 words")
        self.output_word_count.setStyleSheet("color: #666;")
        output_header.addWidget(self.output_word_count)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setMaximumWidth(60)
        self.copy_btn.clicked.connect(self._copy_output)
        self.copy_btn.setEnabled(False)
        output_header.addWidget(self.copy_btn)

        output_layout.addLayout(output_header)

        self.output_text = QPlainTextEdit()
        self.output_text.setPlaceholderText("Humanized text will appear here...")
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("""
            QPlainTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-size: 13px;
                line-height: 1.5;
                background-color: #fafafa;
            }
        """)
        output_layout.addWidget(self.output_text)

        splitter.addWidget(output_widget)
        splitter.setSizes([500, 500])

        layout.addWidget(splitter, 1)

        # Humanize button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.humanize_btn = QPushButton("Humanize Text")
        self.humanize_btn.setMinimumSize(200, 50)
        self.humanize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.humanize_btn.clicked.connect(self._humanize_text)
        self.humanize_btn.setEnabled(False)
        btn_layout.addWidget(self.humanize_btn)

        self.cancel_text_btn = QPushButton("Cancel")
        self.cancel_text_btn.setMinimumSize(100, 50)
        self.cancel_text_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.cancel_text_btn.clicked.connect(self._cancel_text_humanize)
        self.cancel_text_btn.hide()
        btn_layout.addWidget(self.cancel_text_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return tab

    def _create_pdf_tab(self) -> QWidget:
        """Create the PDF paraphraser tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # PDF Settings row
        pdf_settings = QHBoxLayout()
        pdf_settings.addWidget(QLabel("Min words per block:"))
        self.min_words_spin = QSpinBox()
        self.min_words_spin.setRange(1, 50)
        self.min_words_spin.setValue(3)
        self.min_words_spin.setToolTip("Skip text blocks with fewer words than this")
        pdf_settings.addWidget(self.min_words_spin)
        pdf_settings.addStretch()
        layout.addLayout(pdf_settings)

        # Content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Drop zone and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.drop_zone = DropZone(self)
        left_layout.addWidget(self.drop_zone)

        self.process_btn = QPushButton("Process PDF")
        self.process_btn.setMinimumHeight(45)
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

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(45)
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

        preview_label = QLabel("PDF Preview")
        preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(preview_label)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("PDF content preview will appear here...")
        preview_layout.addWidget(self.preview_text)

        content_splitter.addWidget(preview_widget)
        content_splitter.setSizes([300, 500])

        layout.addWidget(content_splitter, 1)

        return tab

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
        self.statusbar.showMessage("Ready - Load a model to begin")

    # Text Tab Methods
    def _update_input_word_count(self):
        """Update the input word count label."""
        text = self.input_text.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        self.input_word_count.setText(f"{word_count} words")
        self._update_humanize_button()

    def _update_humanize_button(self):
        """Update humanize button state."""
        has_text = bool(self.input_text.toPlainText().strip())
        self.humanize_btn.setEnabled(self.model_loaded and has_text)

    def _clear_input(self):
        """Clear the input text."""
        self.input_text.clear()
        self.output_text.clear()
        self.output_word_count.setText("0 words")
        self.copy_btn.setEnabled(False)

    def _copy_output(self):
        """Copy output text to clipboard."""
        text = self.output_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.statusbar.showMessage("Copied to clipboard!", 2000)

    def _humanize_text(self):
        """Start text humanization."""
        text = self.input_text.toPlainText().strip()
        if not text:
            return

        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return

        # Update UI
        self.humanize_btn.hide()
        self.cancel_text_btn.show()
        self.output_text.clear()
        self.output_word_count.setText("0 words")
        self.copy_btn.setEnabled(False)
        self.stage_label.setText("Humanizing text...")
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Start worker
        self.text_worker = TextParaphraseWorker(
            self.paraphraser,
            text,
            self.style_combo.currentText()
        )
        self.text_worker.progress.connect(lambda msg: self.stage_label.setText(msg))
        self.text_worker.finished.connect(self._on_text_humanized)
        self.text_worker.start()

    def _cancel_text_humanize(self):
        """Cancel text humanization."""
        if self.text_worker:
            self.text_worker.cancel()
            self.text_worker.wait(2000)
            self._reset_text_ui()

    def _on_text_humanized(self, success: bool, result: str):
        """Handle text humanization completion."""
        self._reset_text_ui()

        if success and result:
            self.output_text.setPlainText(result)
            word_count = len(result.split())
            self.output_word_count.setText(f"{word_count} words")
            self.copy_btn.setEnabled(True)
            self.stage_label.setText("Done!")
            self.statusbar.showMessage("Text humanized successfully!", 3000)
        elif result and result.startswith("Error:"):
            QMessageBox.critical(self, "Error", result)
            self.stage_label.setText("Error")
        elif not result:
            self.stage_label.setText("No output generated")
            self.statusbar.showMessage("No output was generated. Try again.", 3000)

    def _reset_text_ui(self):
        """Reset text tab UI after operation."""
        self.cancel_text_btn.hide()
        self.humanize_btn.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    # PDF Tab Methods
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

        preview = extract_text_for_preview(file_path)
        self.preview_text.setPlainText(preview)

        self._update_process_button()
        self.statusbar.showMessage(f"Loaded: {filename}")

        # Switch to PDF tab
        self.tab_widget.setCurrentIndex(1)

    def browse_model(self):
        """Open file dialog to select a GGUF model."""
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
        self.model_worker.finished.connect(lambda: self._cleanup_finished_worker('model'))
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
            self._update_humanize_button()
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

        input_path = Path(self.current_pdf_path)
        default_output = input_path.parent / f"{input_path.stem}_humanized.pdf"

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Humanized PDF",
            str(default_output),
            "PDF Files (*.pdf)"
        )

        if not output_path:
            return

        self.process_btn.hide()
        self.cancel_btn.show()
        self.cancel_btn.setEnabled(True)
        self.load_model_btn.setEnabled(False)
        self.drop_zone.setEnabled(False)

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
        self.current_worker.finished.connect(lambda: self._cleanup_finished_worker('current'))
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

        self.stage_label.setText(message)

    def _on_stage_changed(self, stage: str):
        """Handle stage changes."""
        self.stage_label.setText(f"Stage: {stage}")

    def _on_processing_finished(self, success: bool, message: str):
        """Handle processing completion."""
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
            pass

    def _cleanup_finished_worker(self, worker_type: str):
        """Clean up a finished worker thread."""
        if worker_type == 'model' and self.model_worker:
            if not self.model_worker.isRunning():
                self.model_worker.deleteLater()
                self.model_worker = None
        elif worker_type == 'current' and self.current_worker:
            if not self.current_worker.isRunning():
                self.current_worker.deleteLater()
                self.current_worker = None
        elif worker_type == 'text' and self.text_worker:
            if not self.text_worker.isRunning():
                self.text_worker.deleteLater()
                self.text_worker = None

    def closeEvent(self, event):
        """Handle window close."""
        workers_running = (
            (self.current_worker and self.current_worker.isRunning()) or
            (self.model_worker and self.model_worker.isRunning()) or
            (self.text_worker and self.text_worker.isRunning())
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

        self._cleanup_workers()
        self.paraphraser.unload_model()
        self.pdf_processor.close()
        event.accept()

    def _cleanup_workers(self):
        """Safely stop and cleanup all worker threads."""
        for worker in [self.current_worker, self.model_worker, self.text_worker]:
            if worker:
                if worker.isRunning():
                    if hasattr(worker, 'cancel'):
                        worker.cancel()
                    worker.quit()
                    worker.wait(2000)
                    if worker.isRunning():
                        worker.terminate()
                        worker.wait(500)

        self.current_worker = None
        self.model_worker = None
        self.text_worker = None
