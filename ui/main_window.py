"""
Main Window - PyQt6 main application window for LocalWrite.
Privacy-first AI writing assistant with clean, beautiful UI.
Features smart model selection, writing enhancement, and real-time stats.
"""

import os
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit, QPlainTextEdit,
    QFileDialog, QMessageBox, QComboBox, QGroupBox,
    QSplitter, QFrame, QStatusBar, QToolBar, QSpinBox,
    QTabWidget, QApplication, QSlider, QMenu, QMenuBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QAction, QFont, QKeySequence,
    QShortcut, QTextCursor, QTextCharFormat, QColor
)

from src.paraphraser import Paraphraser, ParaphraserConfig
from src.humanizer_v2 import HumanizerV2, HumanizerV2Config
from src.workers import ModelLoaderWorker
from src.text_analyzer import TextAnalyzer, get_stats, get_readability, get_tone
from src.settings_manager import get_settings_manager, save_settings
from src.history_manager import get_history_manager, add_to_history
from src.diff_viewer import DiffViewer, compare_texts, get_diff_html
from src.export_manager import get_export_manager, is_docx_available
from src.synonym_provider import get_synonym_provider, get_synonyms
from src.model_registry import get_model_by_id, get_recommended_model
from src.model_downloader import get_model_manager
from ui.model_selector import ModelSelector


# Mode information - 5 focused modes that actually transform text
MODE_INFO = {
    "professional": ("Professional", "Transforms casual text to formal business language (Hi→Hello, ain't→is not)"),
    "conversational": ("Conversational", "Transforms formal text to casual friendly language (Hello→Hi, cannot→can't)"),
    "scholarly": ("Scholarly", "Transforms to academic style (use→utilize, show→demonstrate)"),
    "creative": ("Creative", "Transforms to vivid, engaging storytelling style"),
    "concise": ("Concise", "Makes text shorter by removing filler and redundancy"),
}


class RefineWorker(QThread):
    """Worker for refining text based on user feedback."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    # Context window settings
    MAX_CONTEXT = 8192  # Most models have 8k context
    CHARS_PER_TOKEN = 4
    MAX_CHUNK_CHARS = 20000

    def __init__(self, model, text: str, instruction: str):
        super().__init__()
        self.model = model
        self.text = text
        self.instruction = instruction
        self._cancelled = False

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // self.CHARS_PER_TOKEN

    def _split_into_chunks(self, text: str) -> list:
        """Split text into manageable chunks."""
        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len + 2 > self.MAX_CHUNK_CHARS:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len + 2

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def run(self):
        try:
            if self._cancelled or not self.model:
                self.finished.emit(False, "")
                return

            self.progress.emit("Refining based on your feedback...")

            # Check if text needs chunking
            chunks = self._split_into_chunks(self.text)

            if len(chunks) == 1:
                # Single chunk - process normally
                result = self._refine_chunk(self.text)
            else:
                # Multiple chunks - process each
                self.progress.emit(f"Refining {len(chunks)} sections...")
                refined_chunks = []
                for i, chunk in enumerate(chunks):
                    if self._cancelled:
                        self.finished.emit(False, "")
                        return
                    self.progress.emit(f"Refining section {i+1}/{len(chunks)}...")
                    refined = self._refine_chunk(chunk)
                    refined_chunks.append(refined)
                result = '\n\n'.join(refined_chunks)

            if self._cancelled:
                self.finished.emit(False, "")
            else:
                self.finished.emit(True, result)

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def _refine_chunk(self, text: str) -> str:
        """Refine a single chunk of text."""
        # Build refinement prompt
        prompt = f"""<|im_start|>system
You are a helpful writing assistant. The user wants you to modify the text based on their specific instruction.
Apply ONLY the requested change. Keep everything else the same.
Do not add unnecessary changes or explanations.
Output the refined text only.<|im_end|>
<|im_start|>user
Here is the current text:
---
{text}
---

Please make this change: {self.instruction}<|im_end|>
<|im_start|>assistant
"""

        # Calculate safe max tokens
        prompt_tokens = self._estimate_tokens(prompt)
        available_tokens = self.MAX_CONTEXT - prompt_tokens - 100
        max_tokens = max(256, min(available_tokens, 4096))

        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )

        return response["choices"][0]["text"].strip()

    def cancel(self):
        self._cancelled = True


class ChatWorker(QThread):
    """Worker for direct chat/generation with conversation memory."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, paraphraser, message: str, conversation_history: list = None):
        super().__init__()
        self.paraphraser = paraphraser
        self.message = message
        self.conversation_history = conversation_history or []
        self._cancelled = False

    def run(self):
        try:
            if self._cancelled:
                self.finished.emit(False, "")
                return

            def progress_callback(msg):
                if not self._cancelled:
                    self.progress.emit(msg)

            result = self.paraphraser.chat(
                self.message,
                self.conversation_history,
                progress_callback
            )

            if self._cancelled:
                self.finished.emit(False, "")
            else:
                self.finished.emit(True, result)

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def cancel(self):
        self._cancelled = True


class TextHumanizeWorker(QThread):
    """Worker for humanizing text in background using advanced humanizer."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, humanizer: HumanizerV2, text: str):
        super().__init__()
        self.humanizer = humanizer
        self.text = text
        self._cancelled = False

    def run(self):
        try:
            if self._cancelled:
                self.finished.emit(False, "")
                return

            def progress_callback(msg):
                if not self._cancelled:
                    self.progress.emit(msg)

            result = self.humanizer.humanize(self.text, progress_callback)

            if self._cancelled:
                self.finished.emit(False, "")
            else:
                self.finished.emit(True, result)

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def cancel(self):
        self._cancelled = True


class StatsPanel(QFrame):
    """Panel showing writing statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsPanel")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        # Create stat items (only accurate/useful metrics)
        self.word_stat = self._create_stat("0", "Words")
        self.sentence_stat = self._create_stat("0", "Sentences")
        self.reading_stat = self._create_stat("0s", "Read Time")
        self.grade_stat = self._create_stat("--", "Grade")
        self.tone_stat = self._create_stat("--", "Tone")

        layout.addWidget(self.word_stat)
        layout.addWidget(self.sentence_stat)
        layout.addWidget(self.reading_stat)
        layout.addWidget(self.grade_stat)
        layout.addWidget(self.tone_stat)
        layout.addStretch()

    def _create_stat(self, value: str, label: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("statItem")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_label)

        label_widget = QLabel(label)
        label_widget.setObjectName("statLabel")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_widget)

        frame.value_label = value_label
        return frame

    def update_stats(self, text: str):
        if not text.strip():
            self.word_stat.value_label.setText("0")
            self.sentence_stat.value_label.setText("0")
            self.reading_stat.value_label.setText("0s")
            self.grade_stat.value_label.setText("--")
            self.tone_stat.value_label.setText("--")
            return

        # Get stats (using accurate metrics only)
        stats = get_stats(text)
        readability = get_readability(text)
        tone = get_tone(text)

        # Update display
        self.word_stat.value_label.setText(str(stats.word_count))
        self.sentence_stat.value_label.setText(str(stats.sentence_count))

        # Format reading time
        if stats.reading_time_seconds < 60:
            self.reading_stat.value_label.setText(f"{stats.reading_time_seconds}s")
        else:
            mins = stats.reading_time_seconds // 60
            self.reading_stat.value_label.setText(f"{mins}m")

        self.grade_stat.value_label.setText(readability.grade_label)
        self.tone_stat.value_label.setText(tone.primary_tone)


class MainWindow(QMainWindow):
    """Main application window with modern UI."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("LocalWrite - Private AI Writing Assistant")
        self.setMinimumSize(1200, 800)

        # Initialize settings
        self.settings_manager = get_settings_manager()
        self.settings = self.settings_manager.settings

        # Initialize components
        self.paraphraser = Paraphraser()
        self.humanizer = HumanizerV2()
        self.text_analyzer = TextAnalyzer()
        self.model_worker = None
        self.text_worker = None
        self.refine_worker = None
        self.chat_worker = None
        self.model_loaded = False
        self.dark_mode = self.settings.theme == "dark"

        # Original text for comparison
        self.original_text = ""

        # Diff view state
        self.diff_viewer = DiffViewer()
        self.diff_view_enabled = False
        self.highlight_enabled = True

        # History manager
        self.history_manager = get_history_manager()

        # Synonym provider
        self.synonym_provider = get_synonym_provider()

        # Export manager
        self.export_manager = get_export_manager()

        # Setup UI
        self._setup_menubar()
        self._setup_ui()
        self._setup_shortcuts()
        self._setup_statusbar()

        # Load settings
        self._apply_settings()

        # Stats update timer (debounce)
        self.stats_timer = QTimer()
        self.stats_timer.setSingleShot(True)
        self.stats_timer.timeout.connect(self._update_stats_delayed)

    def _setup_menubar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_doc_action = QAction("Open Document...", self)
        open_doc_action.setShortcut(QKeySequence("Ctrl+O"))
        open_doc_action.triggered.connect(self._upload_document)
        file_menu.addAction(open_doc_action)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("Export")
        export_txt = QAction("Export as Text (.txt)", self)
        export_txt.triggered.connect(lambda: self._export_output("txt"))
        export_menu.addAction(export_txt)

        export_md = QAction("Export as Markdown (.md)", self)
        export_md.triggered.connect(lambda: self._export_output("md"))
        export_menu.addAction(export_md)

        export_docx = QAction("Export as Word (.docx)", self)
        export_docx.triggered.connect(lambda: self._export_output("docx"))
        export_docx.setEnabled(is_docx_available())
        export_menu.addAction(export_docx)

        export_menu.addSeparator()

        export_comparison = QAction("Export Comparison...", self)
        export_comparison.triggered.connect(self._export_comparison)
        export_menu.addAction(export_comparison)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        copy_action = QAction("Copy Output", self)
        copy_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        copy_action.triggered.connect(self._copy_output)
        edit_menu.addAction(copy_action)

        clear_action = QAction("Clear All", self)
        clear_action.triggered.connect(self._clear_all)
        edit_menu.addAction(clear_action)

        # View menu
        view_menu = menubar.addMenu("View")

        self.dark_mode_action = QAction("Dark Mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.dark_mode)
        self.dark_mode_action.triggered.connect(self._toggle_dark_mode)
        view_menu.addAction(self.dark_mode_action)

        view_menu.addSeparator()

        self.diff_view_action = QAction("Show Diff View", self)
        self.diff_view_action.setCheckable(True)
        self.diff_view_action.setChecked(False)
        self.diff_view_action.triggered.connect(self._toggle_diff_view)
        view_menu.addAction(self.diff_view_action)

        self.highlight_action = QAction("Highlight AI Sentences", self)
        self.highlight_action.setCheckable(True)
        self.highlight_action.setChecked(True)
        self.highlight_action.triggered.connect(self._toggle_highlighting)
        view_menu.addAction(self.highlight_action)

        view_menu.addSeparator()

        history_action = QAction("View History...", self)
        history_action.setShortcut(QKeySequence("Ctrl+H"))
        history_action.triggered.connect(self._show_history)
        view_menu.addAction(history_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        load_custom_action = QAction("Load Custom Model...", self)
        load_custom_action.triggered.connect(self.browse_model)
        tools_menu.addAction(load_custom_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        tutorial_action = QAction("Show Tutorial...", self)
        tutorial_action.triggered.connect(self._show_onboarding)
        help_menu.addAction(tutorial_action)

        about_action = QAction("About LocalWrite", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_ui(self):
        """Setup the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header section
        header = self._create_header()
        main_layout.addWidget(header)

        # Tab widget for Text and Chat modes
        self.tab_widget = QTabWidget()
        text_tab = self._create_text_tab()
        self.tab_widget.addTab(text_tab, "Writing Assistant")

        chat_tab = self._create_chat_tab()
        self.tab_widget.addTab(chat_tab, "AI Chat")

        # Connect tab change to update UI
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        main_layout.addWidget(self.tab_widget, 1)

        # Stats panel (hidden for AI Chat tab)
        self.stats_panel = StatsPanel()
        main_layout.addWidget(self.stats_panel)

        # Progress section
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.stage_label = QLabel("Ready")
        self.stage_label.setObjectName("helperText")
        progress_layout.addWidget(self.stage_label)

        progress_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setFixedHeight(6)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addWidget(progress_widget)

    def _create_header(self) -> QWidget:
        """Create the header with controls."""
        header = QFrame()
        header.setObjectName("headerWidget")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # App title
        title = QLabel("LocalWrite")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        # Privacy badge
        privacy_badge = QLabel("100% Offline")
        privacy_badge.setObjectName("privacyBadge")
        privacy_badge.setStyleSheet("""
            QLabel {
                background-color: #10B981;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        layout.addWidget(privacy_badge)

        layout.addStretch()

        # Model selector (replaces Load Model button)
        layout.addWidget(QLabel("Model:"))
        self.model_selector = ModelSelector()
        self.model_selector.model_changed.connect(self._on_model_selected)
        layout.addWidget(self.model_selector)

        # Mode selector with descriptions
        layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(180)
        for mode_key, (name, desc) in MODE_INFO.items():
            self.mode_combo.addItem(f"{name}", mode_key)
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setToolTip(MODE_INFO["professional"][1])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        # Info icon with tooltip for mode description
        self.mode_info_btn = QPushButton("ⓘ")
        self.mode_info_btn.setFixedSize(24, 24)
        self.mode_info_btn.setToolTip(MODE_INFO["professional"][1])
        self.mode_info_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #6B7280;
                font-size: 16px;
            }
            QPushButton:hover {
                color: #7C3AED;
            }
        """)
        layout.addWidget(self.mode_info_btn)

        # Creativity slider
        layout.addWidget(QLabel("Creativity:"))
        self.creativity_slider = QSlider(Qt.Orientation.Horizontal)
        self.creativity_slider.setMinimum(0)
        self.creativity_slider.setMaximum(100)
        self.creativity_slider.setValue(self.settings.creativity_level)
        self.creativity_slider.setFixedWidth(120)
        self.creativity_slider.valueChanged.connect(self._on_creativity_changed)
        layout.addWidget(self.creativity_slider)

        self.creativity_label = QLabel(f"{self.settings.creativity_level}%")
        self.creativity_label.setFixedWidth(35)
        layout.addWidget(self.creativity_label)

        return header

    def _create_text_tab(self) -> QWidget:
        """Create the text humanizer tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # Input/Output splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Input section
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 8, 0)

        input_header = QHBoxLayout()
        input_label = QLabel("Input Text")
        input_label.setObjectName("sectionTitle")
        input_header.addWidget(input_label)
        input_header.addStretch()

        self.input_word_count = QLabel("0 words")
        self.input_word_count.setObjectName("wordCount")
        input_header.addWidget(self.input_word_count)

        # Upload document button
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setFixedSize(80, 32)
        self.upload_btn.clicked.connect(self._upload_document)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        input_header.addWidget(self.upload_btn)

        self.clear_input_btn = QPushButton("Clear")
        self.clear_input_btn.setFixedSize(70, 32)
        self.clear_input_btn.clicked.connect(self._clear_input)
        self.clear_input_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
        """)
        input_header.addWidget(self.clear_input_btn)

        input_layout.addLayout(input_header)

        self.input_text = QPlainTextEdit()
        self.input_text.setPlaceholderText("Paste or type your AI-generated text here...")
        self.input_text.textChanged.connect(self._on_input_changed)
        input_layout.addWidget(self.input_text)

        splitter.addWidget(input_widget)

        # Output section
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(8, 0, 0, 0)

        output_header = QHBoxLayout()
        output_label = QLabel("Enhanced Output")
        output_label.setObjectName("sectionTitle")
        output_header.addWidget(output_label)
        output_header.addStretch()

        self.output_word_count = QLabel("0 words")
        self.output_word_count.setObjectName("wordCount")
        output_header.addWidget(self.output_word_count)

        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setFixedSize(70, 32)
        self.copy_btn.clicked.connect(self._copy_output)
        self.copy_btn.setEnabled(False)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        output_header.addWidget(self.copy_btn)

        output_layout.addLayout(output_header)

        # Use QTextEdit for rich text highlighting
        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("Enhanced text will appear here...")
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)

        # Chat refinement section
        refine_frame = QFrame()
        refine_frame.setObjectName("refineFrame")
        refine_frame.setStyleSheet("""
            #refineFrame {
                background-color: #F3F4F6;
                border-radius: 8px;
                padding: 8px;
                margin-top: 8px;
            }
        """)
        refine_layout = QVBoxLayout(refine_frame)
        refine_layout.setContentsMargins(8, 8, 8, 8)
        refine_layout.setSpacing(6)

        refine_header = QLabel("Refine Output")
        refine_header.setStyleSheet("font-weight: 600; color: #374151;")
        refine_layout.addWidget(refine_header)

        # Quick refinement buttons
        quick_btn_layout = QHBoxLayout()
        quick_btn_layout.setSpacing(6)

        self.quick_btns = []
        quick_refinements = [
            ("More formal", "Make the text more formal and professional"),
            ("More casual", "Make the text more casual and friendly"),
            ("Shorter", "Make the text more concise, remove unnecessary words"),
            ("Longer", "Expand the text with more details"),
        ]

        for label, instruction in quick_refinements:
            btn = QPushButton(label)
            btn.setFixedHeight(28)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E5E7EB;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #D1D5DB;
                }
            """)
            btn.clicked.connect(lambda checked, inst=instruction: self._quick_refine(inst))
            quick_btn_layout.addWidget(btn)
            self.quick_btns.append(btn)

        quick_btn_layout.addStretch()
        refine_layout.addLayout(quick_btn_layout)

        # Custom refinement input
        custom_layout = QHBoxLayout()
        custom_layout.setSpacing(8)

        self.refine_input = QPlainTextEdit()
        self.refine_input.setPlaceholderText("Type your refinement request... (e.g., 'make paragraph 2 shorter' or 'use simpler words')")
        self.refine_input.setMaximumHeight(50)
        self.refine_input.setStyleSheet("""
            QPlainTextEdit {
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
            }
            QPlainTextEdit:focus {
                border: 2px solid #7C3AED;
            }
        """)
        custom_layout.addWidget(self.refine_input)

        self.refine_btn = QPushButton("Refine")
        self.refine_btn.setObjectName("primaryButton")
        self.refine_btn.setFixedSize(80, 40)
        self.refine_btn.clicked.connect(self._custom_refine)
        self.refine_btn.setEnabled(False)
        custom_layout.addWidget(self.refine_btn)

        refine_layout.addLayout(custom_layout)

        output_layout.addWidget(refine_frame)

        splitter.addWidget(output_widget)
        splitter.setSizes([500, 500])

        layout.addWidget(splitter, 1)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.humanize_btn = QPushButton("Enhance Writing")
        self.humanize_btn.setFixedSize(160, 45)
        self.humanize_btn.clicked.connect(self._humanize_text)
        self.humanize_btn.setEnabled(False)
        self.humanize_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        btn_layout.addWidget(self.humanize_btn)

        # Summarize button
        self.summarize_btn = QPushButton("Summarize")
        self.summarize_btn.setFixedSize(120, 45)
        self.summarize_btn.clicked.connect(self._summarize_text)
        self.summarize_btn.setEnabled(False)
        self.summarize_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6D28D9;
            }
            QPushButton:pressed {
                background-color: #5B21B6;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        btn_layout.addWidget(self.summarize_btn)

        # Expand button
        self.expand_btn = QPushButton("Expand")
        self.expand_btn.setFixedSize(120, 45)
        self.expand_btn.clicked.connect(self._expand_text)
        self.expand_btn.setEnabled(False)
        self.expand_btn.setStyleSheet("""
            QPushButton {
                background-color: #F59E0B;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #D97706;
            }
            QPushButton:pressed {
                background-color: #B45309;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        btn_layout.addWidget(self.expand_btn)

        self.cancel_text_btn = QPushButton("Cancel")
        self.cancel_text_btn.setFixedSize(100, 45)
        self.cancel_text_btn.clicked.connect(self._cancel_text_humanize)
        self.cancel_text_btn.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
        """)
        self.cancel_text_btn.hide()
        btn_layout.addWidget(self.cancel_text_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        return tab

    def _create_chat_tab(self) -> QWidget:
        """Create the AI Chat tab - like ChatGPT but offline."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        # Header with description
        header_layout = QHBoxLayout()
        chat_title = QLabel("AI Assistant")
        chat_title.setObjectName("sectionTitle")
        header_layout.addWidget(chat_title)
        header_layout.addStretch()

        chat_desc = QLabel("Ask questions, generate code, get help - all offline")
        chat_desc.setStyleSheet("color: #6B7280; font-size: 12px;")
        header_layout.addWidget(chat_desc)
        layout.addLayout(header_layout)

        # Document Library Panel (collapsible)
        self.doc_library_frame = QFrame()
        self.doc_library_frame.setStyleSheet("""
            QFrame#docLibrary {
                background-color: #F5F3FF;
                border: 1px solid #DDD6FE;
                border-radius: 8px;
            }
        """)
        self.doc_library_frame.setObjectName("docLibrary")
        doc_lib_layout = QVBoxLayout(self.doc_library_frame)
        doc_lib_layout.setContentsMargins(12, 8, 12, 8)
        doc_lib_layout.setSpacing(8)

        # Library header with toggle
        lib_header = QHBoxLayout()
        lib_header.setSpacing(8)

        self.doc_lib_toggle = QPushButton("📚 Document Library")
        self.doc_lib_toggle.setCheckable(True)
        self.doc_lib_toggle.setChecked(False)
        self.doc_lib_toggle.clicked.connect(self._toggle_doc_library)
        self.doc_lib_toggle.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #5B21B6;
                font-weight: 600;
                font-size: 13px;
                text-align: left;
                padding: 4px 8px;
            }
            QPushButton:hover {
                color: #7C3AED;
            }
        """)
        lib_header.addWidget(self.doc_lib_toggle)

        self.doc_count_label = QLabel("0 documents")
        self.doc_count_label.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        lib_header.addWidget(self.doc_count_label)

        lib_header.addStretch()

        # RAG toggle button
        self.rag_toggle_btn = QPushButton("🔍 RAG: OFF")
        self.rag_toggle_btn.setCheckable(True)
        self.rag_toggle_btn.setChecked(False)
        self.rag_toggle_btn.clicked.connect(self._toggle_rag_mode)
        self.rag_toggle_btn.setToolTip("Toggle document search mode.\nON: AI answers from your documents\nOFF: Regular AI chat")
        self.rag_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #E5E7EB;
                color: #6B7280;
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #10B981;
                color: white;
            }
            QPushButton:hover {
                background-color: #D1D5DB;
            }
            QPushButton:checked:hover {
                background-color: #059669;
            }
        """)
        lib_header.addWidget(self.rag_toggle_btn)

        self.add_doc_btn = QPushButton("+ Add")
        self.add_doc_btn.setFixedHeight(28)
        self.add_doc_btn.clicked.connect(self._add_document_to_library)
        self.add_doc_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #6D28D9;
            }
        """)
        lib_header.addWidget(self.add_doc_btn)

        self.clear_all_docs_btn = QPushButton("Clear All")
        self.clear_all_docs_btn.setFixedHeight(28)
        self.clear_all_docs_btn.clicked.connect(self._clear_all_documents)
        self.clear_all_docs_btn.setStyleSheet("""
            QPushButton {
                background-color: #FEE2E2;
                color: #DC2626;
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #EF4444;
                color: white;
            }
        """)
        lib_header.addWidget(self.clear_all_docs_btn)

        doc_lib_layout.addLayout(lib_header)

        # Document list (hidden by default)
        self.doc_list_widget = QFrame()
        self.doc_list_widget.setStyleSheet("background-color: white; border-radius: 6px; border: 1px solid #E5E7EB;")
        self.doc_list_layout = QVBoxLayout(self.doc_list_widget)
        self.doc_list_layout.setContentsMargins(8, 8, 8, 8)
        self.doc_list_layout.setSpacing(4)

        # Placeholder for empty state
        self.doc_list_empty = QLabel("No documents added yet.\nClick '+ Add' to upload PDFs, Word docs, or text files.")
        self.doc_list_empty.setStyleSheet("color: #9CA3AF; font-size: 12px; padding: 16px;")
        self.doc_list_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.doc_list_layout.addWidget(self.doc_list_empty)

        self.doc_list_widget.hide()  # Hidden by default
        doc_lib_layout.addWidget(self.doc_list_widget)

        layout.addWidget(self.doc_library_frame)

        # Initialize RAG engine (lazy loaded)
        self.rag_engine = None
        self.rag_enabled = False

        # Chat history display
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText(
            "💬 AI Chat - Two Modes Available:\n\n"
            "📚 RAG Mode (Document Q&A):\n"
            "   Add documents above, then ask questions about them.\n"
            "   AI will only answer from your documents.\n\n"
            "💭 Regular Mode:\n"
            "   Chat freely - write code, explain concepts, get help.\n\n"
            "Toggle RAG on/off using the button above."
        )
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #F9FAFB;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(self.chat_history, 1)

        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #F3F4F6;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)

        self.chat_input = QPlainTextEdit()
        self.chat_input.setPlaceholderText("Type your message here... (Enter to send, Shift+Enter for new line)")
        self.chat_input.setMaximumHeight(100)
        self.chat_input.setStyleSheet("""
            QPlainTextEdit {
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QPlainTextEdit:focus {
                border: 2px solid #7C3AED;
            }
        """)
        # Install event filter for Ctrl+Enter shortcut
        self.chat_input.installEventFilter(self)
        input_layout.addWidget(self.chat_input)

        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(80, 50)
        self.send_btn.clicked.connect(self._send_chat_message)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #7C3AED;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6D28D9;
            }
            QPushButton:pressed {
                background-color: #5B21B6;
            }
            QPushButton:disabled {
                background-color: #9CA3AF;
            }
        """)
        input_layout.addWidget(self.send_btn)

        layout.addWidget(input_frame)

        # Subtle action links below input
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(4, 4, 4, 0)

        self.clear_chat_btn = QPushButton("Clear conversation")
        self.clear_chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_chat_btn.clicked.connect(self._clear_chat)
        self.clear_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #9CA3AF;
                border: none;
                font-size: 12px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                color: #EF4444;
            }
        """)
        action_layout.addWidget(self.clear_chat_btn)

        action_layout.addStretch()

        self.copy_chat_btn = QPushButton("Copy last response")
        self.copy_chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_chat_btn.clicked.connect(self._copy_chat_response)
        self.copy_chat_btn.setEnabled(False)
        self.copy_chat_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #9CA3AF;
                border: none;
                font-size: 12px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                color: #7C3AED;
            }
            QPushButton:disabled {
                color: #D1D5DB;
            }
        """)
        action_layout.addWidget(self.copy_chat_btn)
        layout.addLayout(action_layout)

        # Store last response for copying
        self.last_chat_response = ""

        # Conversation history for context memory
        self.conversation_history = []
        self.current_chat_message = ""

        return tab

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Humanize shortcut
        humanize_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        humanize_shortcut.activated.connect(self._humanize_text)

        # Copy shortcut
        copy_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_shortcut.activated.connect(self._copy_output)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def _apply_settings(self):
        """Apply loaded settings."""
        # Window geometry
        geo = self.settings_manager.get_window_geometry()
        if geo['x'] is not None and geo['y'] is not None:
            self.move(geo['x'], geo['y'])
        self.resize(geo['width'], geo['height'])
        if geo['maximized']:
            self.showMaximized()

        # Theme
        if self.settings.theme == "dark":
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

        # Creativity level
        self.creativity_slider.setValue(self.settings.creativity_level)

        # Try to auto-load last used model
        QTimer.singleShot(100, self._try_auto_load_model)

        # Check for first run onboarding
        QTimer.singleShot(500, self._check_first_run)

    def _apply_light_theme(self):
        """Apply light theme stylesheet."""
        style_path = Path(__file__).parent.parent / "resources" / "styles.qss"
        if style_path.exists():
            with open(style_path, 'r') as f:
                self.setStyleSheet(f.read())

    def _apply_dark_theme(self):
        """Apply dark theme stylesheet."""
        style_path = Path(__file__).parent.parent / "resources" / "styles_dark.qss"
        if style_path.exists():
            with open(style_path, 'r') as f:
                self.setStyleSheet(f.read())

    def _toggle_dark_mode(self):
        """Toggle between light and dark themes."""
        self.dark_mode = not self.dark_mode
        self.settings.theme = "dark" if self.dark_mode else "light"
        self.dark_mode_action.setChecked(self.dark_mode)

        if self.dark_mode:
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

        save_settings()

    def _on_creativity_changed(self, value: int):
        """Handle creativity slider change."""
        self.creativity_label.setText(f"{value}%")
        self.settings.creativity_level = value

        # Update humanizer
        if hasattr(self.humanizer, 'creativity_level'):
            self.humanizer.creativity_level = value

    def _on_tab_changed(self, index: int):
        """Handle tab change - show/hide relevant UI elements."""
        # Tab indices: 0=Writing Assistant, 1=AI Chat
        is_chat_tab = (index == 1)

        # Hide stats panel for AI Chat (not relevant)
        self.stats_panel.setVisible(not is_chat_tab)

        # Hide mode selector for AI Chat (not needed)
        self.mode_combo.setVisible(not is_chat_tab)
        self.mode_info_btn.setVisible(not is_chat_tab)

        # Find and hide the "Mode:" label
        for i in range(self.mode_combo.parent().layout().count()):
            item = self.mode_combo.parent().layout().itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QLabel) and widget.text() == "Mode:":
                    widget.setVisible(not is_chat_tab)
                    break

    def _on_mode_changed(self, index: int):
        """Handle mode selection change."""
        mode_key = self.mode_combo.itemData(index)
        if mode_key and mode_key in MODE_INFO:
            _, desc = MODE_INFO[mode_key]
            # Update tooltips with mode description
            self.mode_combo.setToolTip(desc)
            self.mode_info_btn.setToolTip(desc)

    def _on_input_changed(self):
        """Handle input text change."""
        text = self.input_text.toPlainText()
        word_count = len(text.split()) if text.strip() else 0
        self.input_word_count.setText(f"{word_count} words")

        # Update button state
        self._update_humanize_button()

        # Debounce stats update
        self.stats_timer.start(300)

    def _update_stats_delayed(self):
        """Update stats after debounce."""
        text = self.input_text.toPlainText()
        self.stats_panel.update_stats(text)

    def _update_humanize_button(self):
        """Update humanize/summarize/expand button enabled state."""
        has_text = bool(self.input_text.toPlainText().strip())
        # Enable buttons if there's text (will show message if model not loaded)
        self.humanize_btn.setEnabled(has_text)
        self.summarize_btn.setEnabled(has_text)
        self.expand_btn.setEnabled(has_text)

    def _clear_input(self):
        """Clear input text."""
        self.input_text.clear()

    def _clear_all(self):
        """Clear all text areas."""
        self.input_text.clear()
        self.output_text.clear()
        self.copy_btn.setEnabled(False)

    def _upload_document(self):
        """Upload and read a document file."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Document",
            "",
            "Documents (*.txt *.md *.docx *.pdf);;PDF Files (*.pdf);;Text Files (*.txt);;Markdown (*.md);;Word Documents (*.docx);;All Files (*)"
        )

        if not file_path:
            return

        try:
            content = self._read_document(file_path)
            if content:
                self.input_text.setPlainText(content)
                word_count = len(content.split())
                self.statusbar.showMessage(f"Loaded: {file_path.split('/')[-1]} ({word_count} words)", 3000)
            else:
                QMessageBox.warning(self, "Empty Document", "The document appears to be empty.")
        except Exception as e:
            QMessageBox.critical(self, "Error Reading Document", f"Could not read the document:\n{str(e)}")

    def _read_document(self, file_path: str) -> str:
        """Read content from various document formats."""
        import os

        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.txt', '.md']:
            # Plain text files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif ext == '.docx':
            # Word documents
            try:
                from docx import Document
                doc = Document(file_path)
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                return '\n\n'.join(paragraphs)
            except ImportError:
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "python-docx is required to read Word documents.\n\n"
                    "Install it with: pip install python-docx"
                )
                return ""

        elif ext == '.pdf':
            # PDF files
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                return '\n\n'.join(text_parts)
            except ImportError:
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "PyMuPDF is required to read PDF files.\n\n"
                    "Install it with: pip install PyMuPDF"
                )
                return ""

        else:
            # Try reading as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _copy_output(self):
        """Copy output text to clipboard."""
        text = self.output_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.statusbar.showMessage("Copied to clipboard!", 2000)

    # ===== Chat Tab Methods =====

    def _send_chat_message(self):
        """Send a message to the AI assistant with conversation memory."""
        message = self.chat_input.toPlainText().strip()
        if not message:
            self.statusbar.showMessage("Please enter a message", 2000)
            return

        if not self.model_loaded:
            self.statusbar.showMessage("Please select and load a model first", 3000)
            QMessageBox.information(
                self, "Model Required",
                "Please select an AI model from the dropdown first.\n\n"
                "The model will download automatically on first use."
            )
            return

        # Store current message for adding to history after response
        self.current_chat_message = message

        # Add user message to chat display
        self._append_to_chat("You", message)
        self.chat_input.clear()

        # Disable send button while processing
        self.send_btn.setEnabled(False)
        self.send_btn.setText("...")
        self.stage_label.setText("AI is thinking...")
        self.progress_bar.setRange(0, 0)

        # Build the actual prompt - include RAG context if documents are loaded and RAG is enabled
        prompt_message = message
        using_rag = False

        if self.rag_enabled and self.rag_engine is not None and not self.rag_engine.is_empty:
            try:
                # Retrieve relevant context using RAG
                self.stage_label.setText("🔍 Searching documents...")
                context = self.rag_engine.get_context_for_query(message, max_tokens=2500)

                if context:
                    using_rag = True
                    prompt_message = f"""You are a helpful assistant answering questions ONLY based on the provided document excerpts.

STRICT RULES:
1. ONLY use information from the excerpts below - do NOT use any other knowledge
2. If the answer is NOT in the excerpts, respond: "I couldn't find this information in your documents."
3. NEVER make up or infer information not explicitly stated in the excerpts
4. Quote or reference specific parts of the documents when answering
5. If asked about something outside the documents, politely redirect to what's in the documents

DOCUMENT EXCERPTS:
---
{context}
---

USER QUESTION: {message}

Remember: Only answer from the excerpts above. Do not hallucinate or add external knowledge."""

            except Exception as e:
                # Fall back to regular chat if RAG fails
                self.statusbar.showMessage(f"RAG search failed: {str(e)}", 3000)

        # Update status to show mode
        if using_rag:
            self.stage_label.setText("📚 Answering from documents...")

        # Start chat worker with conversation history for context
        self.chat_worker = ChatWorker(
            self.paraphraser,
            prompt_message,
            self.conversation_history.copy()  # Pass copy of history
        )
        self.chat_worker.progress.connect(lambda msg: self.stage_label.setText(msg))
        self.chat_worker.finished.connect(self._on_chat_response)
        self.chat_worker.start()

    def _on_chat_response(self, success: bool, result: str):
        """Handle AI chat response and update conversation history."""
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        if success and result:
            # Add user message and AI response to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": self.current_chat_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": result
            })

            self._append_to_chat("AI", result)
            self.last_chat_response = result
            self.copy_chat_btn.setEnabled(True)
            self.stage_label.setText("Ready")

            # Show conversation context count
            turns = len(self.conversation_history) // 2
            self.statusbar.showMessage(f"Response received (conversation: {turns} turns)", 2000)
        else:
            self._append_to_chat("AI", f"Error: {result}" if result else "Error generating response")
            self.stage_label.setText("Error occurred")

    def _append_to_chat(self, sender: str, message: str):
        """Append a message to the chat history."""
        current = self.chat_history.toHtml()

        if sender == "You":
            formatted = f'''
            <div style="margin: 12px 0; padding: 12px; background-color: #EDE9FE; border-radius: 8px;">
                <b style="color: #7C3AED;">You:</b><br>
                <span style="color: #1F2937;">{self._escape_html(message)}</span>
            </div>
            '''
        else:
            # Format AI response with code block detection
            formatted_message = self._format_ai_message(message)
            formatted = f'''
            <div style="margin: 12px 0; padding: 12px; background-color: #F9FAFB; border-radius: 8px;">
                <b style="color: #10B981;">AI:</b><br>
                <div style="color: #1F2937;">{formatted_message}</div>
            </div>
            '''

        self.chat_history.setHtml(current + formatted)

        # Scroll to bottom
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters to prevent injection."""
        import html
        return html.escape(text)

    def _format_ai_message(self, message: str) -> str:
        """Format AI message with code block styling."""
        import re

        # First, extract and preserve code blocks before escaping
        code_blocks = []
        def save_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks.append((lang, code))
            return placeholder

        # Save code blocks with placeholder
        message = re.sub(r'```(\w*)\n?(.*?)```', save_code_block, message, flags=re.DOTALL)

        # Save inline code
        inline_codes = []
        def save_inline_code(match):
            code = match.group(1)
            placeholder = f"__INLINE_CODE_{len(inline_codes)}__"
            inline_codes.append(code)
            return placeholder

        message = re.sub(r'`([^`]+)`', save_inline_code, message)

        # Now escape HTML for the rest
        message = self._escape_html(message)

        # Convert markdown bold: **text**
        message = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', message)

        # Restore code blocks with dark theme styling
        for i, (lang, code) in enumerate(code_blocks):
            escaped_code = self._escape_html(code.strip())
            lang_label = f"<b style='color: #9CA3AF;'>{lang.upper()}</b><br>" if lang else ""
            code_html = f'''<table width="100%" cellpadding="0" cellspacing="0" style="margin: 8px 0;"><tr><td style="background-color: #1F2937; padding: 12px; border-radius: 6px;"><pre style="margin: 0; color: #E5E7EB; font-family: monospace; white-space: pre-wrap; font-size: 12px;">{lang_label}{escaped_code}</pre></td></tr></table>'''
            message = message.replace(f"__CODE_BLOCK_{i}__", code_html)

        # Restore inline code
        for i, code in enumerate(inline_codes):
            escaped_code = self._escape_html(code)
            inline_html = f'<code style="background-color: #E5E7EB; padding: 2px 4px; font-family: monospace;">{escaped_code}</code>'
            message = message.replace(f"__INLINE_CODE_{i}__", inline_html)

        # Convert line breaks
        message = message.replace('\n', '<br>')

        return message

    def _clear_chat(self):
        """Clear the chat display and conversation memory."""
        self.chat_history.clear()
        self.conversation_history.clear()  # Clear conversation memory
        self.last_chat_response = ""
        self.copy_chat_btn.setEnabled(False)
        self.statusbar.showMessage("Chat and conversation memory cleared", 2000)

    def _copy_chat_response(self):
        """Copy the last AI response to clipboard."""
        if self.last_chat_response:
            QApplication.clipboard().setText(self.last_chat_response)
            self.statusbar.showMessage("Response copied to clipboard!", 2000)

    # ===== Document Library / RAG Methods =====

    def _toggle_doc_library(self):
        """Toggle document library panel visibility."""
        if self.doc_lib_toggle.isChecked():
            self.doc_list_widget.show()
            self.doc_lib_toggle.setText("📚 Document Library ▼")
            self._refresh_doc_list()
        else:
            self.doc_list_widget.hide()
            self.doc_lib_toggle.setText("📚 Document Library")

    def _toggle_rag_mode(self):
        """Toggle RAG mode on/off."""
        if self.rag_toggle_btn.isChecked():
            # Check if there are documents
            if self.rag_engine is None or self.rag_engine.is_empty:
                self.rag_toggle_btn.setChecked(False)
                QMessageBox.information(
                    self, "No Documents",
                    "Please add documents first before enabling RAG mode.\n\n"
                    "Click '+ Add' to upload PDFs, Word docs, or text files."
                )
                return

            self.rag_enabled = True
            self.rag_toggle_btn.setText("🔍 RAG: ON")
            self.chat_history.setPlaceholderText(
                "📚 RAG Mode Active\n\n"
                "Ask questions about your documents:\n"
                "• What is the main topic of this document?\n"
                "• Summarize the key points\n"
                "• What does it say about [topic]?\n"
                "• Find information about [subject]"
            )
            self.statusbar.showMessage("RAG enabled - AI will answer from your documents", 3000)
        else:
            self.rag_enabled = False
            self.rag_toggle_btn.setText("🔍 RAG: OFF")
            self.chat_history.setPlaceholderText(
                "💬 Regular Chat Mode\n\n"
                "Chat freely with AI:\n"
                "• Write code or explain concepts\n"
                "• Get creative writing help\n"
                "• Answer general questions\n\n"
                "Tip: Enable RAG to answer from documents"
            )
            self.statusbar.showMessage("RAG disabled - Regular AI chat mode", 2000)

    def _init_rag_engine(self):
        """Initialize RAG engine (lazy loading)."""
        if self.rag_engine is not None:
            return True

        try:
            from src.rag_engine import RAGEngine, check_rag_dependencies

            # Check dependencies first
            deps_ok, deps_msg = check_rag_dependencies()
            if not deps_ok:
                QMessageBox.warning(
                    self, "Missing Dependencies",
                    f"RAG feature requires additional packages:\n\n{deps_msg}"
                )
                return False

            self.rag_engine = RAGEngine()
            return True

        except Exception as e:
            QMessageBox.critical(self, "RAG Initialization Error", f"Failed to initialize RAG engine:\n{str(e)}")
            return False

    def _add_document_to_library(self):
        """Add a document to the RAG library."""
        from PyQt6.QtWidgets import QFileDialog

        if not self._init_rag_engine():
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Add Document to Library",
            "",
            "Documents (*.txt *.md *.docx *.pdf);;PDF Files (*.pdf);;Word Documents (*.docx);;Text Files (*.txt);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Read document content
            content = self._read_document(file_path)
            if not content:
                QMessageBox.warning(self, "Empty Document", "The document appears to be empty.")
                return

            # Show progress
            self.statusbar.showMessage(f"Processing: {file_path.split('/')[-1]}...")
            self.progress_bar.setRange(0, 0)

            # Add to RAG engine
            def progress_cb(msg):
                self.statusbar.showMessage(msg)

            doc = self.rag_engine.add_document(file_path, content, progress_cb)

            # Update UI
            self.progress_bar.setRange(0, 100)
            self._refresh_doc_list()

            # Show library if hidden
            if not self.doc_lib_toggle.isChecked():
                self.doc_lib_toggle.setChecked(True)
                self._toggle_doc_library()

            self.statusbar.showMessage(f"Added: {doc.name} ({doc.chunk_count} chunks)", 3000)

            # Update placeholder
            self.chat_history.setPlaceholderText(
                f"📚 {self.rag_engine.document_count} document(s) in library\n\n"
                "Ask questions about your documents:\n"
                "• What is the main topic?\n"
                "• Summarize the key points\n"
                "• What does it say about [topic]?\n"
                "• Compare information across documents"
            )

        except Exception as e:
            self.progress_bar.setRange(0, 100)
            QMessageBox.critical(self, "Error Adding Document", f"Failed to add document:\n{str(e)}")

    def _refresh_doc_list(self):
        """Refresh the document list UI."""
        if self.rag_engine is None:
            self.doc_count_label.setText("0 documents")
            self.doc_list_empty.show()
            return

        docs = self.rag_engine.get_documents()
        self.doc_count_label.setText(f"{len(docs)} document{'s' if len(docs) != 1 else ''}")

        # Clear existing items (except empty label)
        for i in reversed(range(self.doc_list_layout.count())):
            item = self.doc_list_layout.itemAt(i)
            if item and item.widget() and item.widget() != self.doc_list_empty:
                item.widget().deleteLater()

        if not docs:
            self.doc_list_empty.show()
            # Disable RAG toggle if no documents
            self.rag_toggle_btn.setChecked(False)
            self.rag_toggle_btn.setText("🔍 RAG: OFF")
            self.rag_enabled = False
        else:
            self.doc_list_empty.hide()
            # Auto-enable RAG when documents are present
            if not self.rag_enabled:
                self.rag_toggle_btn.setChecked(True)
                self.rag_toggle_btn.setText("🔍 RAG: ON")
                self.rag_enabled = True

            for doc in docs:
                doc_item = QFrame()
                doc_item.setStyleSheet("""
                    QFrame {
                        background-color: #F9FAFB;
                        border: 1px solid #E5E7EB;
                        border-radius: 4px;
                        padding: 4px;
                    }
                    QFrame:hover {
                        background-color: #F3F4F6;
                    }
                """)
                item_layout = QHBoxLayout(doc_item)
                item_layout.setContentsMargins(8, 6, 8, 6)
                item_layout.setSpacing(8)

                # Document icon based on extension
                icon = "📄"
                if doc.name.endswith('.pdf'):
                    icon = "📕"
                elif doc.name.endswith('.docx'):
                    icon = "📘"
                elif doc.name.endswith('.md'):
                    icon = "📝"

                icon_label = QLabel(icon)
                item_layout.addWidget(icon_label)

                name_label = QLabel(doc.name)
                name_label.setStyleSheet("color: #1F2937; font-size: 12px;")
                item_layout.addWidget(name_label)

                chunks_label = QLabel(f"{doc.chunk_count} chunks")
                chunks_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
                item_layout.addWidget(chunks_label)

                item_layout.addStretch()

                remove_btn = QPushButton("Remove")
                remove_btn.setFixedHeight(24)
                remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                remove_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FEE2E2;
                        color: #DC2626;
                        border: none;
                        border-radius: 4px;
                        padding: 2px 8px;
                        font-size: 11px;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #EF4444;
                        color: white;
                    }
                """)
                remove_btn.clicked.connect(lambda checked, d=doc: self._remove_document_from_library(d.id, d.name))
                item_layout.addWidget(remove_btn)

                self.doc_list_layout.addWidget(doc_item)

    def _remove_document_from_library(self, doc_id: str, doc_name: str):
        """Remove a document and its embeddings from the RAG library."""
        reply = QMessageBox.question(
            self, "Remove Document",
            f"Remove '{doc_name}' from the library?\n\nThis will delete all embeddings for this document.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.rag_engine.remove_document(doc_id):
                self._refresh_doc_list()
                self.statusbar.showMessage(f"Removed: {doc_name} (embeddings deleted)", 3000)

                # Update placeholder if no documents left
                if self.rag_engine.is_empty:
                    self._reset_chat_placeholder()

    def _clear_all_documents(self):
        """Clear all documents and embeddings from the RAG library."""
        if self.rag_engine is None or self.rag_engine.is_empty:
            self.statusbar.showMessage("No documents to clear", 2000)
            return

        doc_count = self.rag_engine.document_count
        reply = QMessageBox.question(
            self, "Clear All Documents",
            f"Remove all {doc_count} document(s) from the library?\n\n"
            "This will delete all embeddings and cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.rag_engine.clear_all()
            self._refresh_doc_list()
            self._reset_chat_placeholder()
            self.statusbar.showMessage(f"Cleared {doc_count} document(s) and all embeddings", 3000)

    def _reset_chat_placeholder(self):
        """Reset chat placeholder to default state."""
        self.chat_history.setPlaceholderText(
            "💬 AI Chat - Two Modes Available:\n\n"
            "📚 RAG Mode (Document Q&A):\n"
            "   Add documents above, then ask questions about them.\n"
            "   AI will only answer from your documents.\n\n"
            "💭 Regular Mode:\n"
            "   Chat freely - write code, explain concepts, get help.\n\n"
            "Toggle RAG on/off using the button above."
        )

    def _export_output(self, format_type: str):
        """Export output to file."""
        text = self.output_text.toPlainText()
        if not text:
            QMessageBox.warning(self, "Export", "No output to export.")
            return

        if format_type == "txt":
            filter_str = "Text Files (*.txt)"
            ext = ".txt"
        elif format_type == "md":
            filter_str = "Markdown Files (*.md)"
            ext = ".md"
        elif format_type == "docx":
            if not is_docx_available():
                QMessageBox.warning(
                    self, "Export",
                    "DOCX export requires python-docx.\n"
                    "Install with: pip install python-docx"
                )
                return
            filter_str = "Word Documents (*.docx)"
            ext = ".docx"
        else:
            filter_str = "Text Files (*.txt)"
            ext = ".txt"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Output", f"humanized_output{ext}", filter_str
        )

        if file_path:
            metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'mode': self.mode_combo.currentText(),
                'creativity': self.creativity_slider.value()
            }

            result = self.export_manager.export(
                text, file_path, format_type,
                title="Enhanced Text",
                metadata=metadata
            )

            if result.success:
                self.statusbar.showMessage(f"Exported to {file_path}", 3000)
            else:
                QMessageBox.critical(self, "Export Error", result.message)

    def _humanize_text(self):
        """Start text humanization or chat generation."""
        text = self.input_text.toPlainText()
        if not text.strip():
            self.statusbar.showMessage("Please enter some text first", 3000)
            return
        if not self.model_loaded:
            self.statusbar.showMessage("Please select and load a model first", 3000)
            QMessageBox.information(
                self, "Model Required",
                "Please select an AI model from the dropdown first.\n\n"
                "The model will download automatically on first use."
            )
            return

        # Store original for comparison
        self.original_text = text

        # Get mode (use itemData which contains the actual mode key)
        mode = self.mode_combo.currentData() or "professional"

        # Update UI - hide all action buttons during processing
        self.humanize_btn.hide()
        self.summarize_btn.hide()
        self.expand_btn.hide()
        self.cancel_text_btn.show()
        self.output_text.clear()
        self.progress_bar.setRange(0, 0)

        # Enhancement mode - transform text according to selected style
        self.humanizer.creativity_level = self.creativity_slider.value()
        self.humanizer.set_style(mode)  # Set style on humanizer for mode-specific prompts

        self.stage_label.setText(f"Transforming to {mode} style...")
        self.text_worker = TextHumanizeWorker(self.humanizer, text)
        self.text_worker.progress.connect(lambda msg: self.stage_label.setText(msg))
        self.text_worker.finished.connect(self._on_text_humanized)
        self.text_worker.start()

    def _on_text_humanized(self, success: bool, result: str):
        """Handle humanization completion."""
        self.cancel_text_btn.hide()
        self.humanize_btn.show()
        self.summarize_btn.show()
        self.expand_btn.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        if success and result:
            # Display result (with diff highlighting if enabled)
            if self.diff_view_enabled and self.original_text:
                self._show_diff_output(self.original_text, result)
            else:
                self.output_text.setPlainText(result)

            word_count = len(result.split())
            self.output_word_count.setText(f"{word_count} words")
            self.copy_btn.setEnabled(True)
            self.refine_btn.setEnabled(True)  # Enable refinement
            self.stage_label.setText("Complete!")

            # Save to history
            mode = self.mode_combo.currentText()
            creativity = self.creativity_slider.value()
            add_to_history(
                self.original_text, result, mode, creativity,
                None, None
            )
        else:
            self.stage_label.setText("Failed" if not result else result)
            if result and result.startswith("Error"):
                QMessageBox.warning(self, "Error", result)

    def _cancel_text_humanize(self):
        """Cancel text humanization."""
        if self.text_worker and self.text_worker.isRunning():
            self.text_worker.cancel()
            self.text_worker.wait(1000)

        self.cancel_text_btn.hide()
        self.humanize_btn.show()
        self.summarize_btn.show()
        self.expand_btn.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Cancelled")

    def _summarize_text(self):
        """Summarize the input text."""
        text = self.input_text.toPlainText()
        if not text.strip():
            self.statusbar.showMessage("Please enter some text first", 3000)
            return
        if not self.model_loaded:
            self.statusbar.showMessage("Please select and load a model first", 3000)
            QMessageBox.information(
                self, "Model Required",
                "Please select an AI model from the dropdown first.\n\n"
                "The model will download automatically on first use."
            )
            return

        # Store original for comparison
        self.original_text = text

        # Update UI - hide all action buttons during processing
        self.humanize_btn.hide()
        self.summarize_btn.hide()
        self.expand_btn.hide()
        self.cancel_text_btn.show()
        self.output_text.clear()
        self.progress_bar.setRange(0, 0)

        # Set mode to summarize
        self.humanizer.creativity_level = 30  # Lower creativity for summarization
        self.humanizer.set_style("summarize")

        self.stage_label.setText("Summarizing text...")
        self.text_worker = TextHumanizeWorker(self.humanizer, text)
        self.text_worker.progress.connect(lambda msg: self.stage_label.setText(msg))
        self.text_worker.finished.connect(self._on_summarize_finished)
        self.text_worker.start()

    def _on_summarize_finished(self, success: bool, result: str):
        """Handle summarization completion."""
        self.cancel_text_btn.hide()
        self.humanize_btn.show()
        self.summarize_btn.show()
        self.expand_btn.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        if success and result:
            self.output_text.setPlainText(result)
            word_count = len(result.split())
            original_word_count = len(self.original_text.split())
            reduction = int((1 - word_count / original_word_count) * 100) if original_word_count > 0 else 0
            self.output_word_count.setText(f"{word_count} words ({reduction}% shorter)")
            self.copy_btn.setEnabled(True)
            self.refine_btn.setEnabled(True)
            self.stage_label.setText("Summary complete!")

            # Save to history
            add_to_history(
                self.original_text, result, "Summarize", 30,
                None, None
            )
        else:
            self.stage_label.setText("Failed" if not result else result)
            if result and result.startswith("Error"):
                QMessageBox.warning(self, "Error", result)

    def _expand_text(self):
        """Expand the input text with more detail."""
        text = self.input_text.toPlainText()
        if not text.strip():
            self.statusbar.showMessage("Please enter some text first", 3000)
            return
        if not self.model_loaded:
            self.statusbar.showMessage("Please select and load a model first", 3000)
            QMessageBox.information(
                self, "Model Required",
                "Please select an AI model from the dropdown first.\n\n"
                "The model will download automatically on first use."
            )
            return

        # Store original for comparison
        self.original_text = text

        # Update UI - hide all action buttons during processing
        self.humanize_btn.hide()
        self.summarize_btn.hide()
        self.expand_btn.hide()
        self.cancel_text_btn.show()
        self.output_text.clear()
        self.progress_bar.setRange(0, 0)

        # Set mode to expand
        self.humanizer.creativity_level = 60  # Higher creativity for expansion
        self.humanizer.set_style("expand")

        self.stage_label.setText("Expanding text...")
        self.text_worker = TextHumanizeWorker(self.humanizer, text)
        self.text_worker.progress.connect(lambda msg: self.stage_label.setText(msg))
        self.text_worker.finished.connect(self._on_expand_finished)
        self.text_worker.start()

    def _on_expand_finished(self, success: bool, result: str):
        """Handle expansion completion."""
        self.cancel_text_btn.hide()
        self.humanize_btn.show()
        self.summarize_btn.show()
        self.expand_btn.show()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        if success and result:
            self.output_text.setPlainText(result)
            word_count = len(result.split())
            original_word_count = len(self.original_text.split())
            expansion = int((word_count / original_word_count - 1) * 100) if original_word_count > 0 else 0
            self.output_word_count.setText(f"{word_count} words ({expansion}% longer)")
            self.copy_btn.setEnabled(True)
            self.refine_btn.setEnabled(True)
            self.stage_label.setText("Expansion complete!")

            # Save to history
            add_to_history(
                self.original_text, result, "Expand", 60,
                None, None
            )
        else:
            self.stage_label.setText("Failed" if not result else result)
            if result and result.startswith("Error"):
                QMessageBox.warning(self, "Error", result)

    def _on_model_selected(self, model_id: str, model_path: str):
        """Handle model selection from dropdown."""
        self._load_model(model_path)
        # Save selected model ID
        self.settings.selected_model_id = model_id
        save_settings()

    def browse_model(self):
        """Browse for model file (legacy support)."""
        # Start from last model directory or home
        start_dir = ""
        if self.settings.last_model_path:
            start_dir = str(Path(self.settings.last_model_path).parent)
        if not start_dir or not Path(start_dir).exists():
            start_dir = str(Path.home())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File (GGUF format)",
            start_dir,
            "All Files (*);;GGUF Models (*.gguf)"
        )
        if file_path:
            self._load_model(file_path)

    def _load_model(self, model_path: str):
        """Load the LLM model."""
        self.stage_label.setText("Loading model...")
        self.progress_bar.setRange(0, 0)

        self.model_worker = ModelLoaderWorker(self.paraphraser, model_path)
        self.model_worker.progress.connect(lambda msg: self.statusbar.showMessage(msg))
        self.model_worker.finished.connect(self._on_model_loaded)
        self.model_worker.start()

    def _on_model_loaded(self, success: bool, message: str):
        """Handle model load completion."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        if success:
            self.model_loaded = True
            self.stage_label.setText("Ready")

            # Share model with humanizer
            self.humanizer.model = self.paraphraser.model
            self.humanizer.is_loaded = True
            self.humanizer.config.model_path = self.paraphraser.config.model_path

            # Update button state
            self._update_humanize_button()

            # Update settings
            self.settings.last_model_path = self.paraphraser.config.model_path
            save_settings()
        else:
            self.stage_label.setText(message)
            QMessageBox.critical(self, "Model Error", message)

    # ===== New Feature Methods =====

    def _show_diff_output(self, original: str, humanized: str):
        """Show output with diff highlighting."""
        diff_result = compare_texts(original, humanized)
        _, modified_html = self.diff_viewer.to_html(diff_result)

        # Set as HTML
        self.output_text.setHtml(modified_html)

        # Show change summary in status
        summary = self.diff_viewer.get_change_summary(diff_result)
        self.statusbar.showMessage(f"Changes: {summary}", 5000)

    def _toggle_diff_view(self):
        """Toggle diff view mode."""
        self.diff_view_enabled = self.diff_view_action.isChecked()

        # Re-apply to current output if we have both texts
        if self.original_text and self.output_text.toPlainText():
            if self.diff_view_enabled:
                self._show_diff_output(self.original_text, self.output_text.toPlainText())
            else:
                # Remove HTML formatting
                plain = self.output_text.toPlainText()
                self.output_text.setPlainText(plain)

    def _toggle_highlighting(self):
        """Toggle AI sentence highlighting."""
        self.highlight_enabled = self.highlight_action.isChecked()

    def _show_history(self):
        """Show history dialog."""
        from ui.history_dialog import show_history_dialog
        result = show_history_dialog(self)
        if result:
            original, humanized = result
            self.input_text.setPlainText(original)
            self.output_text.setPlainText(humanized)
            self.original_text = original
            self.copy_btn.setEnabled(True)

    def _show_onboarding(self):
        """Show onboarding tutorial."""
        from ui.onboarding_dialog import show_onboarding
        show_onboarding(self)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About LocalWrite",
            "<h2>LocalWrite 1.0</h2>"
            "<p><b>Private AI Writing Assistant</b></p>"
            "<p>Your writing, enhanced locally. No cloud. No compromise.</p>"
            "<p><b>Privacy Promise:</b></p>"
            "<ul>"
            "<li>100% offline - your words never leave your device</li>"
            "<li>No accounts, no API keys, no tracking</li>"
            "<li>Open source (MIT License)</li>"
            "</ul>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>5 curated AI models to choose from</li>"
            "<li>5 writing enhancement modes</li>"
            "<li>Real-time writing statistics</li>"
            "<li>AI Chat assistant</li>"
            "<li>Dark mode support</li>"
            "</ul>"
            "<p style='color: #6B7280;'>A Svetozar Technologies project</p>"
        )

    def _export_comparison(self):
        """Export both original and humanized text."""
        if not self.original_text or not self.output_text.toPlainText():
            QMessageBox.warning(self, "Export", "Need both original and humanized text to export comparison.")
            return

        formats = "Text Files (*.txt);;Markdown Files (*.md)"
        if is_docx_available():
            formats += ";;Word Documents (*.docx)"

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Comparison", "comparison", formats
        )

        if file_path:
            # Determine format from filter or extension
            if "docx" in selected_filter.lower() or file_path.endswith('.docx'):
                fmt = 'docx'
            elif "md" in selected_filter.lower() or file_path.endswith('.md'):
                fmt = 'md'
            else:
                fmt = 'txt'

            metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'mode': self.mode_combo.currentText(),
                'creativity': self.creativity_slider.value()
            }

            result = self.export_manager.export_comparison(
                self.original_text,
                self.output_text.toPlainText(),
                file_path,
                fmt,
                metadata
            )

            if result.success:
                self.statusbar.showMessage(f"Exported to {file_path}", 3000)
            else:
                QMessageBox.critical(self, "Export Error", result.message)

    def _try_auto_load_model(self):
        """Try to auto-load the last used model or any installed model."""
        preferred = getattr(self.settings, 'selected_model_id', None)
        if self.model_selector.try_auto_load(preferred):
            self.statusbar.showMessage("Loading model...", 2000)

    def _quick_refine(self, instruction: str):
        """Apply a quick refinement to the output."""
        self._refine_output(instruction)

    def _custom_refine(self):
        """Apply custom refinement from user input."""
        instruction = self.refine_input.toPlainText().strip()
        if instruction:
            self._refine_output(instruction)
            self.refine_input.clear()

    def _refine_output(self, instruction: str):
        """Refine the output based on user instruction."""
        text = self.output_text.toPlainText()
        if not text.strip():
            self.statusbar.showMessage("No text to refine", 2000)
            return

        if not self.model_loaded:
            self.statusbar.showMessage("Model not loaded", 2000)
            return

        # Disable refinement controls
        self.refine_btn.setEnabled(False)
        for btn in self.quick_btns:
            btn.setEnabled(False)
        self.refine_input.setEnabled(False)

        self.stage_label.setText("Refining...")
        self.progress_bar.setRange(0, 0)

        # Start refine worker
        self.refine_worker = RefineWorker(self.humanizer.model, text, instruction)
        self.refine_worker.progress.connect(lambda msg: self.stage_label.setText(msg))
        self.refine_worker.finished.connect(self._on_refine_complete)
        self.refine_worker.start()

    def _on_refine_complete(self, success: bool, result: str):
        """Handle refinement completion."""
        # Re-enable refinement controls
        self.refine_btn.setEnabled(True)
        for btn in self.quick_btns:
            btn.setEnabled(True)
        self.refine_input.setEnabled(True)

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        if success and result:
            self.output_text.setPlainText(result)
            word_count = len(result.split())
            self.output_word_count.setText(f"{word_count} words")
            self.stage_label.setText("Refined!")
            self.statusbar.showMessage("Text refined successfully", 3000)
        else:
            self.stage_label.setText("Refinement failed")
            if result:
                self.statusbar.showMessage(result, 3000)

    def _check_first_run(self):
        """Check if this is first run and show onboarding."""
        if self.settings_manager.is_first_run():
            from ui.onboarding_dialog import show_onboarding
            show_onboarding(self)
            self.settings_manager.mark_first_run_complete()

    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts for specific widgets."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        if obj == self.chat_input and event.type() == QEvent.Type.KeyPress:
            key_event = event
            # Enter to send message (Shift+Enter for new line)
            if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    # Shift+Enter: insert new line
                    return False  # Let the default handler insert newline
                else:
                    # Enter: send message
                    self._send_chat_message()
                    return True

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """Handle window close."""
        # Save window geometry
        self.settings_manager.save_window_geometry(
            self.width(), self.height(),
            self.x(), self.y(),
            self.isMaximized()
        )
        save_settings()

        # Check for running workers
        workers = [self.model_worker, self.text_worker, self.refine_worker, self.chat_worker]
        running = [w for w in workers if w and w.isRunning()]

        if running:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        # Cleanup workers
        for worker in running:
            if hasattr(worker, 'cancel'):
                worker.cancel()
            worker.quit()
            worker.wait(1000)

        # Unload models
        if self.paraphraser:
            self.paraphraser.unload_model()
        if self.humanizer:
            self.humanizer.unload_model()

        event.accept()
