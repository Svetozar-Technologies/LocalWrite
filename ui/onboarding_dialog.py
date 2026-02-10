"""
Onboarding Dialog - First-run welcome and privacy-focused tutorial.
Shows new users how to use LocalWrite's features with emphasis on privacy.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QWidget, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class OnboardingPage(QFrame):
    """A single page in the onboarding flow."""

    def __init__(self, title: str, description: str, icon_text: str = "",
                 tips: list = None, parent=None):
        super().__init__(parent)
        self.setObjectName("onboardingPage")

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Icon placeholder
        if icon_text:
            icon_label = QLabel(icon_text)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_font = QFont()
            icon_font.setPointSize(48)
            icon_label.setFont(icon_font)
            layout.addWidget(icon_label)

        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #6B7280; font-size: 14px;")
        layout.addWidget(desc_label)

        # Tips list
        if tips:
            tips_frame = QFrame()
            tips_frame.setStyleSheet("""
                QFrame {
                    background-color: #F3F4F6;
                    border-radius: 8px;
                    padding: 16px;
                }
            """)
            tips_layout = QVBoxLayout(tips_frame)

            for tip in tips:
                tip_label = QLabel(f"  {tip}")
                tip_label.setStyleSheet("color: #374151; font-size: 13px;")
                tips_layout.addWidget(tip_label)

            layout.addWidget(tips_frame)

        layout.addStretch()


class OnboardingDialog(QDialog):
    """Multi-page onboarding tutorial dialog with privacy focus."""

    completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to LocalWrite")
        self.setMinimumSize(600, 520)
        self.setModal(True)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget for pages
        self.stack = QStackedWidget()
        self._create_pages()
        layout.addWidget(self.stack)

        # Navigation bar
        nav_bar = QFrame()
        nav_bar.setStyleSheet("background-color: #F9FAFB; border-top: 1px solid #E5E7EB;")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(20, 16, 20, 16)

        # Don't show again checkbox
        self.dont_show_check = QCheckBox("Don't show this again")
        nav_layout.addWidget(self.dont_show_check)

        nav_layout.addStretch()

        # Page indicators
        self.indicators = []
        indicator_layout = QHBoxLayout()
        indicator_layout.setSpacing(8)
        for i in range(self.stack.count()):
            indicator = QLabel()
            indicator.setFixedSize(8, 8)
            indicator.setStyleSheet("""
                background-color: #D1D5DB;
                border-radius: 4px;
            """)
            self.indicators.append(indicator)
            indicator_layout.addWidget(indicator)
        nav_layout.addLayout(indicator_layout)
        self._update_indicators()

        nav_layout.addStretch()

        # Back button
        self.back_btn = QPushButton("Back")
        self.back_btn.setObjectName("secondaryButton")
        self.back_btn.setFixedWidth(80)
        self.back_btn.clicked.connect(self._go_back)
        self.back_btn.setEnabled(False)
        nav_layout.addWidget(self.back_btn)

        # Next/Finish button
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("primaryButton")
        self.next_btn.setFixedWidth(100)
        self.next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self.next_btn)

        layout.addWidget(nav_bar)

    def _create_pages(self):
        """Create all onboarding pages with privacy focus."""

        # Page 1: Welcome
        page1 = OnboardingPage(
            title="Welcome to LocalWrite",
            description="Your Private AI Writing Assistant\n\n"
                       "Enhance your writing with powerful AI that runs\n"
                       "entirely on your device. No cloud. No compromise.",
            icon_text="",
            tips=[
                "100% Offline - Your writing never leaves your device",
                "No Accounts - No sign-up, no API keys, no tracking",
                "Free Forever - MIT licensed, open source",
                "Choose Your AI - 5 curated models for different tasks"
            ]
        )
        self.stack.addWidget(page1)

        # Page 2: Select Model
        page2 = OnboardingPage(
            title="Choose Your AI Model",
            description="Select the best AI for your writing style.\nModels download automatically on first use.",
            icon_text="",
            tips=[
                "Qwen 2.5 7B - Best for general writing (Recommended)",
                "Gemma 2 9B - Great for creative and expressive writing",
                "Llama 3.1 8B - Excellent for professional content",
                "Llama 3.2 3B - Fast and lightweight for quick edits",
                "Mistral 7B - Balanced speed and quality"
            ]
        )
        self.stack.addWidget(page2)

        # Page 3: Privacy Promise
        page3 = OnboardingPage(
            title="Your Privacy, Protected",
            description="We believe your words belong to you.\nHere's our privacy promise:",
            icon_text="",
            tips=[
                "All AI processing happens locally on your Mac",
                "No internet connection required after model download",
                "No telemetry, analytics, or usage tracking",
                "No accounts or registration ever required",
                "Open source code - fully auditable on GitHub"
            ]
        )
        self.stack.addWidget(page3)

        # Page 4: Writing Enhancement
        page4 = OnboardingPage(
            title="Enhance Your Writing",
            description="5 writing modes plus quick actions.",
            icon_text="",
            tips=[
                "Professional - Business and formal documents",
                "Conversational - Friendly, natural tone",
                "Scholarly - Academic and research writing",
                "Creative - Vivid, engaging storytelling",
                "Concise - Shorter, punchier text",
                "Summarize button - Condense to key points",
                "Expand button - Add examples and details"
            ]
        )
        self.stack.addWidget(page4)

        # Page 5: Document Features
        page5 = OnboardingPage(
            title="Chat with Your Documents",
            description="Upload documents and ask questions about them.",
            icon_text="",
            tips=[
                "Upload TXT, Markdown, DOCX, or PDF files",
                "Build a Document Library with multiple files",
                "Toggle RAG mode to chat about your documents",
                "AI answers based on your uploaded content",
                "Great for studying books, papers, or notes"
            ]
        )
        self.stack.addWidget(page5)

        # Page 6: Keyboard Shortcuts
        page6 = OnboardingPage(
            title="Keyboard Shortcuts",
            description="Work faster with these shortcuts.",
            icon_text="",
            tips=[
                "Cmd+Enter: Enhance text",
                "Cmd+Shift+C: Copy output",
                "Enter: Send chat message",
                "View > Dark Mode: Toggle theme"
            ]
        )
        self.stack.addWidget(page6)

        # Page 7: Get Started
        page7 = OnboardingPage(
            title="Start Writing Privately",
            description="You're all set! Here's how to begin:",
            icon_text="",
            tips=[
                "1. Select an AI model from the dropdown",
                "2. Paste or type your text in the input area",
                "3. Choose your preferred writing mode",
                "4. Click 'Enhance Writing' and enjoy!",
                "5. Try AI Chat to ask questions or upload documents"
            ]
        )
        self.stack.addWidget(page7)

    def _update_indicators(self):
        """Update page indicator dots."""
        current = self.stack.currentIndex()
        for i, indicator in enumerate(self.indicators):
            if i == current:
                indicator.setStyleSheet("""
                    background-color: #7C3AED;
                    border-radius: 4px;
                """)
            else:
                indicator.setStyleSheet("""
                    background-color: #D1D5DB;
                    border-radius: 4px;
                """)

    def _go_back(self):
        """Go to previous page."""
        current = self.stack.currentIndex()
        if current > 0:
            self.stack.setCurrentIndex(current - 1)
            self._update_nav()

    def _go_next(self):
        """Go to next page or finish."""
        current = self.stack.currentIndex()
        if current < self.stack.count() - 1:
            self.stack.setCurrentIndex(current + 1)
            self._update_nav()
        else:
            # Finish
            self.completed.emit()
            self.accept()

    def _update_nav(self):
        """Update navigation buttons."""
        current = self.stack.currentIndex()
        self.back_btn.setEnabled(current > 0)

        if current == self.stack.count() - 1:
            self.next_btn.setText("Get Started")
        else:
            self.next_btn.setText("Next")

        self._update_indicators()

    def should_show_again(self) -> bool:
        """Check if dialog should be shown on next launch."""
        return not self.dont_show_check.isChecked()


def show_onboarding(parent=None) -> bool:
    """
    Show onboarding dialog.

    Returns:
        True if user completed onboarding, False if cancelled
    """
    dialog = OnboardingDialog(parent)
    result = dialog.exec()
    return result == QDialog.DialogCode.Accepted
