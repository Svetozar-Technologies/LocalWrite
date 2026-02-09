# LocalWrite

**Private AI Writing Assistant - 100% Offline**

Your writing deserves privacy. LocalWrite enhances your text using powerful local AI models that run entirely on your Mac. No cloud. No accounts. No compromise.

## Why LocalWrite?

- **100% Offline** - Your writing never leaves your device
- **No Accounts** - No sign-up, no API keys, no tracking
- **Free Forever** - MIT licensed, open source
- **Choose Your AI** - Multiple models optimized for different writing tasks

## Features

- **Smart Model Selection** - Choose from 5 curated AI models with recommendations
- **11 Writing Modes** - Professional, Creative, Conversational, Scholarly, and more
- **Real-time Statistics** - Word count, reading time, grade level, tone
- **PDF Enhancement** - Improve documents while preserving layout
- **Batch Processing** - Enhance multiple files at once
- **Dark Mode** - Comfortable writing day and night
- **Export Options** - TXT, Markdown, DOCX

## Available AI Models

| Model | Best For | Size |
|-------|----------|------|
| Qwen 2.5 7B | General writing, essays, articles | 4.7 GB |
| Gemma 2 9B | Creative writing, storytelling | 5.8 GB |
| Llama 3.1 8B | Professional, clean prose | 4.9 GB |
| Llama 3.2 3B | Quick edits, low memory | 2.0 GB |
| Mistral 7B | Balanced speed & quality | 4.4 GB |

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Svetozar-Technologies/LocalWrite.git
   cd localwrite
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run LocalWrite:
   ```bash
   python main.py
   ```

5. Select an AI model from the dropdown (auto-downloads on first use)

6. Paste your text and click "Enhance Writing"

## Requirements

- macOS 11.0 or later
- Python 3.9+
- 8GB RAM minimum (16GB recommended for larger models)
- 2-6 GB disk space per model

## Privacy Promise

LocalWrite is designed with privacy as the foundation:

- **All AI processing happens on your device** - No data is ever sent to the cloud
- **No internet required** - After the one-time model download, works completely offline
- **No telemetry or analytics** - We don't track anything you do
- **No accounts or registration** - Just download and use
- **Open source** - Full transparency, audit the code yourself

## Project Structure

```
localwrite/
├── main.py                 # Application entry point
├── ui/                     # User interface components
│   ├── main_window.py      # Main application window
│   ├── model_selector.py   # AI model selection widget
│   └── onboarding_dialog.py # First-run tutorial
├── src/                    # Core functionality
│   ├── model_registry.py   # Curated AI models
│   ├── model_downloader.py # Auto-download from Hugging Face
│   ├── humanizer_v2.py     # Writing enhancement engine
│   ├── paraphraser.py      # Text rewriting with modes
│   ├── ai_detector.py      # Writing analysis
│   └── text_analyzer.py    # Statistics and metrics
├── resources/              # Stylesheets and assets
│   ├── styles.qss          # Light theme
│   └── styles_dark.qss     # Dark theme
└── docs/                   # GitHub Pages landing site
```

## Writing Enhancement Modes

| Mode | Description |
|------|-------------|
| Enhance | Improve overall clarity and flow |
| Professional | Business and formal tone |
| Conversational | Friendly, natural language |
| Scholarly | Academic and research style |
| Expressive | Creative and storytelling |
| Concise | Shorter, punchier text |
| Elaborate | More detailed explanations |
| Smooth | Improved flow and readability |
| Plain | Simple, easy-to-understand |
| Precise | Technical accuracy |
| Standard | Balanced improvements |

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## License

MIT License - Free to use, modify, and distribute.

---

**Built with privacy in mind.**

A [Svetozar Technologies](https://github.com/Svetozar-Technologies) project.
