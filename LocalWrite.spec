# -*- mode: python ; coding: utf-8 -*-
"""
LocalWrite macOS Application Spec File

Build with: pyinstaller LocalWrite.spec --clean

Optimized for:
- Security: Hardened runtime, code signing ready
- Size: Excludes unnecessary modules
- Performance: UPX compression
"""

import sys
import site
from pathlib import Path

# PyInstaller utilities for collecting package data
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Get the current directory
ROOT = Path(SPECPATH)

# Find llama_cpp library path
def find_llama_libs():
    """Find llama_cpp shared libraries."""
    import llama_cpp
    llama_path = Path(llama_cpp.__file__).parent
    lib_path = llama_path / 'lib'
    binaries = []
    if lib_path.exists():
        for dylib in lib_path.glob('*.dylib'):
            binaries.append((str(dylib), 'llama_cpp/lib'))
        for so in lib_path.glob('*.so'):
            binaries.append((str(so), 'llama_cpp/lib'))
    return binaries

# Collect llama_cpp binaries
llama_binaries = find_llama_libs()

# Find PyQt6 plugins
def find_pyqt6_plugins():
    """Find PyQt6 Qt plugins for macOS."""
    import PyQt6
    pyqt6_path = Path(PyQt6.__file__).parent
    plugins_path = pyqt6_path / 'Qt6' / 'plugins'
    datas = []
    if plugins_path.exists():
        # Include platforms plugin (cocoa for macOS)
        platforms_path = plugins_path / 'platforms'
        if platforms_path.exists():
            datas.append((str(platforms_path), 'PyQt6/Qt6/plugins/platforms'))
        # Include styles plugin
        styles_path = plugins_path / 'styles'
        if styles_path.exists():
            datas.append((str(styles_path), 'PyQt6/Qt6/plugins/styles'))
        # Include imageformats plugin
        imageformats_path = plugins_path / 'imageformats'
        if imageformats_path.exists():
            datas.append((str(imageformats_path), 'PyQt6/Qt6/plugins/imageformats'))
    return datas

pyqt6_plugins = find_pyqt6_plugins()

# Collect data files for ML packages
try:
    sentence_transformers_datas = collect_data_files('sentence_transformers')
except Exception:
    sentence_transformers_datas = []

try:
    chromadb_datas = collect_data_files('chromadb')
except Exception:
    chromadb_datas = []

try:
    tokenizers_datas = collect_data_files('tokenizers')
except Exception:
    tokenizers_datas = []

# Collect data files
datas = [
    (str(ROOT / 'resources'), 'resources'),
    (str(ROOT / 'src'), 'src'),
    (str(ROOT / 'ui'), 'ui'),
] + pyqt6_plugins + sentence_transformers_datas + chromadb_datas + tokenizers_datas

# Hidden imports for Qt and ML libraries
hiddenimports = [
    # PyQt6/Qt essentials
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtSvg',

    # LocalWrite modules
    'src',
    'src.paraphraser',
    'src.humanizer',
    'src.humanizer_v2',
    'src.text_analyzer',
    'src.ai_detector',
    'src.settings_manager',
    'src.model_registry',
    'src.model_downloader',
    'src.export_manager',
    'src.history_manager',
    'src.diff_viewer',
    'src.rag_engine',

    # UI modules
    'ui',
    'ui.main_window',
    'ui.model_selector',
    'ui.onboarding_dialog',
    'ui.history_dialog',

    # llama-cpp-python
    'llama_cpp',

    # DOCX export
    'docx',

    # PDF support
    'fitz',
    'pymupdf',

    # RAG dependencies
    'sentence_transformers',
    'chromadb',
    'rank_bm25',

    # Sentence transformers dependencies
    'torch',
    'tqdm',
    'huggingface_hub',
    'tokenizers',
    'safetensors',

    # ChromaDB dependencies
    'sqlite3',
    'onnxruntime',
    'posthog',
    'opentelemetry',

    # Requests for model download
    'requests',
]

# Exclude unnecessary modules to reduce size
excludes = [
    # Development tools
    'tkinter',
    'matplotlib',
    'IPython',
    'notebook',
    'pytest',
    'black',
    'isort',
    'ruff',
    'mypy',

    # Unused libraries
    'PIL.ImageQt',
    'PIL.ImageTk',
    'scipy',
    'numpy.testing',

    # Exclude other Qt bindings (we use PyQt6)
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PySide6',
    'PySide2',

    # Exclude ML libraries not needed (we use sentence-transformers, not full transformers)
    'tensorflow',
    'keras',
]

a = Analysis(
    [str(ROOT / 'main.py')],
    pathex=[str(ROOT)],
    binaries=llama_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / 'pyi_rth_qt_plugins.py')],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LocalWrite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,  # Set to Apple Developer ID for signing
    entitlements_file=None,   # Set to entitlements.plist for hardened runtime
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalWrite',
)

app = BUNDLE(
    coll,
    name='LocalWrite.app',
    icon='resources/icon.icns',
    bundle_identifier='ai.localwrite.desktop',
    info_plist={
        # Basic app info
        'CFBundleName': 'LocalWrite',
        'CFBundleDisplayName': 'LocalWrite',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleExecutable': 'LocalWrite',
        'CFBundleIdentifier': 'ai.localwrite.desktop',

        # Display settings
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
        'LSMinimumSystemVersion': '11.0',  # macOS Big Sur+

        # Document types (text files and documents)
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Text File',
                'CFBundleTypeExtensions': ['txt', 'md', 'text'],
                'CFBundleTypeRole': 'Editor',
                'LSHandlerRank': 'Alternate',
            },
            {
                'CFBundleTypeName': 'PDF Document',
                'CFBundleTypeExtensions': ['pdf'],
                'CFBundleTypeRole': 'Viewer',
                'LSHandlerRank': 'Alternate',
            },
            {
                'CFBundleTypeName': 'Word Document',
                'CFBundleTypeExtensions': ['docx'],
                'CFBundleTypeRole': 'Editor',
                'LSHandlerRank': 'Alternate',
            },
        ],

        # Privacy permissions
        'NSDesktopFolderUsageDescription': 'LocalWrite needs access to save enhanced documents.',
        'NSDocumentsFolderUsageDescription': 'LocalWrite needs access to open and save documents.',
        'NSDownloadsFolderUsageDescription': 'LocalWrite needs access to save enhanced documents.',

        # Networking (for model downloads)
        'NSAppTransportSecurity': {
            'NSAllowsArbitraryLoads': False,  # Security: only allow HTTPS
            'NSExceptionDomains': {
                'huggingface.co': {
                    'NSExceptionAllowsInsecureHTTPLoads': False,
                    'NSExceptionRequiresForwardSecrecy': True,
                    'NSIncludesSubdomains': True,
                },
            },
        },

        # Security
        'LSApplicationCategoryType': 'public.app-category.productivity',
        'NSPrincipalClass': 'NSApplication',

        # Copyright and info
        'NSHumanReadableCopyright': 'Copyright © 2025 Svetozar Technologies. MIT License.',
        'CFBundleGetInfoString': 'LocalWrite - Private AI Writing Assistant. 100% Offline.',
    },
)
