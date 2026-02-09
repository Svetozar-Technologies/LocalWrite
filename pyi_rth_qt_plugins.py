# Runtime hook to set Qt plugin path
import os
import sys

# Get the base path where the app is running from
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Set Qt plugin path
qt_plugin_path = os.path.join(base_path, 'PyQt6', 'Qt6', 'plugins')
if os.path.exists(qt_plugin_path):
    os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_plugin_path, 'platforms')
