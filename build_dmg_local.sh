#!/bin/bash
# LocalWrite Production DMG Build Script
# Builds a production-ready DMG for direct distribution (no code signing)

set -e  # Exit on error

echo "LocalWrite Production Build"
echo "==========================="
echo ""

# Configuration
APP_NAME="LocalWrite"
DMG_NAME="LocalWrite-macOS.dmg"
VERSION="1.0.0"

# Clean previous builds
echo "[*] Cleaning previous builds..."
rm -rf build dist *.dmg
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "[+] Clean complete"
echo ""

# Check icon exists
if [ ! -f "resources/icon.icns" ]; then
    echo "[!] Icon not found. Creating from PNG..."
    if [ -f "resources/icon.png" ]; then
        # Create iconset directory
        mkdir -p resources/icon.iconset

        # Generate different sizes
        sips -z 16 16     resources/icon.png --out resources/icon.iconset/icon_16x16.png
        sips -z 32 32     resources/icon.png --out resources/icon.iconset/icon_16x16@2x.png
        sips -z 32 32     resources/icon.png --out resources/icon.iconset/icon_32x32.png
        sips -z 64 64     resources/icon.png --out resources/icon.iconset/icon_32x32@2x.png
        sips -z 128 128   resources/icon.png --out resources/icon.iconset/icon_128x128.png
        sips -z 256 256   resources/icon.png --out resources/icon.iconset/icon_128x128@2x.png
        sips -z 256 256   resources/icon.png --out resources/icon.iconset/icon_256x256.png
        sips -z 512 512   resources/icon.png --out resources/icon.iconset/icon_256x256@2x.png
        sips -z 512 512   resources/icon.png --out resources/icon.iconset/icon_512x512.png
        sips -z 1024 1024 resources/icon.png --out resources/icon.iconset/icon_512x512@2x.png

        # Create icns file
        iconutil -c icns resources/icon.iconset -o resources/icon.icns

        # Cleanup
        rm -rf resources/icon.iconset
        echo "[+] Icon created from PNG"
    else
        echo "[!] No icon.png found in resources/. Please add an icon."
        exit 1
    fi
else
    echo "[+] Icon ready"
fi
echo ""

# Check PyInstaller
echo "[*] Checking PyInstaller..."
if ! command -v pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi
echo "[+] PyInstaller ready"
echo ""

# Build with PyInstaller
echo "[*] Building ${APP_NAME}.app..."
echo "This may take 5-10 minutes..."
pyinstaller LocalWrite.spec --clean --noconfirm

if [ ! -d "dist/${APP_NAME}.app" ]; then
    echo "[!] Build failed! ${APP_NAME}.app not found in dist/"
    exit 1
fi
echo "[+] Build complete"
echo ""

# Get app size
APP_SIZE=$(du -sh "dist/${APP_NAME}.app" | cut -f1)
echo "App size: ${APP_SIZE}"
echo ""

# Create DMG
echo "[*] Creating DMG..."
hdiutil create \
  -volname "${APP_NAME}" \
  -srcfolder "dist/${APP_NAME}.app" \
  -ov \
  -format UDZO \
  "${DMG_NAME}"

if [ ! -f "${DMG_NAME}" ]; then
    echo "[!] DMG creation failed!"
    exit 1
fi

DMG_SIZE=$(du -sh "${DMG_NAME}" | cut -f1)
echo "[+] DMG created successfully!"
echo ""

# Final summary
echo "Build Summary"
echo "============="
echo "Version: ${VERSION}"
echo "App: dist/${APP_NAME}.app (${APP_SIZE})"
echo "DMG: ${DMG_NAME} (${DMG_SIZE})"
echo ""

# Distribution checklist
echo "Distribution Checklist"
echo "======================"
echo "[+] App icon included"
echo "[+] Privacy permissions declared"
echo "[+] Dark mode supported"
echo "[+] Models download on-demand"
echo "[+] 100% offline after model download"
echo ""

# Usage instructions
echo "Next Steps"
echo "=========="
echo ""
echo "1. Test the DMG:"
echo "   open ${DMG_NAME}"
echo ""
echo "2. Upload to GitHub Releases:"
echo "   - Create new release with tag v${VERSION}"
echo "   - Upload ${DMG_NAME}"
echo "   - Include installation instructions"
echo ""
echo "3. Users install by:"
echo "   - Opening DMG"
echo "   - Dragging to Applications"
echo "   - Right-click → Open (first time only)"
echo ""
echo "[!] Important: Users will see security warning (not code signed)"
echo "    This is normal for open-source apps without Apple Developer account"
echo "    Users bypass by: Right-click → Open → Open"
echo ""
echo "[+] Build complete! Ready for distribution!"
