# Building the Windows Installer

## Prerequisites
1. **Download Inno Setup**: https://jrsoftware.org/isdl.php
2. Install Inno Setup (it's free and open-source)

## Build Steps

### Option 1: Using Inno Setup GUI
1. Open Inno Setup
2. Click "File" → "Open" → Select `installer.iss`
3. Click "Build" → "Compile"
4. The installer will be created in `installer_output\GandalfUpscaler_Setup.exe`

### Option 2: Command Line
```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

## What the Installer Includes
- ✅ Complete Python 3.9 virtual environment
- ✅ All Python packages (PyTorch, Real-ESRGAN, etc.)
- ✅ GUI application files
- ✅ Desktop shortcut
- ✅ Start Menu entry
- ✅ Automatic uninstaller

## Installer Size
- Approximately 4-6 GB (due to PyTorch and AI dependencies)
- This is normal for AI applications

## Distribution
The final installer (`GandalfUpscaler_Setup.exe`) is a single file that:
1. Users download the .exe
2. Double-click to install
3. Everything is set up automatically
4. No Python installation needed
5. Works on any Windows 10/11 PC

## Testing
After building, test the installer:
1. Run `GandalfUpscaler_Setup.exe`
2. Follow installation wizard
3. Launch from Desktop or Start Menu
4. Verify all features work

## Notes
- The installer bundles your entire `env39_new` virtual environment
- This avoids PyInstaller issues completely
- Users get a proper installed application
- Uninstaller is automatically created
- No PATH modifications needed
