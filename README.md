# Gandalf of Technology Upscaler

AI-powered image upscaling with watermark removal and film grain effects.

## Features
- 🚀 2x/4x upscaling with Real-ESRGAN
- ✨ AI watermark removal (Gemini sparkle)
- 🎞️ Film grain texture control
- 📷 EXIF metadata preservation
- 🎨 Custom output DPI and suffixes
- 👤 Face enhancement (GFPGAN)
- 🖥️ GPU acceleration support

## Quick Start

### Windows
1. Run `setup.bat` (installs Python 3.9 and dependencies)
2. Run `run_gui.bat` (launches the GUI)
3. Download AI models using "📥 Models" button
4. Start upscaling!

### Manual Setup
```bash
# Create virtual environment
python -m venv env

# Activate it
env\Scripts\activate  # Windows
source env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run GUI
python gui.py
```

## Requirements
- Python 3.9+
- NVIDIA GPU with CUDA 11.8 (recommended) or CPU
- 8GB+ RAM (10GB+ recommended)
- Windows 10/11, Linux, or macOS

## Project Structure
```
upscaler/
├── gui.py                 # Main GUI application
├── upscaler.py           # Core upscaling engine
├── watermark_removal.py  # Watermark detection/removal
├── requirements.txt      # Python dependencies
├── run_gui.bat          # Windows launcher
└── models/              # AI models (auto-downloaded)
```

## Usage

### GUI
- Drag & drop images or use browse buttons
- Adjust settings (model, scale, grain, etc.)
- Click "🚀 UPSCALE" button

### Command Line
```python
from upscaler import upscale_image

upscale_image(
    "input.jpg",
    output_path="output_4x.png",
    scale=4,
    grain_strength=0.5
)
```

## Credits
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- GFPGAN: https://github.com/TencentARC/GFPGAN

**Powered by Black Magic** ✨
