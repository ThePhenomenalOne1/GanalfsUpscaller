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

## Installation

> **Note**: This repository uses Git LFS for AI model weights. Make sure you have [Git LFS installed](https://git-lfs.github.com/) before cloning.

### Quick Start (Windows)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThePhenomenalOne1/GanalfsUpscaller.git
   cd GanalfsUpscaller
   ```
   Model weights (~195 MB) will download automatically via Git LFS.

2. **Run automated setup**
   ```bash
   setup.bat
   ```
   This creates a Python 3.9 environment, installs dependencies, and downloads Real-ESRGAN models.

3. **Launch the application**
   ```bash
   run_gui.bat
   ```

### Manual Setup (All Platforms)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThePhenomenalOne1/GanalfsUpscaller.git
   cd GanalfsUpscaller
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv env
   env\Scripts\activate
   
   # Linux/Mac
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the GUI**
   ```bash
   python gui.py
   ```

### Troubleshooting

**Git LFS not installed?**
- Windows: Download from [git-lfs.github.com](https://git-lfs.github.com/)
- Linux: `sudo apt install git-lfs` or `brew install git-lfs`
- After installing, run: `git lfs install`

**Model weights missing?**
If weights didn't download automatically:
```bash
git lfs pull
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

**Created by**: Amir Khalid

This mess was lovingly put together with:
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- GFPGAN: https://github.com/TencentARC/GFPGAN

**Powered by Black Magic** ✨

