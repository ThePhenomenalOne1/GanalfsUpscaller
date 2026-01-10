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

### Option 1: Automated Setup (Windows)

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThePhenomenalOne1/GanalfsUpscaller.git
   cd GanalfsUpscaller
   ```

2. **Download GFPGAN model weights** (required for face enhancement)
   
   Create `gfpgan/weights/` folder and download these files:
   - [detection_Resnet50_Final.pth](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth) (104 MB)
   - [parsing_parsenet.pth](https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth) (81 MB)
   
   Place them in: `gfpgan/weights/`

3. **Run setup**
   ```bash
   setup.bat
   ```
   This will:
   - Create a Python 3.9 virtual environment
   - Install all dependencies
   - Download Real-ESRGAN models automatically

4. **Launch the GUI**
   ```bash
   run_gui.bat
   ```

### Option 2: Manual Setup (All Platforms)

1. **Clone and navigate**
   ```bash
   git clone https://github.com/ThePhenomenalOne1/GanalfsUpscaller.git
   cd GanalfsUpscaller
   ```

2. **Download GFPGAN weights**
   
   Create the weights directory:
   ```bash
   mkdir -p gfpgan/weights
   ```
   
   Download the model files:
   - Windows: Right-click → Save as to `gfpgan/weights/`
   - Linux/Mac: 
     ```bash
     cd gfpgan/weights
     wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
     wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
     cd ../..
     ```

3. **Create virtual environment**
   ```bash
   python -m venv env
   ```

4. **Activate environment**
   ```bash
   # Windows
   env\Scripts\activate
   
   # Linux/Mac
   source env/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the GUI**
   ```bash
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
