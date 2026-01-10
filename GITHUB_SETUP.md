# Connecting to GitHub Repository

## Step 1: Install Git
1. Download Git for Windows: https://git-scm.com/download/win
2. Run the installer (use default settings)
3. Restart your terminal/PowerShell

## Step 2: Configure Git (First Time Only)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Initialize and Connect Repository
Open PowerShell in `c:\Users\Amir\Desktop\upscaler` and run:

```bash
# Initialize git repository
git init

# Add your GitHub repository as remote
git remote add origin https://github.com/ThePhenomenalOne1/GanalfsUpscaller.git

# Add all files (respects .gitignore)
git add -A

# Create initial commit
git commit -m "Initial commit: Gandalf of Technology Upscaler with film grain, watermark removal, and installer"

# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

## Step 4: Future Updates
After making changes, push with:
```bash
git add -A
git commit -m "Description of your changes"
git push
```

## What Gets Uploaded
✅ **Included:**
- Source code (.py files)
- Requirements.txt
- README.md
- Installer script (.iss)
- Documentation

❌ **Excluded (via .gitignore):**
- Virtual environment (env39_new/) - too large
- Build folders (build/, dist/)
- Python cache (__pycache__)
- Model files (.pth)
- Backups

## Important Notes
1. **Large Files**: The virtual environment (4-6GB) is NOT uploaded
2. **Users Clone**: Users will need to:
   - Clone the repository
   - Run `setup.bat` to create their own virtual environment
   - Or use the installer you create with Inno Setup

## Repository Structure on GitHub
```
GanalfsUpscaller/
├── gui.py
├── upscaler.py
├── watermark_removal.py
├── requirements.txt
├── README.md
├── installer.iss
├── INSTALLER_BUILD.md
├── .gitignore
├── run_gui.bat
└── setup.bat
```

Users can then:
1. Clone the repository
2. Run `setup.bat` to install dependencies
3. Or download your Inno Setup installer
