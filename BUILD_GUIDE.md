# BUILD GUIDE
## Creating Windows Executable for Voice Command System

**Author:** Vishal Saha

---

## 📦 Method 1: PyInstaller (Recommended)

### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

### Step 2: Basic Build
```bash
pyinstaller --onefile main.py
```

### Step 3: Optimized Build with All Features
```bash
pyinstaller --onefile \
    --name "VoiceCommandSystem" \
    --add-data "actions.json;." \
    --hidden-import="sklearn.utils._typedefs" \
    --hidden-import="sklearn.neighbors._partition_nodes" \
    --hidden-import="sklearn.utils._cython_blas" \
    --hidden-import="sentence_transformers" \
    --icon=icon.ico \
    main.py
```

### Step 4: Find Your Executable
```
dist/VoiceCommandSystem.exe  (Windows)
dist/VoiceCommandSystem       (Linux/Mac)
```

---

## 📦 Method 2: cx_Freeze (Alternative)

### Step 1: Install cx_Freeze
```bash
pip install cx_Freeze
```

### Step 2: Create setup.py
```python
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": [
        "faster_whisper",
        "sentence_transformers",
        "sounddevice",
        "faiss",
        "numpy"
    ],
    "include_files": ["actions.json"]
}

setup(
    name="VoiceCommandSystem",
    version="1.0",
    description="Offline Voice Command System",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py")]
)
```

### Step 3: Build
```bash
python setup.py build
```

---

## 🎯 Build Options Explained

### --onefile
Creates single executable (vs folder of files)

### --name
Custom name for executable

### --add-data
Include additional files (actions.json)
Format: "source;destination" (Windows) or "source:destination" (Linux/Mac)

### --hidden-import
Import modules not auto-detected

### --icon
Custom icon file (optional)

---

## 📏 Executable Size

### Without Models Bundled
- Size: ~50-100MB
- Models download on first run

### With Models Bundled
- Size: ~500-700MB
- Fully self-contained

### To Bundle Models

```bash
# Download models first
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Then build with model cache
pyinstaller --onefile \
    --add-data "~/.cache/huggingface;huggingface" \
    main.py
```

---

## 🧪 Testing the Executable

### Windows
```cmd
dist\VoiceCommandSystem.exe
```

### Linux/Mac
```bash
./dist/VoiceCommandSystem
```

### With Audio File
```bash
dist\VoiceCommandSystem.exe test_audio.wav
```

---

## ⚠️ Common Build Issues

### Issue 1: Missing DLL
**Error:** "Unable to find X.dll"
**Fix:** Include in --add-binary
```bash
--add-binary "C:\path\to\library.dll;."
```

### Issue 2: Import Errors
**Error:** "No module named 'X'"
**Fix:** Add to --hidden-import
```bash
--hidden-import="module_name"
```

### Issue 3: Large File Size
**Solution:** Use --exclude-module for unused packages
```bash
--exclude-module="matplotlib" \
--exclude-module="PIL"
```

---

## 🚀 Optimization Tips

### 1. UPX Compression
```bash
# Install UPX
# Then build with
pyinstaller --onefile --upx-dir="path/to/upx" main.py
```
Reduces size by 30-50%

### 2. Exclude Unused Modules
```bash
--exclude-module="tkinter" \
--exclude-module="matplotlib"
```

### 3. Strip Debug Symbols
```bash
--strip
```

---

## 📦 Distribution Package

### Create ZIP Package
```bash
# Windows
7z a VoiceCommandSystem.zip dist/VoiceCommandSystem.exe actions.json README.md

# Linux/Mac
zip -r VoiceCommandSystem.zip dist/VoiceCommandSystem actions.json README.md
```

### Include in Package
- VoiceCommandSystem.exe
- actions.json
- README.md
- LICENSE (if applicable)

---

## 🎬 Auto-Download Models on First Run

Add to beginning of `main()`:
```python
def ensure_models():
    """Download models if not present"""
    import os
    from pathlib import Path
    
    model_dir = Path.home() / ".cache" / "huggingface"
    
    if not model_dir.exists():
        print("Downloading models (one-time setup)...")
        # Models will auto-download on first use
        
ensure_models()
```

---

## 🔍 Verification Checklist

After building, verify:
- [ ] Executable runs without errors
- [ ] Models load correctly
- [ ] Audio input works
- [ ] Actions execute properly
- [ ] Logs are created
- [ ] No missing dependencies
- [ ] Works on clean Windows machine

---

## 📊 Build Time Estimates

| Method | Time | Size |
|--------|------|------|
| PyInstaller (basic) | 2-3 min | ~50MB |
| PyInstaller (with models) | 5-10 min | ~600MB |
| cx_Freeze | 3-5 min | ~100MB |

---

## 💾 Final Distribution

### Upload to GitHub Releases
```bash
gh release create v1.0 \
    dist/VoiceCommandSystem.exe \
    --title "Voice Command System v1.0" \
    --notes "Offline speech-to-text with RAG"
```

### Or Share via Google Drive
1. Upload VoiceCommandSystem.zip
2. Set sharing to "Anyone with link"
3. Include link in submission

---

**Build completed! Ready for submission! 🚀**
