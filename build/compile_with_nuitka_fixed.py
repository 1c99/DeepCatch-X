#!/usr/bin/env python3
"""
Fixed Nuitka compilation script that properly handles data files
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def compile_with_proper_data_handling():
    """Compile with all necessary data files included"""
    
    system = platform.system()
    cpu_count = os.cpu_count() or 1
    
    # Base Nuitka command with better data handling
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--show-progress",
        f"--jobs={cpu_count}",
        "--enable-plugin=anti-bloat",
        "--remove-output",
        
        # CRITICAL: Include package data files
        "--include-package-data=pydicom",  # This tells Nuitka to include ALL pydicom data files
        "--include-package-data=nibabel",   # Include nibabel data files
        "--include-package-data=PIL",       # Include PIL/Pillow data files
        "--include-package-data=scipy",     # Include scipy data files
        "--include-package-data=numpy",     # Include numpy data files
        "--include-package-data=torch",     # Include torch data files
        "--include-package-data=torchvision",  # Include torchvision data files
        "--include-package-data=skimage",   # Include scikit-image data files
        "--include-package-data=cv2",       # Include opencv data files
        "--include-package-data=segmentation_models_pytorch",  # Include SMP data files
        
        # Include all packages we use
        "--include-package=torch",
        "--include-package=torchvision", 
        "--include-package=numpy",
        "--include-package=scipy",
        "--include-package=PIL",
        "--include-package=pydicom",
        "--include-package=nibabel",
        "--include-package=skimage",
        "--include-package=pandas",
        "--include-package=yaml",
        "--include-package=cv2",
        "--include-package=segmentation_models_pytorch",
        
        # Include our modules
        "--include-module=src.insights.cardiothoracic_ratio",
        "--include-module=src.insights.peripheral_area",
        "--include-module=src.insights.aorta_diameter",
        
        # Skip problematic imports
        "--nofollow-import-to=matplotlib",
        "--nofollow-import-to=PyQt5",
        "--nofollow-import-to=PyQt6",
        "--nofollow-import-to=PySide2", 
        "--nofollow-import-to=PySide6",
        "--nofollow-import-to=tkinter",
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=jupyter",
    ]
    
    # Platform specific options
    if system == "Windows":
        cmd.extend([
            "--windows-console-mode=force",
            "--low-memory",
            "--plugin-enable=numpy",
            "--plugin-enable=torch",
            "--module-parameter=torch-disable-jit=yes",
        ])
        if os.path.exists("icon.ico"):
            cmd.append("--windows-icon-from-ico=icon.ico")
    
    # Include data directories from our project
    data_dirs = [
        ("checkpoints", "checkpoints"),
        ("src", "src"),
        ("configs", "configs"),
    ]
    
    for src, dest in data_dirs:
        if os.path.exists(src):
            cmd.extend([f"--include-data-dir={src}={dest}"])
    
    # Output directory
    cmd.append("--output-dir=dist/inference_fixed")
    
    # Add the script
    cmd.append("inference.py")
    
    print("🔨 Compiling with comprehensive data file inclusion...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Compilation successful!")
        
        # Create run script
        if system == "Windows":
            run_script = """@echo off
cd /d "%~dp0"
cd inference_fixed\\inference.dist
inference.exe %*
"""
            script_path = "dist/run_inference.bat"
        else:
            run_script = """#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/inference_fixed/inference.dist"
./inference "$@"
"""
            script_path = "dist/run_inference.sh"
            
        with open(script_path, "w") as f:
            f.write(run_script)
            
        if system != "Windows":
            os.chmod(script_path, 0o755)
            
        print(f"✅ Created run script: {script_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation failed: {e}")
        return False

if __name__ == "__main__":
    if not os.path.exists("inference.py"):
        print("❌ Error: inference.py not found. Run this from the project root.")
        sys.exit(1)
        
    compile_with_proper_data_handling()