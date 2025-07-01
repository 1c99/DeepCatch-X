#!/usr/bin/env python3
"""
Nuitka compilation script for DCX inference systems
Compiles both inference.py and inference_ray.py into standalone executables
"""
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_nuitka():
    """Check if Nuitka is installed"""
    try:
        subprocess.run(["python", "-m", "nuitka", "--version"], 
                      capture_output=True, check=True)
        print("✓ Nuitka is installed")
        return True
    except:
        print("✗ Nuitka is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "nuitka"])
        return True

def get_platform_info():
    """Get current platform information"""
    system = platform.system()
    machine = platform.machine()
    cpu_count = os.cpu_count()
    print(f"\n📦 Building for: {system} {machine}")
    print(f"🚀 CPU cores available: {cpu_count} (Nuitka will use all cores)")
    return system, machine

def compile_inference(script_name, output_name=None):
    """Compile a Python script with Nuitka"""
    if output_name is None:
        output_name = script_name.replace('.py', '')
    
    system, machine = get_platform_info()
    
    # Get CPU count for parallel compilation
    cpu_count = os.cpu_count() or 1
    
    # Base Nuitka command
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--show-progress",
        "--show-memory",
        f"--jobs={cpu_count}",  # Use all available CPU cores for faster compilation
        "--enable-plugin=numpy",
        "--enable-plugin=torch",
        "--remove-output",  # Remove build folder after compilation
        "--low-memory",     # Use less memory during compilation
    ]
    
    # Platform-specific options
    if system == "Darwin":  # macOS
        # For now, let's keep it simple without app bundle
        pass
    elif system == "Windows":
        cmd.extend([
            "--windows-disable-console",
        ])
        # Only add icon if it exists
        if os.path.exists("icon.ico"):
            cmd.append("--windows-icon-from-ico=icon.ico")
    
    # Add required packages and modules
    packages_to_include = [
        "torch", "torchvision", "numpy", "scipy", "PIL", 
        "pydicom", "nibabel", "yaml", "skimage",
        "pandas"  # Common packages
    ]
    
    # Only include ray for inference_ray.py
    if "ray" in script_name:
        packages_to_include.append("ray")
    
    for package in packages_to_include:
        cmd.append(f"--include-package={package}")
    
    # Handle cv2 (opencv-python) separately - include as module not package
    cmd.append("--include-module=cv2")
    
    # Include data directories
    data_dirs = [
        ("configs", "configs"),
        ("checkpoints", "checkpoints"),
        ("src", "src"),
    ]
    
    for src, dest in data_dirs:
        if os.path.exists(src):
            cmd.extend([f"--include-data-dir={src}={dest}"])
    
    # Output directory name
    cmd.append(f"--output-dir=dist/{output_name}")
    
    # Add the script to compile (must be last)
    cmd.append(script_name)
    
    print(f"\n🔨 Compiling {script_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ Successfully compiled {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to compile {script_name}: {e}")
        return False

def create_requirements():
    """Create requirements.txt if it doesn't exist"""
    requirements = """torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scipy>=1.5.0
Pillow>=8.0.0
pydicom>=2.0.0
nibabel>=3.0.0
PyYAML>=5.4.0
opencv-python>=4.5.0
scikit-image>=0.18.0
ray>=2.0.0
pandas>=1.3.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ Created requirements.txt")

def create_distribution_package(name):
    """Create a distribution package with all necessary files"""
    dist_path = Path(f"dist/{name}_package")
    dist_path.mkdir(parents=True, exist_ok=True)
    
    # Copy the executable
    exec_path = Path(f"dist/{name}")
    if exec_path.exists():
        shutil.copytree(exec_path, dist_path / name, dirs_exist_ok=True)
    
    # Create run script
    system = platform.system()
    if system == "Windows":
        run_script = f"""@echo off
cd /d "%~dp0"
{name}\\{name}.exe %*
"""
        script_name = f"run_{name}.bat"
    else:
        run_script = f"""#!/bin/bash
cd "$(dirname "$0")"
./{name}/{name} "$@"
"""
        script_name = f"run_{name}.sh"
    
    script_path = dist_path / script_name
    with open(script_path, "w") as f:
        f.write(run_script)
    
    if system != "Windows":
        os.chmod(script_path, 0o755)
    
    # Create README
    readme = f"""# DCX {name.replace('_', ' ').title()} - Compiled Version

## System Requirements
- {platform.system()} {platform.machine()}
- No Python installation required
- Minimum 8GB RAM recommended
- GPU support included (if available)

## Usage

### {platform.system()} 
Run the executable using:
```bash
./{script_name} --input_path sample.dcm --output_dir results --module lung
```

### Full command options:
```bash
./{script_name} --help
```

## Directory Structure
```
{name}_package/
├── {name}/           # Compiled executable and dependencies
├── configs/         # Configuration files (if needed)
├── checkpoints/     # Model weights (add separately)
└── {script_name}    # Convenience run script
```

## Important Notes
1. This executable is platform-specific for {platform.system()} {platform.machine()}
2. Model checkpoint files (.pth, .pt) need to be added separately
3. For other platforms, compile from source
"""
    
    with open(dist_path / "README.md", "w") as f:
        f.write(readme)
    
    print(f"✓ Created distribution package at {dist_path}")

def main():
    """Main compilation process"""
    print("🚀 DCX Nuitka Compilation Tool")
    print("=" * 50)
    
    # Check environment
    if not check_nuitka():
        return
    
    # Create requirements if needed
    if not os.path.exists("requirements.txt"):
        create_requirements()
    
    # Check if we're in the right directory
    if not os.path.exists("inference.py"):
        print("✗ Error: inference.py not found. Run this from the project root.")
        return
    
    # Ask what to compile
    print("\nWhat would you like to compile?")
    print("1. inference.py only")
    print("2. inference_ray.py only")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    success = []
    
    if choice in ["1", "3"]:
        if compile_inference("inference.py", "inference"):
            create_distribution_package("inference")
            success.append("inference")
    
    if choice in ["2", "3"]:
        if compile_inference("inference_ray.py", "inference_ray"):
            create_distribution_package("inference_ray")
            success.append("inference_ray")
    
    if success:
        print("\n✅ Compilation Summary:")
        print("=" * 50)
        for name in success:
            print(f"✓ {name} compiled successfully")
            print(f"  Location: dist/{name}_package/")
        
        print("\n📋 Next Steps:")
        print("1. Copy your checkpoint files to the dist directory")
        print("2. Test the executable with a sample DICOM file")
        print("3. Package for distribution")
    else:
        print("\n❌ Compilation failed. Check the errors above.")

if __name__ == "__main__":
    main()
