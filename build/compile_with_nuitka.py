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

def verify_critical_files():
    """Verify critical files exist before compilation"""
    print("\n🔍 Verifying critical files...")
    issues = []
    
    # Check for pydicom data files
    try:
        import pydicom
        pydicom_path = Path(pydicom.__file__).parent
        urls_json = pydicom_path / "data" / "urls.json"
        if not urls_json.exists():
            issues.append(f"⚠️  Missing pydicom urls.json at {urls_json}")
        else:
            print(f"✓ Found pydicom urls.json")
    except ImportError:
        issues.append("⚠️  pydicom not installed")
    
    # Check for checkpoint files
    if os.path.exists("checkpoints"):
        checkpoint_files = list(Path("checkpoints").glob("**/*.pth")) + \
                          list(Path("checkpoints").glob("**/*.pt"))
        if checkpoint_files:
            print(f"✓ Found {len(checkpoint_files)} checkpoint files")
        else:
            issues.append("⚠️  No checkpoint files found in checkpoints/")
    else:
        issues.append("⚠️  checkpoints/ directory not found")
    
    # Check for config files
    if not os.path.exists("configs"):
        issues.append("⚠️  configs/ directory not found")
    else:
        config_files = list(Path("configs").glob("**/*.yaml"))
        if config_files:
            print(f"✓ Found {len(config_files)} config files")
        else:
            issues.append("⚠️  No YAML config files found in configs/")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   {issue}")
        response = input("\nContinue anyway? (y/n): ")
        return response.lower() == 'y'
    
    print("✓ All critical files verified")
    return True

def get_platform_info():
    """Get current platform information"""
    system = platform.system()
    machine = platform.machine()
    cpu_count = os.cpu_count()
    print(f"\n📦 Building for: {system} {machine}")
    print(f"🚀 CPU cores available: {cpu_count} (Nuitka will use all cores)")
    return system, machine

def embed_configs():
    """Embed configuration files before compilation"""
    try:
        from embed_configs import build_embedded_configs
        print("\n📦 Embedding configuration files...")
        build_embedded_configs()
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not embed configs: {e}")
        print("   Falling back to external config files")
        return False

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
        f"--jobs={cpu_count}",  # Use all available CPU cores for faster compilation
        # Removed deprecated plugins - numpy and torch are auto-detected now
        "--enable-plugin=anti-bloat",  # Helps reduce unnecessary imports
        "--nofollow-import-to=matplotlib",  # Skip matplotlib completely to avoid Qt errors
        "--nofollow-import-to=PyQt5",
        "--nofollow-import-to=PyQt6", 
        "--nofollow-import-to=PySide2",
        "--nofollow-import-to=PySide6",
        "--nofollow-import-to=tkinter",
        "--remove-output",  # Remove build folder after compilation
    ]
    
    # Platform-specific memory options
    if system == "Windows":
        cmd.append("--low-memory")  # Use less memory during compilation
    elif system == "Linux":
        # Skip memory tracking on Linux to avoid yaml assertion error
        pass
    else:  # macOS
        cmd.append("--show-memory")
    
    # Platform-specific options
    if system == "Darwin":  # macOS
        # For now, let's keep it simple without app bundle
        pass
    elif system == "Windows":
        cmd.extend([
            "--windows-console-mode=force",  # Force console window to show output
        ])
        # Only add icon if it exists
        if os.path.exists("icon.ico"):
            cmd.append("--windows-icon-from-ico=icon.ico")
        # Add Windows-specific optimizations
        cmd.extend([
            "--plugin-enable=numpy",  # Explicitly enable numpy plugin for Windows
            "--plugin-enable=torch",  # Explicitly enable torch plugin for Windows
            "--module-parameter=torch-disable-jit=yes",  # Disable Torch JIT for standalone
        ])
    
    # Add required packages and modules
    packages_to_include = [
        "torch", "torchvision", "numpy", "scipy", "PIL",  
        "pydicom", "nibabel", "skimage",
        "pandas",  # Common packages
        # Note: matplotlib is now optional in the insight modules
        # The code will work without it, just without visualization
    ]
    
    # Include package data files (critical for pydicom)
    cmd.extend([
        "--include-package-data=pydicom",  # Include pydicom data files like urls.json
        "--include-package-data=nibabel",
        "--include-package-data=PIL",
        "--include-package-data=scipy",
        "--include-package-data=numpy",
        "--include-package-data=torch",
        "--include-package-data=torchvision",
        "--include-package-data=cv2",
        "--include-package-data=skimage",
        "--include-package-data=segmentation_models_pytorch",
    ])
    
    # Windows-specific: Enable multiprocessing support
    if system == "Windows":
        cmd.extend([
            "--enable-plugin=multiprocessing",  # For Windows multiprocessing support
        ])
    
    # Add yaml carefully to avoid assertion errors
    if system != "Linux":
        packages_to_include.append("yaml")
    
    # Add segmentation_models_pytorch for LAA module
    packages_to_include.append("segmentation_models_pytorch")
    
    # Only include ray for inference_ray.py
    if "ray" in script_name:
        packages_to_include.append("ray")
    
    for package in packages_to_include:
        cmd.append(f"--include-package={package}")
    
    # Handle cv2 (opencv-python) - let Nuitka auto-detect it
    # cv2 will be included automatically when imported
    
    # Include specific modules from src.insights
    insight_modules = [
        "src.insights.cardiothoracic_ratio",
        "src.insights.peripheral_area", 
        "src.insights.aorta_diameter"
    ]
    for module in insight_modules:
        cmd.append(f"--include-module={module}")
    
    # Note: matplotlib is optional and handled conditionally in the code
    
    # Include data directories
    # Check if configs are embedded
    embedded_configs = os.path.exists("src/embedded_configs.py")
    
    # Clean up macOS files from checkpoints before compilation
    if os.path.exists("checkpoints"):
        # Do basic cleanup of macOS files
        for root, dirs, files in os.walk("checkpoints"):
            for file in files:
                if file == ".DS_Store" or file.startswith("._"):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"✓ Removed {file_path}")
            # Remove __MACOSX directories
            if "__MACOSX" in dirs:
                macos_path = os.path.join(root, "__MACOSX")
                shutil.rmtree(macos_path)
                dirs.remove("__MACOSX")
                print(f"✓ Removed directory: {macos_path}")
    
    data_dirs = [
        ("checkpoints", "checkpoints"),
        ("src", "src"),
    ]
    
    # Note: matplotlib config not needed as matplotlib is optional
    
    # Only include configs directory if not embedded
    if not embedded_configs:
        data_dirs.insert(0, ("configs", "configs"))
    
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
        subprocess.run(cmd, check=True)
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
segmentation-models-pytorch>=0.2.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ Created requirements.txt")

def create_distribution_package(name, cleanup=True):
    """Create a distribution package with all necessary files"""
    dist_path = Path(f"dist/{name}_package")
    dist_path.mkdir(parents=True, exist_ok=True)
    
    # Copy the executable
    exec_path = Path(f"dist/{name}")
    if exec_path.exists():
        shutil.copytree(exec_path, dist_path / name, dirs_exist_ok=True)
        
        # Clean up the intermediate directory if requested
        if cleanup:
            print(f"✓ Cleaning up intermediate directory: {exec_path}")
            shutil.rmtree(exec_path)
        else:
            print(f"✓ Keeping intermediate directory: {exec_path}")
    
    # Create run script
    system = platform.system()
    if system == "Windows":
        run_script = f"""@echo off
REM Set threading environment variables to avoid conflicts
set MKL_THREADING_LAYER=sequential
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

cd /d "%~dp0"
cd {name}\\{name}.dist
{name}.exe %*
"""
        script_name = f"run_{name}.bat"
    else:
        run_script = f"""#!/bin/bash

# Set threading environment variables to avoid MKL conflicts
export MKL_THREADING_LAYER=sequential
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/{name}/{name}.dist"

# Run the executable (try both .bin and no extension)
if [ -f "./{name}.bin" ]; then
    ./{name}.bin "$@"
else
    ./{name} "$@"
fi
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
4. Threading is set to single-threaded mode to avoid MKL conflicts
   - To use multiple threads, edit the run script and change OMP_NUM_THREADS
"""
    
    with open(dist_path / "README.md", "w") as f:
        f.write(readme)
    
    print(f"✓ Created distribution package at {dist_path}")

def main():
    """Main compilation process"""
    # Windows console fix
    if platform.system() == "Windows":
        import sys
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
    
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
    
    # Verify critical files
    if not verify_critical_files():
        print("\n❌ Compilation cancelled due to missing files.")
        return
    
    # Ask about config embedding
    print("\nConfiguration file handling:")
    print("1. Embed configs into binary (more secure, no external files)")
    print("2. Keep configs as external files (easier to modify)")
    
    config_choice = input("\nEnter choice (1-2) [default: 2]: ").strip() or "2"
    
    if config_choice == "1":
        # Try to embed configs
        if embed_configs():
            print("✓ Configs will be embedded in the binary")
        else:
            print("⚠️  Continuing with external config files")
    
    # Ask what to compile
    print("\nWhat would you like to compile?")
    print("1. inference.py only")
    print("2. inference_ray.py only")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    # Ask about cleanup
    print("\nCleanup options:")
    print("1. Keep intermediate build directories (for debugging)")
    print("2. Remove intermediate directories (save space)")
    
    cleanup_choice = input("\nEnter choice (1-2) [default: 2]: ").strip() or "2"
    cleanup = (cleanup_choice == "2")
    
    success = []
    
    if choice in ["1", "3"]:
        if compile_inference("inference.py", "inference"):
            create_distribution_package("inference", cleanup)
            success.append("inference")
    
    if choice in ["2", "3"]:
        if compile_inference("inference_ray.py", "inference_ray"):
            create_distribution_package("inference_ray", cleanup)
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



