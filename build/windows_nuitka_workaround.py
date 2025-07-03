#!/usr/bin/env python3
"""
Windows Nuitka workaround for matplotlib issues
"""
import os
import sys
import subprocess
import shutil

def create_minimal_test():
    """Create a truly minimal test without any problematic imports"""
    test_content = '''
import sys
print("Minimal test running!")
print(f"Python: {sys.version}")
print("Success!")
'''
    with open("minimal_test.py", "w") as f:
        f.write(test_content)
    print("✓ Created minimal_test.py")

def compile_minimal():
    """Compile the minimal test"""
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--windows-console-mode=force",
        "--output-dir=minimal_dist",
        "minimal_test.py"
    ]
    
    print("\nCompiling minimal test...")
    try:
        subprocess.run(cmd, check=True)
        print("✓ Minimal test compiled successfully!")
        print("\nTest it with:")
        print("  cd minimal_dist\\minimal_test.dist")
        print("  minimal_test.exe")
        return True
    except:
        print("✗ Even minimal test failed - Nuitka installation issue")
        return False

def uninstall_matplotlib():
    """Temporarily uninstall matplotlib to avoid conflicts"""
    print("\nChecking matplotlib installation...")
    try:
        import matplotlib
        print(f"Found matplotlib {matplotlib.__version__}")
        response = input("Temporarily uninstall matplotlib to test compilation? (y/n): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "matplotlib"])
            print("✓ Matplotlib uninstalled")
            return True
    except ImportError:
        print("✓ Matplotlib not installed")
    return False

def create_nuitka_config():
    """Create a Nuitka configuration file to handle imports"""
    config_content = '''# Nuitka configuration
# Force matplotlib backend
[Environment]
MPLBACKEND = "Agg"

[Compilation]
# Exclude problematic imports
nofollow-import-to = [
    "matplotlib.backends.qt_compat",
    "matplotlib.backends.backend_qt",
    "matplotlib.backends.backend_qt5",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.pyplot",
    "IPython",
    "jupyter",
]

# Force include only needed backends
include-module = [
    "matplotlib.backends.backend_agg",
]
'''
    with open("nuitka-config.cfg", "w") as f:
        f.write(config_content)
    print("✓ Created nuitka-config.cfg")

def compile_with_config():
    """Try compilation with configuration file"""
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--windows-console-mode=force",
        "--assume-yes-for-downloads",
        "--module-parameter=torch-disable-jit=yes",
        "--nofollow-import-to=matplotlib",  # Completely skip matplotlib
        "--nofollow-import-to=IPython",
        "--nofollow-import-to=jupyter",
        "--output-dir=test_dist_no_mpl",
        "test_nuitka.py"
    ]
    
    print("\nCompiling with matplotlib completely excluded...")
    try:
        subprocess.run(cmd, check=True)
        print("✓ Compilation successful without matplotlib!")
        return True
    except:
        print("✗ Compilation failed even without matplotlib")
        return False

def create_inference_launcher():
    """Create a simple launcher that sets environment before running inference"""
    launcher_content = '''@echo off
REM DCX Inference Launcher for Windows

REM Set environment variables
set MPLBACKEND=Agg
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set QT_QPA_PLATFORM=offscreen

REM Disable matplotlib GUI
set MATPLOTLIB_BACKEND=Agg
set MPLCONFIGDIR=%TEMP%

REM Navigate to executable directory
cd /d "%~dp0"

REM Run inference
inference.exe %*
'''
    with open("launch_inference.bat", "w") as f:
        f.write(launcher_content)
    print("✓ Created launch_inference.bat")

def main():
    print("Windows Nuitka Matplotlib Workaround")
    print("=" * 50)
    
    # Create all helper files
    create_minimal_test()
    create_nuitka_config()
    create_inference_launcher()
    
    # Test minimal compilation
    if compile_minimal():
        print("\n✓ Basic Nuitka works!")
    
    # Try without matplotlib
    print("\nAttempting compilation without matplotlib...")
    compile_with_config()
    
    print("\n" + "=" * 50)
    print("SOLUTIONS:")
    print("1. Use launch_inference.bat instead of running .exe directly")
    print("2. Compile with: --nofollow-import-to=matplotlib")
    print("3. Set environment before running:")
    print("   set MPLBACKEND=Agg")
    print("   set QT_QPA_PLATFORM=offscreen")
    print("\n4. Or modify inference.py to make matplotlib optional:")
    print("   try:")
    print("       import matplotlib")
    print("       matplotlib.use('Agg')")
    print("   except ImportError:")
    print("       pass")

if __name__ == "__main__":
    main()