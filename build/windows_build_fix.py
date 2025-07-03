#!/usr/bin/env python3
"""
Windows-specific Nuitka build fixes for DCX
"""
import os
import sys
import subprocess
import platform

def create_test_script():
    """Create a minimal test script to verify Nuitka works"""
    test_content = '''
import sys
print("DCX Test Script Running")
print(f"Python version: {sys.version}")
print(f"Arguments: {sys.argv}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"PyTorch import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")

print("\\nTest completed successfully!")
'''
    
    with open("test_nuitka.py", "w") as f:
        f.write(test_content)
    print("✓ Created test_nuitka.py")

def compile_test_script():
    """Compile the test script with minimal options"""
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--windows-console-mode=force",
        "--assume-yes-for-downloads",
        "--output-dir=test_dist",
        "test_nuitka.py"
    ]
    
    print("Compiling test script...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Test script compiled successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to compile test script: {e}")
        return False

def create_windows_inference_wrapper():
    """Create a Python wrapper that can be compiled more easily"""
    wrapper_content = '''
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force console output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# Print startup message
print("DCX Inference Starting...", flush=True)

try:
    # Import the main inference module
    from inference import main
    
    # Run the main function
    if __name__ == "__main__":
        print("Calling main function...", flush=True)
        main()
except Exception as e:
    print(f"Error during execution: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    with open("inference_wrapper.py", "w") as f:
        f.write(wrapper_content)
    print("✓ Created inference_wrapper.py")

def fix_inference_main():
    """Ensure inference.py has a proper main function"""
    print("\nChecking inference.py for main function...")
    
    # Read current inference.py
    with open("inference.py", "r") as f:
        content = f.read()
    
    # Check if main() exists
    if "def main():" not in content:
        print("⚠️  No main() function found in inference.py")
        print("   You need to add a main() function that wraps the argument parsing")
        print("   Example:")
        print("""
def main():
    parser = argparse.ArgumentParser()
    # ... add arguments ...
    args = parser.parse_args()
    
    # ... rest of the code ...

if __name__ == "__main__":
    main()
""")
        return False
    else:
        print("✓ main() function exists in inference.py")
        return True

def create_debug_batch():
    """Create a debug batch file for Windows"""
    debug_content = '''@echo off
echo DCX Debug Mode
echo ==============
echo.

REM Set environment variables
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

REM Try Python first
echo Testing Python environment:
python --version
echo.

REM Navigate to the executable directory
cd /d "%~dp0\\inference\\inference.dist"

REM List files in directory
echo Files in current directory:
dir /b *.exe
echo.

REM Try running with different methods
echo Method 1: Direct execution
inference.exe --help
echo Exit code: %ERRORLEVEL%
echo.

echo Method 2: With cmd /c
cmd /c inference.exe --help
echo Exit code: %ERRORLEVEL%
echo.

echo Method 3: With start /wait
start /wait inference.exe --help
echo Exit code: %ERRORLEVEL%
echo.

echo Method 4: Python subprocess test
python -c "import subprocess; subprocess.run(['inference.exe', '--help'])"
echo.

pause
'''
    
    with open("debug_inference.bat", "w") as f:
        f.write(debug_content)
    print("✓ Created debug_inference.bat")

def main():
    print("Windows Nuitka Build Diagnostic Tool")
    print("=" * 50)
    
    if platform.system() != "Windows":
        print("⚠️  This script is designed for Windows only")
        return
    
    print("\n1. Creating test files...")
    create_test_script()
    create_windows_inference_wrapper()
    create_debug_batch()
    
    print("\n2. Testing basic Nuitka compilation...")
    if compile_test_script():
        print("\n✓ Basic Nuitka compilation works!")
        print("  Test the compiled executable:")
        print("  cd test_dist\\test_nuitka.dist")
        print("  test_nuitka.exe")
    
    print("\n3. Checking inference.py structure...")
    fix_inference_main()
    
    print("\n4. Recommended steps:")
    print("   a) Test the basic compilation first")
    print("   b) If that works, try compiling inference_wrapper.py instead")
    print("   c) Use debug_inference.bat to diagnose the issue")
    print("   d) Check Windows Event Viewer for application errors")
    print("   e) Disable antivirus temporarily during testing")
    
    print("\n5. Alternative compilation command:")
    print("""
python -m nuitka ^
    --standalone ^
    --windows-console-mode=force ^
    --enable-console ^
    --assume-yes-for-downloads ^
    --plugin-enable=torch ^
    --plugin-enable=numpy ^
    --include-package=pydicom ^
    --include-package=nibabel ^
    --include-data-dir=configs=configs ^
    --include-data-dir=checkpoints=checkpoints ^
    --output-dir=dist_test ^
    inference.py
""")

if __name__ == "__main__":
    main()