#!/usr/bin/env python3
"""Check and update Nuitka version"""
import subprocess
import sys

def check_nuitka_version():
    """Check current Nuitka version"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nuitka", "--version"],
            capture_output=True,
            text=True
        )
        print("Current Nuitka version:")
        print(result.stdout)
        
        # Extract version number
        for line in result.stdout.split('\n'):
            if 'Version' in line:
                version = line.strip()
                break
        
        # Check if version is old
        if "2.7" in version or "2.6" in version or "2.5" in version:
            print("\n⚠️  Your Nuitka version is older. Consider updating:")
            print("  pip install --upgrade nuitka")
            print("\nLatest features require Nuitka 2.8+")
        else:
            print("\n✓ Nuitka version looks good!")
            
    except Exception as e:
        print(f"Error checking version: {e}")

if __name__ == "__main__":
    check_nuitka_version()