#!/usr/bin/env python3
"""
Clean up macOS-specific files from checkpoints directory
"""
import os
import shutil
import sys

def clean_checkpoints(directory="checkpoints"):
    """Remove macOS-specific files and directories"""
    removed_count = 0
    
    # Patterns to remove
    macos_patterns = [
        ".DS_Store",
        "._*",
        "__MACOSX"
    ]
    
    print(f"Cleaning {directory} directory...")
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Remove __MACOSX directories
        if "__MACOSX" in dirs:
            macos_path = os.path.join(root, "__MACOSX")
            print(f"Removing directory: {macos_path}")
            shutil.rmtree(macos_path)
            dirs.remove("__MACOSX")
            removed_count += 1
        
        # Remove .DS_Store and ._ files
        for file in files:
            if file == ".DS_Store" or file.startswith("._"):
                file_path = os.path.join(root, file)
                print(f"Removing file: {file_path}")
                os.remove(file_path)
                removed_count += 1
    
    # Also check if there's a __MACOSX directory at the same level
    if os.path.exists("__MACOSX"):
        print("Removing __MACOSX directory from root")
        shutil.rmtree("__MACOSX")
        removed_count += 1
    
    print(f"\n✓ Cleaned {removed_count} macOS-specific files/directories")
    
    # List remaining checkpoint files
    checkpoint_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.pth', '.pt', '.ckpt')):
                checkpoint_files.append(os.path.join(root, file))
    
    print(f"\n✓ Found {len(checkpoint_files)} checkpoint files:")
    for ckpt in sorted(checkpoint_files):
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        print(f"  - {os.path.basename(ckpt)} ({size_mb:.1f} MB)")

def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "checkpoints"
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found")
        return
    
    clean_checkpoints(directory)

if __name__ == "__main__":
    main()