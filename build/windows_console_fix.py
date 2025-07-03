"""
Add this to the top of inference.py for Windows console compatibility
"""

import sys
import os

# Windows console output fix
if sys.platform == 'win32':
    # Force UTF-8 encoding
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
    
    # Ensure console output is not buffered
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Try to attach to console if available
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.AttachConsole(-1)
    except:
        pass

# Add debug print to verify console works
print("DCX Inference System Initializing...", flush=True)