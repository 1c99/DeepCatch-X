# Windows Build Guide for DCX

## Prerequisites

1. **Python 3.10 or 3.11** (3.12 may have compatibility issues)
   ```cmd
   python --version
   ```

2. **Visual Studio Build Tools** or **Visual Studio Community**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install "Desktop development with C++" workload
   - Required for compiling Python extensions

3. **Clean Python Environment** (Recommended)
   ```cmd
   python -m venv dcx_build_env
   dcx_build_env\Scripts\activate
   pip install -r requirements.txt
   pip install nuitka
   ```

## Common Windows Issues and Solutions

### 1. PyDicom urls.json Error
**Error**: `FileNotFoundError: urls.json`
**Solution**: The build script now includes `--include-package-data=pydicom`

### 2. Console Output Issues
**Error**: No output or garbled text
**Solution**: The script now sets UTF-8 encoding for Windows console

### 3. Missing DLL Errors
**Error**: Missing DLL files at runtime
**Solution**: The script now includes DLL files for torch and opencv

### 4. Antivirus Interference
**Issue**: Build fails or executable is deleted
**Solution**: 
- Temporarily disable Windows Defender during build
- Add build directory to exclusions
- Whitelist the generated executable

### 5. Path Length Limitations
**Error**: Path too long errors
**Solution**: 
- Build from a short path (e.g., `C:\DCX\`)
- Enable long path support in Windows:
  ```cmd
  reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
  ```

## Step-by-Step Build Process

1. **Prepare Environment**
   ```cmd
   cd C:\path\to\dcx_project
   python -m venv build_env
   build_env\Scripts\activate
   pip install -r requirements.txt
   pip install nuitka
   ```

2. **Set Environment Variables**
   ```cmd
   set MPLBACKEND=Agg
   set QT_QPA_PLATFORM=offscreen
   ```

3. **Run Build Script**
   ```cmd
   python build\compile_with_nuitka.py
   ```

4. **Test the Executable**
   ```cmd
   cd dist\inference_package
   run_inference.bat -h
   ```

## Build Script Features

The updated `compile_with_nuitka.py` includes:

- ✅ Automatic package data inclusion
- ✅ Windows console UTF-8 fixes
- ✅ DLL dependency handling
- ✅ File verification before build
- ✅ macOS file cleanup (e.g., .DS_Store)
- ✅ Multiprocessing plugin for Windows
- ✅ Low memory mode for constrained systems

## Troubleshooting

### Build Fails with Memory Error
```cmd
# Use low-memory mode (already enabled for Windows)
# Or increase virtual memory in Windows settings
```

### Missing Package Data
Check that all required data files are included:
```python
--include-package-data=<package_name>
```

### Runtime Errors
1. Check Windows Event Viewer for detailed errors
2. Run with debug output:
   ```cmd
   set PYTHONVERBOSE=1
   run_inference.bat -h
   ```

### Performance Issues
The build script sets single-threaded mode by default to avoid conflicts:
```cmd
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
```

To enable multi-threading, edit `run_inference.bat` and change these values.

## Post-Build Checklist

- [ ] Executable runs without errors
- [ ] All checkpoint files are in place
- [ ] Config files are accessible
- [ ] Test with sample DICOM file
- [ ] Verify output generation
- [ ] Check memory usage
- [ ] Test on clean Windows installation

## Distribution

After successful build:
1. The executable is in `dist\inference_package\`
2. Include all files from this directory
3. Users need the entire directory, not just the .exe
4. No Python installation required on target machine

## Notes

- Build time: 15-30 minutes depending on system
- Final size: ~2-3 GB including PyTorch
- Requires ~8GB RAM during compilation
- Windows Defender may slow down the process