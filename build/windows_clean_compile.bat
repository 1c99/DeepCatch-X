@echo off
REM Clean Windows compilation script for DCX

echo DCX Clean Compilation for Windows
echo =================================
echo.

REM Set environment to prevent matplotlib auto-import
set MPLBACKEND=Agg
set QT_QPA_PLATFORM=offscreen
set MATPLOTLIB_BACKEND=Agg

REM Run the actual compilation with updated script
echo Running updated compilation script...
python build\compile_with_nuitka.py

echo.
echo If compilation fails due to matplotlib, try:
echo 1. Create a new conda environment without matplotlib:
echo    conda create -n dcx_build python=3.11
echo    conda activate dcx_build
echo    pip install nuitka
echo.
echo 2. Then run this script again
pause