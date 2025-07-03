#!/usr/bin/env python3
"""
Fix for Anaconda matplotlib Qt backend issues
"""
import os
import sys
import subprocess
import site

def diagnose_matplotlib():
    """Diagnose matplotlib installation and backends"""
    print("Diagnosing matplotlib installation...")
    
    # Check matplotlib
    try:
        # Set backend BEFORE import
        os.environ['MPLBACKEND'] = 'Agg'
        import matplotlib
        print(f"✓ matplotlib version: {matplotlib.__version__}")
        print(f"  Config dir: {matplotlib.get_configdir()}")
        print(f"  Backend: {matplotlib.get_backend()}")
        
        # Check available backends
        import matplotlib.backends as backends
        backend_dir = os.path.dirname(backends.__file__)
        print(f"  Backends directory: {backend_dir}")
        
    except ImportError:
        print("✗ matplotlib not installed")
        return False
    
    return True

def create_matplotlibrc():
    """Create a matplotlibrc file to force Agg backend"""
    content = """# Force Agg backend for Nuitka compilation
backend : Agg
interactive : False
"""
    
    # Create in current directory
    with open("matplotlibrc", "w") as f:
        f.write(content)
    print("✓ Created matplotlibrc in current directory")
    
    # Also try to create in user config directory
    try:
        import matplotlib
        config_dir = matplotlib.get_configdir()
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, "matplotlibrc"), "w") as f:
            f.write(content)
        print(f"✓ Created matplotlibrc in {config_dir}")
    except:
        pass

def create_site_customize():
    """Create sitecustomize.py to set matplotlib backend early"""
    content = """# Site customization for DCX compilation
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Disable IPython matplotlib integration
os.environ['IPYTHON_MATPLOTLIB'] = 'none'
"""
    
    # Find site-packages directory
    site_packages = site.getsitepackages()[0]
    customize_path = os.path.join(site_packages, "sitecustomize.py")
    
    print(f"\nWould create sitecustomize.py at: {customize_path}")
    print("This would affect ALL Python scripts in this environment.")
    
    response = input("Create sitecustomize.py? (y/n): ")
    if response.lower() == 'y':
        try:
            with open(customize_path, "w") as f:
                f.write(content)
            print("✓ Created sitecustomize.py")
            return customize_path
        except PermissionError:
            print("✗ Permission denied. Try running as administrator.")
    
    return None

def test_import():
    """Test if we can import without Qt errors"""
    print("\nTesting import with Agg backend...")
    
    # Start fresh Python process
    test_script = """
import os
os.environ['MPLBACKEND'] = 'Agg'
print("Environment set")

try:
    import matplotlib
    matplotlib.use('Agg')
    print(f"matplotlib imported successfully, backend: {matplotlib.get_backend()}")
except Exception as e:
    print(f"Error: {e}")
"""
    
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
        env={**os.environ, 'MPLBACKEND': 'Agg'}
    )
    
    print("Output:", result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

def main():
    print("Anaconda Matplotlib Qt Fix")
    print("=" * 50)
    
    # Set environment
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Run diagnostics
    diagnose_matplotlib()
    
    # Create config files
    create_matplotlibrc()
    
    # Test
    if test_import():
        print("\n✓ Matplotlib can be imported without Qt!")
        print("\nNow try compilation with:")
        print("  set MPLBACKEND=Agg")
        print("  python build\\compile_with_nuitka.py")
    else:
        print("\n✗ Still having issues.")
        print("\nTry creating a clean environment:")
        print("  conda create -n dcx_nuitka python=3.11")
        print("  conda activate dcx_nuitka")
        print("  pip install nuitka torch numpy scipy pillow pydicom nibabel opencv-python scikit-image")
        print("  python build\\compile_with_nuitka.py")

if __name__ == "__main__":
    main()