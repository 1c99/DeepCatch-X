#!/usr/bin/env python3
"""
Test script for unified DCX inference system
"""
import os
import sys
import shutil
import argparse

def setup_test_environment():
    """Setup test environment with checkpoints and input files"""
    print("Setting up test environment...")
    
    # Create modules directory structure
    modules_dir = "modules"
    os.makedirs(modules_dir, exist_ok=True)
    
    # Define module mappings
    module_mappings = {
        'lung': '20241226_lung_inference_hjkim',
        'heart': '20241223_heart_inference_hjkim', 
        'airway': '20241226_airway_inferenence_hjkim',
        'covid': '20241223_covid_inference_hjkim',
        'vessel': '20240425_vascular_inference_hjkim',
        'bone': '20250320_bone_inference/20240516_bone_inference_hjkim',
        'heart_volumetry': '20241223_heart_vol_inference_hjkim',
        'lung_volumetry': '20241223_lungreg_inference_hjkim'
    }
    
    # Copy checkpoints and input files
    for module, source_dir in module_mappings.items():
        module_dir = os.path.join(modules_dir, module)
        os.makedirs(module_dir, exist_ok=True)
        
        # Copy checkpoints
        source_checkpoints = os.path.join("..", source_dir, "checkpoints")
        target_checkpoints = os.path.join(module_dir, "checkpoints")
        if os.path.exists(source_checkpoints):
            if os.path.exists(target_checkpoints):
                shutil.rmtree(target_checkpoints)
            shutil.copytree(source_checkpoints, target_checkpoints)
            print(f"Copied checkpoints for {module}")
        
        # Copy sample input files
        source_input = os.path.join("..", source_dir, "input")
        target_input = os.path.join(module_dir, "input")
        if os.path.exists(source_input):
            if os.path.exists(target_input):
                shutil.rmtree(target_input)
            shutil.copytree(source_input, target_input)
            print(f"Copied input files for {module}")
    
    print("Test environment setup complete!")

def run_test(module, input_file=None):
    """Run test for a specific module"""
    print(f"\nTesting {module} module...")
    
    # Find input file
    if not input_file:
        input_dir = f"modules/{module}/input"
        if os.path.exists(input_dir):
            dcm_files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
            if dcm_files:
                input_file = os.path.join(input_dir, dcm_files[0])
            else:
                print(f"No DICOM files found in {input_dir}")
                return False
        else:
            print(f"Input directory not found: {input_dir}")
            return False
    
    # Set output file
    output_file = f"test_output_{module}.nii"
    
    # Prepare command
    cmd = f"python inference.py --module {module} --input {input_file} --output {output_file}"
    
    # Add lung mask for modules that need it
    if module in ['covid', 'vessel']:
        lung_mask_path = f"modules/{module}/input"
        if os.path.exists(lung_mask_path):
            nii_files = [f for f in os.listdir(lung_mask_path) if f.endswith('_lung.nii')]
            if nii_files:
                cmd += f" --lung_mask {os.path.join(lung_mask_path, nii_files[0])}"
            else:
                print(f"No lung mask found for {module}")
                return False
    
    # Run test
    print(f"Running: {cmd}")
    result = os.system(cmd)
    
    if result == 0:
        print(f"✓ {module} test passed")
        return True
    else:
        print(f"✗ {module} test failed")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test unified DCX system')
    parser.add_argument('--setup', action='store_true', help='Setup test environment')
    parser.add_argument('--module', type=str, help='Test specific module')
    parser.add_argument('--all', action='store_true', help='Test all modules')
    parser.add_argument('--input', type=str, help='Specific input file')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_test_environment()
        return
    
    if args.module:
        success = run_test(args.module, args.input)
        sys.exit(0 if success else 1)
    
    if args.all:
        modules = ['lung', 'heart', 'airway', 'covid', 'vessel', 'bone', 'heart_volumetry', 'lung_volumetry']
        successes = 0
        
        for module in modules:
            try:
                if run_test(module):
                    successes += 1
            except Exception as e:
                print(f"Error testing {module}: {e}")
        
        print(f"\nTest results: {successes}/{len(modules)} modules passed")
        sys.exit(0 if successes == len(modules) else 1)
    
    print("Use --setup to setup test environment, --module <name> to test specific module, or --all to test all modules")

if __name__ == '__main__':
    main()