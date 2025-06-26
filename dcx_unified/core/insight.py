#!/usr/bin/env python3
"""
Postprocessing Wrapper for DCX Unified System
Uses original DCX postprocessing algorithms
"""

import os
import sys
import numpy as np
from pathlib import Path
from postprocessing_original import dcx_postprocessing

class PostProcessingWrapper:
    """Wrapper to call original DCX postprocessing algorithms"""
    
    def __init__(self):
        self.core_dir = Path(__file__).parent
        self.postprocessor = dcx_postprocessing
    
    def aorta_diameter(self, aorta_files, output_dir):
        """
        Calculate aorta diameter using original algorithm
        
        Args:
            aorta_files: List of aorta segmentation files (.nii) or single file
            output_dir: Output directory for results
        """
        if isinstance(aorta_files, str):
            aorta_files = [aorta_files]
        
        print(f"Running aorta diameter calculation on {len(aorta_files)} files...")
        
        try:
            # Call original algorithm
            results = self.postprocessor.aorta_diameter(aorta_files, output_dir)
            
            # Print results
            for key, value in results.items():
                print(f"{key}: {value:.2f} mm")
            
            return results
            
        except Exception as e:
            print(f"Aorta diameter calculation failed: {e}")
            return {}
    
    def cardiothoracic_ratio(self, heart_file, lung_file, dicom_file, output_dir):
        """
        Calculate cardiothoracic ratio using original DCX algorithm
        
        Args:
            heart_file: Heart segmentation (.nii)
            lung_file: Lung segmentation (.nii)
            dicom_file: Original DICOM file
            output_dir: Output directory
        """
        print("Running cardiothoracic ratio calculation using original algorithm...")
        
        try:
            # Call original algorithm
            results = self.postprocessor.cardiothoracic_ratio(
                heart_file, lung_file, dicom_file
            )
            
            # Print results
            if 'cardiothoracic_ratio' in results:
                ctr = results['cardiothoracic_ratio']
                print(f"Cardiothoracic Ratio: {ctr:.3f}")
                print(f"Heart width: {results.get('heart_width_pixels', 0)} pixels")
                print(f"Thoracic width: {results.get('thoracic_width_pixels', 0)} pixels")
                
                # Save results file
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "cardiothoracic_ratio_results.txt"), "w") as f:
                    f.write(f"Cardiothoracic Ratio: {ctr:.3f}\n")
                    f.write(f"Heart width: {results.get('heart_width_pixels', 0)} pixels\n")
                    f.write(f"Thoracic width: {results.get('thoracic_width_pixels', 0)} pixels\n")
                    f.write(f"Normal range: 0.45-0.55\n")
                    if ctr > 0.55:
                        f.write("Status: ENLARGED\n")
                    else:
                        f.write("Status: NORMAL\n")
            
            return results
            
        except Exception as e:
            print(f"Cardiothoracic ratio calculation failed: {e}")
            return {}
    
    def peripheral_vessels(self, vessel_file, lung_file, output_dir):
        """
        Calculate peripheral vessel area using original algorithm
        
        Args:
            vessel_file: Vessel segmentation (.nii)
            lung_file: Lung segmentation (.nii)
            output_dir: Output directory
        """
        print("Running peripheral vessel analysis using original algorithm...")
        
        try:
            # Call original algorithm
            results = self.postprocessor.peripheral_vessels(vessel_file, lung_file)
            
            # Print results
            print(f"Total vessel count: {results.get('total_vessel_count', 0)} pixels")
            print(f"Peripheral vessel count: {results.get('peripheral_vessel_count', 0)} pixels")
            print(f"Central vessel count: {results.get('central_vessel_count', 0)} pixels")
            print(f"Peripheral percentage: {results.get('peripheral_vessel_percentage', 0):.2f}%")
            print(f"Vessel density: {results.get('vessel_density_percentage', 0):.2f}%")
            
            # Save results file
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "peripheral_vessel_results.txt"), "w") as f:
                f.write(f"Total vessel count: {results.get('total_vessel_count', 0)} pixels\n")
                f.write(f"Peripheral vessel count: {results.get('peripheral_vessel_count', 0)} pixels\n")
                f.write(f"Central vessel count: {results.get('central_vessel_count', 0)} pixels\n")
                f.write(f"Peripheral percentage: {results.get('peripheral_vessel_percentage', 0):.2f}%\n")
                f.write(f"Vessel density: {results.get('vessel_density_percentage', 0):.2f}%\n")
            
            return results
            
        except Exception as e:
            print(f"Peripheral vessel calculation failed: {e}")
            return {}
    
    def cleanup(self):
        """Clean up any temporary files"""
        pass


def main():
    """CLI interface for postprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DCX Postprocessing Wrapper')
    parser.add_argument('--function', required=True, 
                       choices=['aorta_diameter', 'cardiothoracic_ratio', 'peripheral_vessels'],
                       help='Postprocessing function to run')
    parser.add_argument('--input', nargs='+', required=True,
                       help='Input file(s)')
    parser.add_argument('--output', required=True,
                       help='Output directory')
    parser.add_argument('--dicom', help='Original DICOM file (for CTR)')
    
    args = parser.parse_args()
    
    wrapper = PostProcessingWrapper()
    
    if args.function == 'aorta_diameter':
        results = wrapper.aorta_diameter(args.input, args.output)
    elif args.function == 'cardiothoracic_ratio':
        if len(args.input) != 2:
            print("Error: CTR requires exactly 2 inputs (heart and lung)")
            return
        if not args.dicom:
            print("Error: CTR requires --dicom parameter")
            return
        results = wrapper.cardiothoracic_ratio(
            args.input[0], args.input[1], args.dicom, args.output
        )
    elif args.function == 'peripheral_vessels':
        if len(args.input) != 2:
            print("Error: Peripheral vessels requires exactly 2 inputs (vessel and lung)")
            return
        results = wrapper.peripheral_vessels(
            args.input[0], args.input[1], args.output
        )
    
    # Print results summary
    if results:
        print("\nResults:")
        for key, value in results.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()