#!/usr/bin/env python3
"""
DCX Unified Measurement Collector
Combines all measurements into a single CSV file
"""

import os
import csv
import json
import glob
from datetime import datetime
import re

class MeasurementCollector:
    """Collects and consolidates all DCX measurements into a single CSV"""
    
    def __init__(self, output_dir, dicom_filename):
        self.output_dir = output_dir
        self.dicom_filename = dicom_filename
        self.measurements = {
            'patient_id': os.path.splitext(dicom_filename)[0],
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dicom_file': dicom_filename
        }
        
    def add_segmentation_measurement(self, module_name, area_cm2=None, volume_ml=None):
        """Add segmentation area/volume measurements"""
        if area_cm2 is not None:
            self.measurements[f'{module_name}_area_cm2'] = round(area_cm2, 2)
        if volume_ml is not None:
            self.measurements[f'{module_name}_volume_ml'] = round(volume_ml, 2)
            self.measurements[f'{module_name}_volume_l'] = round(volume_ml/1000, 3)
    
    def add_regression_measurement(self, module_name, values_dict):
        """Add regression measurements (HU values, etc.)"""
        for key, value in values_dict.items():
            if isinstance(value, (int, float)):
                self.measurements[f'{module_name}_{key}'] = round(value, 2)
            else:
                self.measurements[f'{module_name}_{key}'] = value
    
    def add_classification_measurement(self, module_name, probability=None, classification=None, threshold=None):
        """Add classification results"""
        if probability is not None:
            self.measurements[f'{module_name}_probability'] = round(probability, 4)
        if classification is not None:
            self.measurements[f'{module_name}_classification'] = classification
        if threshold is not None:
            self.measurements[f'{module_name}_threshold'] = threshold
    
    def add_postprocessing_measurement(self, postproc_type, values_dict):
        """Add postprocessing measurements"""
        for key, value in values_dict.items():
            if isinstance(value, (int, float)):
                self.measurements[f'{postproc_type}_{key}'] = round(value, 3)
            else:
                self.measurements[f'{postproc_type}_{key}'] = value
    
    def collect_from_files(self):
        """Automatically collect measurements from existing files in output directory"""
        # Collect from LAA CSV if exists
        self._collect_laa_measurements()
        
        # Collect from postprocessing text files
        self._collect_postprocessing_measurements()
        
        # Collect from NIfTI file metadata (pixel spacing, dimensions)
        self._collect_file_metadata()
    
    def _collect_laa_measurements(self):
        """Extract LAA measurements from output_emphysema.csv"""
        laa_csv_path = os.path.join(self.output_dir, 'output_emphysema.csv')
        if os.path.exists(laa_csv_path):
            try:
                with open(laa_csv_path, 'r') as f:
                    content = f.read()
                    # Parse the LAA results (format: filename: {'emphysema_prob': X, 'desc': Y, 'area': Z})
                    if self.dicom_filename in content:
                        import ast
                        # Extract the dictionary part
                        pattern = rf"{re.escape(self.dicom_filename)}:\s*(\{{.*?\}})"
                        match = re.search(pattern, content)
                        if match:
                            result_dict = ast.literal_eval(match.group(1))
                            self.add_classification_measurement('laa_emphysema', 
                                                              result_dict.get('emphysema_prob', 0)/100,
                                                              result_dict.get('desc', 'Unknown'))
                            if 'area' in result_dict:
                                self.measurements['laa_emphysema_area_cm2'] = result_dict['area']
            except Exception as e:
                print(f"Could not parse LAA measurements: {e}")
    
    def _collect_postprocessing_measurements(self):
        """Extract measurements from postprocessing text files"""
        # Cardiothoracic ratio
        ctr_file = os.path.join(self.output_dir, 'cardiothoracic_ratio_results.txt')
        if os.path.exists(ctr_file):
            try:
                with open(ctr_file, 'r') as f:
                    content = f.read()
                    # Extract values using regex
                    heart_width = re.search(r'Heart width:\s*(\d+)\s*pixels', content)
                    thoracic_width = re.search(r'Thoracic width:\s*(\d+)\s*pixels', content)
                    ctr_ratio = re.search(r'Cardiothoracic ratio:\s*([\d.]+)', content)
                    interpretation = re.search(r'INTERPRETATION:\s*(.+?)\\n', content)
                    
                    if heart_width:
                        self.measurements['cardiothoracic_heart_width_pixels'] = int(heart_width.group(1))
                    if thoracic_width:
                        self.measurements['cardiothoracic_thoracic_width_pixels'] = int(thoracic_width.group(1))
                    if ctr_ratio:
                        self.measurements['cardiothoracic_ratio'] = float(ctr_ratio.group(1))
                    if interpretation:
                        self.measurements['cardiothoracic_interpretation'] = interpretation.group(1).strip()
            except Exception as e:
                print(f"Could not parse cardiothoracic measurements: {e}")
        
        # Peripheral vessels
        vessel_file = os.path.join(self.output_dir, 'peripheral_vessels_results.txt')
        if os.path.exists(vessel_file):
            try:
                with open(vessel_file, 'r') as f:
                    content = f.read()
                    vessel_area = re.search(r'Vessel area:\s*([\d.]+)\s*cm²', content)
                    lung_area = re.search(r'Lung area:\s*([\d.]+)\s*cm²', content)
                    vessel_pct = re.search(r'Vessel percentage:\s*([\d.]+)%', content)
                    
                    if vessel_area:
                        self.measurements['peripheral_vessel_area_cm2'] = float(vessel_area.group(1))
                    if lung_area:
                        self.measurements['peripheral_lung_area_cm2'] = float(lung_area.group(1))
                    if vessel_pct:
                        self.measurements['peripheral_vessel_percentage'] = float(vessel_pct.group(1))
            except Exception as e:
                print(f"Could not parse vessel measurements: {e}")
        
        # Aorta diameter results (if any output files exist)
        aorta_files = glob.glob(os.path.join(self.output_dir, '*aorta*diameter*.txt'))
        for aorta_file in aorta_files:
            try:
                with open(aorta_file, 'r') as f:
                    content = f.read()
                    max_diameter = re.search(r'Maximum diameter:\s*([\d.]+)\s*mm', content)
                    if max_diameter:
                        self.measurements['aorta_max_diameter_mm'] = float(max_diameter.group(1))
            except Exception as e:
                print(f"Could not parse aorta measurements: {e}")
    
    def _collect_file_metadata(self):
        """Collect metadata from NIfTI files"""
        try:
            import nibabel as nib
            # Get pixel spacing from any NIfTI file
            nii_files = glob.glob(os.path.join(self.output_dir, '*.nii'))
            if nii_files:
                nii = nib.load(nii_files[0])
                header = nii.header
                if hasattr(header, 'get_zooms'):
                    pixel_spacing = header.get_zooms()
                    self.measurements['pixel_spacing_mm'] = round(pixel_spacing[0], 3)
                
                # Image dimensions
                data_shape = nii.get_fdata().shape
                self.measurements['image_height'] = data_shape[0]
                self.measurements['image_width'] = data_shape[1]
        except Exception as e:
            print(f"Could not collect file metadata: {e}")
    
    def save_to_csv(self, csv_filename='dcx_measurements.csv'):
        """Save all measurements to CSV file"""
        csv_path = os.path.join(self.output_dir, csv_filename)
        
        # Check if CSV exists to determine if we need headers
        file_exists = os.path.exists(csv_path)
        
        # Define column order for better organization
        priority_columns = [
            'patient_id', 'processing_date', 'dicom_file',
            'pixel_spacing_mm', 'image_height', 'image_width'
        ]
        
        # Get all measurement keys and sort them
        all_keys = set(self.measurements.keys())
        remaining_keys = sorted(all_keys - set(priority_columns))
        ordered_keys = priority_columns + remaining_keys
        
        # Write to CSV
        mode = 'a' if file_exists else 'w'
        with open(csv_path, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=ordered_keys)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write measurements
            writer.writerow(self.measurements)
        
        print(f"Measurements saved to: {csv_path}")
        return csv_path
    
    def get_summary(self):
        """Get a summary of all collected measurements"""
        summary = {
            'total_measurements': len(self.measurements),
            'categories': {}
        }
        
        # Categorize measurements
        for key in self.measurements.keys():
            if any(x in key for x in ['area', 'volume']):
                category = 'anatomical_measurements'
            elif any(x in key for x in ['probability', 'classification']):
                category = 'classification_results'
            elif any(x in key for x in ['ratio', 'diameter', 'width', 'percentage']):
                category = 'morphometric_analysis'
            elif any(x in key for x in ['hu', 'density']):
                category = 'density_measurements'
            else:
                category = 'metadata'
            
            if category not in summary['categories']:
                summary['categories'][category] = []
            summary['categories'][category].append(key)
        
        return summary


def create_comprehensive_csv(output_dir, dicom_filename, measurements_data=None):
    """
    Create comprehensive CSV with all measurements
    
    Args:
        output_dir: Directory containing results
        dicom_filename: Original DICOM filename
        measurements_data: Optional dict of measurements to add directly
    """
    collector = MeasurementCollector(output_dir, dicom_filename)
    
    # Add direct measurements if provided
    if measurements_data:
        for key, value in measurements_data.items():
            collector.measurements[key] = value
    
    # Collect from existing files
    collector.collect_from_files()
    
    # Save to CSV
    csv_path = collector.save_to_csv()
    
    # Print summary
    summary = collector.get_summary()
    print(f"\\nMeasurement Summary:")
    print(f"Total measurements: {summary['total_measurements']}")
    for category, keys in summary['categories'].items():
        print(f"  {category}: {len(keys)} measurements")
    
    return csv_path, collector.measurements


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect DCX measurements into CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing DCX results')
    parser.add_argument('--dicom_file', type=str, required=True,
                       help='Original DICOM filename')
    parser.add_argument('--csv_name', type=str, default='dcx_measurements.csv',
                       help='Output CSV filename')
    
    args = parser.parse_args()
    
    create_comprehensive_csv(args.output_dir, args.dicom_file)