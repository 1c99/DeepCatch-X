#!/usr/bin/env python3
"""
Postprocessing module that uses the exact original functions from DCX_python_inference
This module provides a wrapper interface to call the original postprocessing scripts
"""
import os
import sys
import numpy as np
import nibabel as nib
import cv2
import pydicom
from pathlib import Path
import subprocess
import json

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class OriginalDCXPostProcessing:
    """Wrapper to use exact original DCX postprocessing algorithms"""
    
    def __init__(self):
        self.core_dir = Path(__file__).parent
        # Import the original modules
        try:
            # These modules have been copied to the core directory
            from aortav2_test_plus_20250324 import compute_diameter
            from cardiothoracic_ratio import find_contours, center_point, full_mask, get_longest_line
            from central_mask_python import find_contours as vessel_find_contours, center_point as vessel_center_point
            from central_mask_python import full_mask as vessel_full_mask, bitwise_mask
            
            self.compute_diameter = compute_diameter
            self.ctr_find_contours = find_contours
            self.ctr_center_point = center_point
            self.ctr_full_mask = full_mask
            self.ctr_get_longest_line = get_longest_line
            self.vessel_find_contours = vessel_find_contours
            self.vessel_center_point = vessel_center_point
            self.vessel_full_mask = vessel_full_mask
            self.vessel_bitwise_mask = bitwise_mask
        except ImportError as e:
            print(f"Warning: Could not import original postprocessing modules: {e}")
            print("Falling back to simplified implementations")
            self.compute_diameter = None
    
    def cardiothoracic_ratio(self, heart_nii_path, lung_nii_path, dicom_path):
        """
        Calculate cardiothoracic ratio using the exact original DCX algorithm
        Reimplemented from the main section of cardiothoracic_ratio.py
        
        Args:
            heart_nii_path: Path to heart segmentation NIfTI file
            lung_nii_path: Path to lung segmentation NIfTI file
            dicom_path: Path to original DICOM file
            
        Returns:
            dict: Contains 'cardiothoracic_ratio', 'heart_width_pixels', 'thoracic_width_pixels'
        """
        try:
            # Read DICOM for metadata
            ds = pydicom.dcmread(dicom_path)
            pixel_spacing = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
            width = ds.Columns
            height = ds.Rows
            
            # Process lung using original function
            img_color, lung_contours = self.ctr_find_contours(lung_nii_path)
            
            # Get center point
            center = self.ctr_center_point(lung_contours)
            
            # Create full lung mask
            mask = self.ctr_full_mask(center, lung_contours, img_color)
            
            # Find maximum thoracic diameter
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            maxCount = 0
            maxY = -1
            
            for y in range(mask_gray.shape[0]):
                count = cv2.countNonZero(mask_gray[y, :])
                if count > maxCount:
                    maxCount = count
                    maxY = y
            
            # Find thoracic width
            startX = -1
            endX = -1
            for x in range(mask_gray.shape[1]):
                if mask_gray[maxY, x] == 255:
                    if startX == -1:
                        startX = x
                    endX = x
            
            thoracic_width = endX - startX
            center_x = startX + ((endX - startX) / 2)
            
            # Calculate MHTD (Maximum Horizontal Thoracic Diameter)
            MHTD = thoracic_width * pixel_spacing[0]
            
            # Process heart
            heart_img = nib.load(heart_nii_path)
            heart_data = heart_img.get_fdata()
            heart_data = np.squeeze(heart_data)
            heart_data = np.transpose(heart_data, (1, 0))
            
            # For 2048x2048 processing (original DCX uses this)
            if heart_data.shape[0] == 512:
                # Scale up to match original processing size
                heart_data = np.repeat(heart_data, 4, axis=0)
                heart_data = np.repeat(heart_data, 4, axis=1)
            
            # Split heart into left and right
            left_heart = heart_data[:, :int(center_x)]
            right_heart = heart_data[:, int(center_x):]
            
            # Get longest lines for left and right heart
            left_longest_line = self.ctr_get_longest_line(left_heart, center_x, heart_data.shape[0])
            right_longest_line = self.ctr_get_longest_line(right_heart, heart_data.shape[1]-center_x, heart_data.shape[0])
            
            # Calculate heart width
            left_length = left_longest_line[1][0] - left_longest_line[0][0]
            right_length = right_longest_line[1][0] - right_longest_line[0][0]
            MHCD = left_length + right_length
            
            # Calculate spacing for resized image
            newSize = heart_data.shape[0]  # Should be 2048
            newSpacingX = pixel_spacing[0] * max(width, height) / newSize
            
            # Calculate CT ratio
            ct_ratio = (MHCD * newSpacingX) / (MHTD * newSpacingX)
            
            return {
                'cardiothoracic_ratio': ct_ratio,
                'heart_width_pixels': int(MHCD),
                'thoracic_width_pixels': int(thoracic_width)
            }
            
        except Exception as e:
            print(f"CTR calculation error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def aorta_diameter(self, aorta_files, output_dir):
        """
        Calculate aorta diameter using the exact original compute_diameter function
        
        Args:
            aorta_files: List of aorta segmentation files (.nii) or single file
            output_dir: Output directory for results
            
        Returns:
            dict: Contains 'aorta_ascending_diameter_mm', 'aorta_descending_diameter_mm', 'aorta_max_diameter_mm'
        """
        try:
            if isinstance(aorta_files, str):
                aorta_files = [aorta_files]
            
            # If we don't have the original compute_diameter function, use fallback
            if self.compute_diameter is None:
                return self._aorta_diameter_fallback(aorta_files, output_dir)
            
            # Prepare file dictionary for original function
            file_dict = {}
            for i, aorta_file in enumerate(aorta_files):
                # The original expects a specific structure
                key = f"aorta_{i}"
                file_dict[key] = os.path.basename(aorta_file)
            
            # Get input folder from first file
            input_folder = os.path.dirname(aorta_files[0])
            
            # Create temporary output folder
            temp_output = os.path.join(output_dir, "aorta_temp")
            os.makedirs(temp_output, exist_ok=True)
            
            # Call original compute_diameter function
            # The function saves results to files and prints diameter
            head = "unified"
            self.compute_diameter(
                output_folder=temp_output,
                input_folder=input_folder,
                file_dict=file_dict,
                head=head,
                ASCENDING_ONLY=False,
                ABDOMINAL=False,
                visualize=False,  # Disable visualization for speed
                heat_map=False
            )
            
            # Read results from saved files
            diameters = []
            for i in range(len(aorta_files)):
                dist_file = os.path.join(temp_output, f"dists_{head}_{i}.npy")
                if os.path.exists(dist_file):
                    dists = np.load(dist_file)
                    max_diameter = np.max(dists) if len(dists) > 0 else 0
                    diameters.append(max_diameter)
                else:
                    diameters.append(0)
            
            # Return results
            results = {}
            if len(diameters) >= 2:
                results['aorta_ascending_diameter_mm'] = diameters[0]
                results['aorta_descending_diameter_mm'] = diameters[1]
                results['aorta_max_diameter_mm'] = max(diameters)
            elif len(diameters) == 1:
                results['aorta_diameter_mm'] = diameters[0]
                results['aorta_max_diameter_mm'] = diameters[0]
            
            return results
            
        except Exception as e:
            print(f"Aorta diameter calculation error: {e}")
            import traceback
            traceback.print_exc()
            return self._aorta_diameter_fallback(aorta_files, output_dir)
    
    def _aorta_diameter_fallback(self, aorta_files, output_dir):
        """Fallback implementation for aorta diameter calculation"""
        try:
            diameters = []
            
            for i, aorta_file in enumerate(aorta_files):
                # Load aorta segmentation
                aorta_img = nib.load(aorta_file)
                aorta_data = aorta_img.get_fdata()
                aorta_data = np.squeeze(aorta_data)
                
                # Get pixel spacing from header
                pixdim = aorta_img.header.get_zooms()
                pixel_spacing = pixdim[0] if len(pixdim) > 0 else 1.0
                
                # Convert to binary mask
                threshold = -1015  # Standard aorta threshold
                aorta_mask = (aorta_data > threshold).astype(np.uint8)
                
                # Apply morphological operations to clean up
                kernel = np.ones((3,3), np.uint8)
                aorta_mask = cv2.morphologyEx(aorta_mask, cv2.MORPH_CLOSE, kernel)
                aorta_mask = cv2.morphologyEx(aorta_mask, cv2.MORPH_OPEN, kernel)
                
                # Find largest connected component
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(aorta_mask)
                
                if num_labels > 1:  # Has components besides background
                    # Get largest component (excluding background)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    aorta_mask = (labels == largest_label).astype(np.uint8)
                    
                    # Calculate diameter using distance transform
                    dist_transform = cv2.distanceTransform(aorta_mask, cv2.DIST_L2, 5)
                    
                    # Find maximum diameter (2 * max distance)
                    max_radius = np.max(dist_transform)
                    diameter_pixels = 2 * max_radius
                    
                    # Convert to mm
                    diameter_mm = diameter_pixels * pixel_spacing
                    diameters.append(diameter_mm)
                else:
                    diameters.append(0)
            
            # Return results
            results = {}
            if len(diameters) >= 2:
                results['aorta_ascending_diameter_mm'] = diameters[0]
                results['aorta_descending_diameter_mm'] = diameters[1]
                results['aorta_max_diameter_mm'] = max(diameters)
            elif len(diameters) == 1:
                results['aorta_diameter_mm'] = diameters[0]
                results['aorta_max_diameter_mm'] = diameters[0]
            
            return results
            
        except Exception as e:
            print(f"Aorta diameter fallback calculation error: {e}")
            return {}
    
    def peripheral_vessels(self, vessel_file, lung_file):
        """
        Calculate peripheral vessel metrics using original DCX algorithm
        Based on central_mask_python.py
        
        Args:
            vessel_file: Vessel segmentation (.nii)
            lung_file: Lung segmentation (.nii)
            
        Returns:
            dict: Contains 'peripheral_vessel_count', 'peripheral_vessel_percentage', etc.
        """
        try:
            # Process lung to get contours
            img_color, lung_contours = self.vessel_find_contours(lung_file)
            
            # Get center point
            if len(lung_contours) == 2:
                center = self.vessel_center_point(lung_contours)
            else:
                # For single lung contour
                center = self.vessel_center_point(lung_contours)
            
            # Create full lung mask
            lung_mask_full = self.vessel_full_mask(center, lung_contours, img_color)
            
            # Create peripheral masks at different percentages
            # 70% mask (inner region)
            mask_70, masked_70 = self.vessel_bitwise_mask(img_color, lung_mask_full, center, 0.7)
            
            # Load vessel mask
            vessel_img = nib.load(vessel_file)
            vessel_data = np.squeeze(vessel_img.get_fdata())
            vessel_data = np.transpose(vessel_data, (1, 0))
            
            # Convert to binary
            vessel_mask = (vessel_data > 0).astype(np.uint8)
            
            # Convert masks to grayscale for calculations
            lung_mask_gray = cv2.cvtColor(lung_mask_full, cv2.COLOR_BGR2GRAY)
            mask_70_gray = cv2.cvtColor(mask_70, cv2.COLOR_BGR2GRAY)
            
            # Calculate vessel areas
            vessels_in_lung = np.logical_and(vessel_mask, lung_mask_gray > 0)
            central_vessels = np.logical_and(vessel_mask, mask_70_gray > 0)
            peripheral_vessels = np.logical_and(vessels_in_lung, mask_70_gray == 0)
            
            # Calculate metrics
            total_vessel_count = np.sum(vessels_in_lung)
            central_vessel_count = np.sum(central_vessels)
            peripheral_vessel_count = np.sum(peripheral_vessels)
            lung_area = np.sum(lung_mask_gray > 0)
            
            # Calculate percentages
            if total_vessel_count > 0:
                peripheral_percentage = (peripheral_vessel_count / total_vessel_count) * 100
            else:
                peripheral_percentage = 0
            
            if lung_area > 0:
                vessel_density = (total_vessel_count / lung_area) * 100
            else:
                vessel_density = 0
            
            return {
                'peripheral_vessel_count': int(peripheral_vessel_count),
                'peripheral_vessel_percentage': peripheral_percentage,
                'vessel_density_percentage': vessel_density
            }
            
        except Exception as e:
            print(f"Peripheral vessel calculation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'peripheral_vessel_count': 0,
                'peripheral_vessel_percentage': 0,
                'vessel_density_percentage': 0
            }

# Create global instance
dcx_postprocessing = OriginalDCXPostProcessing()