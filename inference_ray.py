#!/usr/bin/env python3
"""
Ray-based DCX Inference System
Each file + module combination is a separate Ray task
Automatically handles resource allocation and scheduling
"""
import os
import sys
import time
import argparse
import glob
import ray
import numpy as np
from datetime import datetime

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))


@ray.remote
class ModuleWorker:
    """Ray actor that processes individual module tasks"""
    
    def __init__(self, device='auto'):
        """Initialize worker with imports and device setup"""
        # Import inside worker to avoid serialization issues
        import torch
        
        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.models_cache = {}
        
    def process_module(self, module, input_file, output_dir, output_format='png', 
                      output_size='original', no_calculations=False, 
                      collect_measurements=False, gpu_id='0'):
        """Process a single module for a single file"""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        
        # Import here to avoid serialization
        from inference import UnifiedDCXInference
        import numpy as np
        
        try:
            print(f"Worker processing: {os.path.basename(input_file)} - {module}")
            
            # Determine config path
            base_module = 'aorta' if module in ['aorta0', 'aorta1'] else module
            config_path = os.path.join(os.path.dirname(__file__), 'configs', f'{base_module}.yaml')
            
            if not os.path.exists(config_path):
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'failed',
                    'error': f'Config not found: {config_path}'
                }
            
            # Create output filename
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            
            # Determine output filename based on module
            if module == 'bone_supp':
                output_filename = f"{input_basename}_{module}.{output_format}"
            elif module == 'aorta0':
                output_filename = f"{input_basename}_aorta_asc.{output_format}"
            elif module == 'aorta1':
                output_filename = f"{input_basename}_aorta_desc.{output_format}"
            else:
                output_filename = f"{input_basename}_{module}.{output_format}"
                
            output_file = os.path.join(output_dir, output_filename)
            
            # Check if module needs lung mask
            lung_mask_path = None
            if module in ['covid', 'vessel', 'tb']:
                # First try to use temp file (always in NII format at 2048x2048)
                temp_lung_path = os.path.join(output_dir, f'{input_basename}_lung_temp2048.nii')
                if os.path.exists(temp_lung_path):
                    lung_mask_path = temp_lung_path
                else:
                    # Fall back to looking for regular lung mask files
                    potential_files = [
                        os.path.join(output_dir, f'{input_basename}_lung.{output_format}'),
                        os.path.join(output_dir, f'{input_basename}_lung.nii'),
                        os.path.join(output_dir, f'{input_basename}_lung.png')
                    ]
                    for f in potential_files:
                        if os.path.exists(f):
                            lung_mask_path = f
                            break
                        
                if not lung_mask_path and module in ['covid', 'vessel']:
                    return {
                        'file': input_file,
                        'module': module,
                        'status': 'skipped',
                        'error': 'Lung mask required but not found'
                    }
            
            # Create inference object (reuse if cached)
            cache_key = f"{module}_{config_path}"
            if cache_key not in self.models_cache:
                print(f"Creating inference for module: {module} with module_name: {module}")
                inference = UnifiedDCXInference(
                    config_path, 
                    device_override=self.device,
                    output_format=output_format,
                    output_size=output_size,
                    module_name=module,
                    unified_csv=collect_measurements,
                    batch_mode=True
                )
                
                # Configure calculations
                if no_calculations:
                    inference.config['calculate_area'] = False
                    inference.config['calculate_volume'] = False
                else:
                    volume_modules = ['heart', 'lung', 'heart_volumetry', 'lung_volumetry', 't12l1_regression']
                    if module not in volume_modules:
                        inference.config['calculate_volume'] = False
                        
                self.models_cache[cache_key] = inference
            else:
                inference = self.models_cache[cache_key]
                # Ensure module_name is set correctly even for cached instances
                inference.module_name = module
            
            # Process the file
            results = inference.process(input_file, output_file, lung_mask_path)
            
            # Special handling for covid and vessel modules
            # These modules need transpose, but it must happen at the right point
            if module in ['covid', 'vessel'] and 'output' in results:
                print(f"\nApplying transpose correction for {module}...")
                # For covid/vessel, we need to ensure proper orientation
                # The preprocessing applies 3x rot90 + fliplr, so output needs adjustment
                output_data = results.get('output')
                if output_data is not None and output_format in ['png', 'dcm']:
                    # First, let's check if we need to reload the saved file
                    # The issue might be that inference.py already saved it with wrong orientation
                    import os
                    import time
                    time.sleep(0.1)  # Small delay to ensure file is written
                    
                    if output_format == 'png' and os.path.exists(output_file):
                        # Load the saved PNG file
                        from PIL import Image
                        img = Image.open(output_file)
                        img_array = np.array(img)
                        print(f"  Original PNG shape: {img_array.shape}")
                        
                        # Apply transpose
                        img_transposed = np.transpose(img_array)
                        print(f"  Transposed PNG shape: {img_transposed.shape}")
                        
                        # Save the corrected image
                        img_corrected = Image.fromarray(img_transposed)
                        img_corrected.save(output_file)
                        print(f"  Saved corrected {module} PNG: {output_file}")
                    
                    elif output_format == 'dcm' and os.path.exists(output_file):
                        # For DICOM, load and correct
                        import pydicom
                        ds = pydicom.dcmread(output_file)
                        pixel_array = ds.pixel_array
                        print(f"  Original DICOM shape: {pixel_array.shape}")
                        
                        # Apply transpose
                        pixel_transposed = np.transpose(pixel_array)
                        print(f"  Transposed DICOM shape: {pixel_transposed.shape}")
                        
                        # Update DICOM with transposed data
                        ds.PixelData = pixel_transposed.astype(ds.pixel_array.dtype).tobytes()
                        ds.Rows, ds.Columns = pixel_transposed.shape
                        
                        # Update pixel spacing if dimensions swapped
                        if hasattr(ds, 'PixelSpacing') and ds.Rows != ds.Columns:
                            ds.PixelSpacing = [ds.PixelSpacing[1], ds.PixelSpacing[0]]
                        
                        # Save corrected DICOM
                        ds.save_as(output_file)
                        print(f"  Saved corrected {module} DICOM: {output_file}")
            
            # Always create temp NII files for specific modules regardless of output format
            # This allows other modules to use them directly
            # Note: COVID and vessel modules handle their own transposition when saving,
            # and they use lung masks as input (not creating temp files themselves)
            if module in ['lung', 'heart', 'aorta', 'aorta0', 'aorta1'] and 'output' in results:
                import nibabel as nib
                import numpy as np
                from PIL import Image
                import pydicom
                
                # Get pixel spacing from DICOM
                try:
                    ds = pydicom.dcmread(input_file)
                    pixel_spacing = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
                    if isinstance(pixel_spacing, (int, float)):
                        pixel_spacing = [pixel_spacing, pixel_spacing]
                except:
                    pixel_spacing = [1.0, 1.0]
                
                # Get the output data from results
                output_data = results.get('output')
                if output_data is not None:
                    # Determine temp file specs based on module
                    if module == 'lung':
                        temp_size = 2048
                        temp_file = os.path.join(output_dir, f"{input_basename}_lung_temp2048.nii")
                    elif module == 'heart':
                        temp_size = 512
                        temp_file = os.path.join(output_dir, f"{input_basename}_heart_temp512.nii")
                    elif module in ['aorta', 'aorta0', 'aorta1']:
                        # For aorta modules, handle them later after checking files
                        temp_size = 2048
                        temp_file = None
                    
                    # Save temp file for lung and heart
                    if module in ['lung', 'heart'] and not os.path.exists(temp_file):
                        # Resize output data to temp size if needed
                        if hasattr(inference, '_resize_output_to_size'):
                            output_temp = inference._resize_output_to_size(output_data, temp_size)
                        else:
                            # Fallback resize
                            from skimage.transform import resize
                            if output_data.ndim == 4:
                                output_2d = output_data[0, 0]
                            else:
                                output_2d = output_data
                            resized_2d = resize(output_2d, (temp_size, temp_size), preserve_range=True)
                            output_temp = resized_2d[np.newaxis, np.newaxis, :, :]
                        
                        # Save as NIfTI with proper formatting from inference.py
                        nii_np = np.transpose(output_temp, axes=[3, 2, 1, 0])
                        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
                        
                        # Set pixel dimensions (from inference.py logic)
                        ratio = float(temp_size) / max(ds.pixel_array.shape)
                        pixel_size_resize_w = pixel_spacing[0] / ratio
                        nii.header['pixdim'] = pixel_size_resize_w
                        
                        # Save file
                        nib.save(nii, temp_file)
                        print(f"Created temp file: {temp_file}")
            
            # For aorta modules, create temp files from the generated asc/desc files
            if module == 'aorta':
                import nibabel as nib
                import numpy as np
                from PIL import Image
                import pydicom
                
                # Get pixel spacing
                try:
                    ds = pydicom.dcmread(input_file)
                    pixel_spacing = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
                    if isinstance(pixel_spacing, (int, float)):
                        pixel_spacing = [pixel_spacing, pixel_spacing]
                except:
                    pixel_spacing = [1.0, 1.0]
                
                # Create temp files for both ascending and descending
                for suffix in ['asc', 'desc']:
                    source_file = os.path.join(output_dir, f"{input_basename}_aorta_{suffix}.{output_format}")
                    temp_file = os.path.join(output_dir, f"{input_basename}_aorta_{suffix}_temp2048.nii")
                    
                    if os.path.exists(source_file) and not os.path.exists(temp_file):
                        # Load the source file
                        if output_format == 'png':
                            img = Image.open(source_file)
                            img_array = np.array(img)
                            if len(img_array.shape) == 3:
                                img_array = img_array[:, :, 0]
                            # Convert to float and expand dims
                            data = img_array.astype(np.float32)[np.newaxis, np.newaxis, :, :]
                        elif output_format == 'nii':
                            nii = nib.load(source_file)
                            data = nii.get_fdata()
                            if data.ndim == 2:
                                data = data[np.newaxis, np.newaxis, :, :]
                            elif data.ndim == 3:
                                data = data[:, :, np.newaxis, np.newaxis]
                        elif output_format == 'dcm':
                            ds_out = pydicom.dcmread(source_file)
                            data = ds_out.pixel_array.astype(np.float32)[np.newaxis, np.newaxis, :, :]
                        
                        # Resize to 2048x2048 if needed
                        if data.shape[2] != 2048 or data.shape[3] != 2048:
                            from skimage.transform import resize
                            output_2d = data[0, 0]
                            resized_2d = resize(output_2d, (2048, 2048), preserve_range=True)
                            data = resized_2d[np.newaxis, np.newaxis, :, :]
                        
                        # Save as NIfTI with proper formatting
                        nii_np = np.transpose(data, axes=[3, 2, 1, 0])
                        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
                        
                        # Set pixel dimensions
                        ratio = float(2048) / max(ds.pixel_array.shape)
                        pixel_size_resize_w = pixel_spacing[0] / ratio
                        nii.header['pixdim'] = pixel_size_resize_w
                        
                        # Save file
                        nib.save(nii, temp_file)
                        print(f"Created temp file: {temp_file}")
            
            elif module in ['aorta0', 'aorta1']:
                # For individual aorta modules, create temp file from their output
                import nibabel as nib
                import numpy as np
                from PIL import Image
                import pydicom
                
                # Get pixel spacing
                try:
                    ds = pydicom.dcmread(input_file)
                    pixel_spacing = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
                    if isinstance(pixel_spacing, (int, float)):
                        pixel_spacing = [pixel_spacing, pixel_spacing]
                except:
                    pixel_spacing = [1.0, 1.0]
                
                suffix = 'asc' if module == 'aorta0' else 'desc'
                temp_file = os.path.join(output_dir, f"{input_basename}_aorta_{suffix}_temp2048.nii")
                
                if os.path.exists(output_file) and not os.path.exists(temp_file):
                    # Load the output file
                    if output_format == 'png':
                        img = Image.open(output_file)
                        img_array = np.array(img)
                        if len(img_array.shape) == 3:
                            img_array = img_array[:, :, 0]
                        data = img_array.astype(np.float32)[np.newaxis, np.newaxis, :, :]
                    elif output_format == 'nii':
                        nii = nib.load(output_file)
                        data = nii.get_fdata()
                        if data.ndim == 2:
                            data = data[np.newaxis, np.newaxis, :, :]
                        elif data.ndim == 3:
                            data = data[:, :, np.newaxis, np.newaxis]
                    elif output_format == 'dcm':
                        ds_out = pydicom.dcmread(output_file)
                        data = ds_out.pixel_array.astype(np.float32)[np.newaxis, np.newaxis, :, :]
                    
                    # Resize to 2048x2048 if needed
                    if data.shape[2] != 2048 or data.shape[3] != 2048:
                        from skimage.transform import resize
                        output_2d = data[0, 0]
                        resized_2d = resize(output_2d, (2048, 2048), preserve_range=True)
                        data = resized_2d[np.newaxis, np.newaxis, :, :]
                    
                    # Save as NIfTI with proper formatting
                    nii_np = np.transpose(data, axes=[3, 2, 1, 0])
                    nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
                    
                    # Set pixel dimensions
                    ratio = float(2048) / max(ds.pixel_array.shape)
                    pixel_size_resize_w = pixel_spacing[0] / ratio
                    nii.header['pixdim'] = pixel_size_resize_w
                    
                    # Save file
                    nib.save(nii, temp_file)
                    print(f"Created temp file: {temp_file}")
            
            return {
                'file': input_file,
                'module': module,
                'status': 'success',
                'output_file': output_file,
                'results': results
            }
            
        except Exception as e:
            import traceback
            return {
                'file': input_file,
                'module': module,
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }


@ray.remote
def process_post_module(module, input_file, output_dir, output_format, all_results):
    """Process post-processing modules (CTR, peripheral, diameter)"""
    # Import required modules inside the function to avoid serialization issues
    import sys
    import os
    import numpy as np
    
    # Add paths for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'insights'))
    
    try:
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        
        if module == 'ctr':
            # Check dependencies - ensure we have the required modules processed
            has_lung = False
            has_heart = False
            
            for m, r in all_results.items():
                if r.get('file') == input_file and r.get('status') == 'success':
                    if m == 'lung':
                        has_lung = True
                    elif m == 'heart':
                        has_heart = True
            
            if not has_lung or not has_heart:
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'skipped',
                    'error': 'Requires lung and heart masks'
                }
            
            # Import CTR-specific modules
            import cv2
            import nibabel as nib
            import pydicom
            from cardiothoracic_ratio import (find_contours, center_point, center_point_one, 
                                              full_mask, bitwise_mask, get_longest_line)
            import cardiothoracic_ratio
            
            # Import the unified inference for creating temp files if needed
            from inference import UnifiedDCXInference
            
            # Use pre-generated temp files from lung/heart modules
            lung_path = os.path.join(output_dir, f"{input_basename}_lung_temp2048.nii")
            heart_path = os.path.join(output_dir, f"{input_basename}_heart_temp512.nii")
            
            # If temp files don't exist, try regular NII files
            if not os.path.exists(lung_path):
                lung_path = os.path.join(output_dir, f"{input_basename}_lung.nii")
            if not os.path.exists(heart_path):
                heart_path = os.path.join(output_dir, f"{input_basename}_heart.nii")
                
            temp_files = []
            
            try:
                
                # Verify files exist before processing
                if not os.path.exists(lung_path):
                    raise Exception(f"Lung NII file not found: {lung_path}")
                if not os.path.exists(heart_path):
                    raise Exception(f"Heart NII file not found: {heart_path}")
                
                # Read DICOM for metadata
                data = pydicom.dcmread(input_file)
                pixel_spacing = data.PixelSpacing if 'PixelSpacing' in data else [0.144, 0.144]
                height = data.Rows
                width = data.Columns
                
                # Process lung contours
                lung_contour = []
                img, lung_contours = find_contours(lung_path)
                
                # CRITICAL: Set the global img variable
                cardiothoracic_ratio.img = img
                
                for c in lung_contours:
                    if len(c) > 1000:
                        lung_contour.append(c)
                
                if len(lung_contour) == 0:
                    raise Exception("No lung contours found for CTR calculation")
                
                # Calculate lung center and measurements
                if len(lung_contour) == 1:
                    center = center_point_one(lung_contour)
                    masks = full_mask(center, lung_contour)
                else:
                    center = center_point(lung_contour)
                    masks = full_mask(center, lung_contour)
                
                mask, masked = bitwise_mask(img, masks, center, 1.0)
                mask = mask[:,:,0]
                
                # Find MHTD
                maxCount = 0
                maxY = -1
                for y in range(mask.shape[0]):
                    count = cv2.countNonZero(mask[y, :])
                    if count > maxCount:
                        maxCount = count
                        maxY = y
                
                startX = -1
                endX = -1
                for x in range(mask.shape[1]):
                    if mask[maxY, x] == 255:
                        if startX == -1:
                            startX = x
                        endX = x
                
                MHTD = (endX - startX) * pixel_spacing[0]
                center_x = startX + ((endX - startX) / 2)
                
                # Process heart
                heart_img = nib.load(heart_path)
                heart_data = heart_img.get_fdata()
                heart_data = np.squeeze(heart_data)
                heart_data = np.transpose(heart_data, (1, 0))
                heart_data = np.repeat(heart_data, 4, axis=0)
                heart_data = np.repeat(heart_data, 4, axis=1)
                
                # Find MHCD
                left_heart = heart_data[:, :int(center_x)]
                right_heart = heart_data[:, int(center_x):]
                
                left_longest_line = get_longest_line(left_heart, center_x, 2048)
                right_longest_line = get_longest_line(right_heart, 2048-center_x, 2048)
                
                left_length = left_longest_line[1][0] - left_longest_line[0][0]
                right_length = right_longest_line[1][0] - right_longest_line[0][0]
                MHCD = left_length + right_length
                
                # Calculate CTR
                newSize = 2048
                newSpacingX = pixel_spacing[0] * max(width, height) / newSize
                ct_ratio = (MHCD * newSpacingX) / MHTD
                
                # Clean up temp files
                lung_base = lung_path[:-4]
                cleanup_files = [
                    f"{lung_base}.png",
                    f"{lung_base}_mask(binary).png",
                    f"{lung_base}_contour.png"
                ]
                cleanup_files.extend(temp_files)
                
                for f in cleanup_files:
                    if os.path.exists(f):
                        os.remove(f)
                
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'success',
                    'results': {
                        'cardiothoracic_ratio': ct_ratio,
                        'mhtd_mm': MHTD,
                        'mhcd_mm': MHCD * newSpacingX,
                        'lung_width_mm': MHTD,
                        'heart_width_mm': MHCD * newSpacingX
                    }
                }
            except Exception as e:
                # Clean up on error
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)
                raise e
            
        elif module == 'peripheral':
            # Build proper results structure
            file_results = {}
            for m, r in all_results.items():
                if r.get('file') == input_file and r.get('status') == 'success':
                    file_results[m] = r.get('results', {})
            
            if 'lung' not in file_results:
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'skipped',
                    'error': 'Requires lung mask'
                }
            
            # Import peripheral-specific modules
            import pydicom as dicom
            from peripheral_area import find_contours, center_point, center_point_one, full_mask, bitwise_mask
            import peripheral_area
            
            # Import the unified inference for creating temp files if needed
            from inference import UnifiedDCXInference
            
            # Use pre-generated temp file from lung module
            lung_path = os.path.join(output_dir, f"{input_basename}_lung_temp2048.nii")
            
            # If temp file doesn't exist, try regular NII file
            if not os.path.exists(lung_path):
                lung_path = os.path.join(output_dir, f"{input_basename}_lung.nii")
                
            temp_files = []
            
            try:
                # Verify file exists before processing
                if not os.path.exists(lung_path):
                    raise Exception(f"Lung NII file not found. Peripheral requires lung mask to be processed first.")
                
                # Process lung mask to generate peripheral masks
                img_color, lung_contours = find_contours(lung_path)
                
                # CRITICAL: Set the global img variable
                peripheral_area.img = img_color
                
                # Filter contours
                lung_contour = []
                for c in lung_contours:
                    if len(c) > 1000:
                        lung_contour.append(c)
                
                if len(lung_contour) == 0:
                    raise Exception("No lung contours found for peripheral mask generation")
                
                # Calculate center point
                if len(lung_contour) == 1:
                    center = center_point_one(lung_contour)
                else:
                    center = center_point(lung_contour)
                
                # Generate full mask
                masks = full_mask(center, lung_contour)
                
                # Generate masks at different percentages
                masks_dict = {}
                for p in [0.5, 0.7]:
                    mask, masked = bitwise_mask(img_color, masks, center, p)
                    masks_dict[p] = mask[:,:,0]
                
                # Calculate areas using pixel spacing from DICOM
                ds = dicom.dcmread(input_file, force=True)
                pixel_spacing = ds.get((0x0028, 0x0030), [0.18, 0.18])
                if isinstance(pixel_spacing, (int, float)):
                    pixel_spacing = [pixel_spacing, pixel_spacing]
                
                # Get the resize ratio for 2048x2048 processing
                original_shape = ds.pixel_array.shape
                ratio = float(2048) / max(original_shape)
                
                # Adjust pixel spacing for the resized 2048x2048 image
                adjusted_pixel_spacing = [ps / ratio for ps in pixel_spacing]
                
                # Calculate areas
                full_area = np.sum(masks[:,:,0] > 0) * adjusted_pixel_spacing[0] * adjusted_pixel_spacing[1] / 100.0
                area_50 = np.sum(masks_dict[0.5] > 0) * adjusted_pixel_spacing[0] * adjusted_pixel_spacing[1] / 100.0
                area_70 = np.sum(masks_dict[0.7] > 0) * adjusted_pixel_spacing[0] * adjusted_pixel_spacing[1] / 100.0
                
                peripheral_50_70 = area_70 - area_50
                peripheral_70_100 = full_area - area_70
                
                # Clean up temp files
                lung_base = lung_path[:-4]
                cleanup_files = [
                    f"{lung_base}.png",
                    f"{lung_base}_mask(binary).png",
                    f"{lung_base}_contour.png"
                ]
                cleanup_files.extend(temp_files)
                
                for f in cleanup_files:
                    if os.path.exists(f):
                        os.remove(f)
                
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'success',
                    'results': {
                        'peripheral_total_area_cm2': full_area,
                        'peripheral_central_area_cm2': area_50,
                        'peripheral_mid_area_cm2': peripheral_50_70,
                        'peripheral_outer_area_cm2': peripheral_70_100
                    }
                }
            except Exception as e:
                # Clean up on error
                for f in temp_files:
                    if os.path.exists(f):
                        os.remove(f)
                raise e
            
        elif module == 'diameter':
            # Build proper results structure
            file_results = {}
            for m, r in all_results.items():
                if r.get('file') == input_file and r.get('status') == 'success':
                    file_results[m] = r.get('results', {})
            
            if 'aorta' not in file_results:
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'skipped',
                    'error': 'Requires aorta mask'
                }
            
            # Use pre-generated temp files from aorta module
            asc_file = os.path.join(output_dir, f"{input_basename}_aorta_asc_temp2048.nii")
            desc_file = os.path.join(output_dir, f"{input_basename}_aorta_desc_temp2048.nii")
            
            # If temp files don't exist, try regular NII files
            if not os.path.exists(asc_file):
                asc_file = os.path.join(output_dir, f"{input_basename}_aorta_asc.nii")
            if not os.path.exists(desc_file):
                desc_file = os.path.join(output_dir, f"{input_basename}_aorta_desc.nii")
            
            # Verify at least one file exists
            if not os.path.exists(asc_file) and not os.path.exists(desc_file):
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'failed',
                    'error': 'Aorta files not found. Diameter requires aorta masks to be processed first.'
                }
            
            # Import the diameter calculation function
            from src.insights.aorta_diameter import compute_diameter
            
            # Prepare file dict for diameter calculation
            diameter_file_dict = {input_basename: []}
            
            if os.path.exists(asc_file):
                diameter_file_dict[input_basename].append(os.path.basename(asc_file))
            if os.path.exists(desc_file):
                diameter_file_dict[input_basename].append(os.path.basename(desc_file))
            
            # Create output directory for diameter results with unique subfolder for each file
            diameter_output_dir = os.path.join(output_dir, 'diameter_results', input_basename)
            os.makedirs(diameter_output_dir, exist_ok=True)
            
            try:
                # Run diameter calculation
                max_diameters = compute_diameter(
                    output_folder=diameter_output_dir,
                    input_folder=output_dir,
                    file_dict=diameter_file_dict,
                    head=input_basename,
                    ASCENDING_ONLY=(len(diameter_file_dict[input_basename]) == 1),
                    visualize=False,
                    heat_map=False
                )
                
                # Build results
                diameter_results = {}
                if max_diameters:
                    if len(max_diameters) >= 1:
                        diameter_results['aorta_ascending_diameter_mm'] = round(max_diameters[0], 1)
                    if len(max_diameters) >= 2:
                        diameter_results['aorta_descending_diameter_mm'] = round(max_diameters[1], 1)
                
                # Clean up only this file's diameter_results subfolder
                import shutil
                if os.path.exists(diameter_output_dir):
                    shutil.rmtree(diameter_output_dir)
                
                return {
                    'file': input_file,
                    'module': module,
                    'status': 'success',
                    'results': diameter_results
                }
                
            except Exception as e:
                # Clean up only this file's diameter_results subfolder on error
                import shutil
                if os.path.exists(diameter_output_dir):
                    shutil.rmtree(diameter_output_dir)
                raise e
        
    except Exception as e:
        return {
            'file': input_file,
            'module': module,
            'status': 'failed',
            'error': str(e)
        }


def get_input_files(input_path):
    """Get list of DICOM files from input path"""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        dicom_extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']
        files = []
        for ext in dicom_extensions:
            files.extend(glob.glob(os.path.join(input_path, ext)))
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(description='Ray-based DCX Medical Imaging Inference')
    
    # Module selection
    parser.add_argument('--module', type=str, action='append',
                       choices=['lung', 'heart', 'airway', 'bone', 'aorta', 't12l1', 'laa',
                               'tb', 'bone_supp', 'covid', 'vessel', 'aorta0', 'aorta1',
                               'ctr', 'peripheral', 'diameter'],
                       help='Module to use for inference (can be specified multiple times)')
    parser.add_argument('--all_modules', action='store_true',
                       help='Run all segmentation modules')
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input DICOM file or directory containing DICOM files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    
    # Ray options
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of Ray workers (auto-determined if not specified)')
    parser.add_argument('--num_gpus', type=float, default=None,
                       help='Number of GPUs per worker (auto-determined if not specified)')
    
    # Processing options
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    
    # Output options
    parser.add_argument('--output_format', type=str, default='png',
                       choices=['nii', 'dcm', 'png'],
                       help='Output format')
    parser.add_argument('--output_size', type=str, default='original',
                       choices=['original', '512', '2048'],
                       help='Output size')
    parser.add_argument('--no_calculations', action='store_true',
                       help='Skip area/volume calculations')
    parser.add_argument('--collect_measurements', action='store_true',
                       help='Collect measurements into CSV')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_modules and not args.module:
        parser.error("Either --module or --all_modules must be specified")
    if args.all_modules and args.module:
        parser.error("Cannot use both --module and --all_modules together")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get input files
    try:
        input_files = get_input_files(args.input)
        if not input_files:
            print(f"No DICOM files found in: {args.input}")
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} DICOM file(s)")
    
    # Determine modules to run
    if args.all_modules:
        modules_to_run = ['lung', 'heart', 'airway', 'bone', 'aorta', 't12l1', 'laa',
                         'tb', 'bone_supp', 'covid', 'vessel', 'ctr', 'peripheral', 'diameter']
    else:
        modules_to_run = args.module
    
    print(f"Modules to run: {', '.join(modules_to_run)}")
    
    # Initialize Ray
    import torch
    if args.num_workers is None:
        # Auto-determine based on system
        import multiprocessing
        args.num_workers = min(multiprocessing.cpu_count(), 8)
        
    if args.num_gpus is None:
        if torch.cuda.is_available():
            args.num_gpus = 1.0 / args.num_workers  # Share GPU among workers
        else:
            args.num_gpus = 0
            
    print(f"\nInitializing Ray with {args.num_workers} workers")
    if args.num_gpus > 0:
        print(f"GPUs per worker: {args.num_gpus}")
        
    ray.init(num_cpus=args.num_workers, num_gpus=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0)
    
    # Create workers
    workers = [ModuleWorker.remote(device=args.device) for _ in range(args.num_workers)]
    
    # Module dependencies
    independent_modules = ['lung', 'heart', 'airway', 'bone', 'aorta', 't12l1', 'laa', 'bone_supp']
    dependent_modules = ['tb', 'covid', 'vessel']  # Need lung
    post_modules = ['ctr', 'peripheral', 'diameter']  # Post-processing
    
    start_time = time.time()
    
    # Submit all tasks
    print("\nSubmitting tasks to Ray...")
    all_tasks = []
    task_info = []
    
    # Phase 1: Submit independent modules for all files
    for file_idx, input_file in enumerate(input_files):
        for module in modules_to_run:
            if module in independent_modules:
                worker = workers[len(all_tasks) % len(workers)]
                task = worker.process_module.remote(
                    module, input_file, args.output_dir, 
                    args.output_format, args.output_size,
                    args.no_calculations, args.collect_measurements, args.gpu
                )
                all_tasks.append(task)
                task_info.append((input_file, module))
    
    # Wait for lung modules to complete if we have dependent modules
    lung_results = {}
    if any(m in modules_to_run for m in dependent_modules):
        print("\nWaiting for lung segmentation to complete...")
        
        # Get lung results first
        completed = 0
        for task, (file, module) in zip(all_tasks, task_info):
            if module == 'lung':
                result = ray.get(task)
                lung_results[file] = result
                completed += 1
                print(f"  Lung {completed}/{len(input_files)}: {os.path.basename(file)} - {result['status']}")
    
    # Phase 2: Submit dependent modules
    for input_file in input_files:
        # Check if lung was successful for this file
        if input_file in lung_results and lung_results[input_file]['status'] == 'success':
            for module in modules_to_run:
                if module in dependent_modules:
                    worker = workers[len(all_tasks) % len(workers)]
                    task = worker.process_module.remote(
                        module, input_file, args.output_dir,
                        args.output_format, args.output_size,
                        args.no_calculations, args.collect_measurements, args.gpu
                    )
                    all_tasks.append(task)
                    task_info.append((input_file, module))
    
    # Process all tasks and collect results
    print(f"\nProcessing {len(all_tasks)} tasks...")
    results = []
    completed = 0
    
    # Use ray.wait for better progress tracking
    remaining_tasks = all_tasks
    remaining_info = task_info
    
    while remaining_tasks:
        # Wait for any task to complete
        ready, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        
        # Get the result
        result = ray.get(ready[0])
        results.append(result)
         
        # Find which task completed
        task_idx = all_tasks.index(ready[0])
        file, module = task_info[task_idx]
        
        completed += 1
        status_symbol = "✓" if result['status'] == 'success' else "✗" if result['status'] == 'failed' else "⚠"
        print(f"  [{completed}/{len(all_tasks)}] {status_symbol} {os.path.basename(file)} - {module} ({result['status']})")
        
        # Update remaining info
        remaining_info = [info for i, info in enumerate(task_info) if all_tasks[i] in remaining_tasks]
    
    # Phase 3: Post-processing modules
    if any(m in modules_to_run for m in post_modules):
        print("\nRunning post-processing modules...")
        post_tasks = []
        
        # Convert results to dict for easy lookup
        results_dict = {(r['file'], r['module']): r for r in results}
        
        for input_file in input_files:
            for module in modules_to_run:
                if module in post_modules:
                    # Prepare results for this file
                    file_results = {m: r for (f, m), r in results_dict.items() if f == input_file}
                    
                    task = process_post_module.remote(
                        module, input_file, args.output_dir, 
                        args.output_format, file_results
                    )
                    post_tasks.append((task, input_file, module))
        
        # Wait for post-processing
        for task, file, module in post_tasks:
            result = ray.get(task)
            results.append(result)
            status_symbol = "✓" if result['status'] == 'success' else "✗" if result['status'] == 'failed' else "⚠"
            print(f"  {status_symbol} {os.path.basename(file)} - {module} ({result['status']})")
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"\n{'='*80}")
    print("RAY PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Files processed: {len(input_files)}")
    print(f"Average time per file: {total_time/len(input_files):.1f}s")  
    
    # Show failed tasks
    if failed > 0:
        print("\nFailed tasks:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {os.path.basename(r['file'])} - {r['module']}: {r.get('error', 'Unknown error')}")
    
    # Collect measurements if requested
    if args.collect_measurements and successful > 0:
        print("\nCollecting measurements into CSV...")
        import csv
        from collections import OrderedDict
        
        # Group results by file
        from collections import defaultdict
        file_results = defaultdict(dict)
        for r in results:
            if r['status'] == 'success' and 'results' in r:
                file_results[r['file']][r['module']] = r['results']
        
        # Collect all measurements for all files
        all_measurements = []
        all_columns = OrderedDict()
        
        # Always include these columns first
        for col in ['patient_id', 'dicom_file', 'processing_date']:
            all_columns[col] = True
        
        # Process each file
        for input_file, modules_results in sorted(file_results.items()):
            measurements_data = OrderedDict()
            measurements_data['patient_id'] = os.path.splitext(os.path.basename(input_file))[0]
            measurements_data['dicom_file'] = os.path.basename(input_file)
            measurements_data['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Collect all measurements from all modules
            for module, module_results in sorted(modules_results.items()):
                if isinstance(module_results, dict):
                    # Area measurements
                    if 'area' in module_results:
                        col_name = f'{module}_area_cm2'
                        measurements_data[col_name] = round(module_results['area'], 2)
                        all_columns[col_name] = True
                    
                    # Volume measurements
                    if 'volume' in module_results and module in ['heart', 'lung']:
                        col_name_ml = f'{module}_volume_ml'
                        col_name_l = f'{module}_volume_l'
                        measurements_data[col_name_ml] = round(module_results['volume'], 2)
                        measurements_data[col_name_l] = round(module_results['volume']/1000, 3)
                        all_columns[col_name_ml] = True
                        all_columns[col_name_l] = True
                    
                    # Metadata
                    for key in ['pixel_spacing_mm', 'image_height', 'image_width']:
                        if key in module_results and key not in measurements_data:
                            measurements_data[key] = module_results[key]
                            all_columns[key] = True
                    
                    # Special module measurements
                    if module == 't12l1' or module == 't12l1_regression':
                        if 'area_t12' in module_results:
                            measurements_data['t12_area_cm2'] = round(module_results['area_t12'], 2)
                            all_columns['t12_area_cm2'] = True
                        if 'area_l1' in module_results:
                            measurements_data['l1_area_cm2'] = round(module_results['area_l1'], 2)
                            all_columns['l1_area_cm2'] = True
                        if 'area_t12_pixels' in module_results:
                            measurements_data['t12_area_pixels'] = module_results['area_t12_pixels']
                            all_columns['t12_area_pixels'] = True
                        if 'area_l1_pixels' in module_results:
                            measurements_data['l1_area_pixels'] = module_results['area_l1_pixels']
                            all_columns['l1_area_pixels'] = True
                        if 'bone_density_t12' in module_results:
                            measurements_data['bone_density_t12'] = round(module_results['bone_density_t12'], 2)
                            all_columns['bone_density_t12'] = True
                        if 'bone_density_l1' in module_results:
                            measurements_data['bone_density_l1'] = round(module_results['bone_density_l1'], 2)
                            all_columns['bone_density_l1'] = True
                    
                    elif module == 'laa':
                        if 'emphysema_probability' in module_results:
                            measurements_data['laa_emphysema_probability'] = round(module_results['emphysema_probability'], 4)
                            all_columns['laa_emphysema_probability'] = True
                        if 'emphysema_classification' in module_results:
                            measurements_data['laa_emphysema_classification'] = module_results['emphysema_classification']
                            all_columns['laa_emphysema_classification'] = True
                        if 'emphysema_area' in module_results:
                            measurements_data['laa_emphysema_area_cm2'] = round(module_results['emphysema_area'], 2)
                            all_columns['laa_emphysema_area_cm2'] = True
                        if 'laa_percentage' in module_results:
                            measurements_data['laa_percentage'] = round(module_results['laa_percentage'], 2)
                            all_columns['laa_percentage'] = True
                    
                    elif module == 'tb':
                        if 'probability' in module_results:
                            measurements_data['tb_probability'] = round(module_results['probability'], 4)
                            all_columns['tb_probability'] = True
                        if 'prediction' in module_results:
                            measurements_data['tb_classification'] = 'Positive' if module_results['prediction'] else 'Negative'
                            all_columns['tb_classification'] = True
                    
                    elif module == 'ctr':
                        if 'cardiothoracic_ratio' in module_results:
                            measurements_data['cardiothoracic_ratio'] = round(module_results['cardiothoracic_ratio'], 3)
                            all_columns['cardiothoracic_ratio'] = True
                        if 'mhtd_mm' in module_results:
                            measurements_data['mhtd_mm'] = round(module_results['mhtd_mm'], 2)
                            all_columns['mhtd_mm'] = True
                        if 'mhcd_mm' in module_results:
                            measurements_data['mhcd_mm'] = round(module_results['mhcd_mm'], 2)
                            all_columns['mhcd_mm'] = True
                        if 'lung_width_mm' in module_results:
                            measurements_data['lung_width_mm'] = round(module_results['lung_width_mm'], 2)
                            all_columns['lung_width_mm'] = True
                        if 'heart_width_mm' in module_results:
                            measurements_data['heart_width_mm'] = round(module_results['heart_width_mm'], 2)
                            all_columns['heart_width_mm'] = True
                    
                    elif module == 'peripheral':
                        for key in ['peripheral_total_area_cm2', 'peripheral_central_area_cm2',
                                   'peripheral_mid_area_cm2', 'peripheral_outer_area_cm2']:
                            if key in module_results:
                                measurements_data[key] = round(module_results[key], 2)
                                all_columns[key] = True
                    
                    elif module == 'diameter':
                        if 'aorta_ascending_diameter_mm' in module_results:
                            measurements_data['aorta_ascending_diameter_mm'] = round(module_results['aorta_ascending_diameter_mm'], 1)
                            all_columns['aorta_ascending_diameter_mm'] = True
                        if 'aorta_descending_diameter_mm' in module_results:
                            measurements_data['aorta_descending_diameter_mm'] = round(module_results['aorta_descending_diameter_mm'], 1)
                            all_columns['aorta_descending_diameter_mm'] = True
            
            all_measurements.append(measurements_data)
        
        # Write single CSV file with all measurements
        csv_filename = f"dcx_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(args.output_dir, csv_filename)
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = list(all_columns.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for measurements in all_measurements:
                # Fill in empty values for missing columns
                row = {col: measurements.get(col, '') for col in fieldnames}
                writer.writerow(row)
        
        print(f"  All measurements saved to: {csv_path}")
        print(f"  Total rows: {len(all_measurements)}")
    
    print(f"\nProcessing complete! Results saved to: {args.output_dir}")
    
    # Clean up temporary files
    print("\nCleaning up temporary files...")
    temp_patterns = [
        "*_temp2048.nii",
        "*_temp512.nii",
        "*_lung_temp2048.nii",
        "*_heart_temp512.nii",
        "*_aorta_asc_temp2048.nii",
        "*_aorta_desc_temp2048.nii",
        "*_temp2048.png",
        "*_temp512.png",
        "*_mask(binary).png",
        "*_contour.png",
        "*_fullMask.png"
    ]
    
    for pattern in temp_patterns:
        temp_files = glob.glob(os.path.join(args.output_dir, pattern))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"  Removed: {os.path.basename(temp_file)}")
            except Exception as e:
                print(f"  Failed to remove {os.path.basename(temp_file)}: {e}")
    
    # Clean up the main diameter_results folder if it exists
    diameter_results_folder = os.path.join(args.output_dir, 'diameter_results')
    if os.path.exists(diameter_results_folder):
        import shutil
        try:
            shutil.rmtree(diameter_results_folder)
            print(f"  Removed: diameter_results folder")
        except Exception as e:
            print(f"  Failed to remove diameter_results folder: {e}")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == '__main__':
    main()


