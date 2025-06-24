#!/usr/bin/env python3
import os
import sys
import numpy as np
import pydicom as dicom
import torch
import nibabel as nib
import argparse
import yaml
from PIL import Image

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from base_options import BaseOptions
from core.test import create_model_v2 as create_model
from scipy.ndimage import gaussian_filter


def resize_keep_ratio(img, size):
    """Resize image while keeping aspect ratio"""
    old_size = img.shape[:2]
    ratio = float(size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = np.array(Image.fromarray(img).resize((new_size[1], new_size[0]), resample=Image.Resampling.BILINEAR))
    
    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0]
    new_im = np.pad(im, pad_width=((top, bottom), (left, right)), constant_values=color[0])
    return new_im, ratio


def pad_image(img, size):
    """Pad image to target size"""
    old_size = img.shape[:2]
    delta_w = size - old_size[1]
    delta_h = size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0]
    new_im = np.pad(img, pad_width=((top, bottom), (left, right)), constant_values=color[0])
    return new_im


def histogram_normalization(data):
    """Apply histogram normalization"""
    sigma = 1.2  # FWHM_sigma equivalent
    blur_data = gaussian_filter(data.astype(np.float32), sigma=sigma, truncate=2.0)
    gmin = np.percentile(blur_data[blur_data > 0], 0.1)
    gmax = np.percentile(blur_data[blur_data > 0], 99.9)
    data = data.astype(float)
    data = (data - gmin) / (gmax - gmin)
    data[data < 0] = 0
    return data


def get_biggest_connected_region(mask, n_region=2):
    """Extract the n largest connected regions"""
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    sizes = ndimage.sum(mask, labeled_array, range(num_features + 1))
    sizes_list = sizes.tolist()
    sizes_list[0] = 0
    
    biggest_regions = []
    for _ in range(min(n_region, num_features)):
        max_idx = sizes_list.index(max(sizes_list))
        biggest_regions.append(max_idx)
        sizes_list[max_idx] = 0
    
    result_mask = np.zeros_like(mask)
    for idx in biggest_regions:
        result_mask[labeled_array == idx] = 1
    
    return result_mask


class UnifiedInference:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Parse base options
        opt = BaseOptions()
        opt.initialize()
        self.opt = opt.parse(save=False)
        
        # Override with config values
        for key, value in self.config.items():
            if hasattr(self.opt, key):
                setattr(self.opt, key, value)
        
        # Set paths - map config name to module directory
        name_to_module = {
            'xray2lung': 'lung',
            'xray2heart': 'heart', 
            'xray2airwaynan': 'airway',
            'lung2covid': 'covid',
            'lung2vessel': 'vessel',
            'xray2bone': 'bone',
            'heart_volumetry': 'heart_volumetry',
            'lung_volumetry': 'lung_volumetry'
        }
        
        module_name = name_to_module.get(self.config['name'], self.config['name'])
        checkpoint_path = os.path.join("modules", module_name, self.config['checkpoint_path'])
        
        self.opt.checkpoint_path = checkpoint_path
        self.opt.checkpoints_dir = os.path.dirname(checkpoint_path)
        self.opt.name = self.config['name']
        
        # Add missing attributes for pix2pixHD model
        self.opt.cuda1 = 0
        self.opt.continue_train = False
        
        # Set CPU mode if CUDA not available
        if not torch.cuda.is_available():
            self.opt.gpu_ids = []
        
        # Create model
        self.model = create_model(self.opt)
        self.model.eval()
        
        # Load regression model if needed
        if self.config.get('calculate_volume', False):
            self.load_regression_model()
    
    def load_regression_model(self):
        """Load regression model for volumetry"""
        if 'heart' in self.config['name']:
            from heartregression_model import HeartRegressionModel
            self.regression_model = HeartRegressionModel()
        else:
            from lungregression_model import LungRegressionModel
            self.regression_model = LungRegressionModel()
        
        checkpoint = torch.load(self.config['regression_checkpoint_path'])
        self.regression_model.load_state_dict(checkpoint['model'])
        self.regression_model.eval()
        if torch.cuda.is_available():
            self.regression_model = self.regression_model.cuda()
    
    def preprocess_dicom(self, dicom_path, lung_mask_path=None):
        """Load and preprocess DICOM file"""
        # Read DICOM
        ds = dicom.dcmread(dicom_path)
        img = ds.pixel_array.astype(np.float32)
        
        # Get metadata
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        if isinstance(pixel_spacing, (int, float)):
            pixel_spacing = [pixel_spacing, pixel_spacing]
        
        # Resize to target size
        target_size = self.config['loadSize']
        img_resized, ratio = resize_keep_ratio(img, target_size)
        
        # Apply normalization based on config
        if self.config['normalization_method'] == 'percentile':
            pmin = np.percentile(img_resized[img_resized > 0], self.config['percentile_min'])
            pmax = np.percentile(img_resized[img_resized > 0], self.config['percentile_max'])
            img_normalized = (img_resized - pmin) / (pmax - pmin)
        else:  # minmax
            img_normalized = (img_resized - self.config['input_min']) / \
                            (self.config['input_max'] - self.config['input_min'])
        
        img_normalized = np.clip(img_normalized, 0, 1)
        
        # Process lung mask if required
        lung_mask = None
        if self.config.get('requires_lung_mask', False):
            if not lung_mask_path:
                raise ValueError(f"Module {self.config['name']} requires lung mask")
            
            lung_nii = nib.load(lung_mask_path)
            lung_mask = lung_nii.get_fdata().astype(np.float32)
            
            # Handle multi-dimensional NIfTI files by extracting 2D slice
            if lung_mask.ndim > 2:
                # Squeeze extra dimensions and take first slice if needed
                lung_mask = np.squeeze(lung_mask)
                if lung_mask.ndim > 2:
                    lung_mask = lung_mask[:, :, 0]
            
            lung_mask = (lung_mask > 0).astype(np.float32)
            lung_mask = pad_image(lung_mask, target_size)
            
            if self.config.get('use_histogram_normalization', False):
                lung_mask = histogram_normalization(lung_mask)
            
            # Ensure values are 0 or 1
            lung_mask = np.where(lung_mask > 0.5, 1.0, 0.0)
        
        return img_normalized, lung_mask, pixel_spacing, ratio
    
    def run_inference(self, img, lung_mask=None):
        """Run model inference"""
        # Prepare input tensor
        if self.config.get('requires_lung_mask', False):
            # For COVID and vessel modules, use the lung mask as the input
            if lung_mask is None:
                raise ValueError("Lung mask is required but not provided")
            input_tensor = torch.from_numpy(lung_mask).unsqueeze(0).unsqueeze(0).float()
        else:
            input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        if torch.cuda.is_available() and len(self.opt.gpu_ids) > 0:
            input_tensor = input_tensor.cuda()
        
        # Run inference
        with torch.no_grad():
            generated = self.model(input_tensor)
        
        # Convert to numpy
        output = generated.cpu().numpy()
        
        # Denormalize
        if self.config['normalization_method'] == 'percentile':
            # Use output range for denormalization
            output_min, output_max = self.config['output_range']
            output = output * (output_max - output_min) + output_min
        else:  # minmax
            output = output * (self.config['output_max'] - self.config['output_min']) + \
                    self.config['output_min']
        
        return output
    
    def postprocess(self, output):
        """Apply postprocessing based on config"""
        # Apply threshold
        mask = output[0] > self.config['threshold']
        
        # Apply connected regions if configured
        if self.config.get('use_connected_regions', False):
            n_regions = self.config.get('n_regions', 2)
            mask = get_biggest_connected_region(mask, n_regions)
        
        return mask, output
    
    def calculate_area(self, mask, pixel_spacing, ratio):
        """Calculate area in cm²"""
        if not self.config.get('calculate_area', False):
            return None
        
        area_pixels = np.sum(mask)
        pixel_size_resize_w = pixel_spacing[0] / ratio
        pixel_size_resize_h = pixel_spacing[1] / ratio
        area_cm2 = area_pixels * pixel_size_resize_w * pixel_size_resize_h / 100
        
        return area_cm2
    
    def calculate_volume(self, img, mask):
        """Calculate volume using regression model"""
        if not self.config.get('calculate_volume', False):
            return None
        
        # Prepare input for regression model
        img_resized = np.array(Image.fromarray(img).resize((512, 512), resample=Image.Resampling.BILINEAR))
        mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((512, 512), resample=Image.Resampling.BILINEAR))
        
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
        
        with torch.no_grad():
            volume = self.regression_model(img_tensor, mask_tensor)
        
        return volume.item()
    
    def save_output(self, output, output_path, dicom_path):
        """Save output as NIfTI file"""
        # Handle output shape properly
        if output.ndim == 4:  # (batch, channel, height, width)
            output_data = output[0, 0]  # Take first batch and channel
        elif output.ndim == 3:  # (channel, height, width)
            output_data = output[0]     # Take first channel
        else:  # (height, width)
            output_data = output
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(output_data[:, :, np.newaxis], np.eye(4))
        
        # Save file
        nib.save(nii_img, output_path)
        print(f"Output saved to: {output_path}")
    
    def process(self, dicom_path, output_path, lung_mask_path=None):
        """Main processing pipeline"""
        print(f"Processing: {dicom_path}")
        
        # Preprocess
        img, lung_mask, pixel_spacing, ratio = self.preprocess_dicom(dicom_path, lung_mask_path)
        
        # Run inference
        output = self.run_inference(img, lung_mask)
        
        # Postprocess
        mask, output_full = self.postprocess(output)
        
        # Calculate metrics
        area = self.calculate_area(mask, pixel_spacing, ratio)
        if area is not None:
            print(f"Area: {area:.2f} cm²")
        
        volume = self.calculate_volume(img, mask)
        if volume is not None:
            print(f"Volume: {volume:.2f} mL ({volume/1000:.3f} L)")
        
        # Save output
        self.save_output(output_full, output_path, dicom_path)
        
        return {
            'output': output_full,
            'mask': mask,
            'area': area,
            'volume': volume
        }


def main():
    parser = argparse.ArgumentParser(description='Unified DCX Inference')
    parser.add_argument('--module', type=str, required=True, 
                       choices=['lung', 'heart', 'airway', 'covid', 'vessel', 'bone', 
                               'heart_volumetry', 'lung_volumetry'],
                       help='Module to use for inference')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input DICOM file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output NIfTI file')
    parser.add_argument('--lung_mask', type=str, default=None,
                       help='Path to lung mask (required for covid/vessel modules)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Get config path
    config_path = os.path.join(os.path.dirname(__file__), 'configs', f'{args.module}.yaml')
    
    # Create inference object and process
    inference = UnifiedInference(config_path)
    results = inference.process(args.input_file, args.output_file, args.lung_mask)
    
    print("Processing complete!")


if __name__ == '__main__':
    main()