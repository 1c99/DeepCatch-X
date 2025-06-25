#!/usr/bin/env python3
"""
Unified DCX Medical Imaging Inference System
Supports 4 groups of modules with exact original functionality
"""
import os
import sys
import numpy as np
import pydicom as dicom
import torch
import nibabel as nib
import argparse
import yaml
from PIL import Image
from math import ceil, floor
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import glob

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from base_options import BaseOptions
from core.test import create_model_v2 as create_model


# ============================================================================
# GROUP 1: BASIC SEGMENTATION (heart, lung, airway, bone)
# ============================================================================

def resize_keep_ratio_pil(img_pil, target_size, interpolation="LANCZOS"):
    """Resize PIL image while keeping aspect ratio (Group 1)"""
    old_size = img_pil.size  # (width, height)
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    
    # Map string to PIL interpolation constant
    interp_map = {
        "NEAREST": Image.NEAREST,
        "LANCZOS": Image.LANCZOS,
        "BILINEAR": Image.BILINEAR,
        "BICUBIC": Image.BICUBIC
    }
    interp_method = interp_map.get(interpolation, Image.LANCZOS)
    
    try:
        im = img_pil.resize(new_size, interp_method)
    except:
        im = img_pil.resize(new_size, Image.NEAREST)
    return im, ratio


def pad_image_pil(img_pil, target_size, pad_value=0):
    """Pad PIL image to target size (Group 1)"""
    old_size = img_pil.size
    pad_size_w = (target_size - old_size[0]) / 2
    pad_size_h = (target_size - old_size[1]) / 2
    
    if pad_size_w.is_integer():
        wl = wr = int(pad_size_w)
    else:
        wl = floor(pad_size_w)
        wr = ceil(pad_size_w)
    
    if pad_size_h.is_integer():
        ht = hb = int(pad_size_h)
    else:
        ht = ceil(pad_size_h)
        hb = floor(pad_size_h)
    
    pad_transform = transforms.Pad((wl, ht, wr, hb), fill=pad_value)
    return pad_transform


def get_biggest_connected_region(gen_lung, n_region=2):
    """Get n biggest connected regions (Group 1: lung, airway)"""
    from skimage.measure import label
    labels = label(gen_lung)
    n_connected_region = np.bincount(labels.flat)
    if n_connected_region[0] != np.max(n_connected_region):
        n_connected_region[0] = np.max(n_connected_region) + 1
    biggest_regions_index = (-n_connected_region).argsort()[1:n_region + 1]

    biggest_regions = np.array([])
    for ind in biggest_regions_index:
        if biggest_regions.size == 0:
            biggest_regions = labels == ind
        else:
            biggest_regions += labels == ind
    return biggest_regions


# ============================================================================
# GROUP 2: LUNG-BASED WITH TRANSFORMS (covid, vessel)
# ============================================================================

def histogram_normalization(arr):
    """Histogram normalization for vessel module (Group 2)"""
    try:
        arr = arr.astype(np.float)
        a_norm = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.int)
        if len(a_norm.shape) == 4:
            a_norm = a_norm[:, :, 0, 0]
        elif len(a_norm.shape) == 3:
            a_norm = a_norm[:, :, 0]
        a_norm = a_norm[:, :, None]
        a_norm = np.tile(a_norm, 3)

        hist, bins = np.histogram(a_norm.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        arr_histnorm = cdf[a_norm]
        arr_denorm = (arr_histnorm / 255) * (arr.max() - arr.min()) + arr.min()

        return arr_denorm[:, :, 0]
    except:
        return arr


# ============================================================================
# GROUP 4: DIFFUSION MODEL (bone_supp)
# ============================================================================

def adjust_size(image, N):
    """Adjust image size for diffusion model (Group 4)"""
    _, _, H, W = image.shape
    pad_h = (N - H % N) if H % N != 0 else 0
    pad_w = (N - W % N) if W % N != 0 else 0

    pad_left = pad_right = pad_top = pad_bottom = 0
    if pad_h > 0:
        pad_bottom = pad_h
    if pad_w > 0:
        pad_right = pad_w

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    if pad_right > 0 or pad_bottom > 0:
        image = torch.nn.functional.pad(image, padding, mode='constant', value=0)

    return image, padding


def remove_padding(image, padding):
    """Remove padding for diffusion model (Group 4)"""
    pad_left, pad_right, pad_top, pad_bottom = padding
    if pad_right == 0 and pad_bottom == 0:
        return image

    _, _, H_padded, W_padded = image.shape
    H_original = H_padded - pad_bottom
    W_original = W_padded - pad_right
    return image[:, :, :H_original, :W_original]


def ddim_sample(model, condition, device='cuda'):
    """DDIM sampling for diffusion model (Group 4)"""
    num_timesteps = 1000
    sample_step = 30
    new_timesteps = torch.linspace(num_timesteps - 1, 0, steps=sample_step, device=device).long()

    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    x = torch.randn_like(condition)

    for step, i in enumerate(new_timesteps):
        t = torch.tensor([i], device=device, dtype=torch.long)

        with torch.no_grad():
            eps_theta = model(x, condition, t)

        alpha = alpha_cumprod[i]
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha)
        x0_pred = (x - sqrt_one_minus_alpha * eps_theta) / sqrt_alpha

        if step < sample_step - 1:
            next_i = new_timesteps[step + 1]
            alpha_next = alpha_cumprod[next_i]

            coef1 = torch.sqrt(alpha_next) * x0_pred
            coef2 = torch.sqrt(1 - alpha_next) * eps_theta
            x = coef1 + coef2
        else:
            x = x0_pred

    return x


# ============================================================================
# UNIFIED INFERENCE CLASS
# ============================================================================

class UnifiedDCXInference:
    def __init__(self, config_path, device_override=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.module_type = self.config.get('module_type', 'basic_segmentation')
        
        # Device selection with MPS support for Mac
        if device_override and device_override != 'auto':
            if device_override == 'mps':
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    print("Using Mac MPS (Metal Performance Shaders) - forced")
                else:
                    print("MPS not available, falling back to CPU")
                    self.device = torch.device("cpu")
            elif device_override == 'cuda':
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print("Using CUDA GPU - forced")
                else:
                    print("CUDA not available, falling back to CPU")
                    self.device = torch.device("cpu")
            else:  # cpu
                self.device = torch.device("cpu")
                print("Using CPU - forced")
        else:
            # Automatic device selection
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA GPU")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Mac MPS (Metal Performance Shaders)")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        
        # Initialize based on module type
        if self.module_type == 'diffusion':
            self._init_diffusion_model()
        else:
            self._init_segmentation_model()
    
    def _init_segmentation_model(self):
        """Initialize segmentation-based models (Groups 1, 2, 3)"""
        # Parse base options
        opt = BaseOptions()
        opt.initialize()
        self.opt = opt.parse(save=False)
        
        # Override with config values
        for key, value in self.config.items():
            if hasattr(self.opt, key):
                setattr(self.opt, key, value)
        
        # Set paths
        name_to_module = {
            'xray2lung': 'lung', 'xray2heart': 'heart', 'xray2airwaynan': 'airway',
            'lung2covid': 'covid', 'lung2vessel': 'vessel', 'xray2bone': 'bone',
            'heart_volumetry': 'heart_volumetry', 'lung_volumetry': 'lung_volumetry'
        }
        
        module_name = name_to_module.get(self.config['name'], self.config['name'])
        checkpoint_path = os.path.join("modules", module_name, self.config['checkpoint_path'])
        
        self.opt.checkpoint_path = checkpoint_path
        self.opt.checkpoints_dir = os.path.dirname(checkpoint_path)
        self.opt.name = self.config['name']
        
        # Model setup
        self.opt.cuda1 = 0
        self.opt.continue_train = False
        
        # Set GPU IDs based on device availability
        if torch.cuda.is_available():
            self.opt.gpu_ids = [0]  # Use first GPU
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.opt.gpu_ids = [0]  # MPS acts like GPU ID 0
        else:
            self.opt.gpu_ids = []   # CPU only
        
        # Create segmentation model
        self.model = create_model(self.opt)
        self.model.eval()
        
        # Move model to appropriate device
        if self.device.type == 'cuda':
            self.model = self.model.cuda()
        elif self.device.type == 'mps':
            self.model = self.model.to('mps')
        
        # Load regression model if needed (Group 3: volumetry)
        if self.config.get('calculate_volume', False):
            self._load_regression_model()
    
    def _init_diffusion_model(self):
        """Initialize diffusion model (Group 4)"""
        from eff_unet_shuffle import EfficientUNet
        
        checkpoint_path = self.config['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = EfficientUNet(in_channels=2, out_channels=1, init_features=64).to(self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
    
    def _load_regression_model(self):
        """Load regression model for volumetry (Group 3)"""
        if 'heart' in self.config['name']:
            from heartregression_model import UNetRFull
            # Heart: 512*512 + 1 feature = 262145 total (need args='x,y' → len-1=1)
            self.regression_model = UNetRFull(n_channels=1, n_classes=1, args='x,y', isHeart=True)
        else:
            from lungregression_model import UNetRFull
            # Lung: 2048*2048 + 3 features = 4194307 total (need args='x,y,z,w' → len-1=3)
            self.regression_model = UNetRFull(n_channels=1, n_classes=1, args='x,y,z,w', isHeart=False)
        
        # Build full path for regression checkpoint
        name_to_module = {
            'xray2lung': 'lung', 'xray2heart': 'heart', 'xray2airwaynan': 'airway',
            'lung2covid': 'covid', 'lung2vessel': 'vessel', 'xray2bone': 'bone',
            'heart_volumetry': 'heart_volumetry', 'lung_volumetry': 'lung_volumetry'
        }
        module_name = name_to_module.get(self.config['name'], self.config['name'])
        regression_checkpoint_path = os.path.join("modules", module_name, self.config['regression_checkpoint_path'])
        checkpoint = torch.load(regression_checkpoint_path, map_location=self.device)
        self.regression_model.load_state_dict(checkpoint['weight'])
        self.regression_model.eval()
        
        # Keep regression model on CPU due to hardcoded .cpu() calls in the model
        self.regression_model = self.regression_model.cpu()
    
    def process(self, dicom_path, output_path, lung_mask_path=None):
        """Main processing pipeline"""
        print(f"Processing: {dicom_path}")
        
        # Validate inputs
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"Input file not found: {dicom_path}")
        
        # Check if lung mask is required
        if self.config.get('requires_lung_mask', False) and not lung_mask_path:
            module_name = self.config.get('name', 'unknown')
            raise ValueError(f"Module '{module_name}' requires a lung mask. Use --lung_mask parameter.")
        
        if lung_mask_path and not os.path.exists(lung_mask_path):
            raise FileNotFoundError(f"Lung mask file not found: {lung_mask_path}")
        
        if self.module_type == 'diffusion':
            return self._process_diffusion(dicom_path, output_path)
        else:
            return self._process_segmentation(dicom_path, output_path, lung_mask_path)
    
    def _process_diffusion(self, dicom_path, output_path):
        """Process diffusion model (Group 4: bone_supp)"""
        # Load DICOM
        ds = dicom.dcmread(dicom_path)
        dcm_arr = np.array(ds.pixel_array, dtype='float32')
        
        # IQR clipping
        Q3 = np.percentile(dcm_arr, 75)
        Q1 = np.percentile(dcm_arr, 25)
        IQR = Q3 - Q1
        upper_boundary = Q3 + 1.5 * IQR
        dcm_clip = np.clip(dcm_arr, 0, upper_boundary)
        outlier_residual = dcm_arr - dcm_clip
        
        # Normalization
        max_val = np.max(dcm_clip)
        dcm_max_norm = dcm_clip / max_val
        mean_val = np.mean(dcm_max_norm)
        std_val = np.std(dcm_max_norm)
        dcm_norm = (dcm_max_norm - mean_val) / std_val
        
        # To tensor
        dcm_torch = torch.from_numpy(dcm_norm).unsqueeze(0).unsqueeze(0)
        dcm_torch = dcm_torch.to(self.device).float()
        
        # Adaptive downsampling
        b, c, h, w = dcm_torch.shape
        max_size = np.maximum(h, w)
        ds_factor = int(np.round(max_size / 512))
        
        # Padding and interpolation
        dcm_torch, padding = adjust_size(dcm_torch, ds_factor)
        dcm_torch = torch.nn.functional.interpolate(dcm_torch, scale_factor=1/ds_factor, mode='bilinear')
        
        # DDIM sampling
        x = ddim_sample(model=self.model, condition=dcm_torch, device=self.device)
        
        # Interpolation and remove padding
        output = torch.nn.functional.interpolate(x, scale_factor=ds_factor, mode='bilinear')
        output = remove_padding(output, padding)
        
        # To numpy and denormalization
        output = output.squeeze(0).squeeze(0).detach().cpu().numpy()
        output_denorm = (output * std_val + mean_val) * max_val
        output_denorm += outlier_residual
        output_denorm = np.clip(output_denorm, 0, upper_boundary)
        
        # Save as DICOM
        output_denorm = output_denorm.astype(str(ds.pixel_array.dtype))
        ds.PixelData = output_denorm.tobytes()
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory path
            os.makedirs(output_dir, exist_ok=True)
        ds.save_as(output_path)
        
        print(f"Bone suppression output saved to: {output_path}")
        return {'output': output_denorm}
    
    def _process_segmentation(self, dicom_path, output_path, lung_mask_path=None):
        """Process segmentation-based models (Groups 1, 2, 3)"""
        # Get input and metadata
        img, pixel_spacing, ratio, original_input_shape = self._preprocess_input(dicom_path, lung_mask_path)
        
        # Run inference
        output = self._run_inference(img)
        
        # Postprocess
        output_full, mask = self._postprocess(output)
        
        # Calculate metrics
        area = volume = None
        if self.config.get('calculate_area', False):
            area = self._calculate_area(mask, pixel_spacing, ratio)
            print(f"Area: {area:.2f} cm²")
        
        if self.config.get('calculate_volume', False):
            volume = self._calculate_volume(img, mask)
            print(f"Volume: {volume:.2f} mL ({volume/1000:.3f} L)")
        
        # Save output
        self._save_output(output_full, output_path, pixel_spacing, ratio)
        
        return {'output': output_full, 'mask': mask, 'area': area, 'volume': volume}
    
    def _preprocess_input(self, dicom_path, lung_mask_path=None):
        """Preprocess input based on module group"""
        # Read DICOM for metadata
        ds = dicom.dcmread(dicom_path, force=True)
        
        # Get metadata
        pixel_spacing = getattr(ds, 'PixelSpacing', None)
        if not pixel_spacing:
            pixel_spacing = getattr(ds, 'ImagerPixelSpacing', [0.18, 0.18])
        if isinstance(pixel_spacing, (int, float)):
            pixel_spacing = [pixel_spacing, pixel_spacing]
        
        target_size = self.config['loadSize']
        
        # Group 2: Lung-based (covid, vessel)
        if self.config.get('requires_lung_mask', False):
            return self._preprocess_lung_based(lung_mask_path, target_size, ds.pixel_array.shape, pixel_spacing)
        
        # Group 1 & 3: Standard DICOM processing
        return self._preprocess_dicom(ds, target_size, pixel_spacing)
    
    def _preprocess_lung_based(self, lung_mask_path, target_size, original_shape, pixel_spacing):
        """Preprocess lung-based inputs (Group 2: covid, vessel)"""
        if not lung_mask_path:
            raise ValueError(f"Module {self.config['name']} requires lung mask")
        
        # Load lung NIfTI
        nii_image = nib.load(lung_mask_path)
        image_array = nii_image.get_fdata()
        
        # Handle multi-dimensional arrays
        if image_array.shape[-2:] == (1, 1):
            image_array = image_array[..., 0, 0]
        
        # Apply transformations (3 rotations + flip)
        image_array = np.rot90(image_array, 3)
        image_array = np.fliplr(image_array)
        
        # Apply histogram normalization for vessel
        if 'vessel' in self.config['name']:
            image_array = histogram_normalization(image_array)
        
        # Resize and pad
        A = Image.fromarray(image_array)
        interpolation = self.config.get('interpolation', 'LANCZOS')
        A, ratio = resize_keep_ratio_pil(A, target_size, interpolation)
        img_pad = pad_image_pil(A, target_size=target_size, pad_value=0)
        A_ = np.array(img_pad(A))
        
        # Apply fixed normalization
        eps = 1e-10
        a_min_val, a_max_val = self.config['input_min'], self.config['input_max']
        normalized_a = (A_ - a_min_val) / ((a_max_val - a_min_val) + eps)
        
        # Calculate ratio
        ratio = float(target_size) / max(original_shape)
        
        return normalized_a, pixel_spacing, ratio, original_shape
    
    def _preprocess_dicom(self, ds, target_size, pixel_spacing):
        """Preprocess DICOM images (Groups 1 & 3)"""
        # Read pixel array
        try:
            image_array = ds.pixel_array
        except:
            ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
            image_array = ds.pixel_array
        
        image_array = np.array(image_array, dtype=np.int32)
        original_input_shape = image_array.shape
        
        # Handle MONOCHROME1
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            image_array = image_array * -1 + image_array.max()
        
        # Resize and pad
        A = Image.fromarray(image_array)
        interpolation = self.config.get('interpolation', 'LANCZOS')
        A, ratio = resize_keep_ratio_pil(A, target_size, interpolation)
        img_pad = pad_image_pil(A, target_size=target_size, pad_value=0)
        A_ = np.array(img_pad(A))
        
        # Percentile normalization
        eps = 1e-10
        mean, std = A_.mean(), A_.std()
        A_neg2std = np.where(A_ < mean - (2 * std), mean - (2 * std), A_)
        percentile0 = np.percentile(A_neg2std, 0)
        percentile99 = np.percentile(A_neg2std, 99)
        normalized_a = (A_ - percentile0) / ((percentile99 - percentile0) + eps)
        
        return normalized_a, pixel_spacing, ratio, original_input_shape
    
    def _run_inference(self, img):
        """Run model inference"""
        # Convert to tensor
        to_tensor = transforms.ToTensor()
        normalized_a = img.astype(np.float32)
        normalized_a = to_tensor(normalized_a)
        normalized_a = normalized_a.unsqueeze(0)
        
        # Move to appropriate device
        if self.device.type == 'cuda' and len(self.opt.gpu_ids) > 0:
            normalized_a = normalized_a.cuda()
        elif self.device.type == 'mps' and len(self.opt.gpu_ids) > 0:
            normalized_a = normalized_a.to('mps')
        
        # Run inference
        with torch.no_grad():
            output = self.model(normalized_a)
            output = output.cpu().numpy()
        
        return output
    
    def _postprocess(self, output):
        """Apply postprocessing based on module configuration"""
        # Get output range
        if 'covid' in self.config['name']:
            b_min_val, b_max_val = -1100, -400
        elif 'airway' in self.config['name']:
            b_min_val, b_max_val = -1000, -500
        else:
            b_min_val, b_max_val = -1100, -500
        
        # Denormalize
        denormalize_gen = output * (b_max_val - b_min_val) + b_min_val
        
        # Apply thresholding
        threshold = self.config['threshold']
        replace_value = -1024
        denormalize_gen = np.where(denormalize_gen[0] < threshold, replace_value, denormalize_gen)
        
        # Create mask
        denormalize_gen_mask = np.where(denormalize_gen[0, 0] < threshold, 0, 1)
        
        # Apply connected regions (Group 1: lung, airway)
        if self.config.get('use_connected_regions', False):
            n_regions = self.config.get('n_regions', 2)
            denormalize_gen_mask = get_biggest_connected_region(denormalize_gen_mask, n_regions)
            if 'lung' in self.config['name']:
                connected_lung = np.where(denormalize_gen_mask, denormalize_gen[0, 0], -1024)
                denormalize_gen = connected_lung[np.newaxis, np.newaxis]
        
        # COVID special processing (Group 2)
        elif 'covid' in self.config['name']:
            if np.sum(denormalize_gen_mask.flatten()) == denormalize_gen_mask.size:
                denormalize_gen_mask = np.zeros_like(denormalize_gen_mask)
        
        return denormalize_gen, denormalize_gen_mask
    
    def _calculate_area(self, mask, pixel_spacing, ratio):
        """Calculate area in cm²"""
        area = np.sum(mask.flatten())
        pixel_size_resize_w = pixel_spacing[0] / ratio
        pixel_size_resize_h = pixel_spacing[1] / ratio
        area_cm2 = area * pixel_size_resize_w * pixel_size_resize_h / 100
        return area_cm2
    
    def _calculate_volume(self, img, mask):
        """Calculate volume using regression model (Group 3)"""
        # Determine size based on heart vs lung
        target_size = 512 if 'heart' in self.config['name'] else 2048
        
        # Resize for regression - use mask as single channel input
        mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((target_size, target_size)))
        
        # Single channel input (mask)
        input_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        # Create features based on module type (exact match with original)
        if 'heart' in self.config['name']:
            features = torch.tensor([[0.5]], dtype=torch.float32)  # 1 feature for heart
        else:
            features = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)  # 3 features for lung
        
        # Keep everything on CPU for regression model
        input_tensor = input_tensor.cpu()
        features = features.cpu()
        
        with torch.no_grad():
            volume = self.regression_model(input_tensor, features)
        
        return volume.item()
    
    def _save_output(self, output, output_path, pixel_spacing, ratio):
        """Save output as NIfTI file"""
        # Transpose to match original format
        nii_np = np.transpose(output, axes=[3, 2, 1, 0])
        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
        
        # Set pixel dimensions
        pixel_size_resize_w = pixel_spacing[0] / ratio
        nii.header['pixdim'] = pixel_size_resize_w
        
        # Save file
        nib.save(nii, output_path)
        print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified DCX Medical Imaging Inference')
    parser.add_argument('--module', type=str, required=True, 
                       choices=['lung', 'heart', 'airway', 'bone', 'covid', 'vessel', 
                               'heart_volumetry', 'lung_volumetry', 'bone_supp'],
                       help='Module to use for inference')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input DICOM file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output file')
    parser.add_argument('--lung_mask', type=str, default=None,
                       help='Path to lung mask (required for covid/vessel modules)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Force specific device (auto=automatic selection)')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Get config path
    config_path = os.path.join(os.path.dirname(__file__), 'configs', f'{args.module}.yaml')
    
    # Create inference object and process
    inference = UnifiedDCXInference(config_path, args.device)
    results = inference.process(args.input_file, args.output_file, args.lung_mask)
    
    print("Processing complete!")


if __name__ == '__main__':
    main()