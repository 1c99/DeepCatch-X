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
import cv2

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
    """Pad PIL image to target size (Group 1) - exact original DCX logic"""
    old_size = img_pil.size
    pad_size_w = (target_size - old_size[0]) / 2
    pad_size_h = (target_size - old_size[1]) / 2
    
    # Original DCX logic: use % 2 == 0 check and different ceil/floor order
    if pad_size_w % 2 == 0:
        wl, wr = int(pad_size_w), int(pad_size_w)
    else:
        wl = ceil(pad_size_w)
        wr = floor(pad_size_w)
    
    if pad_size_h % 2 == 0:
        ht, hb = int(pad_size_h), int(pad_size_h)
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
    def __init__(self, config_path, device_override=None, output_format='nii', output_size='512', module_name=None, unified_csv=False, batch_mode=False, ctr_requested=False):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Store module name for special handling (aorta0, aorta1)
        self.module_name = module_name
        
        # Control whether to create separate CSV files (for unified measurement collection)
        self.unified_csv = unified_csv
        
        # Track if we're in batch mode (for lung+covid+vessel processing)
        self.batch_mode = batch_mode
        
        # Track if CTR calculation is requested (to create 512 heart mask)
        self.ctr_requested = ctr_requested
        
        self.module_type = self.config.get('module_type', 'basic_segmentation')
        self.output_format = output_format
        self.output_size = output_size
        
        # Keep original config loadSize for model inference (never change this)
        # output_size only affects the final output dimensions, not model input
        
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
        elif self.module_type == 'segmentation_models_pytorch':
            self._init_smp_model()
        elif self.module_type == 'efficientnet_classification':
            self._init_efficientnet_model()
        elif self.module_type == 'dual_model_segmentation':
            self._init_dual_model_segmentation()
        elif self.module_type == 'dual_segmentation_with_regression':
            self._init_dual_segmentation_with_regression()
        elif self.module_type == 'tb_with_lung_preprocessing':
            self._init_tb_with_lung_preprocessing()
        elif self.module_type == 'segmentation_models_pytorch_laa':
            self._init_laa_model()
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
        # Use unified checkpoints directory
        checkpoint_path = self.config['checkpoint_path']
        
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
            try:
                self._load_regression_model()
                print(f"✓ Regression model loaded for {self.config['name']}")
            except Exception as e:
                print(f"⚠ Failed to load regression model for {self.config['name']}: {e}")
                self.regression_model = None
    
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
        # Use unified checkpoints directory for regression
        regression_checkpoint_path = self.config['regression_checkpoint_path']
        checkpoint = torch.load(regression_checkpoint_path, map_location=self.device)
        
        # Load model parameters from checkpoint (DCX_python_inference exact)
        n_classes = checkpoint.get('n_class', 1)
        input_feature = checkpoint.get('input_feature', 'x,y')
        
        if 'heart' in self.config['name']:
            from core.heartregression_model import UNetRFull
            self.regression_model = UNetRFull(n_channels=1, n_classes=n_classes, args=input_feature, isHeart=True)
        else:
            from core.lungregression_model import UNetRFull
            self.regression_model = UNetRFull(n_channels=1, n_classes=n_classes, args=input_feature, isHeart=False)
        
        self.regression_model.load_state_dict(checkpoint['weight'])
        self.regression_model.eval()
        
        # Keep regression model on CPU due to hardcoded .cpu() calls in the model
        self.regression_model = self.regression_model.cpu()
    
    def _init_smp_model(self):
        """Initialize segmentation_models_pytorch model (LAA)"""
        import segmentation_models_pytorch as smp
        
        model_name = self.config.get('model_type', 'efficientnet-b0')
        self.model = smp.Unet(encoder_name=model_name, encoder_weights=None, classes=1, in_channels=3)
        
        checkpoint_path = self.config['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _init_efficientnet_model(self):
        """Initialize EfficientNet classification model (TB)"""
        from model import EfficientNet
        
        model_name = self.config.get('model_type', 'efficientnet-b5')
        checkpoint_path = self.config['checkpoint_path']
        
        self.model = EfficientNet.from_pretrained(model_name, num_classes=1)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _init_dual_model_segmentation(self):
        """Initialize dual model segmentation (T12L1)"""
        # Parse base options for both models
        opt = BaseOptions()
        opt.initialize()
        self.opt = opt.parse(save=False)
        
        # Override with config values
        for key, value in self.config.items():
            if hasattr(self.opt, key) and not key.endswith('_l1'):
                setattr(self.opt, key, value)
        
        # Use unified checkpoints directory for dual models
        checkpoint_path_t12 = self.config['checkpoint_path']
        checkpoint_path_l1 = self.config['checkpoint_path_l1']
        
        self.opt.checkpoint_path = checkpoint_path_t12
        self.opt.checkpoints_dir = os.path.dirname(checkpoint_path_t12)
        self.opt.name = self.config['name']
        
        # Model setup
        self.opt.cuda1 = 0
        self.opt.continue_train = False
        
        # Set GPU IDs based on device availability
        if torch.cuda.is_available():
            self.opt.gpu_ids = [0]
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.opt.gpu_ids = [0]
        else:
            self.opt.gpu_ids = []
        
        # Create T12 model
        self.model_t12 = create_model(self.opt)
        self.model_t12.eval()
        
        # Create L1 model with different checkpoint
        self.opt.checkpoint_path = checkpoint_path_l1
        self.model_l1 = create_model(self.opt)
        self.model_l1.eval()
        
        # Move models to appropriate device
        if self.device.type == 'cuda':
            self.model_t12 = self.model_t12.cuda()
            self.model_l1 = self.model_l1.cuda()
        elif self.device.type == 'mps':
            self.model_t12 = self.model_t12.to('mps')
            self.model_l1 = self.model_l1.to('mps')
    
    def _init_dual_segmentation_with_regression(self):
        """Initialize dual segmentation with regression (T12L1 with bone density)"""
        # Initialize dual segmentation models first
        self._init_dual_model_segmentation()
        
        # Load regression models
        from heartregression_model import UNetRFull
        
        # Load T12 regression model with correct args from checkpoint
        t12_reg_path = self.config['regression_checkpoint_t12']
        t12_checkpoint = torch.load(t12_reg_path, map_location='cpu')
        t12_input_feature = t12_checkpoint.get('input_feature', 'x')
        t12_n_classes = t12_checkpoint.get('n_class', 1)
        self.regression_model_t12 = UNetRFull(n_channels=1, n_classes=t12_n_classes, args=t12_input_feature, isHeart=True)
        self.regression_model_t12.load_state_dict(t12_checkpoint['weight'])
        self.regression_model_t12.eval()
        self.regression_model_t12 = self.regression_model_t12.cpu()
        
        # Load L1 regression model with correct args from checkpoint
        l1_reg_path = self.config['regression_checkpoint_l1']
        l1_checkpoint = torch.load(l1_reg_path, map_location='cpu')
        l1_input_feature = l1_checkpoint.get('input_feature', 'x,y')
        l1_n_classes = l1_checkpoint.get('n_class', 1)
        self.regression_model_l1 = UNetRFull(n_channels=1, n_classes=l1_n_classes, args=l1_input_feature, isHeart=True)
        self.regression_model_l1.load_state_dict(l1_checkpoint['weight'])
        self.regression_model_l1.eval()
        self.regression_model_l1 = self.regression_model_l1.cpu()
    
    def _init_tb_with_lung_preprocessing(self):
        """Initialize TB model with lung preprocessing pipeline"""
        # Initialize lung segmentation model first
        opt = BaseOptions()
        opt.initialize()
        self.opt = opt.parse(save=False)
        
        # Use unified checkpoints directory for TB lung model
        lung_checkpoint_path = self.config['lung_checkpoint_path']
        
        self.opt.checkpoint_path = lung_checkpoint_path
        self.opt.checkpoints_dir = os.path.dirname(lung_checkpoint_path)
        self.opt.name = 'xray2lung'  # Use lung model configuration
        self.opt.cuda1 = 0
        self.opt.continue_train = False
        
        # Set GPU IDs
        if torch.cuda.is_available():
            self.opt.gpu_ids = [0]
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.opt.gpu_ids = [0]
        else:
            self.opt.gpu_ids = []
        
        # Create lung model for preprocessing
        self.lung_model = create_model(self.opt)
        self.lung_model.eval()
        
        # Load TB model (EfficientNet)
        tb_checkpoint_path = self.config['checkpoint_path']
        
        # Try to use efficientnet_pytorch package first (common in older models)
        try:
            from efficientnet_pytorch import EfficientNet
            # Create EfficientNet-B5 model
            self.tb_model = EfficientNet.from_name('efficientnet-b5', num_classes=2)
            
            # Load the state dict
            checkpoint = torch.load(tb_checkpoint_path, map_location=self.device)
            self.tb_model.load_state_dict(checkpoint, strict=False)
            
        except ImportError:
            # Fall back to torchvision implementation
            import torchvision.models as models
            from torchvision.models import efficientnet_b5
            
            # Create EfficientNet-B5 model for TB classification
            self.tb_model = efficientnet_b5(weights=None)
            # Modify classifier for TB (binary classification: TB/Normal)
            num_features = self.tb_model.classifier[1].in_features
            self.tb_model.classifier = torch.nn.Linear(num_features, 2)
            
            # Load the state dict with less strict matching
            checkpoint = torch.load(tb_checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Try to load with strict=False to handle key mismatches
            self.tb_model.load_state_dict(state_dict, strict=False)
        self.tb_model = self.tb_model.to(self.device)
        self.tb_model.eval()
        
        # Move models to device
        if self.device.type == 'cuda':
            self.lung_model = self.lung_model.cuda()
        elif self.device.type == 'mps':
            self.lung_model = self.lung_model.to('mps')
    
    def _init_laa_model(self):
        """Initialize LAA model (segmentation_models_pytorch with auxiliary params)"""
        import segmentation_models_pytorch as smp
        
        # Auxiliary parameters (DCX_python_inference exact)
        aux_params = dict(
            pooling='avg',
            dropout=0.2,
            activation='softmax',
            classes=2,
        )
        
        # Create UNet model with auxiliary parameters
        self.model = smp.Unet(
            encoder_name=self.config.get('model_type', 'efficientnet-b0'),
            encoder_weights=None,
            in_channels=self.config.get('in_channels', 3),
            classes=self.config.get('classes', 1),
            aux_params=aux_params,
            activation=self.config.get('activation', 'sigmoid')
        )
        
        # Load checkpoint (DCX_python_inference exact weight loading)
        checkpoint_path = self.config['checkpoint_path']
        weight = torch.load(checkpoint_path, map_location='cpu')
        
        # Process state dict (remove "model." prefix)
        new_state_dict = {}
        for key in weight.keys():
            new_state_dict[key.replace("model.", "")] = weight[key]
        
        self.model.load_state_dict(state_dict=new_state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
    
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
        elif self.module_type == 'segmentation_models_pytorch':
            return self._process_smp(dicom_path, output_path)
        elif self.module_type == 'efficientnet_classification':
            return self._process_efficientnet(dicom_path, output_path)
        elif self.module_type == 'dual_model_segmentation':
            return self._process_dual_model_segmentation(dicom_path, output_path)
        elif self.module_type == 'dual_segmentation_with_regression':
            return self._process_dual_segmentation_with_regression(dicom_path, output_path)
        elif self.module_type == 'tb_with_lung_preprocessing':
            return self._process_tb_with_lung_preprocessing(dicom_path, output_path)
        elif self.module_type == 'segmentation_models_pytorch_laa':
            return self._process_laa_model(dicom_path, output_path)
        else:
            return self._process_segmentation(dicom_path, output_path, lung_mask_path)
    
    def _process_diffusion(self, dicom_path, output_path):
        """Process diffusion model (Group 4: bone_supp)"""
        # Load DICOM
        ds = dicom.dcmread(dicom_path)
        dcm_arr = np.array(ds.pixel_array, dtype='float32')
        original_shape = dcm_arr.shape
        
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
        
        # Apply output size resizing if needed
        if hasattr(self, 'output_size') and self.output_size != 'original':
            output_denorm = self._resize_output_to_size(output_denorm, self.output_size)
        
        if self.output_format == 'dcm':
            # Save as DICOM (exactly like original DCX_python_inference)
            output_denorm = output_denorm.astype(str(ds.pixel_array.dtype))
            ds.PixelData = output_denorm.tobytes()
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create if there's a directory path
                os.makedirs(output_dir, exist_ok=True)
            ds.save_as(output_path)
        else:
            # Save as NIfTI or PNG using standard method
            pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
            ratio = 1.0  # No resizing ratio for bone_supp
            
            # Check PhotometricInterpretation for proper inversion (like DICOM saving)
            photometric_interpretation = getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2')
            output_for_nii = output_denorm.copy()
            
            # Invert if MONOCHROME1 (like original DCX logic)
            if photometric_interpretation == 'MONOCHROME1':
                # Get the data type max value for inversion
                if ds.pixel_array.dtype == np.uint8:
                    max_val = 255
                elif ds.pixel_array.dtype == np.uint16:
                    max_val = 65535
                else:
                    max_val = np.iinfo(ds.pixel_array.dtype).max
                
                output_for_nii = max_val - output_for_nii
                print(f"Inverted output for MONOCHROME1 (max_val: {max_val})")
            
            # Reshape for _save_output (expects 4D: [batch, channel, height, width])
            output_4d = output_for_nii[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
            
            # Use standard save method to support NIfTI and PNG formats
            self._save_output(output_4d, output_path, pixel_spacing, ratio)
        
        print(f"Bone suppression output saved to: {output_path}")
        return {'output': output_denorm}
    
    def _process_segmentation(self, dicom_path, output_path, lung_mask_path=None):
        """Process segmentation-based models (Groups 1, 2, 3)"""
        # Check if we can reuse existing segmentation for volumetry modules
        if self.module_name in ['heart_volumetry', 'lung_volumetry']:
            base_module = 'heart' if 'heart' in self.module_name else 'lung'
            output_dir = os.path.dirname(output_path)
            input_basename = os.path.splitext(os.path.basename(dicom_path))[0]
            existing_mask_path = os.path.join(output_dir, f"{input_basename}_{base_module}.nii")
            
            if os.path.exists(existing_mask_path):
                print(f"Using existing {base_module} segmentation for volume calculation...")
                return self._process_volumetry_from_existing_mask(dicom_path, output_path, existing_mask_path)
        
        # Load DICOM data for potential volume calculation
        ds = dicom.dcmread(dicom_path, force=True)
        dicom_array = ds.pixel_array.astype(np.float32)
        dicom_pixel_spacing = ds.get((0x0028, 0x0030), [0.18, 0.18])
        if isinstance(dicom_pixel_spacing, (int, float)):
            dicom_pixel_spacing = [dicom_pixel_spacing, dicom_pixel_spacing]
        
        # Get input and metadata
        img, pixel_spacing, ratio, original_input_shape = self._preprocess_input(dicom_path, lung_mask_path)
        
        # Run inference
        output = self._run_inference(img)
        
        # Postprocess
        output_full, mask = self._postprocess(output)
        
        # Handle aorta0/aorta1 single channel extraction
        if self.module_name == 'aorta0':
            # Extract only ascending aorta (channel 0)
            single_channel = output_full[:, 0:1, :, :]
            self._save_output(single_channel, output_path, pixel_spacing, ratio)
        elif self.module_name == 'aorta1':
            # Extract only descending aorta (channel 1) 
            single_channel = output_full[:, 1:2, :, :]
            self._save_output(single_channel, output_path, pixel_spacing, ratio)
        else:
            # Regular processing for combined aorta
            if self.config.get('multi_channel_output', False):
                # For multi-channel like aorta, sum the channels for combined output
                combined_output = np.sum(output_full, axis=1, keepdims=True)  # Sum across channel dimension
                self._save_output(combined_output, output_path, pixel_spacing, ratio)
                
                # Also save individual channels
                self._save_multi_channel_output(output_full, output_path, pixel_spacing, ratio)
            else:
                # Regular single-channel output
                self._save_output(output_full, output_path, pixel_spacing, ratio)
        
        # Calculate metrics
        area = volume = None
        if self.config.get('calculate_area', False):
            area = self._calculate_area(mask, pixel_spacing, ratio)
            print(f"Area: {area:.2f} cm²")
        
        if self.config.get('calculate_volume', False):
            if hasattr(self, 'regression_model') and self.regression_model is not None:
                try:
                    print(f"🔄 Starting volume calculation with area={area:.2f} cm²...")
                    volume = self._calculate_volume(dicom_array, dicom_pixel_spacing, area)
                    print(f"Volume: {volume:.2f} mL ({volume/1000:.3f} L)")
                except Exception as e:
                    print(f"⚠ Volume calculation failed: {e}")
                    volume = None
            else:
                print("⚠ Volume calculation skipped - regression model not loaded")
                volume = None
        
        # Process COVID and vessel modules using in-memory lung data if in batch mode
        covid_results = vessel_results = None
        if self.batch_mode and self.config.get('name') == 'xray2lung':
            covid_results, vessel_results = self._process_lung_dependent_modules(
                output_full, dicom_path, output_path, pixel_spacing, ratio
            )
        
        results = {'output': output_full, 'mask': mask, 'area': area, 'volume': volume}
        if covid_results:
            results['covid'] = covid_results
        if vessel_results:
            results['vessel'] = vessel_results
        
        # Add metadata and postprocessing measurements when available
        results.update(self._extract_metadata_and_measurements(dicom_array, dicom_pixel_spacing, output_full, mask))
            
        return results
    
    def _preprocess_input(self, dicom_path, lung_mask_path=None):
        """Preprocess input based on module group"""
        # Read DICOM for metadata
        ds = dicom.dcmread(dicom_path, force=True)
        
        # Store original DICOM path for DICOM output format and postprocessing
        self._original_dicom_path = dicom_path
        
        # Get metadata (DCX_python_inference exact extraction)
        pixel_spacing = ds.get((0x0028, 0x0030), [0.18, 0.18])
        if isinstance(pixel_spacing, (int, float)):
            pixel_spacing = [pixel_spacing, pixel_spacing]
        
        # Always use the config's loadSize for model inference (exactly as original)
        target_size = self.config['loadSize']
        
        # Store original dimensions for 'original' size mode (used only for output resizing)
        if self.output_size == 'original':
            self._original_dicom_shape = ds.pixel_array.shape
        
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
        
        # Apply transformations: 3x rot90 + fliplr (DCX_python_inference exact order)
        image_array = np.rot90(image_array)
        image_array = np.rot90(image_array)
        image_array = np.rot90(image_array)
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
        
        # Handle MONOCHROME1 with outlier clipping (DCX_python_inference exact)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            mean = np.mean(image_array)
            std_dev = np.std(image_array)
            threshold = mean + 2 * std_dev
            replacement_value = mean + 2 * std_dev
            image_array = np.where(image_array > threshold, replacement_value, image_array)
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
        # Get output range from config or use module-specific defaults
        if 'output_min_max' in self.config:
            # Parse output_min_max from config (e.g., "-1100,-500")
            min_max_str = self.config['output_min_max'].split(',')
            b_min_val, b_max_val = int(min_max_str[0]), int(min_max_str[1])
        elif 'output_range' in self.config:
            # Parse output_range from config (e.g., [-1100, -500])
            b_min_val, b_max_val = self.config['output_range'][0], self.config['output_range'][1]
        else:
            # Fallback to module-specific defaults (exactly as original)
            if 'covid' in self.config['name']:
                b_min_val, b_max_val = -1100, -400
            elif 'aorta' in self.config['name']:
                b_min_val, b_max_val = -1100, -700
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
    
    def _calculate_volume(self, dicom_data, dicom_pixel_spacing, area):
        """Calculate volume using regression model (Group 3) - DCX_python_inference exact"""
        # Load state and get parameters
        if 'heart' in self.config['name']:
            checkpoint_path = self.config['regression_checkpoint_path']
        else:
            checkpoint_path = self.config['regression_checkpoint_path']
        
        # Use unified checkpoints directory for volumetry regression
        checkpoint_full_path = checkpoint_path
        
        state = torch.load(checkpoint_full_path, map_location='cpu')
        ww = state.get('ww')
        wl = state.get('wl')
        
        # Use original DICOM data directly (no file loading needed)
        arr = dicom_data.copy()
        pixel_spacing = dicom_pixel_spacing
        
        # Resize input according to pixel size (lung regression specific)
        if 'lung' in self.config['name']:
            max_pixsize = 0.319333  # Original lung regression max pixel size
            target_resize = pixel_spacing[0] / max_pixsize * 2048
        else:
            max_pixsize = 0.9355469  # Heart regression max pixel size
            target_resize = pixel_spacing[0] / max_pixsize * 512
        
        input_ = Image.fromarray(arr)
        input_, _ = resize_keep_ratio_pil(input_, target_resize, "LANCZOS")
        
        # Use appropriate padding size based on module type
        if 'lung' in self.config['name']:
            pad_size = 2048
        else:
            pad_size = 512
        img_pad = pad_image_pil(input_, pad_size, pad_value=0)
        input_ = np.array(img_pad(input_))
        
        # Normalization (DCX_python_inference exact)
        realMinHU = np.amin(input_)
        if realMinHU > 100:
            wl = (realMinHU + 1024) + wl
        
        minHU = wl - (ww / 2)
        maxHU = wl + (ww / 2)
        
        x = np.clip(input_, minHU, maxHU)
        input_norm = (x - minHU) / (maxHU - minHU)
        
        # Prepare inputs (lung regression needs [area, sex, age])
        if 'lung' in self.config['name']:
            # Use default values for sex (0=male) and age (50) when not provided
            sex = 0
            age = 50
            input_feat = torch.from_numpy(np.array([[area, sex, age]], dtype=np.float32))
        else:
            input_feat = torch.from_numpy(np.array([[area]], dtype=np.float32))
        input_img = torch.from_numpy(input_norm[np.newaxis, np.newaxis]).float()
        
        # Run regression
        self.regression_model.eval()
        with torch.no_grad():
            reg_pred = self.regression_model(input_img, input_feat)
        
        result = reg_pred.item()
        
        # Cleanup
        self.regression_model.cpu()
        input_img.cpu()
        input_feat.cpu()
        torch.cuda.empty_cache()
        
        return result
    
    def _save_output(self, output, output_path, pixel_spacing, ratio):
        """Save output as NIfTI, DICOM, or PNG file"""
        # Resize output based on output_size option
        if self.output_size == 'original' and hasattr(self, '_original_dicom_shape'):
            output = self._resize_output_to_original(output)
        elif self.output_size in ['512', '2048']:
            # Resize to specific size (512x512 or 2048x2048)
            target_output_size = int(self.output_size)
            if output[0, 0].shape != (target_output_size, target_output_size):
                output = self._resize_output_to_size(output, target_output_size)
        # If output_size matches config loadSize, no resizing needed
        
        if self.output_format == 'dcm':
            # Save as DICOM using the same approach as bone_supp
            self._save_as_dicom(output, output_path)
        elif self.output_format == 'png':
            # Save as PNG image
            self._save_as_png(output, output_path)
        else:
            # Save as NIfTI (default)
            nii_np = np.transpose(output, axes=[3, 2, 1, 0])
            nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
            
            # Set pixel dimensions
            pixel_size_resize_w = pixel_spacing[0] / ratio
            nii.header['pixdim'] = pixel_size_resize_w
            
            # Save file
            nib.save(nii, output_path)
        
        print(f"Output saved to: {output_path}")
        
        # Create 512x512 version for CTR calculation only for heart when CTR is requested
        # (CTR expects 2048x2048 lung, 512x512 heart)
        if self.module_name == 'heart' and self.output_size != '512' and self.ctr_requested:
            self._save_512_version_for_ctr(output, output_path, pixel_spacing, ratio)
    
    def _save_512_version_for_ctr(self, output, original_path, pixel_spacing, ratio):
        """Save a 512x512 version of heart mask for CTR calculation"""
        # Resize to 512x512
        output_512 = self._resize_output_to_size(output, 512)
        
        # Create filename with _512 suffix
        dir_path = os.path.dirname(original_path)
        basename = os.path.basename(original_path)
        name_parts = os.path.splitext(basename)
        
        # Remove any existing size suffix and add _512
        base_name = name_parts[0]
        if base_name.endswith('_2048'):
            base_name = base_name[:-5]  # Remove _2048
        output_filename_512 = f"{base_name}_512{name_parts[1]}"
        output_path_512 = os.path.join(dir_path, output_filename_512)
        
        # Save as NIfTI (CTR expects NIfTI files)
        nii_np = np.transpose(output_512, axes=[3, 2, 1, 0])
        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
        
        # Set pixel dimensions
        pixel_size_resize_w = pixel_spacing[0] / ratio
        nii.header['pixdim'] = pixel_size_resize_w
        
        # Save file
        nib.save(nii, output_path_512)
        print(f"512x512 version saved for CTR: {output_path_512}")
    
    def _resize_output_to_original(self, output):
        """Resize output to original DICOM dimensions"""
        from skimage.transform import resize
        
        # Get 2D output
        output_2d = output[0, 0]
        
        # Resize to original dimensions
        resized_2d = resize(output_2d, self._original_dicom_shape, preserve_range=True)
        
        # Reconstruct 4D array
        resized_output = resized_2d[np.newaxis, np.newaxis, :, :]
        
        return resized_output
    
    def _resize_output_to_size(self, output, target_size):
        """Resize output to specific size (512x512 or 2048x2048)"""
        from skimage.transform import resize
        
        # Convert target_size to int if it's a string
        if isinstance(target_size, str):
            target_size = int(target_size)
        
        # Handle different input shapes
        if output.ndim == 2:
            # 2D array (from bone_supp)
            resized_output = resize(output, (target_size, target_size), preserve_range=True)
        elif output.ndim == 4:
            # 4D array (from segmentation models)
            output_2d = output[0, 0]
            resized_2d = resize(output_2d, (target_size, target_size), preserve_range=True)
            resized_output = resized_2d[np.newaxis, np.newaxis, :, :]
        else:
            raise ValueError(f"Unsupported output shape: {output.shape}")
        
        return resized_output
        
    def _save_as_dicom(self, output, output_path):
        """Save output as DICOM using optimized creation method"""
        if not hasattr(self, '_original_dicom_path'):
            print("Warning: Original DICOM path not available for DICOM output. Using NIfTI format instead.")
            # Fallback to NIfTI-like saving but with .dcm extension
            nii_np = np.transpose(output, axes=[3, 2, 1, 0])
            nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
            nib.save(nii, output_path.replace('.dcm', '.nii'))
            return
            
        # Get 2D array from 4D segmentation output
        output_2d = output[0, 0]  # Extract 2D array
        
        # Use the optimized DICOM creation function
        self._create_dicom_file_optimized(output_2d, self._original_dicom_path, output_path)
    
    def _create_dicom_file_optimized(self, nii_np, original_dicom_path, output_filename):
        """Create DICOM following the simple approach from DCX_python_inference"""
        try:
            # Read original DICOM
            ds = dicom.dcmread(original_dicom_path)
            
            # Get original data type
            dicom_dtype = ds.pixel_array.dtype
            
            # Clean data first to handle NaN values
            nii_clean = np.nan_to_num(nii_np, nan=-1024.0, posinf=0.0, neginf=-1024.0)
            
            # Ensure the output array has the same shape as the original
            if nii_clean.shape != ds.pixel_array.shape:
                # If shapes don't match, resize to original dimensions
                from PIL import Image
                img = Image.fromarray(nii_clean)
                img_resized = img.resize((ds.pixel_array.shape[1], ds.pixel_array.shape[0]), Image.NEAREST)
                nii_clean = np.array(img_resized)
            
            # Convert to original DICOM data type
            # For uint16, we need to shift negative values to positive range
            if dicom_dtype == np.uint16:
                # Shift HU values to positive range: -1024 becomes 0, 0 becomes 1024, etc.
                # This matches how medical imaging typically handles uint16 storage
                output_data = nii_clean + 1024
                output_data = np.clip(output_data, 0, 65535).astype(np.uint16)
                
                # Update RescaleIntercept and RescaleSlope for proper HU interpretation
                # These tags tell viewers how to convert stored values back to HU
                ds.RescaleIntercept = -1024
                ds.RescaleSlope = 1
            else:
                # For signed types, just convert directly
                output_data = nii_clean.astype(dicom_dtype)
            
            # Ensure data is C-contiguous (required for DICOM)
            output_data = np.ascontiguousarray(output_data)
            
            # Update the DICOM's PixelData
            ds.PixelData = output_data.tobytes()
            ds.Rows, ds.Columns = output_data.shape
            
            # Remove window settings to let viewer auto-adjust
            # Most medical viewers handle this better than hardcoded values
            if hasattr(ds, 'WindowCenter'):
                delattr(ds, 'WindowCenter')
            if hasattr(ds, 'WindowWidth'):
                delattr(ds, 'WindowWidth')
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_filename)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the modified DICOM file
            ds.save_as(output_filename)
            print(f"DICOM saved to: {output_filename}")
            
        except Exception as e:
            print(f"Error creating DICOM: {e}")
            # Fallback to NIfTI if DICOM creation fails
            nii_np_4d = np.expand_dims(np.expand_dims(nii_np, axis=2), axis=3)
            nii = nib.Nifti1Image(nii_np_4d.astype(np.int16), affine=None)
            nib.save(nii, output_filename.replace('.dcm', '.nii'))
    
    def _save_as_png(self, output, output_path):
        """Save output as PNG image"""
        # Get 2D array from 4D segmentation output
        output_2d = output[0, 0]  # Extract 2D array
        
        # Normalize to 0-255 range for PNG
        output_min, output_max = output_2d.min(), output_2d.max()
        if output_max > output_min:
            normalized = ((output_2d - output_min) / (output_max - output_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(output_2d, dtype=np.uint8)
        
        # Convert to PIL Image and save
        img = Image.fromarray(normalized)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Change extension to .png
        png_path = output_path.replace('.nii', '.png').replace('.dcm', '.png')
        img.save(png_path)
        print(f"PNG saved to: {png_path}")
    
    def _load_segmentation_file(self, file_path):
        """Load segmentation file in any format (NIfTI, DICOM, PNG)"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.nii':
            # Load NIfTI file
            nii = nib.load(file_path)
            return nii.get_fdata()
        elif file_ext == '.dcm':
            # Load DICOM file
            ds = dicom.dcmread(file_path)
            return ds.pixel_array.astype(np.float64)
        elif file_ext == '.png':
            # Load PNG file
            img = Image.open(file_path)
            return np.array(img).astype(np.float64)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _process_lung_dependent_modules(self, lung_output, dicom_path, lung_output_path, pixel_spacing, ratio):
        """Process COVID and vessel modules using in-memory lung data"""
        covid_results = vessel_results = None
        
        # Process COVID module
        print(f"\n{'='*60}")
        print("Processing module: covid (using in-memory lung data)")
        print(f"{'='*60}")
        
        try:
            covid_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'covid.yaml')
            covid_inference = UnifiedDCXInference(covid_config_path, self.device, self.output_format, self.output_size)
            covid_inference.batch_mode = True  # Ensure temporary NIfTI files are created
            
            # BUGFIX: Pass original DICOM shape and path to COVID/vessel modules
            if hasattr(self, '_original_dicom_shape'):
                covid_inference._original_dicom_shape = self._original_dicom_shape
            if hasattr(self, '_original_dicom_path'):
                covid_inference._original_dicom_path = self._original_dicom_path
            
            # Transform lung data for COVID processing
            lung_array = lung_output[0, 0]  # Extract 2D array
            
            # Apply COVID-specific preprocessing (no histogram normalization for COVID)
            lung_array = self._apply_lung_transforms(lung_array)
            
            # Run COVID inference
            covid_output = self._run_covid_vessel_inference(lung_array, covid_inference)
            
            # Apply transpose for proper orientation (like other modules)
            covid_output = np.transpose(covid_output, axes=[0, 1, 3, 2])
            
            # Save COVID output
            output_dir = os.path.dirname(lung_output_path)
            input_basename = os.path.splitext(os.path.basename(dicom_path))[0]
            size_suffix = "_2048" if self.output_size == '2048' else ""
            covid_output_path = os.path.join(output_dir, f"{input_basename}_covid{size_suffix}.{self.output_format}")
            
            covid_inference._save_output(covid_output, covid_output_path, pixel_spacing, ratio)
            
            # Calculate area
            covid_mask = covid_output[0, 0] > -1015  # COVID threshold
            covid_area = covid_inference._calculate_area(covid_mask, pixel_spacing, ratio) if covid_inference.config.get('calculate_area') else None
            if covid_area:
                print(f"COVID Area: {covid_area:.2f} cm²")
            
            covid_results = {'output': covid_output, 'mask': covid_mask, 'area': covid_area}
            print("✓ Module covid completed successfully")
            
        except Exception as e:
            print(f"✗ Module covid failed: {str(e)}")
            covid_results = {'error': str(e)}
        
        # Process vessel module
        print(f"\n{'='*60}")
        print("Processing module: vessel (using in-memory lung data)")
        print(f"{'='*60}")
        
        try:
            vessel_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'vessel.yaml')
            vessel_inference = UnifiedDCXInference(vessel_config_path, self.device, self.output_format, self.output_size)
            vessel_inference.batch_mode = True  # Ensure temporary NIfTI files are created
            
            # BUGFIX: Pass original DICOM shape and path to vessel module
            if hasattr(self, '_original_dicom_shape'):
                vessel_inference._original_dicom_shape = self._original_dicom_shape
            if hasattr(self, '_original_dicom_path'):
                vessel_inference._original_dicom_path = self._original_dicom_path
            
            # Transform lung data for vessel processing
            lung_array = lung_output[0, 0]  # Extract 2D array
            
            # Apply vessel-specific preprocessing (WITH histogram normalization for vessel)
            lung_array = self._apply_lung_transforms(lung_array)
            lung_array = histogram_normalization(lung_array)  # Vessel-specific step
            
            # Run vessel inference
            vessel_output = self._run_covid_vessel_inference(lung_array, vessel_inference)
            
            # Apply transpose for proper orientation (like other modules)
            vessel_output = np.transpose(vessel_output, axes=[0, 1, 3, 2])
            
            # Save vessel output
            size_suffix = "_2048" if self.output_size == '2048' else ""
            vessel_output_path = os.path.join(output_dir, f"{input_basename}_vessel{size_suffix}.{self.output_format}")
            vessel_inference._save_output(vessel_output, vessel_output_path, pixel_spacing, ratio)
            
            # Calculate area
            vessel_mask = vessel_output[0, 0] > -1015  # Vessel threshold
            vessel_area = vessel_inference._calculate_area(vessel_mask, pixel_spacing, ratio) if vessel_inference.config.get('calculate_area') else None
            if vessel_area:
                print(f"Vessel Area: {vessel_area:.2f} cm²")
            
            # Calculate peripheral vessel measurements using lung and vessel data
            peripheral_measurements = self._calculate_peripheral_vessels(vessel_output, lung_output, pixel_spacing)
            
            vessel_results = {'output': vessel_output, 'mask': vessel_mask, 'area': vessel_area}
            vessel_results.update(peripheral_measurements)
            print("✓ Module vessel completed successfully")
            
        except Exception as e:
            print(f"✗ Module vessel failed: {str(e)}")
            vessel_results = {'error': str(e)}
        
        return covid_results, vessel_results
    
    def _apply_lung_transforms(self, lung_array):
        """Apply lung transforms for COVID/vessel processing"""
        # Apply transformations: 3x rot90 + fliplr (DCX_python_inference exact order)
        lung_array = np.rot90(lung_array)
        lung_array = np.rot90(lung_array)
        lung_array = np.rot90(lung_array)
        lung_array = np.fliplr(lung_array)
        return lung_array
    
    def _run_covid_vessel_inference(self, lung_array, inference_obj):
        """Run inference on transformed lung data for COVID/vessel"""
        # Resize and pad like original preprocessing
        target_size = inference_obj.config['loadSize']
        A = Image.fromarray(lung_array)
        interpolation = inference_obj.config.get('interpolation', 'LANCZOS')
        A, ratio = resize_keep_ratio_pil(A, target_size, interpolation)
        img_pad = pad_image_pil(A, target_size=target_size, pad_value=0)
        A_ = np.array(img_pad(A))
        
        # Convert to tensor and run inference
        to_tensor = transforms.ToTensor()
        normalized_a = A_.astype(np.float32)
        normalized_a = to_tensor(normalized_a)
        normalized_a = normalized_a.unsqueeze(0)
        
        # Move to appropriate device
        if inference_obj.device.type == 'cuda' and len(inference_obj.opt.gpu_ids) > 0:
            normalized_a = normalized_a.cuda()
        elif inference_obj.device.type == 'mps' and len(inference_obj.opt.gpu_ids) > 0:
            normalized_a = normalized_a.to('mps')
        
        # Run inference
        with torch.no_grad():
            output = inference_obj.model(normalized_a)
            output = output.cpu().numpy()
        
        # Postprocess
        output_full, mask = inference_obj._postprocess(output)
        
        return output_full
    
    def _extract_metadata_and_measurements(self, dicom_array, dicom_pixel_spacing, output_full, mask):
        """Extract metadata and calculate module-specific postprocessing measurements"""
        measurements = {}
        
        # Extract metadata
        measurements['pixel_spacing_mm'] = f"{dicom_pixel_spacing[0]:.3f}"
        measurements['image_height'] = dicom_array.shape[0]
        measurements['image_width'] = dicom_array.shape[1]
        
        # Module-specific postprocessing measurements
        module_name = self.config.get('name', '')
        
        if module_name == 'xray2aorta':
            # Calculate aorta diameter during aorta processing
            aorta_measurements = self._calculate_aorta_diameter(output_full, dicom_pixel_spacing)
            print(f"DEBUG: Aorta measurements returned: {aorta_measurements}")
            measurements.update(aorta_measurements)
        
        # Postprocessing measurements are now handled by reading saved .nii files
        
        return measurements
    
    def _calculate_aorta_diameter(self, aorta_output, pixel_spacing):
        """Placeholder for aorta diameter - actual calculation done in postprocessing"""
        # Aorta diameter will be calculated by postprocessing using saved .nii files
        # This ensures exact compatibility with original DCX algorithms
        return {}
    
    def _store_heart_data_for_ctr(self, heart_output, heart_mask, pixel_spacing):
        """Store heart data for cardiothoracic ratio calculation"""
        self._heart_output = heart_output
        self._heart_mask = heart_mask
        self._heart_pixel_spacing = pixel_spacing
    
    def _calculate_cardiothoracic_ratio_if_ready(self, lung_output, lung_mask, pixel_spacing):
        """Postprocessing removed - focusing on segmentation and regression only"""
        return {}
    
    def _calculate_peripheral_vessels(self, vessel_output, lung_output, pixel_spacing):
        """Postprocessing removed - focusing on segmentation and regression only"""
        return {}
    
    def _process_smp(self, dicom_path, output_path):
        """Process segmentation_models_pytorch model (LAA)"""
        # Load and preprocess DICOM
        ds = dicom.dcmread(dicom_path)
        image_array = ds.pixel_array.astype(np.float32)
        
        # Normalize to 0-1
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        
        # Convert to RGB (3 channels)
        image_rgb = np.stack([image_array] * 3, axis=-1)
        
        # Resize to 512x512
        image_pil = Image.fromarray((image_rgb * 255).astype(np.uint8))
        image_resized = image_pil.resize((512, 512))
        
        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            output = torch.sigmoid(output)  # Apply sigmoid for binary segmentation
        
        # Convert to numpy and save (keep as float values like original LAA)
        output_np = output.cpu().numpy()[0, 0]
        
        # Resize to original dimensions while preserving float values
        # Use PIL resize but with float array
        output_pil = Image.fromarray(output_np, mode='F')  # 'F' mode for float32
        output_resized = output_pil.resize(ds.pixel_array.shape[::-1], Image.NEAREST)
        output_array = np.array(output_resized)
        
        # Save as NIfTI with float values (like original)
        nii_np = output_array[np.newaxis, np.newaxis, :, :]
        nii = nib.Nifti1Image(nii_np.astype(np.float32), affine=None)
        nib.save(nii, output_path)
        
        print(f"LAA segmentation saved to: {output_path}")
        return {'output': output_array}
    
    def _process_efficientnet(self, dicom_path, output_path):
        """Process EfficientNet classification model (TB)"""
        # Load and preprocess DICOM
        ds = dicom.dcmread(dicom_path)
        image_array = ds.pixel_array.astype(np.float32)
        
        # Resize and normalize
        image_pil = Image.fromarray(image_array).convert('RGB')
        image_resized = image_pil.resize((512, 512))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image_resized).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output).item()
        
        print(f"TB Probability: {probability:.4f}")
        return {'probability': probability, 'prediction': probability > 0.5}
    
    def _process_dual_model_segmentation(self, dicom_path, output_path):
        """Process dual model segmentation (T12L1)"""
        # Get input and metadata
        img, pixel_spacing, ratio, original_input_shape = self._preprocess_input(dicom_path, None)
        
        # Run T12 inference
        output_t12 = self._run_inference_dual(img, 't12')
        output_t12_full, mask_t12 = self._postprocess_dual(output_t12, 't12')
        
        # Run L1 inference
        output_l1 = self._run_inference_dual(img, 'l1')
        output_l1_full, mask_l1 = self._postprocess_dual(output_l1, 'l1')
        
        # Save both outputs with consistent naming
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Simply append _t12 and _l1 to the base filename (keeps t12l1 for clarity)
        extension = self.output_format
        t12_path = os.path.join(output_dir, f"{base_name}_t12.{extension}")
        l1_path = os.path.join(output_dir, f"{base_name}_l1.{extension}")
        
        self._save_output(output_t12_full, t12_path, pixel_spacing, ratio)
        self._save_output(output_l1_full, l1_path, pixel_spacing, ratio)
        
        # Calculate areas
        area_t12 = area_l1 = None
        if self.config.get('calculate_area', False):
            area_t12 = self._calculate_area(mask_t12, pixel_spacing, ratio)
            area_l1 = self._calculate_area(mask_l1, pixel_spacing, ratio)
            print(f"T12 Area: {area_t12:.2f} cm²")
            print(f"L1 Area: {area_l1:.2f} cm²")
        
        return {'output_t12': output_t12_full, 'output_l1': output_l1_full, 
                'mask_t12': mask_t12, 'mask_l1': mask_l1, 
                'area_t12': area_t12, 'area_l1': area_l1}
    
    def _run_inference_dual(self, img, model_type):
        """Run inference for dual models (T12 or L1)"""
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
        
        # Select model
        model = self.model_t12 if model_type == 't12' else self.model_l1
        
        # Run inference
        with torch.no_grad():
            output = model(normalized_a)
            output = output.cpu().numpy()
        
        return output
    
    def _postprocess_dual(self, output, model_type):
        """Apply postprocessing for dual models (T12 or L1)"""
        # Get output range based on model type
        if model_type == 't12':
            b_min_val, b_max_val = -1100, -500
        else:  # L1
            b_min_val, b_max_val = -1024, 1000
        
        # Denormalize
        denormalize_gen = output * (b_max_val - b_min_val) + b_min_val
        
        # Apply thresholding
        threshold = self.config['threshold']
        replace_value = -1024
        denormalize_gen = np.where(denormalize_gen[0] < threshold, replace_value, denormalize_gen)
        
        # Create mask
        denormalize_gen_mask = np.where(denormalize_gen[0, 0] < threshold, 0, 1)
        
        return denormalize_gen, denormalize_gen_mask
    
    def _save_multi_channel_output(self, output_full, output_path, pixel_spacing, ratio):
        """Save multi-channel output as separate files (aorta: ascending + descending)"""
        if output_full.shape[1] < 2:
            print("Warning: Expected multi-channel output but only found single channel")
            return
        
        output_channels = self.config.get('output_channels', ['channel_0', 'channel_1'])
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Remove the module suffix if it exists to get original filename
        if base_name.endswith('_aorta_2048'):
            original_name = base_name[:-11].rstrip('_')  # Remove '_aorta_2048' and trailing underscores
        elif base_name.endswith('_aorta'):
            original_name = base_name[:-6].rstrip('_')  # Remove '_aorta' and trailing underscores
        else:
            original_name = base_name
        
        for i, channel_name in enumerate(output_channels):
            if i >= output_full.shape[1]:
                break
                
            # Extract single channel
            channel_output = output_full[:, i:i+1, :, :]
            
            # Apply output size resizing if needed
            if self.output_size == 'original' and hasattr(self, '_original_dicom_shape'):
                channel_output = self._resize_output_to_original(channel_output)
            elif self.output_size in ['512', '2048']:
                # Resize to specific size (512x512 or 2048x2048)
                target_output_size = int(self.output_size)
                if channel_output[0, 0].shape != (target_output_size, target_output_size):
                    channel_output = self._resize_output_to_size(channel_output, target_output_size)
            
            # Save as separate NIfTI file
            nii_np = np.transpose(channel_output, axes=[3, 2, 1, 0])
            nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
            
            # Set pixel dimensions
            pixel_size_resize_w = pixel_spacing[0] / ratio
            nii.header['pixdim'] = pixel_size_resize_w
            
            # Create channel filename with correct extension and size suffix
            extension = self.output_format
            size_suffix = "_2048" if self.output_size == '2048' else ""
            channel_path = os.path.join(output_dir, f"{original_name}_{channel_name}{size_suffix}.{extension}")
            
            # Use standard save method to support all formats
            self._save_output(channel_output, channel_path, pixel_spacing, ratio)
            print(f"Channel {i} ({channel_name}) saved to: {channel_path}")

    def _process_dual_segmentation_with_regression(self, dicom_path, output_path):
        """Process dual segmentation with regression (T12L1 + bone density)"""
        # First do dual segmentation (same as T12L1)
        result = self._process_dual_model_segmentation(dicom_path, output_path)
        
        # Get the saved NIfTI paths for regression  
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        # Remove the module suffix if it exists to get original filename  
        if base_name.endswith('_t12l1_regression'):
            original_name = base_name[:-17].rstrip('_')  # Remove '_t12l1_regression' and trailing underscores
        else:
            original_name = base_name
            
        # Use the actual generated filenames with correct suffix
        extension = self.output_format
        t12_path = os.path.join(output_dir, f"{original_name}_t12l1_regression_t12.{extension}")
        l1_path = os.path.join(output_dir, f"{original_name}_t12l1_regression_l1.{extension}")
        
        # Run T12 regression
        t12_hu = None
        l1_hu = None
        
        if self.config.get('calculate_regression', False):
            try:
                t12_hu = self._calculate_bone_regression(t12_path, result['area_t12'], 't12')
                print(f"T12 HU: {t12_hu:.2f}")
            except Exception as e:
                print(f"T12 regression failed: {e}")
                t12_hu = -2
            
            try:
                l1_hu = self._calculate_bone_regression(l1_path, result['area_l1'], 'l1')
                print(f"L1 HU: {l1_hu:.2f}")
            except Exception as e:
                print(f"L1 regression failed: {e}")
                l1_hu = -2
        
        # Add regression results to output
        result['t12_hu'] = t12_hu
        result['l1_hu'] = l1_hu
        
        return result
    
    def _calculate_bone_regression(self, nifti_path, area, bone_type):
        """Calculate bone density regression (T12 or L1)"""
        # Select appropriate regression model
        regression_model = self.regression_model_t12 if bone_type == 't12' else self.regression_model_l1
        
        # Load state parameters (similar to heart/lung regression)
        checkpoint_key = 'regression_checkpoint_t12' if bone_type == 't12' else 'regression_checkpoint_l1'
        checkpoint_path = self.config[checkpoint_key]
        
        state = torch.load(checkpoint_path, map_location='cpu')
        ww = state.get('ww')
        wl = state.get('wl')
        
        # Load and process NIfTI (DCX_python_inference exact)
        nii = nib.load(nifti_path)
        arr = np.transpose(np.array(nii.dataobj)[:, :, 0, 0], axes=[1, 0])
        header = nii.header
        pixel_spacing = header.get_zooms()
        
        # Resize according to pixel spacing
        max_pixsize = 0.9355469
        target_resize = pixel_spacing[0] / max_pixsize * 512
        
        input_ = Image.fromarray(arr)
        input_, _ = resize_keep_ratio_pil(input_, target_resize, "LANCZOS")
        img_pad = pad_image_pil(input_, 512, pad_value=0)
        input_ = np.array(img_pad(input_))
        
        # Normalization (DCX_python_inference exact)
        realMinHU = np.amin(input_)
        if realMinHU > 100:
            wl = (realMinHU + 1024) + wl
        
        minHU = wl - (ww / 2)
        maxHU = wl + (ww / 2)
        
        x = np.clip(input_, minHU, maxHU)
        input_norm = (x - minHU) / (maxHU - minHU)
        
        # Prepare inputs based on bone type (matching original implementation)
        # Get the expected input feature from the model's stored parameters
        regression_model = self.regression_model_t12 if bone_type == 't12' else self.regression_model_l1
        checkpoint_key = 'regression_checkpoint_t12' if bone_type == 't12' else 'regression_checkpoint_l1'
        checkpoint_path = self.config[checkpoint_key]
        state = torch.load(checkpoint_path, map_location='cpu')
        input_feature = state.get('input_feature', 'x')
        
        # Create input features based on the model's expected args
        if input_feature == 'x':  # T12 - no additional features
            input_feat = torch.from_numpy(np.array([[]], dtype=np.float32))
        elif 'y' in input_feature:  # L1 - includes area
            input_feat = torch.from_numpy(np.array([[area]], dtype=np.float32))
        else:
            input_feat = torch.from_numpy(np.array([[]], dtype=np.float32))
            
        input_img = torch.from_numpy(input_norm[np.newaxis, np.newaxis]).float()
        
        # Run regression
        regression_model.eval()
        with torch.no_grad():
            reg_pred = regression_model(input_img, input_feat)
        
        result = reg_pred.item()
        
        # Cleanup
        regression_model.cpu()
        input_img.cpu()
        input_feat.cpu()
        torch.cuda.empty_cache()
        
        return result
    
    def _process_volumetry_from_existing_mask(self, dicom_path, output_path, existing_mask_path):
        """Process volumetry calculation using existing segmentation mask"""
        import nibabel as nib
        
        # Load existing mask
        nii = nib.load(existing_mask_path)
        mask_data = nii.get_fdata()
        
        # Handle different NIfTI shapes (squeeze to 2D, then add dims back)
        if mask_data.ndim > 2:
            mask_data = np.squeeze(mask_data)  # Remove singleton dimensions
        
        # Ensure 2D
        if mask_data.ndim != 2:
            raise ValueError(f"Mask data should be 2D after squeezing, got shape: {mask_data.shape}")
        
        # Create binary mask (same logic as _postprocess)
        threshold = self.config.get('threshold', -1015)
        mask = np.where(mask_data > threshold, 1, 0)
        
        # Get pixel spacing from NIfTI header
        pixel_spacing = nii.header.get_zooms()[:2]  # Take first 2 dimensions
        ratio = 1.0  # No ratio adjustment needed for existing mask
        
        # Copy existing mask to volumetry output path (to maintain consistency)
        output_4d = mask_data[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
        self._save_output(output_4d, output_path, pixel_spacing, ratio)
        
        # Calculate metrics
        area = volume = None
        if self.config.get('calculate_area', False):
            area = self._calculate_area(mask, pixel_spacing, ratio)
            print(f"Area: {area:.2f} cm²")
        
        if self.config.get('calculate_volume', False):
            volume = self._calculate_volume(existing_mask_path, area)  # Use existing mask for volume calc
            print(f"Volume: {volume:.2f} mL ({volume/1000:.3f} L)")
        
        return {'output': output_4d, 'mask': mask, 'area': area, 'volume': volume, 'pixel_spacing': pixel_spacing}
    
    def _process_tb_with_lung_preprocessing(self, dicom_path, output_path):
        """Process TB detection with lung preprocessing (DCX_python_inference exact)"""
        import cv2
        
        # Check if lung segmentation already exists in output directory (for --all_modules)
        output_dir = os.path.dirname(output_path)
        input_basename = os.path.splitext(os.path.basename(dicom_path))[0]
        
        # Look for existing lung segmentation in current output format
        existing_lung_path = os.path.join(output_dir, f"{input_basename}_lung.{self.output_format}")
        # Fallback to .nii if not found in current format
        if not os.path.exists(existing_lung_path):
            existing_lung_path = os.path.join(output_dir, f"{input_basename}_lung.nii")
        
        if os.path.exists(existing_lung_path):
            print(f"Step 1: Using existing lung segmentation: {existing_lung_path}")
            temp_lung_path = existing_lung_path
        else:
            # Step 1: Run lung segmentation preprocessing
            print("Step 1: Running lung segmentation for TB preprocessing...")
            
            # Load and preprocess DICOM for lung segmentation
            img, pixel_spacing, ratio, original_input_shape = self._preprocess_input(dicom_path, None)
            
            # Run lung segmentation using lung model
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
            
            # Run lung inference
            with torch.no_grad():
                lung_output = self.lung_model(normalized_a)
                lung_output = lung_output.cpu().numpy()
            
            # Use lung config parameters for postprocessing
            original_config = self.config.copy()
            lung_config = {
                'threshold': -1015,
                'output_range': [-1100, -500],
                'name': 'xray2lung',
                'use_connected_regions': True,
                'n_regions': 2
            }
            self.config.update(lung_config)
            
            lung_output_full, lung_mask = self._postprocess(lung_output)
            
            # Restore original config
            self.config = original_config
            
            # Save temporary lung segmentation
            base_path = os.path.splitext(output_path)[0]
            temp_lung_path = base_path + f'_temp_lung.{self.output_format}'
            self._save_output(lung_output_full, temp_lung_path, pixel_spacing, ratio)
        
        # Step 2: Process TB using lung segmentation (DCX_python_inference exact pipeline)
        print("Step 2: Running TB detection with lung-segmented input...")
        
        # Load original DICOM
        ds = dicom.dcmread(dicom_path)
        dcm_array = ds.pixel_array.astype(np.float64)
        
        # Normalize DICOM
        dcm_normalized = (dcm_array - dcm_array.min()) / (dcm_array.max() - dcm_array.min())
        
        # Load lung segmentation (support multiple formats)
        lung_array = self._load_segmentation_file(temp_lung_path)
        lung_array = np.squeeze(lung_array)
        lung_array = np.expand_dims(lung_array, axis=2)
        lung_array = cv2.rotate(lung_array, cv2.ROTATE_90_CLOCKWISE)
        lung_array = cv2.flip(lung_array, 1)
        
        # Normalize lung segmentation
        lung_normalized = (lung_array - lung_array.min()) / (lung_array.max() - lung_array.min())
        lung_resized = self._resize_and_pad(lung_normalized, (2048, 2048))
        lung_final = np.clip(lung_resized, 0.0, 1.0).astype(np.float64)
        
        # Resize DICOM and apply lung mask
        dcm_resized = self._resize_and_pad(dcm_normalized, (2048, 2048))
        dcm_final = np.clip(dcm_resized, 0.0, 1.0).astype(np.float64)
        
        # Apply lung mask (DCX_python_inference exact)
        lung_binary = np.where(lung_final > 0.05, 1, 0)
        dcm_segmented = dcm_final * lung_binary
        
        # Prepare for TB model input
        dcm_segmented = np.stack((dcm_segmented,) * 3, axis=-1)
        dcm_segmented = np.clip(dcm_segmented, 0, 1).astype(np.float64)
        dcm_segmented = cv2.resize(dcm_segmented, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        dcm_segmented = dcm_segmented[np.newaxis, ...]  # Add batch dimension
        
        # Convert to tensor and run TB inference
        dcm_tensor = torch.from_numpy(dcm_segmented).float().to(self.device)
        dcm_tensor = dcm_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        with torch.no_grad():
            tb_output = self.tb_model(dcm_tensor)
            # TB output has 2 classes [Normal, TB], take TB probability
            tb_probabilities = torch.softmax(tb_output, dim=1)
            tb_probability = tb_probabilities[0, 1].item()  # Take TB class probability
        
        # Clean up temporary files only if we created them (not existing lung files)
        if not os.path.exists(existing_lung_path) and os.path.exists(temp_lung_path):
            os.remove(temp_lung_path)
        
        print(f"TB Probability: {tb_probability:.4f}")
        
        return {'probability': tb_probability, 'prediction': tb_probability > 0.5}
    
    def _process_laa_model(self, dicom_path, output_path):
        """Process LAA segmentation (DCX_python_inference exact)"""
        # Store original DICOM path for DICOM output format
        self._original_dicom_path = dicom_path
        
        # Load DICOM and convert to RGB array (DCX_python_inference exact)
        ds = dicom.dcmread(dicom_path, force=True)
        
        # Handle DICOM reading errors (original DCX logic)
        try:
            image_arr = ds.pixel_array
        except:
            ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
            image_arr = ds.pixel_array
        
        image_arr = np.array(image_arr, dtype=np.int16)
        pixel_spacing = ds.PixelSpacing
        
        # BUGFIX: Store original DICOM dimensions BEFORE any processing
        original_width, original_height = image_arr.shape[0], image_arr.shape[1]
        
        # Handle MONOCHROME1 (original DCX logic)
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            image_arr = image_arr * -1 + image_arr.max()
        
        # Normalize image_arr value to 0-255 (original DCX logic)
        image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min()) * 255
        # Make sure image_arr has 3 channels
        image_arr = np.stack((image_arr,)*3, axis=-1).astype(np.int16)
        
        # Resize if not 512x512 (original DCX logic)
        if not (image_arr.shape[0] == 512 and image_arr.shape[1] == 512):
            import cv2
            s = max(image_arr.shape[0:2])
            f = np.zeros((s, s, 3), np.float64)
            ax, ay = (s - image_arr.shape[1]) // 2, (s - image_arr.shape[0]) // 2
            f[ay:image_arr.shape[0] + ay, ax:ax + image_arr.shape[1], ...] = image_arr
            image_arr = cv2.resize(f, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        
        image_arr = np.clip(image_arr / 255, 0, 1).astype(np.float32)
        
        # RGB image checks (original DCX logic)
        assert image_arr.shape[-1] == 3
        assert image_arr.shape[0] == 512 and image_arr.shape[1] == 512
        
        # Convert to model input format (original DCX logic)
        img = np.moveaxis(image_arr, 0, -1)
        img = np.moveaxis(img, 1, 0)
        
        assert img.max() <= 1 and img.min() >= 0
        assert img.dtype == np.float32
        
        img = torch.Tensor(img).to(self.device)
        img = torch.unsqueeze(img, 0)  # Add batch dimension
        
        # Run LAA inference (original DCX logic with auxiliary outputs)
        output_dict = {}
        with torch.no_grad():
            out = self.model(img)
            
            batch_size = 1
            for batch in range(0, batch_size):
                # Emphysema binary classification probability (original DCX logic)
                emphysema_prob = out[1][batch][1].cpu().numpy()
                print("Emphysema binary classification : %.1f%%" % (emphysema_prob * 100))
                output_dict['emphysema_prob'] = emphysema_prob * 100
                print("EMPHYSEMA PROB : ", emphysema_prob * 100)
                
                # Classification threshold (original DCX exact)
                if emphysema_prob >= 0.0262980695:
                    print("EMPHYSEMA : POSITIVE")
                    output_dict['desc'] = "EMPHYSEMA : POSITIVE"
                else:
                    print("EMPHYSEMA : NEGATIVE")
                    output_dict['desc'] = "EMPHYSEMA : NEGATIVE"
                
                # Emphysema mask (original DCX logic)
                emphysema_mask = out[0][batch]
                emphysema_mask = emphysema_mask.cpu().numpy()
                emphysema_mask = np.squeeze(emphysema_mask)
                emphysema_mask = np.moveaxis(emphysema_mask, 1, 0)
                
                # Use original array for output (original DCX logic)
                emphysema_ori = np.array(emphysema_mask)
                
                # Resize output to match output_size setting
                if hasattr(self, 'output_size') and self.output_size == 'original':
                    # Resize back to original DICOM dimensions
                    emphysema_resized = Image.fromarray(emphysema_ori)
                    emphysema_resized = emphysema_resized.resize((original_height, original_width), Image.NEAREST)
                    emphysema_ori = np.array(emphysema_resized)
                elif hasattr(self, 'output_size') and self.output_size in ['2048']:
                    # Resize to specified size
                    target_size = int(self.output_size)
                    emphysema_resized = Image.fromarray(emphysema_ori)
                    emphysema_resized = emphysema_resized.resize((target_size, target_size), Image.NEAREST)
                    emphysema_ori = np.array(emphysema_resized)
                # For 512 or default, keep as-is (already 512x512)
                
                # Create binary mask with 0.001 threshold for area calculation (original DCX logic)
                emphysema001 = np.where(emphysema_ori >= 0.001, 1, 0)
                count_ones = np.sum(emphysema001)
                
                # Area calculation (original DCX exact formula)
                print(original_width)
                print(original_height)
                
                # Adjust area calculation based on actual output size
                if hasattr(self, 'output_size') and self.output_size == 'original':
                    # When output is at original size, no resize factor needed
                    area = count_ones * pixel_spacing[0] * pixel_spacing[1] / 100.0
                elif hasattr(self, 'output_size') and self.output_size in ['2048']:
                    # When output is at 2048x2048
                    target_size = int(self.output_size)
                    resize_factor = target_size / max(original_width, original_height)
                    area = count_ones * (pixel_spacing[0] / resize_factor) * (pixel_spacing[1] / resize_factor) / 100.0
                else:
                    # Default: output is at 512x512 (original DCX formula)
                    area = count_ones * (pixel_spacing[0] * max(original_width, original_height) / 512.0) * (pixel_spacing[1] * max(original_width, original_height) / 512.0) / 100.0
                
                print("EMPHYSEMA AREA : ", area)
                output_dict['area'] = area
        
        # Save main output exactly like original LAA (float values with transpose)
        # Check output format
        if self.output_format == 'png':
            # For PNG, normalize the float values to 0-255 range for visualization
            # This matches how NIfTI viewers display the data
            # No transpose needed for PNG - keep original orientation
            if emphysema_ori.max() > 0:
                # Normalize to full range based on actual values
                emphysema_normalized = (emphysema_ori / emphysema_ori.max() * 255).astype(np.uint8)
            else:
                emphysema_normalized = np.zeros_like(emphysema_ori, dtype=np.uint8)
            Image.fromarray(emphysema_normalized, mode='L').save(output_path)
        elif self.output_format == 'dcm':
            # For DICOM, use standard save method with proper formatting
            # Reshape LAA output to 4D format expected by _save_output
            emphysema_4d = emphysema_ori[np.newaxis, np.newaxis, :, :]
            # Use standard DICOM save method
            self._save_as_dicom(emphysema_4d, output_path)
        else:
            # For NIfTI, save with transpose as float values
            nii_np = np.transpose(emphysema_ori, axes=[1, 0])
            nii = nib.Nifti1Image(nii_np.astype(np.float32), affine=None)
            nib.save(nii, output_path)
        
        # Save CSV output (original DCX logic) - only if not using unified CSV
        if not self.unified_csv:
            output_csv_path = os.path.join(os.path.dirname(output_path), 'output_emphysema.csv')
            image_name = os.path.basename(dicom_path)
            with open(output_csv_path, 'a') as fd:
                fd.write("\n" + image_name + ": " + str(output_dict))
            
            print(f"LAA segmentation saved to: {output_path}")
            print(f"Results saved to: {output_csv_path}")
        else:
            print(f"LAA segmentation saved to: {output_path}")
        
        return output_dict
    
    def _resize_and_pad(self, img, size, padColor=0):
        """Resize and pad image (DCX_python_inference exact for TB)"""
        import cv2
        
        h, w = img.shape[:2]
        sh, sw = size
        
        # Interpolation method
        if h > sh or w > sw:  # shrinking image
            interp = cv2.INTER_AREA
        else:  # stretching image
            interp = cv2.INTER_CUBIC
        
        # Aspect ratio of image
        aspect = w/h
        
        # Compute scaling and pad sizing
        if aspect > 1:  # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1:  # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else:  # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
        
        # Set pad color
        if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
            padColor = [padColor]*3
        
        # Scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=padColor)
        
        return scaled_img
    

# Postprocessing functions removed - focusing on segmentation and regression only


def main():
    parser = argparse.ArgumentParser(description='Unified DCX Medical Imaging Inference')
    parser.add_argument('--module', type=str, 
                       choices=['lung', 'heart', 'airway', 'bone', 'covid', 'vessel', 
                               'heart_volumetry', 'lung_volumetry', 'bone_supp',
                               'aorta', 'aorta0', 'aorta1', 't12l1', 'laa', 'tb', 't12l1_regression'],
                       help='Module to use for inference (required unless --all_modules is used)')
    parser.add_argument('--all_modules', action='store_true',
                       help='Run all segmentation modules in a single execution')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input DICOM file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--lung_mask', type=str, default=None,
                       help='Path to lung mask (required for covid/vessel modules)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Force specific device (auto=automatic selection)')
    parser.add_argument('--output_format', type=str, default='nii',
                       choices=['nii', 'dcm', 'png'],
                       help='Output format: nii (NIfTI), dcm (DICOM), or png (PNG image)')
    parser.add_argument('--output_size', type=str, default='512',
                       choices=['original', '512', '2048'],
                       help='Output size: original (keep input size), 512, or 2048')
    parser.add_argument('--no_calculations', action='store_true',
                       help='Skip area/volume calculations (only generate masks)')
    # Postprocessing removed - focusing on segmentation and regression only
    parser.add_argument('--collect_measurements', action='store_true',
                       help='Collect all measurements into a single CSV file')
    parser.add_argument('--diameter', action='store_true',
                       help='Calculate aorta max diameter (forces 2048x2048 for aorta module)')
    parser.add_argument('--ctr', action='store_true',
                       help='Calculate cardiothoracic ratio (CTR) when running heart module')
    parser.add_argument('--peripheral', action='store_true',
                       help='Generate peripheral masks (50% and 70%) when running lung module')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_modules and not args.module:
        parser.error("Either --module or --all_modules must be specified")
    if args.all_modules and args.module:
        parser.error("Cannot use both --module and --all_modules together")
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which modules to run
    if args.all_modules:
        # Run all available modules - COVID and vessel will be processed during lung processing
        modules_to_run = ['lung', 'heart', 'airway', 'bone', 
                         'aorta', 't12l1', 'laa', 'tb', 'bone_supp']
        print(f"Running all modules: {', '.join(modules_to_run)}")
        print("Note: COVID and vessel modules will be processed during lung segmentation")
    else:
        modules_to_run = [args.module]
    
    # Process each module
    all_results = {}
    for module in modules_to_run:
        print(f"\n{'='*60}")
        print(f"Processing module: {module}")
        print(f"{'='*60}")
        
        try:
            # Get config path - handle aorta0/aorta1 mapping to aorta config
            base_module = module
            if module in ['aorta0', 'aorta1']:
                base_module = 'aorta'
            config_path = os.path.join(os.path.dirname(__file__), 'configs', f'{base_module}.yaml')
            
            # Generate output filename based on input filename and module
            input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
            
            # Handle output format and file extensions
            if module == 'bone_supp':
                # Bone suppression supports all formats now
                extension = args.output_format
                output_filename = f"{input_basename}_{module}.{extension}"
            elif module in ['tb']:
                # TB only outputs classification results (no file output)
                output_filename = f"{input_basename}_{module}_result"  # Dummy filename for consistency
            elif module == 'aorta0':
                # Ascending aorta
                extension = args.output_format
                size_suffix = "_2048" if args.output_size == '2048' else ""
                output_filename = f"{input_basename}_aorta_asc{size_suffix}.{extension}"
            elif module == 'aorta1':
                # Descending aorta
                extension = args.output_format
                size_suffix = "_2048" if args.output_size == '2048' else ""
                output_filename = f"{input_basename}_aorta_desc{size_suffix}.{extension}"
            else:
                # Use user-specified format
                extension = args.output_format
                size_suffix = "_2048" if args.output_size == '2048' else ""
                output_filename = f"{input_basename}_{module}{size_suffix}.{extension}"
            
            output_file = os.path.join(args.output_dir, output_filename)
            
            # Force 2048x2048 output for aorta if diameter calculation is requested
            output_size_override = args.output_size
            if args.diameter and module == 'aorta':
                output_size_override = '2048'
                print("📏 Diameter calculation requested - forcing 2048x2048 output for aorta module")
            
            # Create inference object and process
            # Use unified CSV when collect_measurements is enabled
            # Pass ctr_requested flag for heart module
            ctr_requested = args.ctr and module == 'heart'
            inference = UnifiedDCXInference(config_path, args.device, args.output_format, output_size_override, module, 
                                          unified_csv=args.collect_measurements, batch_mode=args.all_modules, 
                                          ctr_requested=ctr_requested)
            
            # Skip calculations if requested
            if args.no_calculations:
                # Temporarily disable calculations in config
                inference.config['calculate_area'] = False
                inference.config['calculate_volume'] = False
            else:
                # Volume calculations only for modules with regression models
                volume_modules = ['heart', 'lung', 'heart_volumetry', 'lung_volumetry', 't12l1_regression']
                if module not in volume_modules:
                    inference.config['calculate_volume'] = False
            
            # Auto-detect lung mask for covid/vessel modules
            lung_mask_path = args.lung_mask
            if module in ['covid', 'vessel'] and not lung_mask_path:
                # First try to find lung mask with same basename as input
                input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
                
                # Look for lung mask with matching basename in priority order
                potential_lung_files = [
                    os.path.join(args.output_dir, f'{input_basename}_lung.{args.output_format}'),  # Same format as requested
                    os.path.join(args.output_dir, f'{input_basename}_lung.nii'),  # NIfTI fallback
                    os.path.join(args.output_dir, f'{input_basename}_lung.png'),  # PNG fallback
                    os.path.join(args.output_dir, f'{input_basename}_lung.dcm')   # DICOM fallback
                ]
                
                # Find first existing file
                for potential_file in potential_lung_files:
                    if os.path.exists(potential_file):
                        lung_mask_path = potential_file
                        print(f"🔍 Auto-detected lung mask for {module}: {lung_mask_path}")
                        break
                
                # If no exact match, look for any lung files (batch mode fallback)
                if not lung_mask_path:
                    lung_files = (glob.glob(os.path.join(args.output_dir, f'*lung.{args.output_format}')) or
                                 glob.glob(os.path.join(args.output_dir, '*lung.nii')) or  # fallback to nii
                                 glob.glob(os.path.join(args.output_dir, '*lung.*')))  # any format as last resort
                    if lung_files:
                        lung_mask_path = lung_files[0]
                        print(f"🔍 Auto-detected lung mask for {module}: {lung_mask_path}")
                
                # Warning if still no lung mask found
                if not lung_mask_path:
                    print(f"⚠ No lung mask found for {module} module. Please run lung segmentation first or provide --lung_mask parameter.")
            
            results = inference.process(args.input_file, output_file, lung_mask_path)
            all_results[module] = results
            
            print(f"✓ Module {module} completed successfully")
            
            # Run peripheral mask generation if requested
            if args.peripheral and module == 'lung':
                print(f"\n{'='*60}")
                print("🫁 Generating peripheral masks (50% and 70%)...")
                print(f"{'='*60}")
                
                try:
                    # Import the peripheral mask functions
                    sys.path.append(os.path.join(os.path.dirname(__file__), 'postprocessing', 'vessels_peripheral area'))
                    from central_mask_python import find_contours, center_point, center_point_one, full_mask, bitwise_mask, save_mask
                    
                    # Force 2048x2048 for lung if peripheral calculation requested
                    lung_2048_path = None
                    if args.output_size != '2048':
                        # Need to generate 2048 version
                        print("⚡ Generating 2048x2048 lung mask for peripheral calculation...")
                        lung_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'lung.yaml')
                        lung_inference = UnifiedDCXInference(lung_config_path, args.device, args.output_format, '2048', 'lung')
                        lung_2048_path = os.path.join(args.output_dir, f"{input_basename}_lung_2048.{args.output_format}")
                        lung_results = lung_inference.process(args.input_file, lung_2048_path, None)
                    else:
                        # Already have 2048 version
                        lung_2048_path = output_file
                    
                    # Set global variable for the functions
                    import central_mask_python
                    
                    # Process lung mask to generate peripheral masks
                    img_color, lung_contours = find_contours(lung_2048_path)
                    central_mask_python.img = img_color
                    
                    # Filter contours (keep only significant ones > 1000 pixels)
                    lung_contour = []
                    for c in lung_contours:
                        if len(c) > 1000:
                            lung_contour.append(c)
                    
                    if len(lung_contour) == 0:
                        print("⚠️  No significant lung contours found")
                        raise Exception("No lung contours found for peripheral mask generation")
                    
                    # Calculate center point
                    if len(lung_contour) == 1:
                        center = center_point_one(lung_contour)
                    else:
                        center = center_point(lung_contour)
                    
                    # Generate full mask
                    masks = full_mask(center, lung_contour)
                    
                    # Track files to clean up
                    temp_files_to_delete = []
                    
                    # Generate masks at different percentages
                    masks_dict = {}
                    for p in [0.5, 0.7]:
                        mask, masked = bitwise_mask(img_color, masks, center, p)
                        masks_dict[p] = mask[:,:,0]
                    
                    # Calculate areas using pixel spacing from DICOM
                    # Load the original DICOM to get pixel spacing
                    ds = dicom.dcmread(args.input_file, force=True)
                    pixel_spacing = ds.get((0x0028, 0x0030), [0.18, 0.18])
                    if isinstance(pixel_spacing, (int, float)):
                        pixel_spacing = [pixel_spacing, pixel_spacing]
                    
                    # Always calculate areas when we have masks
                    full_area = np.sum(masks[:,:,0] > 0) * pixel_spacing[0] * pixel_spacing[1] / 100.0
                    area_50 = np.sum(masks_dict[0.5] > 0) * pixel_spacing[0] * pixel_spacing[1] / 100.0
                    area_70 = np.sum(masks_dict[0.7] > 0) * pixel_spacing[0] * pixel_spacing[1] / 100.0
                    
                    peripheral_50_70 = area_70 - area_50
                    peripheral_70_100 = full_area - area_70
                    
                    # Print results (similar to aorta diameter output)
                    print(f"\n✓ Peripheral area calculation completed")
                    print(f"\n=== Peripheral Area Results ===")
                    print(f"Total lung area: {full_area:.2f} cm²")
                    print(f"Central area (0-50%): {area_50:.2f} cm²")
                    print(f"Mid-peripheral area (50-70%): {peripheral_50_70:.2f} cm²")
                    print(f"Peripheral area (70-100%): {peripheral_70_100:.2f} cm²")
                    
                    # Calculate percentages
                    central_pct = (area_50 / full_area) * 100
                    mid_pct = (peripheral_50_70 / full_area) * 100
                    peripheral_pct = (peripheral_70_100 / full_area) * 100
                    
                    print(f"\nArea distribution:")
                    print(f"Central: {central_pct:.1f}%")
                    print(f"Mid-peripheral: {mid_pct:.1f}%")
                    print(f"Peripheral: {peripheral_pct:.1f}%")
                    
                    # Update results for CSV
                    all_results[module]['peripheral_total_area_cm2'] = full_area
                    all_results[module]['peripheral_central_area_cm2'] = area_50
                    all_results[module]['peripheral_mid_area_cm2'] = peripheral_50_70
                    all_results[module]['peripheral_outer_area_cm2'] = peripheral_70_100
                    
                    # Clean up all temporary files
                    # Remove PNG files created by find_contours
                    temp_png_files = [
                        lung_2048_path.replace('.nii', '.png'),
                        lung_2048_path.replace('.nii', '_mask(binary).png'),
                        lung_2048_path.replace('.nii', '_contour.png'),
                        lung_2048_path.replace('.nii', '_fullMask.png')
                    ]
                    
                    for temp_file in temp_png_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            temp_files_to_delete.append(temp_file)
                    
                    # Clean up temporary 2048 file if created
                    if args.output_size != '2048' and os.path.exists(lung_2048_path):
                        os.remove(lung_2048_path)
                        temp_files_to_delete.append(lung_2048_path)
                    
                    if temp_files_to_delete:
                        print("\n🧹 Cleaning up temporary files...")
                        for temp_file in temp_files_to_delete:
                            print(f"🗑️  Removed: {os.path.basename(temp_file)}")
                    
                except Exception as e:
                    print(f"⚠️  Peripheral mask generation failed: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
            
            # Run aorta diameter calculation if requested
            if args.diameter and module == 'aorta':
                print(f"\n{'='*60}")
                print("📏 Calculating aorta max diameter...")
                print(f"{'='*60}")
                
                try:
                    # Import the diameter calculation function
                    sys.path.append(os.path.join(os.path.dirname(__file__), 'postprocessing', 'aorta_max diameter'))
                    from aortav2_test_plus_20250324 import compute_diameter
                    
                    # Prepare file dict for diameter calculation
                    diameter_file_dict = {input_basename: []}
                    
                    # Find the generated 2048 files
                    combined_file = f"{input_basename}_aorta_2048.{args.output_format}"
                    asc_file = f"{input_basename}_aorta_asc_2048.{args.output_format}"
                    desc_file = f"{input_basename}_aorta_desc_2048.{args.output_format}"
                    
                    # Track all 2048 files to remove
                    all_2048_files = []
                    
                    # Check which files exist
                    if os.path.exists(os.path.join(args.output_dir, asc_file)):
                        diameter_file_dict[input_basename].append(asc_file)
                        all_2048_files.append(asc_file)
                    if os.path.exists(os.path.join(args.output_dir, desc_file)):
                        diameter_file_dict[input_basename].append(desc_file)
                        all_2048_files.append(desc_file)
                    if os.path.exists(os.path.join(args.output_dir, combined_file)):
                        all_2048_files.append(combined_file)
                    
                    if diameter_file_dict[input_basename]:
                        # Create output directory for diameter results
                        diameter_output_dir = os.path.join(args.output_dir, 'diameter_results')
                        os.makedirs(diameter_output_dir, exist_ok=True)
                        
                        # Run diameter calculation
                        compute_diameter(
                            output_folder=diameter_output_dir,
                            input_folder=args.output_dir,
                            file_dict=diameter_file_dict,
                            head=input_basename,
                            ASCENDING_ONLY=(len(diameter_file_dict[input_basename]) == 1),
                            visualize=False,  # Disable visualization to avoid errors
                            heat_map=False
                        )
                        
                        print("✓ Aorta diameter calculation completed")
                        
                        # Remove temporary 2048 files if not originally requested
                        if args.output_size != '2048':
                            for temp_file in all_2048_files:
                                temp_path = os.path.join(args.output_dir, temp_file)
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                    print(f"🗑️  Removed temporary file: {temp_file}")
                        
                        # Also remove diameter_results directory
                        if os.path.exists(diameter_output_dir):
                            import shutil
                            shutil.rmtree(diameter_output_dir)
                            print(f"🗑️  Removed diameter results directory")
                    else:
                        print("⚠️  No aorta files found for diameter calculation")
                        
                except Exception as e:
                    print(f"⚠️  Diameter calculation failed: {e}")
                    # Continue processing other modules even if diameter fails
            
            # Run CTR calculation if requested
            if args.ctr and module == 'heart':
                print(f"\n{'='*60}")
                print("💓 Calculating cardiothoracic ratio (CTR)...")
                print(f"{'='*60}")
                
                try:
                    # Track files to potentially delete after CTR calculation
                    temp_files_to_delete = []
                    lung_was_generated = False
                    
                    # Check if lung mask exists
                    lung_2048_file = f"{input_basename}_lung_2048.{args.output_format}"
                    lung_2048_path = os.path.join(args.output_dir, lung_2048_file)
                    
                    # If no 2048 lung mask exists, generate it automatically
                    if not os.path.exists(lung_2048_path):
                        print("📦 No lung mask found. Automatically generating lung segmentation at 2048x2048...")
                        
                        # Run lung segmentation
                        lung_config_path = os.path.join(os.path.dirname(__file__), 'configs', 'lung.yaml')
                        lung_inference = UnifiedDCXInference(lung_config_path, args.device, args.output_format, '2048', 'lung')
                        
                        # Process lung
                        lung_results = lung_inference.process(args.input_file, lung_2048_path, None)
                        if 'error' in lung_results:
                            raise Exception(f"Lung segmentation failed: {lung_results['error']}")
                        
                        print("✓ Lung segmentation completed automatically")
                        lung_was_generated = True
                        temp_files_to_delete.append(lung_2048_path)
                        
                        # Also track any intermediate files created by lung segmentation
                        lung_base = os.path.join(args.output_dir, f"{input_basename}_lung_2048")
                        for ext in ['.png', '_contour.png', '_mask(binary).png']:
                            temp_file = lung_base + ext
                            if os.path.exists(temp_file):
                                temp_files_to_delete.append(temp_file)
                    
                    # Import CTR calculation function
                    sys.path.append(os.path.join(os.path.dirname(__file__), 'postprocessing', 'heart_cardiothoracic ratio'))
                    from cardiothoracic_ratio import (find_contours, center_point, center_point_one, 
                                                      full_mask, bitwise_mask, get_longest_line)
                    import cardiothoracic_ratio
                    
                    # Get heart mask path (512x512)
                    heart_512_file = f"{input_basename}_heart_512.{args.output_format}"
                    heart_512_path = os.path.join(args.output_dir, heart_512_file)
                    
                    # Track heart_512 if it's temporary (not originally requested size)
                    if args.output_size != '512':
                        temp_files_to_delete.append(heart_512_path)
                    
                    # Check if 512 heart mask exists
                    if not os.path.exists(heart_512_path):
                        # Check for default size heart mask
                        heart_file = f"{input_basename}_heart.{args.output_format}"
                        heart_path = os.path.join(args.output_dir, heart_file)
                        if os.path.exists(heart_path) and args.output_size == '512':
                            heart_512_path = heart_path
                        else:
                            print("⚠️  No 512x512 heart mask found. CTR requires 512x512 heart mask.")
                            raise Exception("512x512 heart mask required for CTR calculation")
                    
                    # Read DICOM for metadata
                    import pydicom
                    data = pydicom.dcmread(args.input_file)
                    pixel_spacing = data.PixelSpacing if 'PixelSpacing' in data else [0.144, 0.144]
                    height = data.Rows
                    width = data.Columns
                    
                    # Process lung contours
                    lung_contour = []
                    img, lung_contours = find_contours(lung_2048_path)
                    
                    # Track intermediate files created by find_contours
                    lung_base = lung_2048_path[:-4]  # Remove .nii extension
                    ctr_intermediate_files = [
                        f"{lung_base}.png",
                        f"{lung_base}_mask(binary).png",
                        f"{lung_base}_contour.png"
                    ]
                    for intermediate_file in ctr_intermediate_files:
                        if os.path.exists(intermediate_file):
                            temp_files_to_delete.append(intermediate_file)
                    
                    # Set global img variable for cardiothoracic_ratio module
                    cardiothoracic_ratio.img = img
                    
                    for c in lung_contours:
                        if len(c) > 1000:
                            lung_contour.append(c)
                    
                    if len(lung_contour) == 0:
                        print("⚠️  No significant lung contours found")
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
                    
                    # Find MHTD (Maximum Horizontal Thoracic Diameter)
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
                    center = startX + ((endX - startX) / 2)
                    
                    # Process heart (512x512 -> repeated to 2048x2048)
                    heart_img = nib.load(heart_512_path)
                    heart_data = heart_img.get_fdata()
                    heart_data = np.squeeze(heart_data)
                    heart_data = np.transpose(heart_data, (1, 0))
                    heart_data = np.repeat(heart_data, 4, axis=0)
                    heart_data = np.repeat(heart_data, 4, axis=1)
                    
                    # Find MHCD (Maximum Horizontal Cardiac Diameter)
                    left_heart = heart_data[:, :int(center)]
                    right_heart = heart_data[:, int(center):]
                    
                    left_longest_line = get_longest_line(left_heart, center, 2048)
                    right_longest_line = get_longest_line(right_heart, 2048-center, 2048)
                    
                    left_length = left_longest_line[1][0] - left_longest_line[0][0]
                    right_length = right_longest_line[1][0] - right_longest_line[0][0]
                    MHCD = left_length + right_length
                    
                    # Calculate CTR
                    newSize = 2048
                    newSpacingX = pixel_spacing[0] * max(width, height) / newSize
                    ct_ratio = (MHCD * newSpacingX) / MHTD
                    
                    # Print results (similar to aorta diameter output)
                    print(f"\n✓ CTR calculation completed")
                    print(f"\n=== Cardiothoracic Ratio Results ===")
                    print(f"MHTD: {MHTD:.2f} mm")
                    print(f"MHCD: {MHCD * newSpacingX:.2f} mm")
                    print(f"CT Ratio: {ct_ratio:.3f}")
                    print(f"Interpretation: {'Normal (<0.50)' if ct_ratio < 0.50 else 'Borderline (0.50-0.55)' if ct_ratio <= 0.55 else 'Enlarged (>0.55)'}")
                    
                    # Update results for CSV
                    if 'heart' in all_results:
                        all_results['heart']['ctr'] = ct_ratio
                        all_results['heart']['mhtd_mm'] = MHTD
                        all_results['heart']['mhcd_mm'] = MHCD * newSpacingX
                    
                    # Clean up temporary files
                    if temp_files_to_delete:
                        print("\n🧹 Cleaning up temporary files...")
                        for temp_file in temp_files_to_delete:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                                print(f"🗑️  Removed: {os.path.basename(temp_file)}")
                    
                except Exception as e:
                    print(f"⚠️  CTR calculation failed: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    # Continue processing even if CTR fails
            
        except Exception as e:
            print(f"✗ Module {module} failed: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            all_results[module] = {'error': str(e)}
            continue
    
    # Summarize results
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    successful_modules = [m for m, r in all_results.items() if 'error' not in r]
    failed_modules = [m for m, r in all_results.items() if 'error' in r]
    
    print(f"Successful modules ({len(successful_modules)}): {', '.join(successful_modules)}")
    if failed_modules:
        print(f"Failed modules ({len(failed_modules)}): {', '.join(failed_modules)}")
    
    # Use results from the first successful module or single module for postprocessing
    results = all_results.get(modules_to_run[0], {}) if modules_to_run else {}
    
    # Postprocessing removed - focusing on segmentation and regression only
    
    # Collect measurements into CSV if requested
    if args.collect_measurements and not args.no_calculations:
        print("\nCollecting measurements into CSV...")
        from core.measurement_collector import create_comprehensive_csv
        
        # Postprocessing removed - focusing on segmentation and regression only
        
        # Extract measurements from all module results
        measurements_data = {}
        for module, module_results in all_results.items():
            if 'error' not in module_results:
                if 'area' in module_results:
                    measurements_data[f'{module}_area_cm2'] = module_results['area']
                # Volume only for modules with regression models
                volume_modules = ['heart', 'lung', 't12l1_regression']
                if 'volume' in module_results and module in volume_modules:
                    measurements_data[f'{module}_volume_ml'] = module_results['volume']
                
                # Handle COVID and vessel results from lung module in batch mode
                if module == 'lung':
                    if 'covid' in module_results and 'error' not in module_results['covid']:
                        covid_data = module_results['covid']
                        if 'area' in covid_data:
                            measurements_data['covid_area_cm2'] = covid_data['area']
                    
                    if 'vessel' in module_results and 'error' not in module_results['vessel']:
                        vessel_data = module_results['vessel']
                        if 'area' in vessel_data:
                            measurements_data['vessel_area_cm2'] = vessel_data['area']
                
                # Extract metadata and postprocessing measurements from module results
                # Debug: print all keys to see what's available
                if module == 'aorta':
                    print(f"DEBUG: Aorta module results keys: {list(module_results.keys())}")
                    
                for key, value in module_results.items():
                    if key in ['pixel_spacing_mm', 'image_height', 'image_width']:
                        measurements_data[key] = value
        
        # Handle module-specific measurements
        
        # Handle special cases with multiple values (continue from previous loop)
        for module, module_results in all_results.items():
            if 'error' not in module_results:
                if module == 't12l1':
                    if 'area_t12' in module_results:
                        measurements_data['t12_area_cm2'] = module_results['area_t12']
                    if 'area_l1' in module_results:
                        measurements_data['l1_area_cm2'] = module_results['area_l1']
                elif module == 'laa':
                    # LAA returns data directly, not nested
                    if 'emphysema_prob' in module_results:
                        measurements_data['laa_emphysema_probability'] = module_results['emphysema_prob']
                    if 'desc' in module_results:
                        measurements_data['laa_emphysema_classification'] = module_results['desc']
                    if 'area' in module_results:
                        measurements_data['laa_emphysema_area_cm2'] = module_results['area']
                elif module == 'tb':
                    # TB classification results
                    if 'probability' in module_results:
                        measurements_data['tb_probability'] = module_results['probability']
                    if 'prediction' in module_results:
                        measurements_data['tb_classification'] = 'Positive' if module_results['prediction'] else 'Negative'
                elif module == 'lung':
                    # Peripheral measurements
                    if 'peripheral_total_area_cm2' in module_results:
                        measurements_data['peripheral_total_area_cm2'] = module_results['peripheral_total_area_cm2']
                    if 'peripheral_central_area_cm2' in module_results:
                        measurements_data['peripheral_central_area_cm2'] = module_results['peripheral_central_area_cm2']
                    if 'peripheral_mid_area_cm2' in module_results:
                        measurements_data['peripheral_mid_area_cm2'] = module_results['peripheral_mid_area_cm2']
                    if 'peripheral_outer_area_cm2' in module_results:
                        measurements_data['peripheral_outer_area_cm2'] = module_results['peripheral_outer_area_cm2']
                elif module == 'heart':
                    # CTR measurements
                    if 'ctr' in module_results:
                        measurements_data['cardiothoracic_ratio'] = module_results['ctr']
                    if 'mhtd_mm' in module_results:
                        measurements_data['mhtd_mm'] = module_results['mhtd_mm']
                    if 'mhcd_mm' in module_results:
                        measurements_data['mhcd_mm'] = module_results['mhcd_mm']
        
        # Postprocessing removed - focusing on segmentation and regression only
        
        # Debug: print all measurements before CSV creation
        print(f"\nDEBUG: Final measurements being saved to CSV:")
        for key, value in sorted(measurements_data.items()):
            print(f"  {key}: {value}")
        
        # Create comprehensive CSV
        dicom_filename = os.path.basename(args.input_file)
        csv_path, _ = create_comprehensive_csv(
            args.output_dir, dicom_filename, measurements_data
        )
        
        print(f"All measurements collected in: {csv_path}")
    elif args.collect_measurements and args.no_calculations:
        print("CSV collection skipped due to --no_calculations flag")
    
    print(f"\nProcessing complete! Generated {len([m for m in all_results.values() if 'error' not in m])} masks"
          f"{' with measurements' if args.collect_measurements and not args.no_calculations else ''}")


if __name__ == '__main__':
    main()