import os
import glob
import pydicom
import torch
from eff_unet_shuffle import EfficientUNet
from torch.nn import functional as F
import numpy as np

def adjust_size(image, N):

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
        image = F.pad(image, padding, mode='constant', value=0)

    return image, padding

def remove_padding(image, padding):
    pad_left, pad_right, pad_top, pad_bottom = padding
    # no padding return original
    if pad_right == 0 and pad_bottom == 0:
        return image

    _, _, H_padded, W_padded = image.shape

    H_original = H_padded - pad_bottom
    W_original = W_padded - pad_right
    return image[:, :, :H_original, :W_original]


def ddim_sample(model, condition, device='cuda'):
    # Time Step
    num_timesteps = 1000
    sample_step = 30
    new_timesteps = torch.linspace(num_timesteps - 1, 0, steps=sample_step, device=device).long()

    # Diffusion Params
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

weight_dir = "./bone_supp_diff.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(weight_dir, map_location=device)

model = EfficientUNet(in_channels=2, out_channels=1, init_features=64).to(device)
model.load_state_dict(checkpoint)
model.eval()

root_path = "C:\\Users\\medicalip\\Downloads\\20241022 MIP 전달 ChestPA dcm 모음"#./sample"
out_path = "./results"

dcm_list = glob.glob1(root_path, "*.dcm")

for num, dcm_name in enumerate(dcm_list, 1):
    print(f"{num}/{len(dcm_list)} File Name: {dcm_name}")
    dcm_path = os.path.join(root_path, dcm_name)
    if not os.path.exists(dcm_path):
        print(f"File does not exist or is not readable: {dcm_path}")
        continue
    ds = pydicom.read_file(dcm_path)
    dcm_arr = np.array(ds.pixel_array, dtype='float32')
    # Clipping outliers
    Q3 = np.percentile(dcm_arr, 75)
    Q1 = np.percentile(dcm_arr, 25)
    IQR = Q3-Q1
    upper_boundary = Q3 + 1.5 *IQR
    dcm_clip = np.clip(dcm_arr, 0, upper_boundary)
    outlier_residual = dcm_arr - dcm_clip
    # Normalization
    max_val = np.max(dcm_clip)
    dcm_max_norm = dcm_clip / max_val
    mean_val = np.mean(dcm_max_norm)
    std_val = np.std(dcm_max_norm)
    dcm_norm = (dcm_max_norm - mean_val) / std_val
    # to tensor
    dcm_torch = torch.from_numpy(dcm_norm).unsqueeze(0).unsqueeze(0)
    dcm_torch = dcm_torch.to(device).float()
    # check downsampling level
    b, c, h, w = dcm_torch.shape
    max_size = np.maximum(h, w)
    ds_factor = int(np.round(max_size / 512))
    # padding and interpolation
    dcm_torch, padding = adjust_size(dcm_torch, ds_factor)
    dcm_torch = F.interpolate(dcm_torch, scale_factor=1/ds_factor, mode='bilinear')
    # Diffusion Process
    x = ddim_sample(model=model,
                    condition=dcm_torch,
                    device=device)
    # interpolation and remove padding
    output = F.interpolate(x, scale_factor=ds_factor, mode='bilinear')
    output = remove_padding(output, padding)
    # to numpy
    output = output.squeeze(0).squeeze(0).detach().cpu().numpy()
    # denormalization
    output_denorm = (output * std_val + mean_val) * max_val
    output_denorm += outlier_residual
    output_denorm = np.clip(output_denorm, 0, upper_boundary)
    # Save
    output_denorm = output_denorm.astype(str(ds.pixel_array.dtype))
    ds.PixelData = output_denorm.tobytes()

    os.makedirs(out_path, exist_ok=True)
    ds.save_as(os.path.join(out_path, dcm_name))