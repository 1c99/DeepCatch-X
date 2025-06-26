"""
initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021
===modified: 03-September-2021
===modified by Han-Jae Chung: 05-November-2021
"""
from torch.autograd import Variable

import logging
import pathlib
from pathlib import Path
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import gc
import os
import datetime
from math import ceil, floor
import nibabel as nib
import numpy as np
import torch
import cv2
import pydicom
import sys
from sklearn.linear_model import LinearRegression

from src.build_model import build_model_G
from src.args import Args
from src.create_data import create_data_loaders

def resize_back(nii_np, file_path, height, width):
    
    #print("From resize_back ", height, width)
    target_size = max(nii_np.shape)
    
    original_height = height
    original_width = width

    ratio = float(target_size) / max([original_width, original_height])
    new_size = tuple([int(x * ratio) for x in [original_height, original_width]])

    #print(new_size)
    pad_size_w = (target_size - new_size[0]) / 2
    pad_size_h = (target_size - new_size[1]) / 2
    #print(pad_size_h, pad_size_w)
    wl, wr = (ceil(pad_size_w), floor(pad_size_w)) if pad_size_w % 2 else (int(pad_size_w), int(pad_size_w))
    ht, hb = (ceil(pad_size_h), floor(pad_size_h)) if pad_size_h % 2 else (int(pad_size_h), int(pad_size_h))

    eps = 1e-10
    nii_np_min, nii_np_max = nii_np.min(), nii_np.max()
    
    nii_np = ((nii_np - nii_np_min) / ((nii_np_max - nii_np_min) + eps))

    if file_path.endswith('dcm') or file_path.endswith('.dicom'):
        img_masked = nii_np[:, :, 0:1, :][wl:target_size - wr, ht:target_size - hb, :]  # dcm
        resize_dims = (original_width, original_height)
    elif file_path.endswith('.nii') or file_path.endswith('.gz'):
        img_masked = nii_np[:, :, 0:1, :][ht:target_size - hb, wl:target_size - wr, :]  # nii
        resize_dims = [original_height, original_width]
    else:
        img_masked = nii_np[:, :, 0:1, :][wl:target_size - wr, ht:target_size - hb, :]
        resize_dims = (original_width, original_height)

    img_masked = img_masked[:, :, 0, 0]

    try:
        img_resized = cv2.resize(img_masked, resize_dims, interpolation=cv2.INTER_CUBIC)
    except:
        img_resized = cv2.resize(img_masked, resize_dims, interpolation=cv2.INTER_NEAREST)

    nii_np = img_resized * (nii_np_max - nii_np_min) + nii_np_min
    nii_np = np.clip(nii_np, 0, np.max(nii_np))
    # nii_np = img_resized * 4095
    # min_val, max_val = nii_np.min(), nii_np.max()
    # processed_data_normalized = (nii_np - min_val) / (max_val - min_val) * 4095
    # processed_data_uint16 = processed_data_normalized.astype(np.uint16)
    
    # processed_pixel_value = processed_data_uint16[tuple([pixel_coords_x, pixel_coords_y])]

    # #processed_pixel_value = processed_data_uint16[pixel_coords_y, pixel_coords_x]
    # pixel_value_diff = pixel_value_A - processed_pixel_value
    
    # print(f'{os.path.basename(file_path)} | {pixel_value_diff} = {pixel_value_A} - {processed_pixel_value}')
    
    # adjusted_data = (processed_data_uint16 + pixel_value_diff).clip(0, 4095).astype(np.uint16)
    # adjusted_data_min, adjusted_data_max = adjusted_data.min(), adjusted_data.max()
    # adjusted_data_normalized = (adjusted_data - adjusted_data_min) / (adjusted_data_max - adjusted_data_min) * 4095
    # adjusted_data_normalized = np.clip(adjusted_data_normalized, 0, 4095).astype(np.uint16)
    
    # return processed_data_uint16
    return nii_np

def match_pixel_range(processed_img, ori_file_path, method='percentile', lower_percentile=1, upper_percentile=99):
    
    if ori_file_path.endswith('.dcm') or ori_file_path.endswith('.dicom'):
        dicom = pydicom.dcmread(ori_file_path)
        original_image = dicom.pixel_array
        if hasattr(dicom, 'PhotometricInterpretation') and dicom.PhotometricInterpretation == 'MONOCHROME1':
            original_image = original_image * -1 + original_image.max()
    elif ori_file_path.endswith('.nii') or ori_file_path.endswith('.nii.gz'):
        original_image = nib.load(ori_file_path).get_fdata().T
    else:
        raise ValueError("Unsupported file type. Please provide a DICOM or NIfTI file.")
    
    if original_image.shape != processed_img.shape:
        raise ValueError("The dimensions of the DICOM and NIfTI images do not match.")
    
    if method == 'percentile':
        minA, maxA = np.percentile(original_image, lower_percentile), np.percentile(original_image, upper_percentile)
        minB, maxB = np.percentile(processed_img, lower_percentile), np.percentile(processed_img, upper_percentile)
        B_normalized = (processed_img - minB) / (maxB - minB)
        B_normalized = np.clip(B_normalized, 0, 1)
        adjusted_imageB = B_normalized * (maxA - minA) + minA
    elif method == 'zscore':
        mean = np.mean(processed_img)
        std = np.std(processed_img)
        B_normalized = (processed_img - mean ) / std
        adjusted_imageB = B_normalized * np.std(original_image) + np.mean(original_image)
    else:
        raise ValueError("Invalid normalization method specified.")
    
    adjusted_image = adjusted_imageB 
    
    
    # ori_arr = np.array(original_image, dtype=original_image.dtype)
    # min_val, max_val = ori_arr.min(), ori_arr.max()

    # # ori_arr_normalized = (ori_arr - min_val) / (max_val - min_val) * 4095
    # original_pixel_data = original_image
    # processed_pixel_data = adjusted_image
    # # original_pixel_data = ori_arr_normalized
    # # processed_pixel_data = img_arr * 4095

    # original_pixels_flat = original_pixel_data.flatten()
    # processed_pixels_flat = processed_pixel_data.flatten()

    # model = LinearRegression()
    # model.fit(processed_pixels_flat.reshape(-1, 1), original_pixels_flat)

    # slope = model.coef_[0]
    # intercept = model.intercept_

    # #print(f'{os.path.basename(file_path)} | Slope: {slope}, Intercept: {intercept}')
    # #print(original_pixel_data.dtype, processed_pixel_data.dtype)

    # adjusted_pixels = slope * processed_pixel_data + intercept
    # adjusted_pixels = np.clip(adjusted_pixels, 0, 65535).astype(np.uint16)
    
    return adjusted_image

def load_model(checkpoint_file):
    checkpoint_file = pathlib.Path(checkpoint_file)
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file)
        args = checkpoint['args']
        model_G = build_model_G(args)

        if args.data_parallel:
            model_G = torch.nn.DataParallel(model_G)

        model_G.load_state_dict(checkpoint['model_G'])
        
        del checkpoint

        return model_G
    else:
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} not found")

def extract_dcm_net(args, model_G, display_loader):

    model_G.eval()
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    Tensor = args.Tensor
    num_dataset = len(display_loader.dataset)
    with torch.no_grad():
        for iter2, (img_A_torch, rescale_slope, rescale_intercept, file_path_A, file_name_A, height_A, width_A, max_val_A, min_val_A) in enumerate(display_loader):
            
            progress = (iter2 + 1) / num_dataset * 100
            sys.stdout.write(f"\r{iter2 + 1}/{num_dataset} | {progress:.2f}%")
            sys.stdout.flush()

            print(img_A_torch.shape)
            real_A = Variable(img_A_torch.type(Tensor)) # Source
            print("input shape : ", real_A.shape)
            
            fake_A = model_G(real_A)
            fake_A = model_G(fake_A)
            fake_A = model_G(fake_A)

            fake_A = fake_A.squeeze(1).squeeze(0).detach().cpu().numpy()

            print("rescale_slope : ", rescale_slope.dtype)
            fake_A_denorm = fake_A * (rescale_slope.cpu().numpy()) + rescale_intercept.cpu().numpy()
            #fake_A_denorm = fake_A * (-500 + 1000) - 1000
            #fake_A_denorm = np.where(fake_A_denorm < -1015, -3000, fake_A_denorm)
            fake_A_denorm = np.clip(fake_A_denorm, min_val_A.item(), max_val_A.item())
            if file_path_A[0].endswith('.gz') or file_path_A[0].endswith('.nii'):
                print("gz")
                ds_A_fake = nib.load(file_path_A[0])
                affine = ds_A_fake.header.get_base_affine()
                fake_A = np.expand_dims(fake_A_denorm, axis=(-1, -2)) 
                fake_A_resized = resize_back(fake_A, file_name_A[0], width_A.item(), height_A.item())
                
            elif file_path_A[0].endswith('.dcm') or file_path_A[0].endswith('.dicom'):
                print("dcm")
                affine = np.eye(4)
                fake_A = np.expand_dims(fake_A_denorm, axis=(-1, -2)) 
                fake_A_resized = resize_back(fake_A, file_name_A[0], height_A.item(), width_A.item())        
            else:
                raise ValueError(f"Unsupported file format: {file_path_A[0]}")
            
            # img_arr_normalized = fake_A_resized
            # eps = 1e-10
            # mean, std = img_arr_normalized.mean(), img_arr_normalized.std()
            # img_arr_normalized = np.where(img_arr_normalized < mean - (2 * std), mean - (2 * std), img_arr_normalized)
            # img_arr_normalized = np.clip(img_arr_normalized, np.percentile(img_arr_normalized, 0), np.percentile(img_arr_normalized, 99))
            # min_val, max_val = np.min(img_arr_normalized), np.max(img_arr_normalized)
            # img_arr_normalized = ((img_arr_normalized - min_val) / ((max_val - min_val) + eps))
            
            #adjusted_pixels = match_pixel_range(fake_A_resized, file_path_A[0])
            # min_v, max_v = fake_A_resized.min(), fake_A_resized.max()
            # print(min_v, max_v)
            # fake_A_resized = (fake_A_resized - min_v) / (max_v - min_v) * max_val_A.item() + min_val_A.item()
            
            
            new_nii_ds = nib.Nifti1Image(fake_A_resized.T, affine=affine)
            result_folder_path = os.path.join(args.val_data_save_path, current_time)
            nii_file_name = pathlib.Path(file_name_A[0]).with_suffix('.nii.gz').name
            result_nii_path = os.path.join(result_folder_path, nii_file_name)

            os.makedirs(result_folder_path, exist_ok=True)
            nib.save(new_nii_ds, result_nii_path)
            
def main(args):
    gc.collect()
    torch.cuda.empty_cache()

    model_G = load_model(args.checkpoint)

    # logging.info(args)
    # logging.info(model_G)

    display_loader = create_data_loaders(args)
    
    extract_dcm_net(args, model_G, display_loader)
    
if __name__ == '__main__':
    
    args = Args().parse(save=False)
    args.device = 'cuda'
    args.Tensor = torch.cuda.FloatTensor
    args.data_parallel
    args.resume
    args.val_A_data_path = './input'
    args.val_data_save_path = './output'
    args.checkpoint = './checkpoints/best_model_2024_05_24_13_21_59.pt'  
    args.batch_size = 1 

    print(f'GRU RUNNING: {True if torch.cuda.is_available() else False}')
    print(f'DEVICE NAME: {torch.cuda.get_device_name(0)} | DEVICE_COUNT: {torch.cuda.device_count()}')

    # logging.basicConfig(level=logging.INFO, filename='log_unet_v0_val.txt')
    # logger = logging.getLogger(__name__)

    main(args)
    print("\nPROCESS DONE")