#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from data_loader import CustomDatasetDataLoader
import torch
from torch.autograd import Variable
torch.cuda.current_device()
torch.cuda._initialized = True
import numpy as np
import nibabel as nib
from skimage.measure import label
from skimage.io import imsave
import pydicom
from PIL import Image
from math import ceil, floor
import torchvision.transforms as transforms
from model import EfficientNet, resnet18
from skimage.io import imsave
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

from base_options import BaseOptions
from test import create_model_v2 as create_model

def resize_keep_ratio(img, target_size):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    try:
        # im = img.resize(new_size, Image.LANCZOS) # this one changes the min max value of the image
        im = img.resize(new_size, Image.NEAREST)
    except:
        im = img.resize(new_size, Image.NEAREST)
    return im


def pad_image(img, target_size, pad_value=-1024):
    old_size = img.size
    pad_size_w = (target_size - old_size[0]) / 2
    pad_size_h = (target_size - old_size[1]) / 2

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

    print("pad values: ", wl, ht, wr, hb)

    return transforms.Compose(
        [
            transforms.Pad((wl, ht, wr, hb), fill=pad_value),
            # transforms.ToTensor(),
        ]
    )

def get_biggest_connected_region(gen_lung, n_region=2):
    """ return n_biggest connected region -> similar to region growing in Medip """
    labels = label(gen_lung)  # label each connected region with index from 0 - n of connected region found
    n_connected_region = np.bincount(labels.flat)  # number of pixel for each connected region
    if n_connected_region[0] != np.max(n_connected_region):  # if number of background's pixel is not the biggest
        n_connected_region[0] = np.max(n_connected_region) + 1  # make it the biggest
    biggest_regions_index = (-n_connected_region).argsort()[1:n_region + 1]  # get n biggest regions index without BG

    biggest_regions = np.array([])
    for ind in biggest_regions_index:
        if biggest_regions.size == 0:
            biggest_regions = labels == ind
        else:
            biggest_regions += labels == ind
    return biggest_regions

if __name__ == "__main__":
    image_path = "./input/AN_ID_20210526104509_1.dcm" # change input filename
    file_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join("./output", f"{file_name}.nii")
    image = pydicom.dcmread(image_path, force=True)
    try:
        image_array = image.pixel_array
    except:
        image.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        image_array = image.pixel_array
    image_array = np.array(image_array, dtype=np.int32)
    if image.PhotometricInterpretation == 'MONOCHROME1':
        image_array = image_array * -1 + image_array.max()

    rows = image.Rows
    cols = image.Columns

    original_input_shape = image.pixel_array.shape
    pixel_spacing = image.ImagerPixelSpacing if "PixelSpacing" in image else [0.18, 0.18]

    print(pixel_spacing)
    
    A = Image.fromarray(image_array)
    A = resize_keep_ratio(A, 2048)
    img_pad = pad_image(A, target_size=2048, pad_value=0)
    A_ = np.array(img_pad(A))
    print("A_ shape", A_.shape)
    
    # prof normalization
    eps = 1e-10
    mean, std = A_.mean(), A_.std()
    print("mean, std", mean, std)
    A_neg2std = np.where(A_ < mean - (2 * std), mean - (2 * std), A_)
    percentile0, percentile99 = np.percentile(A_neg2std, 0), np.percentile(A_neg2std, 99)
    normalized_a = (A_ - percentile0) / ((percentile99 - percentile0) + eps)
    # normalized image min max -0.39650070649878066 1.306639044019536
    print("normalized image min max", normalized_a.min(), normalized_a.max())
    
    to_tensor = transforms.ToTensor()
    normalized_a = normalized_a.astype(np.float32)
    normalized_a = to_tensor(normalized_a)

    inst_tensor = torch.tensor(0).cpu()
    feat_tensor = torch.tensor(0).cpu()
    normalized_a = normalized_a.unsqueeze(0).cpu()

    normalized_a = normalized_a.type(torch.cuda.FloatTensor)

    print("normalized_a shape", normalized_a.shape)
    
    opt = BaseOptions().parse(save=False)
    opt.get_lung_area = True
    opt.hn = False
    opt.output_min_max = "-1100,-500"
    opt.checkpoint_path = "./checkpoints/xray2lung.pth"
    opt.threshold = -1015
    opt.save_input = True
    opt.profnorm = True
    opt.check_xray = False
    opt.age = 76
    opt.sex = "M"
    opt.pixel_spacing = None
    opt.use_gpu = 0
    device = 'cuda:' + str(0)
    opt.cuda0 = device
    opt.cuda1 = device
    
    model = create_model(opt)
    model.eval()
    model.float()
    
    with torch.no_grad():
        output = model(normalized_a)
        output = output.cpu().numpy()
        b_min_val, b_max_val = -1100, -500
        denormalize_gen = output * (b_max_val - b_min_val) + b_min_val
        threshold = -1015.0
        replace_value = -1024
        denormalize_gen = np.where(denormalize_gen[0] < threshold, np.full_like(denormalize_gen, replace_value), denormalize_gen)
        denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)
        if opt.get_lung_area:  # find connected region
            denormalize_gen_mask = get_biggest_connected_region(denormalize_gen_mask)
            connected_lung = np.where(denormalize_gen_mask, denormalize_gen[0, 0], -1024)
            denormalize_gen = connected_lung[np.newaxis, np.newaxis]
        # calculate area
        original_width = original_input_shape[0]
        original_height = original_input_shape[1]
        ratio = float(2048) / max(original_width, original_height)

        pixel_size_resize_w = pixel_spacing[0] / ratio
        pixel_size_resize_h = pixel_spacing[1] / ratio

        area = np.sum(denormalize_gen_mask.flatten())
        area = area * pixel_size_resize_w * pixel_size_resize_h / 100
        print("area", area)
        nii_np = np.transpose(denormalize_gen, axes=[3, 2, 1, 0])
        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
        nii.header['pixdim'] = pixel_size_resize_w
        nib.save(nii, output_path) #change output filename
        print('output saved.')

