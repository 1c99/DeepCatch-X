import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
from base_options import BaseOptions
from test import create_model_v2 as create_model

def resize_keep_ratio( img, target_size):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    try:
        im = img.resize(new_size, Image.LANCZOS)
    except:
        im = img.resize(new_size, Image.NEAREST)
    return im

def pad_image( img, target_size, pad_value=-1024):
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

    return transforms.Compose(
        [
            transforms.Pad((wl, ht, wr, hb), fill=pad_value),
            # transforms.ToTensor(),
        ])


if __name__ == "__main__":
    image_path = "./input/AN_ID_20210526112758_1.dcm"
    suffix = "_lung_c"
    new_extension=".nii"
    image = pydicom.dcmread(image_path, force=True)
    rows = image.Rows
    cols = image.Columns
    image_array = image.pixel_array
    image_array = np.array(image_array)

    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    file_name, ext = os.path.splitext(base_name)
    lung_image_name = f"{file_name}{suffix}{new_extension}"
    lung_image_path = os.path.join(dir_name, lung_image_name)
    image_path = lung_image_path
    
    nii_image = nib.load(image_path)
    image_array = nii_image.get_fdata()
    if image_array.shape[-2:] == (1, 1):
        image_array = image_array[..., 0, 0]
    image_array = np.rot90(image_array)
    image_array = np.rot90(image_array)
    image_array = np.rot90(image_array)
    image_array = np.fliplr(image_array)

    original_input_shape = image.pixel_array.shape
    pixel_spacing = image.PixelSpacing if "PixelSpacing" in image else [0.18, 0.18]

    A = Image.fromarray(image_array)
    A = resize_keep_ratio(A, 2048)
    img_pad = pad_image(A, target_size=2048, pad_value=0)
    A_ = np.array(img_pad(A))

    eps = 1e-10
    mean, std = A_.mean(), A_.std()
    print("mean, std", mean, std)
    a_min_val = -1100
    a_max_val = -500 
    normalized_a = (A_ - a_min_val) / ((a_max_val - a_min_val) + eps)

    to_tensor = transforms.ToTensor()
    normalized_a = normalized_a.astype(np.float32)
    normalized_a = to_tensor(normalized_a)
    
    inst_tensor = torch.tensor(0).cpu()
    feat_tensor = torch.tensor(0).cpu()
    normalized_a = normalized_a.unsqueeze(0).cpu()
    
    normalized_a = normalized_a.type(torch.cuda.FloatTensor)

    opt = BaseOptions().parse(save=False)
    opt.hn = False
    opt.input_min_max = "-1100,-500"
    opt.output_min_max = "-1100,-400"
    opt.checkpoint_path = "./checkpoints/lung2covid.pth"
    opt.threshold = -950
    opt.save_input = False
    opt.profnorm = False
    opt.check_xray = False
    opt.age = 50
    opt.sex = "F"
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
        b_min_val, b_max_val = opt.output_min_max.split(',')
        b_min_val, b_max_val = int(b_min_val), int(b_max_val)
        denormalize_gen = output * (b_max_val - b_min_val) + b_min_val
        threshold = -950.0 ##
        replace_value = -1024 ##
        denormalize_gen = np.where(denormalize_gen[0] < threshold, np.full_like(denormalize_gen, replace_value), denormalize_gen)
        denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)
        if opt.get_covid_area:
            if np.sum(denormalize_gen_mask.flatten()) == 2048*2048:
                denormalize_gen_mask = np.zeros_like(denormalize_gen_mask)
        # calculate area
        original_width = original_input_shape[0]
        original_height = original_input_shape[1]
        ratio = float(2048) / max(original_width, original_height)
        pixel_size_resize_w = pixel_spacing[0] / ratio
        pixel_size_resize_h = pixel_spacing[0] / ratio

        area = np.sum(denormalize_gen_mask.flatten())
        print("area sum", area)
        area = area * pixel_size_resize_w * pixel_size_resize_h / 100
        print("area", area)
        nii_np = np.transpose(denormalize_gen, axes=[3, 2, 1, 0])
        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
        nii.header['pixdim'] = pixel_spacing[0]
        output_path = f"output/{file_name}.nii"
        nib.save(nii, output_path)
        print('output saved.')