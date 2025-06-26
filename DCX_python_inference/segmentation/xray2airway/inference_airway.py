import datetime
import gc
import os
import nibabel as nib
from skimage.measure import label
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pathlib
from collections import OrderedDict
from test import create_model_v2 as create_model
from base_options import BaseOptions
import pydicom
import numpy as np
from PIL import Image
from math import ceil, floor
from torchvision import transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Important : C++ needs the following version of pytorch:
"""
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
"""
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

checkpoint_file = 'checkpoints\\xray2airwaynan.pth'


def normalize(image):
    """
    Min-Max Normalization
    :param image: 3D image
    """
    img_3d = image
    mean_val = torch.mean(img_3d)
    std_val = torch.std(img_3d)

    rescale_slope = std_val
    rescale_intercept = mean_val

    return (img_3d - rescale_intercept) / rescale_slope


def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def build_model(checkpoint):

    input_filename = "./input/JB0006_CXR_0base_201229.dcm"
    opt = BaseOptions().parse(save=False)
    opt.test_input = input_filename
    opt.test_output = "./output/JB0006_CXR_0base_201229_temp.dcm"
    opt.get_lung_area = True
    opt.output_min_max = "-1100,-500"
    opt.checkpoint_path = checkpoint
    opt.threshold = -1015
    opt.save_input = True
    opt.profnorm = True
    opt.check_xray = False
    opt.age = None
    opt.sex = None
    opt.pixel_spacing = None
    opt.hn = False
    opt.loadSize = 512
    opt.output_nc = 1
    opt.netG = 'local'
    opt.use_gpu = 0
    device ='cuda:' + str(0)
    opt.cuda0 = device
    opt.cuda1 = device

    model = create_model(opt)
    return model

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

def histogram_normalization( arr):
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
        # cdf_normalized = cdf * hist.max()/ cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        arr_histnorm = cdf[a_norm]

        arr_denorm = (arr_histnorm / 255) * (arr.max() - arr.min()) + arr.min()

        # print(cdf)
        # print(a_norm[0:5])
        # print(arr_histnorm[0:5])

        return arr_denorm[:, :, 0]
    except:
        return arr

if __name__ == "__main__":

    image_path = "./input/AN_ID_20210526112758_1.dcm"
    image = pydicom.dcmread(image_path, force=True)
    try:
        image_array = image.pixel_array
    except:
        image.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        image_array = image.pixel_array
    image_array = np.array(image_array, dtype=np.int32)

    if image.PhotometricInterpretation == 'MONOCHROME1':
        mean = np.mean(image_array)
        std_dev = np.std(image_array)
        threshold = mean + 2 * std_dev
        replacement_value = mean + 2 * std_dev
        image_array = np.where(image_array > threshold, replacement_value, image_array)
        image_array = image_array * -1 + image_array.max()

    sex = 'F'
    age = 50
    original_input_shape = image.pixel_array.shape
    print(original_input_shape)
    pixel_spacing = image.PixelSpacing

    A = Image.fromarray(image_array)
    A = resize_keep_ratio(A, 512)
    img_pad = pad_image(A, target_size=512, pad_value=0)
    A_ = np.array(img_pad(A))

    print("A_ shape", A_.shape)

    # prof normalization
    eps = 1e-10
    mean, std = A_.mean(), A_.std()
    print("mean, std", mean, std)
    A_neg2std = np.where(A_ < mean - (2 * std), mean - (2 * std), A_)
    percentile0, percentile99 = np.percentile(A_neg2std, 0), np.percentile(A_neg2std, 99)
    normalized_a = (A_ - percentile0) / ((percentile99 - percentile0) + eps)
    print("normalized image min max", normalized_a.min(), normalized_a.max())

    to_tensor = transforms.ToTensor()
    normalized_a = normalized_a.astype(np.float32)
    normalized_a = to_tensor(normalized_a)
    
    normalized_a = normalized_a.unsqueeze(0).cpu()

    print("normalized_a shape", normalized_a.shape)


    # load torchscript model
    model = build_model(checkpoint_file)
    model = model.cuda()
    model.eval()
    model.float()

    print('loading model successful.')

    # inference
    with torch.no_grad():
        output = model(normalized_a)
        output = output.cpu().numpy()
        b_min_val, b_max_val = -1100, -500
        denormalize_gen = output * (b_max_val - b_min_val) + b_min_val
        threshold = -1015.0 
        replace_value = -1024 
        denormalize_gen_th = np.where(denormalize_gen[0] < threshold, np.full_like(denormalize_gen, replace_value), denormalize_gen)
        denormalize_gen_mask = np.where(denormalize_gen_th[0, 0] < -1015, 0, 1)
        denormalize_gen_mask = get_biggest_connected_region(denormalize_gen_mask, 1)
        connected_lung = np.where(denormalize_gen_mask, denormalize_gen_th[0, 0], -1024)
        denormalize_gen_th = connected_lung[np.newaxis, np.newaxis]
        # calculate area
        original_width = original_input_shape[0]
        original_height = original_input_shape[1]
        ratio = float(512) / max(original_width, original_height)

        pixel_size_resize_w = pixel_spacing[0] / ratio
        pixel_size_resize_h = pixel_spacing[1] / ratio
        print(pixel_size_resize_w)
        print(pixel_size_resize_h)
        
        area = np.sum(denormalize_gen_mask.flatten())
        print('area sum', area)
        area = area * pixel_size_resize_w * pixel_size_resize_h / 100
        print('area', area)
        nii_np = np.transpose(denormalize_gen_th, axes=[3, 2, 1, 0])
        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
        nii.header['pixdim'] = pixel_size_resize_w

        file_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join("./output", f"{file_name}.nii")
        
        nib.save(nii, output_path)
        print('output saved.')