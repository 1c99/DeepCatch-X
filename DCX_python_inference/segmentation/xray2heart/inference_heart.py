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
    image_path = "./input/AN_ID_20210526104509_1.dcm"
    file_name = image_path.split("/")[-1].rsplit(".", 1)[0]
    output_path = f"output/{file_name}.nii"
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
    rows = image.Rows
    cols = image.Columns

    original_input_shape = image.pixel_array.shape
    pixel_spacing = image.PixelSpacing if "PixelSpacing" in image else [0.18, 0.18]
    print(pixel_spacing)
    sex = image.PatientSex if "PatientSex" in image else "F"
    age = image.PatientAge if "PatientAge" in image else 50
    if (sex == ""):
        sex = "F"
    if (age == ""):
        age = 50

    print("rows", rows)
    print("cols", cols)
    print("shape", original_input_shape)
    print("pixel_spaicng", pixel_spacing)
    print("sex", sex)
    print("age", age)

    A = Image.fromarray(image_array)
    A = resize_keep_ratio(A, 512)
    img_pad = pad_image(A, target_size=512, pad_value=0)
    A_ = np.array(img_pad(A))

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
    
    inst_tensor = torch.tensor(0).cpu()
    feat_tensor = torch.tensor(0).cpu()
    normalized_a = normalized_a.unsqueeze(0).cpu()
    
    normalized_a = normalized_a.type(torch.cuda.FloatTensor)

    from base_options import BaseOptions
    from test import create_model_v2 as create_model
    opt = BaseOptions().parse(save=False)

    opt.get_lung_area = True
    opt.hn = False
    opt.output_min_max = "-1100,-500"
    opt.checkpoint_path = "./checkpoints/xray2heart.pth"
    opt.threshold = -1015
    opt.save_input = True
    opt.profnorm = True
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
        b_min_val, b_max_val = -1100, -500
        pixel_size_resize_w = 0.18517382812499997
        denormalize_gen = output * (b_max_val - b_min_val) + b_min_val
        threshold = -1015.0
        replace_value = -1024
        denormalize_gen = np.where(denormalize_gen[0] < threshold, np.full_like(denormalize_gen, replace_value), denormalize_gen)
        denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)
        # calculate area
        original_width = rows
        original_height = cols
        ratio = float(512) / max(original_width, original_height)
        pixel_size_resize_w = pixel_spacing[0] / ratio
        pixel_size_resize_h = pixel_spacing[1] / ratio
        area = np.sum(denormalize_gen_mask.flatten())
        area = area * pixel_size_resize_w * pixel_size_resize_h / 100
        print ("area : ", area, "cm2")
        nii_np = np.transpose(denormalize_gen, axes=[3, 2, 1, 0])
        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
        nii.header['pixdim'] = pixel_size_resize_w
        nib.save(nii, output_path)
        print('output saved.')