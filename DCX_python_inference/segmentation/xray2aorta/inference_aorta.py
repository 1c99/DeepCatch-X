import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from .base_options import BaseOptions
#from pix2pixHD_model import InferenceModel
#from lungregression_model import UNetRFull
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

if __name__ == '__main__':
    file_path = "./input/JB0006_CXR_0base_201229.dcm"
    image = pydicom.dcmread(file_path, force=True)
    image_array = image.pixel_array
    if image.PhotometricInterpretation == 'MONOCHROME1':
        mean = np.mean(image_array)
        std_dev = np.std(image_array)
        threshold = mean + 2 * std_dev
        replacement_value = mean + 2 * std_dev
        image_array = np.where(image_array > threshold, replacement_value, image_array)
        image_array = image_array * -1 + image_array.max()
    image_array = np.array(image_array)
    
    original_input_shape = image.pixel_array.shape
    pixel_spacing = image.get((0x0028, 0x0030), [0.18, 0.18])
    
    A = Image.fromarray(image_array)
    A = resize_keep_ratio(A, 512)
    img_pad = pad_image(A, target_size=512, pad_value=0)
    A_ = np.array(img_pad(A))

    eps = 1e-10
    mean, std = A_.mean(), A_.std()
    A_neg2std = np.where(A_ < mean - (2*std), mean - (2*std), A_)
    percentile0, percentile99 = np.percentile(A_neg2std, 0), np.percentile(A_neg2std, 99)
    A_neg2std = np.where(A_neg2std > mean + (2*std), mean + (2*std), A_neg2std)
    normalized_a = (A_neg2std - percentile0) / ((percentile99 - percentile0) + eps)

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

    opt.get_lung_area = False
    opt.hn = False
    opt.output_min_max = "-1100,-700"
    opt.checkpoint_path = "./checkpoints/xray2aorta.pth"
    opt.threshold = -1015
    opt.save_input = False
    opt.profnorm = True
    opt.check_xray = False
    opt.age = 50
    opt.sex = "F"
    opt.pixel_spacing = None
    opt.use_gpu = 0
    device = 'cuda:' + str(0)
    opt.cuda0 = device
    opt.cuda1 = device
    opt.loadSize = 512
    opt.output_nc = 2
    opt.netG = 'local_spade'

    model = create_model(opt)
    model.eval()
    model.float()

    with torch.no_grad():
        output = model(normalized_a)
        output = output.cpu().numpy()
        b_min_val, b_max_val = -1100, -700
        pixel_size_resize_w = 0.18517382812499997
        denormalize_gen = output * (b_max_val - b_min_val) + b_min_val
        threshold = -1015.0
        replace_value = -1024
        denormalize_gen = np.where(denormalize_gen[0] < threshold, np.full_like(denormalize_gen, replace_value), denormalize_gen)
        # calculate area
        denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)
        original_width = original_input_shape[0]
        original_height = original_input_shape[1]
        ratio = float(512) / max(original_width, original_height)
        pixel_size_resize_w = pixel_spacing[0] / ratio
        pixel_size_resize_h = pixel_spacing[1] / ratio
        area = np.sum(denormalize_gen_mask.flatten())
        print("area sum", area)
        area = area * pixel_size_resize_w * pixel_size_resize_h / 100
        print("area", area)
        nii_np = np.transpose(denormalize_gen, axes=[3, 2, 1, 0])
        nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
        nii.header['pixdim'] = pixel_size_resize_w
        file_name = os.path.basename(file_path).split('.')[0]
        output_path = os.path.join("./output", f"{file_name}.nii")
        nib.save(nii, output_path)
        print('output saved.')

