import datetime
import gc
import os
import nibabel as nib
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

checkpoint_file = 'checkpoints\\lung2covid.pth'
scripted_model_path = 'checkpoints\\lung2covid_111.mipx'


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
    opt.get_lung_area = False
    opt.input_min_max = "-1100,-500"
    opt.output_min_max = "-1100,-400"
    opt.checkpoint_path = checkpoint
    opt.threshold = -950
    opt.save_input = False
    opt.profnorm = False
    opt.check_xray = False
    opt.age = None
    opt.sex = None
    opt.pixel_spacing = None
    opt.hn = False
    opt.use_gpu = 0
    device ='cuda:' + str(0)
    opt.cuda0 = device
    opt.cuda1 = device

    model = create_model(opt)
    return model


def scriptify_tiseptb2():
    """
    modelInputImage shape:  torch.Size([1, 3, 1024, 1024])
    device:  cuda:0
    pytorch_model_path:  ./checkpoints/tiseptb2.pth

    """

    modelInputImage = torch.rand(1, 3, 1024, 1024).type(torch.cuda.FloatTensor)
    device = torch.device('cuda:' + str(0))
    pytorch_model_path = './checkpoints/tiseptb2.pth'
    model = torch.load(pytorch_model_path)
    model = model.to(device)
    modelInputImage = modelInputImage.to(device)
    model.eval()
    model.float()
    print('modelInputImage shape: ', modelInputImage.shape)
    print('device: ', device)
    print('pytorch_model_path: ', pytorch_model_path)

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, modelInputImage, check_trace=True)
        traced_script_module.save('./checkpoints/tiseptb2_cuda.mipx')
        print('done')

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

    model = build_model(checkpoint_file)
    dumy_data = torch.rand(1, 1, 2048, 2048).type(torch.cuda.FloatTensor)
    #dumy_inst = torch.rand(1).type(torch.cuda.FloatTensor)
    #dumy_img = torch.rand(1).type(torch.cuda.FloatTensor)
    """
    model = model.cpu()
    dumy_data = dumy_data.cpu()
    dumy_inst = dumy_inst.cpu()
    dumy_img = dumy_img.cpu()
    """
    model = model.cuda()
    dumy_data = dumy_data.cuda()
    #dumy_inst = dumy_inst.cuda()
    #dumy_img = dumy_img.cuda()
    
    """
    label shape:  torch.Size([1, 1, 2048, 2048])
    inst shape:  torch.Size([1])
    image shape:  torch.Size([1])
    """
    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, (dumy_data), check_trace=True)
        traced_script_module.save(scripted_model_path)
        print('done')


    # ----------------------- Scriptify tiseptb2 -----------------------
    # scriptify_tiseptb2()


    # ---------------------- TEST ----------------------
    # Prepare input image
    """
    image_path = "./input/JB0006_CXR_0base_201229.dcm"
    image = pydicom.dcmread(image_path, force=True)
    image_array = image.pixel_array
    image_array = np.array(image_array, dtype=np.int32)

    sex = 'M'
    age = 76
    original_input_shape = image.pixel_array.shape
    pixel_spacing = image.PixelSpacing
    print("before hist min max", image_array.min(), image_array.max())
    image_array = histogram_normalization(image_array)

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

    print("normalized_a shape", normalized_a.shape)


    # load torchscript model
    scripted_model_path = './checkpoints/xray2lung_cpu_v2.mipx'
    model = torch.jit.load(scripted_model_path)
    model = model.cpu()
    # model.eval()
    # model.float()

    print('loading model successful.')

    # inference
    with torch.no_grad():
        output = model(normalized_a, inst_tensor, feat_tensor)
        print("output shape", output.shape)
        output = output.squeeze(0).cpu().numpy().astype(np.uint8)
        print("output shape", output.shape)
        pil_img = Image.fromarray(output)
        pil_img = pil_img.resize(original_input_shape)
        pil_img.save("./output/JB0006_CXR_0base_201229_sc.png")
    """



