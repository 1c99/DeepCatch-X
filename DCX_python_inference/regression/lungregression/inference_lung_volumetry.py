import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from base_options import BaseOptions
from test import test, test_lung_regression, print_gpu_stats
import os, hashlib, random, datetime
from PIL import Image
from functools import wraps
import numpy as np
import io
import pydicom
import nibabel as nib
import os
import time
import matplotlib.cm as cm
import torch
from torch.autograd import Variable
#from model import EfficientNet, resnet18
import pdb
from skimage.io import imsave

def predict(input):
    opt = BaseOptions().parse(save=False)

    WEB_DEPLOY = False
    age = None
    sex = None
    pixel_spacing = None
    
    #input_dir
    input_dir = './input'
    #output dir
    output_dir = './output'
    
    input_filename = os.path.join(input_dir, input)
    lungoutput_filename = input + '_lung.png'
    
    #pdb.set_trace()
    
    lungnii = os.path.join(output_dir, lungoutput_filename + '.nii') # ./output\AN_ID_20210526104509_1.dcm_lung.png.nii
    lungpath = os.path.join(output_dir, lungoutput_filename)

    message = 'Success'
    success = True

    opt.test_input = input_filename
    opt.test_output = lungpath
    opt.get_lung_area = True
    opt.hn = False
    opt.output_min_max = "-1100,-500"
    opt.checkpoint_path = "./checkpoints/xray2lung.pth"
    opt.threshold = -1015
    opt.save_input = True
    opt.profnorm = True
    opt.check_xray = False
    opt.age = age
    opt.sex = sex
    opt.pixel_spacing = pixel_spacing
    
    try:
        lungarea, is_xray, sex, age, default_sex_age, original_width, original_height = test(opt)
        print("Lung Area : ", lungarea)

    except Exception as e:
        is_xray = False
        lungarea = -2
        lungoutput_filename = ''
        message = 'TiSepX Lung Prediction Exception: ' + str(e)
        success = False

        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        #print(e)

    # lungarea = test(opt, device_num=1)

    if is_xray:
        try:
            lungvolume = test_lung_regression(lungnii, lungarea, sex, age)
        except Exception as e:
            is_xray = False
            lungvolume = -2
            lungoutput_filename = ''
            message = 'TiSepX Lung Volume Prediction Error: ' + str(e)
            success = False

            import sys, traceback
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            #print(e)

        eps = 1e-10
        lung = nib.load(lungnii)

        lung_arr = np.transpose(np.array(lung.dataobj)[..., 0, 0], axes=[1, 0])
        lung_img = ((lung_arr - lung_arr.min()) / ((lung_arr.max() - lung_arr.min()) + eps)) * 255
        imsave(lungpath, lung_img.astype(np.uint8))
        #os.remove(lungnii)
        breakpoint()
                    
    else:
        lungvolume = -2

    print_gpu_stats("before predict() empty_cache()")
    torch.cuda.empty_cache()
    print_gpu_stats("after predict() empty_cache()")
    return lungvolume
    
if __name__ == "__main__":
    # 도커 컨테이너와 통신하기 위해 ip를 0.0.0.0으로 설정
    # input = "AN_ID_20210526104509_1.dcm"
    input = "JB0006_CXR_0base_201229.dcm"
    lung_volume = predict(input)
    print(input + ": Lung Volume: " + str(lung_volume/1000) + "L")