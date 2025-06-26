#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from base_options import BaseOptions
from test import test, color_overlay, test_heart_regression, print_gpu_stats
import os
from PIL import Image
from functools import wraps
import numpy as np
import nibabel as nib
import os
import torch
from torch.autograd import Variable
from skimage.io import imsave

import glob, ntpath, argparse
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def predict(input, input_dir, output_dir):
    opt = BaseOptions().parse(save=False)

    WEB_DEPLOY = False
    age = None
    sex = None
    pixel_spacing = None
    
    #input_dir
    #input_dir = './input'
    #output dir
    #output_dir = './output_indvbnan'
    
    data = {"success": True}
    data["message"] = 'Success'
    #success = True

    #T12 Predict
    input_filename = os.path.join(input_dir, input)
    t12output_filename = input + '_T12.png'
    
    t12nii = os.path.join(output_dir, t12output_filename + '.nii')
    t12path = os.path.join(output_dir, t12output_filename)
    print(input_filename, t12output_filename)    
    
    opt.test_input = input_filename
    opt.test_output = t12path
    opt.get_heart_area = True
    opt.hn = False
    opt.output_min_max = "-1024,1000"
    opt.checkpoint_path = "./checkpoints/xray2T12nan.pth"
    opt.threshold = -1015
    opt.save_input = True
    opt.profnorm = True
    opt.check_xray = False
    opt.age = age
    opt.sex = sex
    opt.pixel_spacing = pixel_spacing
    opt.loadSize = 512
    
    try:
        data["t12area"], is_xray, _, _, _, original_width, original_height = test(opt)
    except Exception as e:
        is_xray = False
        data["t12area"] = -2
        heartoutput_filename = ''
        # covidoutput_filename = ''
        data["message"] = 'TiSepX Heart Prediction Exception: ' + str(e)
        data["success"] = False

        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        #print(e)
    
    #L1 Predict
    if data["success"]:
        input_filename = os.path.join(input_dir, input)
        l1output_filename = input + '_L1.png'
        
        l1nii = os.path.join(output_dir, t12output_filename + '.nii')
        l1path = os.path.join(output_dir, l1output_filename)
        print(input_filename, l1output_filename)    
        
        opt.test_input = input_filename
        opt.test_output = l1path
        opt.get_heart_area = True
        opt.hn = False
        opt.output_min_max = "-1024,1000"
        opt.checkpoint_path = "./checkpoints/xray2L1nan.pth"
        opt.threshold = -1015
        opt.save_input = True
        opt.profnorm = True
        opt.check_xray = False
        opt.age = age
        opt.sex = sex
        opt.pixel_spacing = pixel_spacing
        opt.loadSize = 512
        
        try:
            data["l1area"], is_xray, _, _, _, original_width, original_height = test(opt)
        except Exception as e:
            is_xray = False
            data["l1area"] = -2
            heartoutput_filename = ''
            # covidoutput_filename = ''
            data["message"] = 'TiSepX Heart Prediction Exception: ' + str(e)
            data["success"] = False

            import sys, traceback
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            #print(e)
    
    
    # lungarea = test(opt, device_num=1)
    #Regression
    '''
    if is_xray:
        try:
            heartvolume = test_heart_regression(heartnii, heartarea)
        except Exception as e:
            is_xray = False
            heartvolume = -2
            lungoutput_filename = ''
            covidoutput_filename = ''
            data["message"] = 'TiSepX Heart Volume Prediction Error: ' + str(e)
            data["success"] = False

            import sys, traceback
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            #print(e)

        #color_overlay(lungnii, covidnii, covidpath, original_width, original_height)
        # eps = 1e-10
        # lung = nib.load(lungnii)
        # lung_arr = np.transpose(np.array(lung.dataobj)[..., 0, 0], axes=[1, 0])
        # lung_img = ((lung_arr - lung_arr.min()) / ((lung_arr.max() - lung_arr.min()) + eps)) * 255
        # imsave(lungpath, lung_img.astype(np.uint8))
        # os.remove(lungnii)
        #os.remove(covidnii)
                    
    else:
        # covidarea = -1
        heartvolume = -2
    #'''
    #End Regression
    
    #print_gpu_stats("before predict() empty_cache()")
    torch.cuda.empty_cache()
    #print_gpu_stats("after predict() empty_cache()")
    
    return data

# Limit file upload size
def limit_content_length(max_length):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cl = request.content_length
            if cl is not None and cl > max_length:
                abort(413)
            return f(*args, **kwargs)

        return wrapper

    return decorator


def resize_and_pad(img, size, padColor=0):
    import cv2

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

if __name__ == "__main__":
    # 도커 컨테이너와 통신하기 위해 ip를 0.0.0.0으로 설정
    '''
    input = "AN_ID_20210526104509_1.DCM"
    
    heart_volume = predict(input)
    print(input + ": Heart Volume: " + str(heart_volume/1000) + "L")
    #'''
    #input_dir
    input_dir = './input'
    #input_dir = args.input_dir
    #output dir
    output_dir = './output_indvbnan'
    #output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    dcms = glob.glob(input_dir + '\*.dcm')
    #dcms = glob.glob(input_dir + '\*.dicom')

    for dcm in dcms:
        input = path_leaf(dcm)
        output = predict(input, input_dir, output_dir)
        print(input + ": Output: " + str(output))
        #with open(output_dir + '\output_indvb_meanhu.csv','a') as fd:
        #    fd.write("\n" + input + ": Output: " + str(output))
            
