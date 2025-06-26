#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from base_options import BaseOptions
from test import test, test_lung_regression, print_gpu_stats, remove_private_tags, load_initial_models, grad_cam_efficientnet_tb, check_xray
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
from model import EfficientNet#, resnet18
import pdb
from skimage.io import imsave
import cv2

import glob, ntpath, argparse
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


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


def npz_batch_from_dcm_and_tisepx(dcm_file, tisepx_file, resolution=(1024, 1024), precision=np.float64, channel_first=False):
    import cv2
    import nibabel as nib

    # load tisepx
    img = nib.load(tisepx_file+".nii")
    pixel_array = img.get_fdata()
    pixel_array = np.squeeze(pixel_array)
    pixel_array = np.expand_dims(pixel_array, axis=2)
    pixel_array = cv2.rotate(pixel_array, cv2.ROTATE_90_CLOCKWISE)
    pixel_array = cv2.flip(pixel_array, 1)

    normalized_array = (lambda x:(x-x.min())/(x.max()-x.min()))(pixel_array.astype(np.float64))
    resized = resize_and_pad(normalized_array, (2048,2048))
    lung = np.clip(resized, 0.0, 1.0).astype(np.float64)

    resized = resize_and_pad(dcm_file, (2048,2048))
    cxr = np.clip(resized, 0.0, 1.0).astype(np.float64)
    lung_binary_array = np.where(lung > 0.05, 1, 0)
    cxr_segmented = cxr * lung_binary_array

    cxr_segmented = cxr_segmented.astype(precision)
    cxr_segmented = np.stack((cxr_segmented,) * 3, axis=-1)
    cxr_segmented = np.clip(cxr_segmented, 0, 1).astype(precision)
    cxr_segmented = cv2.resize(cxr_segmented, dsize=resolution, interpolation=cv2.INTER_LINEAR)
    #cxr_segmented (1024, 1024, 3)
    if channel_first:
        cxr_segmented = np.moveaxis(cxr_segmented, -1, 0)
        #cxr_segmented (3, 1024, 1024)
    cxr_segmented = cxr_segmented[np.newaxis, ...]
    #cxr_segmented (1, 3, 1024, 1024)
    return cxr_segmented

def npz_batch_cpp(dcm_file, tisepx_file, resolution=(1024, 1024), precision=np.float64, channel_first=False):
    import cv2
    import nibabel as nib

    # load tisepx
    img = nib.load(tisepx_file+".nii")
    pixel_array = img.get_fdata()
    pixel_array = np.squeeze(pixel_array)
    pixel_array = np.expand_dims(pixel_array, axis=2)
    pixel_array = cv2.rotate(pixel_array, cv2.ROTATE_90_CLOCKWISE)
    pixel_array = cv2.flip(pixel_array, 1)

    normalized_array = (lambda x:(x-x.min())/(x.max()-x.min()))(pixel_array.astype(np.float64))
    resized = resize_and_pad(normalized_array, (2048,2048))
    lung = np.clip(resized, 0.0, 1.0).astype(np.float64)

    resized = resize_and_pad(dcm_file, (2048,2048))
    cxr = np.clip(resized, 0.0, 1.0).astype(np.float64)
    lung_binary_array = np.where(lung > 0.05, 1, 0)
    cxr_segmented = cxr * lung_binary_array

    cxr_segmented = cxr_segmented.astype(precision)
    cxr_segmented = np.stack((cxr_segmented,) * 3, axis=-1)
    cxr_segmented = np.clip(cxr_segmented, 0, 1).astype(precision)
    cxr_segmented = cv2.resize(cxr_segmented, dsize=resolution, interpolation=cv2.INTER_LINEAR)
    #cxr_segmented (1024, 1024, 3)
    if channel_first:
        cxr_segmented = np.moveaxis(cxr_segmented, -1, 0)
        #cxr_segmented (3, 1024, 1024)
    cxr_segmented = cxr_segmented[np.newaxis, ...]
    #cxr_segmented (1, 3, 1024, 1024)
    fileobj = open(f'{tisepx_file}_test.raw', mode='wb')
    off = np.array(cxr_segmented[0][0], dtype=precision)
    off.tofile(fileobj)
    fileobj.close()
    
    return cxr_segmented


def resize_ori(rgb_arr, height, width):
    from math import ceil, floor
    # _, file_extension = os.path.splitext(input_filename)

    target_size = max(rgb_arr.shape[0], rgb_arr.shape[1])
    ratio = float(target_size) / max(height, width)
    new_size = tuple([int(x * ratio) for x in [height, width]])

    pad_size_w = (target_size - new_size[0]) / 2
    pad_size_h = (target_size - new_size[1]) / 2

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

    eps = 1e-10
    nii_np_min, nii_np_max = rgb_arr.min(), rgb_arr.max()
    rgb_arr = ((rgb_arr - nii_np_min) / ((nii_np_max - nii_np_min) + eps))
    img_masked = rgb_arr[wl:target_size - wr, ht:target_size - hb, :]
    # img_masked = rgb_arr[wl:target_size - wr, ht:target_size - hb]

    img_ = Image.fromarray((img_masked * 255).astype(np.uint8))
    try:
        img_ = img_.resize((width, height), Image.LANCZOS)
    except:
        img_ = img_.resize((width, height), Image.NEAREST)
    return np.array(img_)


def read_dcm_to_img_arr(dcm_file, output_size_rectangle=(1024,1024), precision=np.float64, channel_first=False):
    import cv2

    resized = cv2.resize(dcm_file, dsize=output_size_rectangle, interpolation=cv2.INTER_LINEAR)
    resized = np.stack((resized,) * 3, axis=-1)
    resized = np.clip(resized, 0, 1).astype(precision)
    if channel_first:
        resized = np.moveaxis(resized, -1, 0)

    return resized

def save_rgb(rgb_arr, input_filename, output_file):
    from pydicom.dataset import Dataset
    from pydicom.uid import generate_uid
    # Saving to DICOM
    try:  # if file_extension.lower() == '.dcm' or file_extension.lower() == '.dicom':
        ds_ori = pydicom.dcmread(input_filename, force=True)
        try:
            try:
                arr = ds_ori.pixel_array
            except:
                ds_ori.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                arr = ds_ori.pixel_array
            # print(arr.shape)
            # pdb.set_trace()
            rgb_arr = resize_ori(rgb_arr, arr.shape[0], arr.shape[1])
        except:
            print("Failed to resize")
            pass

    except:  # else:
        if file_extension.lower() == '.nii' or file_extension.lower() == '.gz':
            nii = nib.load(input_filename)
            nii_shape = len(np.array(nii.dataobj).shape)

            if nii_shape == 2:
                arr = np.transpose(
                    np.array(nii.dataobj).astype(np.int16), axes=[1, 0])
            elif nii_shape == 3:
                arr = np.transpose(np.array(nii.dataobj).astype(
                    np.int16), axes=[2, 1, 0])[0, :, :]
            elif nii_shape == 4:
                arr = np.transpose(np.array(nii.dataobj).astype(
                    np.int16), axes=[3, 2, 1, 0])[0, 0, :, :]

            header = nii.header
            pixel_spacing = header.get_zooms()
            # original_input_shape = np.transpose(arr, axes=[1, 0]).shape
        else:
            im = Image.open(input_filename).convert('L')
            arr = np.array(im)
            pixel_spacing = [1, 1]

        try:
            rgb_arr = resize_ori(rgb_arr, arr.shape[0], arr.shape[1])
            # ds.PixelData = rgb_arr.tobytes()
        except:
            pdb.set_trace()

        ds.PixelSpacing = [pixel_spacing[0], pixel_spacing[0]]
        # ds.Rows, ds.Columns, _ = rgb_arr.shape

    
    
    Image.fromarray(rgb_arr).save(output_file, format="png")

def pytorch_inference_with_custom_grad_cam(dcm_file, tisepx_file, input_filename, pytorch_model_path, output_grad_cam_png_path, output_ori_path, device):

    batch = npz_batch_from_dcm_and_tisepx(dcm_file=dcm_file, tisepx_file=tisepx_file, resolution=(1024, 1024),
                                          precision=np.float32, channel_first=True)
    
    batch2 = npz_batch_cpp(dcm_file=dcm_file, tisepx_file=tisepx_file, resolution=(1024, 1024), precision=np.float32, channel_first=True)
    #compare modified data for C++ port
    
   # pytorch_model = torch.load(pytorch_model_path)
    
    
    #model = load_initial_models()
    #New TB
    #model = EfficientNet.from_pretrained('efficientnet-b5', weights_path='./checkpoints/tiseptb2_pytorch.pth', num_classes=2)
    #Old TB
    model = EfficientNet.from_pretrained('efficientnet-b5', weights_path='./checkpoints/efficientnet-b5.pth', num_classes=2)
    
    ONNX_PYTORCH = False
    #ONNX_PYTORCH = True
    
    model_input_image = torch.tensor(batch).to(device)
    if ONNX_PYTORCH:
        pytorch_model.to(device)
        pytorch_model.eval()
        #pytorch_model.double()
        
        y = pytorch_model(model_input_image).detach().cpu().numpy()
        prediction = y[0][1]
        print("PYTORCH positive prediction : %.3f" % float(prediction))
        gradients = []
        activations = []

        def hook_fn(module, input, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))
        for name, module in pytorch_model.named_modules():
            if name == 'Mul_StatefulPartitionedCall/model/top_activation/mul:0':  # 대상 레이어 이름으로 대체
                module.register_forward_hook(hook_fn)
                break

        output = pytorch_model(model_input_image)
        target = output[0][1]
        
        pytorch_model.zero_grad()
        target.backward()
        
        gradients = gradients[0].cpu().data.numpy()[0]
        activations = activations[0].cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        #pdb.set_trace()
        cam = np.maximum(cam, 0)

        cam = cv2.resize(cam, (1024, 1024))
    else:
        activation = {}

        def get_activation(name):
            def hook(model, input, prediction):
                activation[name] = prediction

            return hook

        model._softmax.register_forward_hook(
            get_activation('_softmax'))
        model._last_swish.register_forward_hook(
            get_activation('_last_swish'))

        model.to(device)
        model.eval()
        # with torch.no_grad():
        # Perform prediction
        y = model(model_input_image)
        #pdb.set_trace()
        prediction = y[0][1].cpu().detach().numpy()
        print(prediction)
        # Generate CAM
        classIndex = 1#new weight's TB index
        imageSize = 1024
        cam = grad_cam_efficientnet_tb(img=model_input_image, model=model, class_index=classIndex, activation_layer=activation, imageSize=imageSize)

        # Generate color image from CAM array
        cam = np.clip(cam / cam.max(), 0, 1.0)
        #pdb.set_trace()
        #y = pytorch_model(model_input_image).detach().cpu().numpy()
        #prediction = y[0][1]
        #print("PYTORCH positive prediction : %.3f" % float(prediction))
        '''
        new_params = pytorch_model.named_parameters()
        for name, module in pytorch_model.named_modules():
            print(name)
            pdb.set_trace()
        for name, buffer in pytorch_model.named_buffers():
            print(name)
            #pdb.set_trace()
            with open('./onnx_buffers.csv','a') as fd:
                try:
                    fd.write("\n" + name + "|" + str(buffer.shape))
                except:
                    fd.write("\n" + name)
            print(name)
        for name, buffer in model.named_buffers():
            print(name)
            #pdb.set_trace()
            with open('./prev_bufferms.csv','a') as fd:
                try:
                    fd.write("\n" + name + "|" + str(buffer.shape))
                except:
                    fd.write("\n" + name)
            print(name)
        for name, param in pytorch_model.named_parameters():
            print(name)
            #pdb.set_trace()
            with open('./onnx_params.csv','a') as fd:
                try:
                    fd.write("\n" + name + "|" + str(param.shape))
                except:
                    fd.write("\n" + name)
            print(name)
        for name, param in model.named_parameters():
            print(name)
            #pdb.set_trace()
            with open('./prev_params.csv','a') as fd:
                try:
                    fd.write("\n" + name + "|" + str(param.shape))
                except:
                    fd.write("\n" + name)
            print(name)
        pdb.set_trace()
        for name, module in pytorch_model.named_modules():
            with open('./onnx2pytorch.csv','a') as fd:
                try:
                    fd.write("\n" + name + "|" + str(module) + "|" + str(module.weight.shape))
                except:
                    fd.write("\n" + name + "|" + str(module))
            print(name)
            #pdb.set_trace()
        #pdb.set_trace()
        #'''
        
    
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam2 = np.clip(cam / cam.max() * prediction, 0, 1)
    tb_pixel_count = np.sum(np.where(cam2 >= 0.16, 1, 0).flatten())

    image = read_dcm_to_img_arr(dcm_file=dcm_file)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    alpha = prediction / 3
    #if alpha < 0.15:
    #    alpha = 0.15
    output_image = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    Image.fromarray(image).save(output_ori_path, format="png")
    save_rgb(output_image, input_filename=input_filename, output_file=output_grad_cam_png_path)

    model_input_image.cpu()
    #pytorch_model.cpu()
    
    return prediction * 100, int(tb_pixel_count)

"""
def pytorch_inference_with_grad_cam(dcm_file, tisepx_file, input_filename, pytorch_model_path, output_grad_cam_png_path, output_ori_path, device):
    import torch, cv2
    from pytorch_grad_cam import GradCAM, EigenGradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from PIL import Image
    
    #torch.set_default_dtype(torch.float64)

    batch = npz_batch_from_dcm_and_tisepx(dcm_file=dcm_file,tisepx_file=tisepx_file,resolution=(1024,1024),
                                                    precision=np.float32,channel_first=True)#np.float64,channel_first=True)
    pytorch_model = torch.load(pytorch_model_path)
    pytorch_model.to(device)
    pytorch_model.eval()
    #pytorch_model.double()
    modelInputImage = torch.tensor(batch).to(device)
    #modelInputImage = torch.tensor(batch)
    y = pytorch_model(modelInputImage).detach().cpu().numpy()
    prediction = y[0][1]
    print("PYTORCH positive prediction : %.3f" % float(prediction))

    for name, module in pytorch_model.named_modules():
        if name == 'Mul_StatefulPartitionedCall/model/top_activation/mul:0':
            break

    #cam = GradCAM(model=pytorch_model, target_layers=[module
    cam = GradCAMPlusPlus(model=pytorch_model, target_layers=[module])
    #cam = EigenGradCAM(model=pytorch_model, target_layers=[module])
    targets = [ClassifierOutputTarget(1)]
    #heatmap = cam(input_tensor=torch.tensor(batch), targets=targets)
    heatmap = cam(input_tensor=modelInputImage, targets=targets, aug_smooth = False, eigen_smooth = False)
    #True False
    heatmap = np.moveaxis(heatmap,0,-1)
    #import matplotlib.pyplot as plt
    #plt.imshow(np.where(heatmap >= 0.16, 1, 0))
    #plt.imshow(batch[0][0])
    #plt.show()
    
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap2 = np.clip(heatmap / heatmap.max() * prediction, 0, 1)
    tb_pixel_count = np.sum(np.where(heatmap2 >= 0.16, 1, 0).flatten())
    image = read_dcm_to_img_arr(dcm_file=dcm_file)
    #lung only
    #image = np.stack((batch[0][0],) * 3, axis=-1)
    #image = np.clip(image, 0, 1).astype(np.float64)
    
    #image = np.moveaxis(batch, 0, -1)
    
    heatmap = np.clip((1-heatmap) * 255.0, 0, 255).astype(np.uint8)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cam = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    alpha = prediction / 3
    if alpha < 0.15:
        alpha = 0.15
    #alpha = prediction * 3
    output = cv2.addWeighted(image, 1 - alpha, cam, alpha, 0)
    Image.fromarray(image).save(output_ori_path, format="png")
    #overlay = Image.fromarray(output)
    #overlay.save(output_grad_cam_png_path, format="png")
    save_rgb(output, input_filename=input_filename, output_file=output_grad_cam_png_path)
    modelInputImage.cpu()
    pytorch_model.cpu()
    return prediction * 100, int(tb_pixel_count)
"""

def predict(input, input_dir, output_dir):
    opt = BaseOptions().parse(save=False)

    WEB_DEPLOY = False
    age = None
    sex = None
    pixel_spacing = None
    
    #input_dir
    #input_dir = './input'
    #output dir
    #output_dir = './output'
    
    input_filename = os.path.join(input_dir, input)
    covidoutput_filename = input + '_covid.png' 
    lungoutput_filename = input + '_lung.png'
    
    
    lungnii = os.path.join(output_dir, lungoutput_filename + '.nii')
    lungpath = os.path.join(output_dir, lungoutput_filename)
    covidnii = os.path.join(output_dir, covidoutput_filename + '.nii')
    covidpath = os.path.join(output_dir, covidoutput_filename)
    print(input_filename, lungoutput_filename, covidoutput_filename)

    output = {}
    data = {"success": False}
    start_time = time.time()

    device_num = 0
    device = 'cuda:' + str(device_num)
    

    # Current model name
    data["model"] = "SNUH_TB 202301"

    # WEB_DEPLOY = True
    WEB_DEPLOY = False
    
    #######################
    # Predict TiSepX lung #
    #######################
    opt = BaseOptions().parse(save=False)
    opt.test_input = input_filename
    opt.test_output = input_filename + 'temp'
    opt.get_lung_area = True
    opt.output_min_max = "-1100,-500"
    opt.checkpoint_path = "./checkpoints/xray2lung.pth"
    opt.threshold = -1015
    opt.save_input = False
    opt.profnorm = True
    opt.check_xray = False
    opt.age = None
    opt.sex = None
    opt.pixel_spacing = None
    opt.hn = True

    try:
        lungarea, _, _, _, _, _, _ = test(opt)
        #os.remove(input_filename + 'temp' + '.nii')
        data['success'] = True
    except Exception as e:
        is_xray = False
        lungarea = -2
        lungoutput_filename = ''
        covidoutput_filename = ''
        message = 'TiSepX Lung Prediction Exception: ' + str(e)
        success = False

        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        # print(e)

    #######################
    # Predict TiSepX lung #
    #######################

    #######################
    ##### Predict TB ######
    #######################

    # Generate random word for temporary file name
    def randomword(length):
        from datetime import datetime
        import random, string
        letters = string.hexdigits
        return datetime.today().strftime("%Y%m%d") + "_" + ''.join(random.choice(letters) for i in range(length))
    
    #pdb.set_trace()
    opt.pytorch_model_path = "./checkpoints/tiseptb2.pth"
    opt.tisepx_lung = input_filename + 'temp'
    
    #random = randomword(20)
    #opt.tb_output = './output_tb/' + random + '-overlay.png'
    #opt.ori_output = './output_tb/' + random + '-original.png'
    
    opt.tb_output = os.path.join(output_dir, input + '_tb_overlay.png')
    opt.ori_output = os.path.join(output_dir, input + '_original.png')
    
    data["color"] = opt.tb_output
    data["original"] = opt.ori_output

    if WEB_DEPLOY:
        print(file.content_type)
        content_type = file.content_type
        ext = file.filename.lower().split(".")[-1]
    else:
        content_type = ''
        ext = input_filename.lower().split(".")[-1]

    try:
        data['pixel_spacing'] = [1, 1]
        data['original_input_shape'] = [2048, 2048]
        if content_type in ["image/png", "image/jpeg", "image/bmp", "image/gif", "image/tif"] or \
                ext in ["png", "jpeg", "jpg", "bmp", "gif", "tif"]:
            if WEB_DEPLOY:
                imageBytes = flask.request.files["file"].read()
                imgobj = Image.open(io.BytesIO(imageBytes))
            else:
                imgobj = Image.open(input_filename)

            imgobjmode = imgobj.mode

            # Convert image to grayscale image
            if imgobj.mode in ["RGB", "RGBA"]:
                imgobjmode = "L"
                image = np.array(imgobj.convert("L"))
            elif imgobj.mode in ["L"]:
                imgobjmode = "L"
                image = np.array(imgobj.convert("L"))
            else:
                # Preserve 16 bit color image
                image = np.array(imgobj.convert("I"))

            if image.shape[0] is None or image.shape[1] is None:
                image = np.array(imgobj.convert("L"))
            if image.shape[0] is None or image.shape[1] is None:
                data["success"] = False
                data["message"] = "Corrupted File"
                return flask.jsonify(data)
            # image = Image.fromarray(image, mode=imgobjmode)
            f = (image - np.min(image)) / (np.max(image) - np.min(image))
            import matplotlib.pyplot as plt
            plt.imshow(f)

        # If uploaded content is dicom file :
        elif content_type in ["application/dicom"] or ext in ["dcm", "dc3", "dicom", "nii"]:
            if WEB_DEPLOY:
                tempDCMPath = "tempdcm/" + randomword(20)
                file.save(tempDCMPath)
            else:
                tempDCMPath = input_filename

            if ext in ["dcm", "dc3", "dicom"]:
                dataset = pydicom.dcmread(tempDCMPath, force=True)
                # dataset = remove_private_tags(dataset, tempDCMPath)
                new_array = dataset.pixel_array

                try:
                    assert dataset.ImagerPixelSpacing[0] == dataset.ImagerPixelSpacing[1]
                    pixel_spacing = dataset.ImagerPixelSpacing[:2]
                except:
                    try:
                        assert dataset.PixelSpacing[0] == dataset.PixelSpacing[1]
                        pixel_spacing = dataset.PixelSpacing[:2]
                    except:
                        pixel_spacing = [1, 1]
                data['pixel_spacing'] = pixel_spacing
                data['original_input_shape'] = new_array.shape

            elif ext in ["nii"]:
                dataset = nib.load(tempDCMPath)
                nii_shape = len(np.array(dataset.dataobj).shape)

                if nii_shape == 2:
                    new_array = np.transpose(np.array(dataset.dataobj), axes=[1, 0])
                elif nii_shape == 3:
                    new_array = np.transpose(np.array(dataset.dataobj), axes=[2, 1, 0])[0, :, :]
                elif nii_shape == 4:
                    new_array = np.transpose(np.array(dataset.dataobj), axes=[3, 2, 1, 0])[0, 0, :, :]

                header = dataset.header
                data['pixel_spacing'] = [float(h) for h in header.get_zooms()]
                data['original_input_shape'] = np.transpose(new_array, axes=[1, 0]).shape

            mean = np.mean(new_array)
            std = np.std(new_array)

            new_array[new_array < mean - std * 2] = mean - std * 2
            np05 = np.nanpercentile(new_array, 0)
            np95 = np.nanpercentile(new_array, 99)

            # normalized_array = (new_array - np05) * 65535.0 / (np95 - np05)
            # normalized_array[normalized_array > 1 * 65535] = 65535
            # normalized_array[normalized_array < 0 * 65535] = 0 * 65535
            normalized_array = (new_array - np05) / (np95 - np05)
            normalized_array = np.clip(normalized_array, 0, 1)

            if ext in ["dcm", "dc3", "dicom"] and str(dataset[0x28, 0x04].value) == "MONOCHROME1":
                # normalized_array = 65535 - normalized_array
                normalized_array = 1 - normalized_array

            s = max(normalized_array.shape[0:2])
            f = np.zeros((s, s), np.float64)
            ax, ay = (s - normalized_array.shape[1]) // 2, (s - normalized_array.shape[0]) // 2
            f[ay:normalized_array.shape[0] + ay, ax:ax + normalized_array.shape[1]] = normalized_array

        else:
            data["success"] = False
            data["message"] = "Unidentifiable file"
            return flask.jsonify(data)
        # data['tb_pixel_count']
        data["prediction"], data['tb_pixel_count'] = pytorch_inference_with_custom_grad_cam(dcm_file=f,
                                                                                 tisepx_file=opt.tisepx_lung,
                                                                                 input_filename=input_filename,
                                                                                 pytorch_model_path=opt.pytorch_model_path,
                                                                                 output_grad_cam_png_path=opt.tb_output,
                                                                                 output_ori_path=opt.ori_output, device=device)

        os.remove(input_filename + 'temp' + '.nii')
        # Generate "decision" string to display on client.
        # decision = 'N/A'
        if data["prediction"] > 80:
            decision = "Definite active Tb"
        elif data["prediction"] > 50.84:
            decision = "Probable active Tb"
        elif data["prediction"] > 25:
            decision = "Probable healed Tb"
        else:
            decision = "Definite healed Tb"

        data['decision'] = decision

        #######################
        # Calculate TB Extent #
        #######################
        original_height = data['original_input_shape'][0]
        original_width = data['original_input_shape'][1]
        ratio = float(1024) / max(original_width, original_height)

        pixel_size_resize_w = data['pixel_spacing'][0] / ratio
        pixel_size_resize_h = data['pixel_spacing'][1] / ratio
        tb_area = data['tb_pixel_count'] * pixel_size_resize_w * pixel_size_resize_h / 100
        
        tb_extent = tb_area / lungarea * 100
        data['tb_extent'] = tb_extent
        data['lungarea'] = lungarea
        data['tbarea'] = tb_area
        #######################
        # Calculate TB Extent #
        #######################

    except Exception as e:
        # Handling all kinds of error
        data["success"] = False
        data["message"] = "Error " + str(e)

        import sys, traceback
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        print(e)

    #######################
    ##### Predict TB ######
    #######################

    #os.remove(opt.tisepx_lung +".nii")

    # Calculating elapsed time during calculation.
    elapsed_time = time.time() - start_time
    data['elapsed_time'] = elapsed_time

    # Prevent memory leak, dealloc variables
    import gc
    gc.collect()

    print_gpu_stats("before predict_snuhtb() empty_cache()")
    torch.cuda.empty_cache()
    print_gpu_stats("after predict_snuhtb() empty_cache()")
    
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

if __name__ == "__main__":
    '''
    input = "AN_ID_20210526104509_1.dcm"
    lungarea, covidarea = predict(input)
    print(input + ": Lung Area: " + str(lungarea) + "cm2" + " | COVID Area: " + str(covidarea) + "cm2")
    #'''
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--input_dir', type=str, help='input data path', default = './input')
    parser.add_argument('--output_dir', type=str, help='outfolder path', default = './output')
    args = parser.parse_args()
    
    #input_dir
    #input_dir = args.input_dir
    #input_dir = './input'
    input_dir = './input_debug'    
    #output dir
    #output_dir = args.output_dir
    #output_dir = './output'
    output_dir = './output_debug'
    #output_dir = "./output_pytorch2"
    #output_dir = "./debug_pytorch2"
    #output_dir = "./debug_onnx2"
    os.makedirs(output_dir, exist_ok=True)
    
    dcms = glob.glob(input_dir + '\*.dcm')
    #dcms = glob.glob(input_dir + '\*.dicom')

    for dcm in dcms:
        input = path_leaf(dcm)
        output = predict(input, input_dir, output_dir)
        print(input + ", " + str(output))
        with open(output_dir + '\output_tb.csv','a') as fd:
            fd.write("\n" + input + ": " + str(output))
        
        