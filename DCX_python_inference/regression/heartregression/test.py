import os
# from .base_options import BaseOptions
#from pix2pixHD_model import InferenceModel
from pix2pixHD_model import BuildModel
from heartregression_model import UNetRFull
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
from collections import OrderedDict

def CreateDataLoader(opt):
    dataloader = CustomDatasetDataLoader()
    dataloader.initialize(opt)
    return dataloader


def print_gpu_stats(title=""):
    t = torch.cuda.get_device_properties(0).total_memory
    #r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    #f = r-a # free inside reserved
    print (title)
    print ('total_memory :'+str(t))
    #print ('memory_reserved :'+str(r))
    #print ('free inside reserved :'+str(f))
    print ('memory_allocated :'+str(a))

# remove private information
def remove_private_tags(dataset, savepath, type='dicom'):
    try:
        if type == 'dicom':
            dataset.StudyDate = ''
            dataset.SeriesDate = ''
            dataset.AcquisitionDate = ''
            dataset.ContentDate = ''
            dataset.Manufacturer = ''
            dataset.InstitutionName = ''
            dataset.InstitutionAddress = ''
            dataset.ReferringPhysicianName = ''
            dataset.StationName = ''
            dataset.AttendingPhysicianName = ''
            dataset.PatientName = ''
            dataset.PatientID = ''
            dataset.PatientBirthDate = ''
            #dataset.PatientSex = ''
            dataset.OtherPatientIDs = ''
            dataset.OtherPatientNames = ''
            #dataset.PatientAge = ''
            dataset.PatientSize = ''
            dataset.PatientWeight = ''
            dataset.BodyPartExamined = ''
            dataset.save_as(savepath)

        print('Private Tag Removed')
    except:
        print('Failed to remove private tag')
    return dataset

def load_xrayct_models():
    xrayct_model = resnet18(num_classes=2, num_channels=1)
    xrayct_model.load_state_dict(torch.load('./checkpoints/xrayctcls.pth').get('weight', False))

    return xrayct_model


# Check if input is xray image
def check_xray(input_arr, device, th=0.31):
    xrayct_model = load_xrayct_models()
    xrayct_model.eval()
    xrayct_model.to(device)

    normalized_input = (input_arr - input_arr.mean()) / input_arr.std()
    normalized_input = Variable(torch.from_numpy(normalized_input.astype(np.float32))).to(device)
    prediction = xrayct_model(normalized_input[np.newaxis, np.newaxis])
    softmax = torch.nn.Softmax(dim=1)
    xray_prob = softmax(prediction).detach().cpu().numpy()[0, 1]
    is_xray = xray_prob > th
    print('xray probability: ', xray_prob)

    xrayct_model.cpu()
    normalized_input = normalized_input.cpu()
    
    torch.cuda.empty_cache()
    return is_xray


#### SNUH TB
# Initialize model
def load_initial_models():
    # Keras model path
    # efficientTBModelPath = "model-0.84-0.3845_single.weights"

    # global model
    # graph = tf.get_default_graph()

    # Load model from file
    # model = load_model(efficientTBModelPath)
    model = EfficientNet.from_pretrained('efficientnet-b5', weights_path='./checkpoints/efficientnet-b5.pth', num_classes=2)
    # model.set_swish(memory_efficient=False)
    return model

# Calculate gradient-class-activation-map
def grad_cam_efficientnet_tb(img, model, class_index, activation_layer, imageSize):
    input_model = model

    # y_c = input_model.get_layer('fc_out').output[0, class_index]
    y_c = activation_layer['_softmax'][0, class_index]
    # conv_output = input_model.get_layer(activation_layer).output
    conv_output = activation_layer['_last_swish']
    # grads = K.gradients(y_c, conv_output)[0]

    grads = torch.autograd.grad(y_c, conv_output)
    # gradient_function = K.function([input_model.get_layer('input_1').input], [conv_output, grads])

    # output, grads_val = gradient_function([img])
    output, grads_val = conv_output[0, :], grads[0][0, :, :, :]
    output = np.transpose(output.cpu().detach().numpy(), axes=(1, 2, 0))
    grads_val = np.transpose(grads_val.cpu().detach().numpy(), axes=(1, 2, 0))

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = Image.fromarray(cam)
    cam = cam.resize((imageSize, imageSize), resample=Image.BILINEAR)
    cam = np.array(cam)
    # cam = cv2.resize(cam, (imageSize, imageSize), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = np.clip(cam / cam.max(), 0, 1)

    return cam



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


def create_model(opt):
    model = InferenceModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.data_parallel:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    if opt.use_gpu == 0:
        model.cuda('cuda:0')
    elif opt.use_gpu == 1:
        model.cuda('cuda:1')

    return model

def create_model_v2(opt):
    model = BuildModel(opt)
    # if opt.data_parallel:
        # model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    if opt.use_gpu == 0:
        model.cuda('cuda:0')
    elif opt.use_gpu == 1:
        model.cuda('cuda:1')

    # model = remove_netG(model, model.state_dict())
    state = torch.load(opt.checkpoint_path)

    ordered_state_dict = OrderedDict()
    for k, v in state.get('weight').items():
        #print(ordered_state_dict["model.netG." + k])
        name = "model.netG." + k
        ordered_state_dict[name] = v

    model.load_state_dict(ordered_state_dict)

    return model

def test(opt, device_num=0):
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # device_count = torch.cuda.device_count()
    opt.use_gpu = device_num
    device = 'cuda:' + str(device_num)
    opt.cuda0 = device
    opt.cuda1 = device

    # epoch_list = opt.test_epoch.split(',')

    b_min_val, b_max_val = opt.output_min_max.split(',')
    b_min_val, b_max_val = int(b_min_val), int(b_max_val)

    # for epoch in epoch_list:
    # lung_info = []
    # opt.which_epoch = epoch
    os.makedirs("%s/" % 'output', exist_ok=True)

    model = create_model_v2(opt)
    is_xray = True

    for i, data in enumerate(dataset):
        # minibatch = 1
        if opt.check_xray:
            #from predict import check_xray
            #is_xray = check_xray(data['arr'].numpy()[0], device)
            is_xray = True
        if is_xray:
            sex = data['sex']
            age = data['age']
            default_sex_age = data['default_sex_age']
            model.eval()
            model.float()
            with torch.no_grad():
                #traced_cell = torch.jit.trace(model, (data['label'].type(torch.cuda.FloatTensor), data['inst'].type(torch.cuda.FloatTensor)) )
                #generated = traced_cell(data['label'].type(torch.cuda.FloatTensor), data['inst'].type(torch.cuda.FloatTensor) )
                generated = model(data['label'].type(torch.cuda.FloatTensor))
                #generated = model.inference(data['label'].type(torch.cuda.FloatTensor), data['inst'].type(torch.cuda.FloatTensor), data['image'].type(torch.cuda.FloatTensor))

            generated_np = generated.detach().cpu().numpy()
            if opt.standardization:
                denormalize_gen = generated_np * data['std'].cpu().numpy() + data['mean'].cpu().numpy()
            else:
                denormalize_gen = generated_np * (b_max_val - b_min_val) + b_min_val

            # filename = os.path.basename(data['path'][0])  # .split("/")[-1]  # .split(".")[0]
            # filename = filename[0:filename.rfind(".")]

            # apply threshold
            denormalize_gen = np.where(denormalize_gen < opt.threshold, -1024, denormalize_gen)
            denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)
            if opt.get_covid_area:
                if np.sum(denormalize_gen_mask.flatten()) == 512*512:
                    denormalize_gen_mask = np.zeros_like(denormalize_gen_mask)

            if opt.get_lung_area:  # find connected region
                # denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)
                # denormalize_gen_mask = np.where(denormalize_gen == -1024, 0, 1)
                denormalize_gen_mask = get_biggest_connected_region(denormalize_gen_mask)
                connected_lung = np.where(denormalize_gen_mask, denormalize_gen[0, 0], -1024)
                denormalize_gen = connected_lung[np.newaxis, np.newaxis]

            # calculate area
            original_width = data['original_input_shape'][0].numpy()[0]
            original_height = data['original_input_shape'][1].numpy()[0]
            ratio = float(512) / max(original_width, original_height)

            pixel_size_resize_w = data['pixel_spacing'][0].numpy()[0] / ratio
            pixel_size_resize_h = data['pixel_spacing'][1].numpy()[0] / ratio

            area = np.sum(denormalize_gen_mask.flatten())
            area = area * pixel_size_resize_w * pixel_size_resize_h / 100

            # lung_info.append(
            #     [opt.test_output, original_width, original_height, ratio, data['pixel_spacing'][0].numpy()[0],
            #      data['pixel_spacing'][1].numpy()[0],
            #      pixel_size_resize_w, pixel_size_resize_h,
            #      area, opt.threshold])
            
            nii_np = np.transpose(denormalize_gen, axes=[3, 2, 1, 0])
            nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
            nii.header['pixdim'] = pixel_size_resize_w
            nib.save(nii, opt.test_output + '.nii')

            if opt.save_input:
                arr, _, _, _, _, _, _, _ = data_loader.dataset.read_input(opt.test_input, opt)
                eps = 1e-10
                #norm
                #mean, std = arr.mean(), arr.std()
                #arr = np.where(arr < mean - (2*std), mean - (2*std), arr)
                #arr = np.clip(arr, np.percentile(arr, 0), np.percentile(arr, 99))
                #end
                input_img = ((arr - arr.min()) / ((arr.max() - arr.min()) + eps)) * 255
                path = os.path.join(os.path.dirname(opt.test_output), os.path.basename(opt.test_input) + '.png')
                imsave(path, input_img.astype(np.uint8))
                print(path, "saved")
                
                img_ = Image.fromarray((input_img).astype(np.uint8))

            # imsave(opt.test_output, denormalize_gen.astype(np.int16)[0, 0])
            # minVal = np.amin(generated_np[0,0])
            # maxVal = np.amax(generated_np[0,0])
            # generated_np[0,0] = (generated_np[0,0] - minVal) / (maxVal - minVal)
            # imsave(opt.test_output, generated_np[0,0])
        else:
            area = -1
    # log_df = pd.DataFrame(lung_info, columns=['filename', 'original_width', 'original_height', 'ratio_to_2048',
    #                                           'original_pixel_size_w', 'original_pixel_size_resize_h',
    #                                           'pixel_size_resize_w', 'pixel_size_resize_h', 'area',
    #                                           'threshold'])
    # log_df.to_csv("%s/%s_%s.csv" % ('./output', filename, opt.name))
    # log_df.to_csv(opt.test_output + ".csv")

    # return filename + '_' + opt.name + '.nii'

    model.cpu()
    data = {}
    torch.cuda.empty_cache()
    return area, is_xray, sex, age, default_sex_age, original_width, original_height


def color_overlay(lung_path, covid_path, output_path, original_width, original_height):
    from skimage import color
    # import pdb
    # import matplotlib.pyplot as plt

    lung = nib.load(lung_path)
    lung_arr = np.transpose(np.array(lung.dataobj)[..., 0, 0], axes=[1, 0])
    covid = nib.load(covid_path)
    covid_arr = np.transpose(np.array(covid.dataobj)[..., 0, 0], axes=[1, 0])
    eps = 1e-10

    lung_img = ((lung_arr - lung_arr.min()) / ((lung_arr.max() - lung_arr.min()) + eps))  # * 255

    # covid_alpha = (covid_arr - covid_arr.min()) / (covid_arr.max() - covid_arr.min())
    second_min = np.amin(covid_arr)
    if covid_arr[covid_arr != second_min].shape[0] != 0:
        second_min = np.amin(covid_arr[covid_arr != second_min])
    
    covid_arr_no_bg = np.where(covid_arr < second_min, second_min, covid_arr) # change BG to second smallest value
    # normalize to alpha_range
    alpha_range = [0.4, 0.5]
    covid_alpha_ = (alpha_range[1] - alpha_range[0]) * \
                   ((covid_arr_no_bg - covid_arr_no_bg.min()) / (covid_arr_no_bg.max() - covid_arr_no_bg.min() + eps)) + \
                   alpha_range[0]
    #print(covid_alpha_.min())
    #print(covid_alpha_.max())
    covid_alpha = np.where(covid_arr == covid_arr.min(), 0, covid_alpha_)  # change BG value to 0
    covid_img = covid_alpha * 255

    # Construct RGB version of grey-level image
    lung_img_color = np.dstack((lung_img, lung_img, lung_img))

    # Construct a colour image to superimpose
    color_mask = np.zeros((covid_arr.shape[0], covid_arr.shape[1], 3))
    red255 = 255
    green255 = 0
    blue255 = 0
    red_value = red255 / 255.0
    green_value = green255 / 255.0
    blue_value = blue255 / 255.0
    # Red block
    color_mask[:, :, 0] = np.where(covid_img > 0, red_value, 0)
    # Green block
    color_mask[:, :, 1] = np.where(covid_img > 0, green_value, 0)
    # Blue block
    color_mask[:, :, 2] = np.where(covid_img > 0, blue_value, 0)

    # Convert the input image and color mask to Hue Saturation Value (HSV) colorspace
    img_hsv = color.rgb2hsv(lung_img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * covid_alpha

    img_masked = color.hsv2rgb(img_hsv)

    # Resize back to original shape
    target_size = max(img_masked.shape)
    ratio = float(target_size) / max(original_width, original_height)
    new_size = tuple([int(x * ratio) for x in [original_width, original_height]])

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
        
    img_masked = img_masked[wl:target_size-wr, ht:target_size-hb, :]
    img_ = Image.fromarray((img_masked * 255).astype(np.uint8))
    try:
        img_ = img_.resize((original_height, original_width), Image.LANCZOS)
    except:
        img_ = img_.resize((original_height, original_width), Image.NEAREST)

    imsave(output_path, np.array(img_).astype(np.uint8))

    thumb_size = 200, 200
    img_.thumbnail(thumb_size, Image.ANTIALIAS)
    img_.save(output_path[:-4] + "_thumbnail" + ".png", format="png")

    # save nii with original shape
    # lung_img = lung_img[wl:target_size-wr, ht:target_size-hb]
    # img_ = Image.fromarray(lung_img)
    # img_ = img_.resize((original_height, original_width), Image.LANCZOS)
    # arr = np.array(img_)
    # arr = arr * (lung_arr.max() - lung_arr.min()) + lung_arr.min()
    # header = lung.header
    # nii = nib.Nifti1Image(np.transpose(arr, axes=[1, 0]).astype(np.int16), affine=None)  # header=header
    # nib.save(nii, output_path.replace('.png', '.nii'))


def resize_keep_ratio(img, target_size):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # im.thumbnail(new_size, Image.ANTIALIAS)
    try:
        im = img.resize(new_size, Image.NEAREST)
        #im = img.resize(new_size, Image.LANCZOS)
    except:
        im = img.resize(new_size, Image.NEAREST)
    return im


def pad_image(img, load_size, pad_value=-1024):
    old_size = img.size
    target_size = load_size

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
        ]
    )


def test_heart_regression(heart, heartarea):
    state = torch.load("./checkpoints/heartregression.pth")
    ww = state.get('ww')
    wl = state.get('wl')
    n_class = state.get('n_class')
    input_feature = state.get('input_feature')

    weight = state.get('weight', False)
    net = UNetRFull(n_channels=1, n_classes=n_class, model_parallelism=False, args=input_feature, isHeart=True)
    net.load_state_dict(weight)

    nii = nib.load(heart)
    arr = np.transpose(np.array(nii.dataobj)[:, :, 0, 0], axes=[1, 0])
    header = nii.header
    pixel_spacing = header.get_zooms()

    # Resize input according to pixel size
    max_pixsize = 0.9355469
    target_resize = pixel_spacing[0] / max_pixsize * 512
    
    input_ = Image.fromarray(arr)
    input_ = resize_keep_ratio(input_, target_resize)

    img_pad = pad_image(input_, 512)
    input_ = np.array(img_pad(input_))

    # save resized input for debugging purpose
    # nii_np = np.transpose(input_[:, :, np.newaxis], axes=[1, 0, 2])
    # nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
    # nib.save(nii, lung + '_resize.nii')

    # Normalization
    realMinHU = np.amin(input_)
    if realMinHU > 100:
        wl = (realMinHU + 1024) + wl

    minHU = wl - (ww / 2)
    maxHU = wl + (ww / 2)

    x = np.clip(input_, minHU, maxHU)
    input_norm = (x - minHU) / (maxHU - minHU)

    # print(heartarea, sex, age)
    input_feat = torch.from_numpy(np.array([[heartarea]], dtype=np.float32))
    input_img = torch.from_numpy(input_norm[np.newaxis, np.newaxis]).float()
    #input_img = input_img.cuda()

    net.eval()
    # print(input_feat)
    # plt.imshow(input_img[0,0].cpu().detach().numpy())
    # plt.show()
    with torch.no_grad():
        reg_pred = net(input_img, input_feat)
    reg_pred = reg_pred.cpu()
    net.cpu()
    input_img.cpu()
    input_feat.cpu()
    net, input_img, input_feat = None, None, None
    result = reg_pred.item()
    torch.cuda.empty_cache()
    return result

if __name__ == '__main__':
    lung = "./output/2020.04.10_CHEST AP00001_lung.png.nii"
    covid = "./output/2020.04.10_CHEST AP00001_covid.png.nii"

    color_overlay(lung, covid, covid + '.png')
