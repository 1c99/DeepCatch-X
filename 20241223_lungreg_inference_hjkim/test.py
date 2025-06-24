import os
#below is for aorta
#from pix2pixHD_model_new import BuildModel
from pix2pixHD_model import BuildModel
from lungregression_model import UNetRFull
from data_loader import CustomDatasetDataLoader
import torch
torch.cuda.current_device()
torch.cuda._initialized = True
import numpy as np
import nibabel as nib
from skimage.measure import label
from skimage.io import imsave
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


def remove_netG(model, old_state_dict):

    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[11:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
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

    opt.use_gpu = device_num
    device = 'cuda:' + str(device_num)
    opt.cuda0 = device
    opt.cuda1 = device

    b_min_val, b_max_val = opt.output_min_max.split(',')
    b_min_val, b_max_val = int(b_min_val), int(b_max_val)

    os.makedirs("%s/" % 'output', exist_ok=True)
    model = create_model_v2(opt)

    is_xray = True

    for i, data in enumerate(dataset):
        is_xray = True
        if is_xray:
            sex = data['sex']
            age = data['age']
            default_sex_age = data['default_sex_age']
            model.eval()
            model.float()

            with torch.no_grad():
                generated = model(data['label'].type(torch.cuda.FloatTensor))

                # torch.jit.trace(model, (data['label'].type(torch.cuda.FloatTensor)),check_trace=True).save( "new_xray2lung.mipx")

                # load traced model and make inference
                # traced_model = torch.jit.load("new_xray2lung.mipx")
                #
                # trace_gen = traced_model(data['label'].type(torch.cuda.FloatTensor))

            generated_np = generated.detach().cpu().numpy()

            if opt.standardization:
                denormalize_gen = generated_np * data['std'].cpu().numpy() + data['mean'].cpu().numpy()
            else:
                denormalize_gen = generated_np * (b_max_val - b_min_val) + b_min_val

            # apply threshold
            denormalize_gen = np.where(denormalize_gen < opt.threshold, -1024, denormalize_gen)
            denormalize_gen_mask = np.where(denormalize_gen[0, 0] < opt.threshold, 0, 1)
            # print('denormalize_gen.shape', denormalize_gen.shape)
            # print('denormalize_gen_mask.shape', denormalize_gen_mask.shape)
            # print('denormalize_gen_mask min, max', np.min(denormalize_gen_mask), np.max(denormalize_gen_mask))
            # np.save('denormalize_gen_mask.npy', denormalize_gen_mask)


            if opt.get_covid_area:
                if np.sum(denormalize_gen_mask.flatten()) == 2048*2048:
                    denormalize_gen_mask = np.zeros_like(denormalize_gen_mask)

            if opt.get_lung_area:  # find connected region
                denormalize_gen_mask = get_biggest_connected_region(denormalize_gen_mask)
                # print('denormalize_gen_mask.shape', denormalize_gen_mask.shape)
                connected_lung = np.where(denormalize_gen_mask, denormalize_gen[0, 0], -1024)
                denormalize_gen = connected_lung[np.newaxis, np.newaxis]
                # print('denormalize_gen.shape', denormalize_gen.shape)

            # denorm_mask_test = denormalize_gen_mask[np.newaxis, np.newaxis]

            # calculate area
            original_width = data['original_input_shape'][0].numpy()[0]
            original_height = data['original_input_shape'][1].numpy()[0]
            ratio = float(2048) / max(original_width, original_height)

            pixel_size_resize_w = data['pixel_spacing'][0].numpy()[0] / ratio
            pixel_size_resize_h = data['pixel_spacing'][1].numpy()[0] / ratio

            area = np.sum(denormalize_gen_mask.flatten())
            area = area * pixel_size_resize_w * pixel_size_resize_h / 100
            # print('area', area)
            # print('pixel_size_resize_w', pixel_size_resize_w)
            # print('pixel_size_resize_h', pixel_size_resize_h)

            nii_np = np.transpose(denormalize_gen, axes=[3, 2, 1, 0])

            nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
            nii.header['pixdim'] = pixel_size_resize_w
            nib.save(nii, opt.test_output + '.nii')


            if opt.save_input:
                arr, _, _, _, _, _, _, _ = data_loader.dataset.read_input(opt.test_input, opt)
                eps = 1e-10
                input_img = ((arr - arr.min()) / ((arr.max() - arr.min()) + eps)) * 255
                path = os.path.join(os.path.dirname(opt.test_output), os.path.basename(opt.test_input) + '.png')
                imsave(path, input_img.astype(np.uint8))
                
                img_ = Image.fromarray((input_img).astype(np.uint8))

        else:
            area = -1

    model.cpu()
    torch.cuda.empty_cache()
    return area, is_xray, sex, age, default_sex_age, original_width, original_height

def resize_keep_ratio(img, target_size):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

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

def test_lung_regression(lung, lungarea, sex, age):
    
    # state = torch.load("./checkpoints/lungregression.pth", map_location='cuda:0')
    state = torch.load("./checkpoints/lungregression.pth")
    ww = state.get('ww')
    wl = state.get('wl')
    n_class = state.get('n_class')
    input_feature = state.get('input_feature')
    weight = state.get('weight', False)
    net = UNetRFull(n_channels=1, n_classes=n_class, model_parallelism=False, args=input_feature)
    net.load_state_dict(weight)

    nii = nib.load(lung)
    arr = np.transpose(np.array(nii.dataobj)[:, :, 0, 0], axes=[1, 0]) # [2048, 2048]
    header = nii.header
    pixel_spacing = header.get_zooms()


    # Resize input according to pixel size
    max_pixsize = 0.319333
    target_resize = pixel_spacing[0] / max_pixsize * 2048
    input_ = Image.fromarray(arr)
    input_ = resize_keep_ratio(input_, target_resize)
    img_pad = pad_image(input_, 2048)
    input_ = np.array(img_pad(input_))
    #

    # save resized input for debugging purpose
    # nii_np = np.transpose(input_[:, :, np.newaxis], axes=[1, 0, 2])
    # nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None)
    # nib.save(nii, lung + '_resize.nii')

    # Normalization
    realMinHU = np.amin(input_) #-1024
    if realMinHU > 100:
        wl = (realMinHU + 1024) + wl

    minHU = wl - (ww / 2) # -1025
    maxHU = wl + (ww / 2) # -775


    x = np.clip(input_, minHU, maxHU)
    input_norm = (x - minHU) / (maxHU - minHU)

    sex = 0 if 'f' in sex[0].lower() else 1
    age = float(age)/100

    input_feat = torch.from_numpy(np.array([[lungarea, sex, age]], dtype=np.float32))
    input_img = torch.from_numpy(input_norm[np.newaxis, np.newaxis]).float()
    input_img = input_img.cpu()
    # pdb.set_trace()

    net.eval()
    with torch.no_grad():
        device_hj = next(net.parameters()).device
        reg_pred = net(input_img, input_feat)
    
    result = reg_pred.item()
    return result
