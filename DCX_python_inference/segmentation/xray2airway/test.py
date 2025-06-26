import os
#below is for aorta
#from pix2pixHD_model_new import BuildModel
from pix2pixHD_model import BuildModel
#from lungregression_model import UNetRFull
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
            print('saved_nii ', opt.test_output + '.nii')


            if opt.save_input:
                arr, _, _, _, _, _, _, _ = data_loader.dataset.read_input(opt.test_input, opt)
                eps = 1e-10
                input_img = ((arr - arr.min()) / ((arr.max() - arr.min()) + eps)) * 255
                path = os.path.join(os.path.dirname(opt.test_output), os.path.basename(opt.test_input) + '.png')
                imsave(path, input_img.astype(np.uint8))
                print(path, "saved")
                
                img_ = Image.fromarray((input_img).astype(np.uint8))

        else:
            area = -1

    model.cpu()
    torch.cuda.empty_cache()
    return area, is_xray, sex, age, default_sex_age, original_width, original_height

def resize_keep_ratio(img, target_size):
    print("resize and keep ratio target size: ", target_size)
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

    print('-------------- regression part --------------')
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
    print(np.amax(input_)) #= -711
    print(np.amin(input_)) #= -1024
    print('target size', target_resize) #  1334.9168094182348
    input_ = resize_keep_ratio(input_, target_resize)
    print("input_ shape", input_.size) # (1335, 1335)
    print(np.amax(input_)) # -713
    print(np.amin(input_)) # -1024
    img_pad = pad_image(input_, 2048)
    input_ = np.array(img_pad(input_))
    
    print(np.amax(input_)) # -713
    print(np.amin(input_)) # -1024
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

    print('minHU', minHU, 'maxHU', maxHU)


    x = np.clip(input_, minHU, maxHU)
    input_norm = (x - minHU) / (maxHU - minHU)
    print('input_norm', np.amax(input_norm), np.amin(input_norm)) # 1.0 0.004

    sex = 0 if 'f' in sex[0].lower() else 1
    print('sex: ', sex)
    print('age: ', age)
    age = float(age)/100

    print(lungarea, sex, age) # 492.48332911555406 1 0.84
    input_feat = torch.from_numpy(np.array([[lungarea, sex, age]], dtype=np.float32))
    input_img = torch.from_numpy(input_norm[np.newaxis, np.newaxis]).float()
    input_img = input_img.cuda()

    print("input feat", input_feat.shape) # torch.Size([1, 3])
    print("input img", input_img.shape) # torch.Size([1, 1, 2048, 2048])
    # pdb.set_trace()

    # print('input_img', input_img.shape) # torch.Size([1, 1, 2048, 2048])
    # print('input_feat', input_feat.shape) # torch.Size([1, 3])
    net.eval()
    with torch.no_grad():
        reg_pred = net(input_img, input_feat)


    net.cpu()
    input_img.cpu()
    input_feat.cpu()

    torch.jit.trace(net, (input_img, input_feat)).save("./checkpoints/lungregression_cpu.mipx")
    reg_pred = reg_pred.cpu()
    print('reg_pred', reg_pred.item()) # reg_pred 3443.654541015625')
    scripted_net = torch.jit.load("./checkpoints/lungregression_cpu.mipx")
    # check the scripted model device
    print(next(scripted_net.parameters()).is_cuda) # False
    scripted_net.eval()
    reg_pred2 = scripted_net(input_img, input_feat)
    # print('reg value', reg_pred2.item()) # reg value 3443.654541015625
    # --------------------------------------------------
    net.cpu()
    input_img.cpu()
    input_feat.cpu()
    net, input_img, input_feat = None, None, None
    result = reg_pred.item()
    torch.cuda.empty_cache()
    return result