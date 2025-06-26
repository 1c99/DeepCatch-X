# Written by Seowoo Lee (2023-09-28)
# E-mail : seowoo.md@gmail.com

import torch, os
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2
import pydicom
import nibabel as nib
import pdb
from math import ceil, floor

checkpoint_name = "cxr_cxr_effb0_512_3_b4_f32_lbce-epoch=303-valid_loss=0.4251-valid_iou_score=0.0000-ct_valid_rocauc=0.0000-cxr_valid_rocauc=0.8547.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def save_rgb_dicom_safe(rgb_arr, input_filename, output_dicom_file):
    from pydicom.dataset import Dataset
    from pydicom.uid import generate_uid
    # Saving to DICOM
    ds = Dataset()
    # Set required DICOM tags
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.PhotometricInterpretation = 'RGB'
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'RGB']
    ds.SamplesPerPixel = 3
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0  # Unsigned
    ds.PlanarConfiguration = 0  # Interleaved
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

    # Set other relevant DICOM attributes
    #print(rgb_arr.shape)
    #pdb.set_trace()
    
    ds.Rows, ds.Columns, _ = rgb_arr.shape
    ds.PixelData = rgb_arr.tobytes()

    # Set byte order and VR
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Generate unique SOP Instance UID and Series Instance UID
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    Image.fromarray(rgb_arr).save(output_dicom_file[:-4]+".png", format="png")
    # Save the DICOM dataset to a file
    pydicom.filewriter.dcmwrite(output_dicom_file, ds)


import glob, ntpath, argparse
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


if __name__ == '__main__':

    # Load model
    aux_params = dict(
        pooling='avg',
        dropout=0.2,
        activation='softmax',
        classes=2,
    )
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        aux_params=aux_params,
        activation='sigmoid'
    )
    model.eval()
    model = model.to(device)

    model_path = os.path.join(os.path.dirname(__file__),"model",checkpoint_name)
    weight = torch.load(model_path, map_location='cpu')
    weight.weights_only = True
    new_state_dict = {}
    for key in weight.keys():
        new_state_dict[key.replace("model.","")] = weight[key]

    model.load_state_dict(state_dict=new_state_dict, strict=True)

    
    #Single image
    
    #multiple input
    #'''
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--input_dir', type=str, help='input data path', default = './input')
    parser.add_argument('--output_dir', type=str, help='outfolder path', default = './output')
    args = parser.parse_args()
    
    #input_dir = "./image"
    #output_dir = "./output"
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    paths = glob.glob(input_dir + '\*.dcm')
    paths = glob.glob(input_dir + '\*.dicom')
    paths = glob.glob(input_dir + '\*')
    #pdb.set_trace()
    #'''
    for path in paths:
        try:
            # load image
            # Image는 512,512 해상도의 이미지만 사용 가능합니다.
            # 이외의 해상도 이미지의 경우 512x512로 변환하여 사용하여야 합니다.
            _, file_extension = os.path.splitext(path)
            image_name = path_leaf(path)
            output_dict = {}
            if file_extension.lower() == '.dcm' or file_extension.lower() == '.dicom':
                # dicom data
                print(path)
                dicom = pydicom.dcmread(path, force=True)
                #dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                #image_arr = dicom.pixel_array
                try:
                    image_arr = dicom.pixel_array
                except:
                    dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                    image_arr = dicom.pixel_array = dicom.pixel_array
                # arr = np.transpose(np.array(arr, dtype=np.int16), axes=[1, 0])
                image_arr = np.array(image_arr, dtype=np.int16)
                pixel_spacing = dicom.PixelSpacing
                if dicom.PhotometricInterpretation == 'MONOCHROME1':
                    image_arr = image_arr * -1 + image_arr.max()

                # normalize image_arr value to 0-255
                image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min()) * 255
                # make sure image_arr has 3 channels
                image_arr = np.stack((image_arr,)*3, axis=-1).astype(np.int16)
            elif file_extension.lower() == '.nii' or file_extension.lower() == '.gz':
                nii = nib.load(path)
                header = nii.header
                pixel_spacing = header.get_zooms()
                #pdb.set_trace()
                nii_shape = len(np.array(nii.dataobj).shape)

                if nii_shape == 2:
                    arr = np.transpose(np.array(nii.dataobj).astype(np.int16), axes=[1, 0])
                elif nii_shape == 3:
                    arr = np.transpose(np.array(nii.dataobj).astype(np.int16), axes=[2, 1, 0])[0, :, :]
                elif nii_shape == 4:
                    arr = np.transpose(np.array(nii.dataobj).astype(np.int16), axes=[3, 2, 1, 0])[0, 0, :, :]
                # normalize image_arr value to 0-255
                image_arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                # make sure image_arr has 3 channels
                image_arr = np.stack((image_arr,)*3, axis=-1).astype(np.int16)
            else:
                image_arr = Image.open(os.path.join(os.path.dirname(__file__),"image",image_name)).convert("RGB")
                image_arr = np.array(image_arr)
            # pdb.set_trace()

            # input 이미지 해상도 512x512가 아닌 경우 resize
            original_width, original_height = image_arr.shape[0], image_arr.shape[1]
            if not (image_arr.shape[0] ==512 and image_arr.shape[1] ==512):
                s = max(image_arr.shape[0:2])
                f = np.zeros((s, s, 3), np.float64)
                ax, ay = (s - image_arr.shape[1]) // 2, (s - image_arr.shape[0]) // 2
                f[ay:image_arr.shape[0] + ay, ax:ax + image_arr.shape[1], ...] = image_arr
                image_arr = cv2.resize(f, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)

            image_arr = np.clip(image_arr / 255, 0, 1).astype(np.float32)

            # RGB image인지 확인
            assert image_arr.shape[-1] == 3

            # 512x512 이미지인지 확인
            assert image_arr.shape[0] == 512 and image_arr.shape[1] == 512

            # last channel로 변환
            img = np.moveaxis(image_arr,0, -1)
            img = np.moveaxis(img,1,0)

            assert img.max() <=1 and img.min() >=0
            assert img.dtype == np.float32

            img = torch.Tensor(img).to(device)

            # batch dimension 추가
            img = torch.unsqueeze(img, 0)

            with torch.no_grad():
                #scripted_model = torch.jit.load("model/anynet_1280x720_b1.traced.mipx")
                #scripted_model.eval()
                #scripted_model.float()
                #out = scripted_model(img)
                out = model(img)

                batch_size = 1
                for batch in range(0,batch_size):

                    # Emphysema binary classification probability
                    emphysema_prob = out[1][batch][1].cpu().numpy()
                    print("Emphysema binary classification : %.1f%%"%(emphysema_prob*100))
                    output_dict['emphysema_prob'] = emphysema_prob*100
                    print("EMPHYSEMA PROB : ", emphysema_prob*100)
                    if emphysema_prob >= 0.0262980695:
                        print("EMPHYSEMA : POSITIVE")
                        output_dict['desc'] = "EMPHYSEMA : POSITIVE"
                    else:
                        print("EMPHYSEMA : NEGATIVE")
                        output_dict['desc'] = "EMPHYSEMA : NEGATIVE"

                    # Emphysema mask
                    emphysema_mask = out[0][batch]
                    emphysema_mask = emphysema_mask.cpu().numpy()
                    emphysema_mask = np.squeeze(emphysema_mask)
                    emphysema_mask = np.moveaxis(emphysema_mask,1,0)
                    
                    #pdb.set_trace()
                    # Resize back to original shape
                    """
                    target_size = max(emphysema_mask.shape)
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
                    emphysema_ori = emphysema_mask[wl:target_size-wr, ht:target_size-hb]
                    img_ = Image.fromarray(emphysema_ori)
                    #img_ = img_.resize((original_height, original_width), Image.LANCZOS)
                    img_ = img_.resize((original_height, original_width), Image.Resampling.NEAREST)
                    """
                    emphysema_ori = np.array(emphysema_mask)
                    #emphysema_ori = resize_ori(emphysema_mask, image_arr.shape[0], image_arr.shape[1])
                    os.makedirs(os.path.join(output_dir, 'pred_empysemafloatmask'), exist_ok=True)
                    nii = nib.Nifti1Image(np.transpose(emphysema_ori, axes=[1, 0]), affine=None)
                    nib.save(nii, os.path.join(output_dir,'pred_empysemafloatmask', image_name+"_emphysema_mask.nii.gz"))
                    
                    emphysema001 = np.where(emphysema_ori >= 0.001, 1, 0)
                    count_ones = np.sum(emphysema001)
                    print(original_width)
                    print(original_height)
                    area = count_ones * (pixel_spacing[0] / (512.0) / max(original_width, original_height)) * (pixel_spacing[1] / (512.0) / max(original_width, original_height)) / 100.0;
                    print("EMPHYSEMA AREA : ", area)
                    
                    os.makedirs(os.path.join(output_dir, 'pred_empysemamask001'), exist_ok=True)
                    nii = nib.Nifti1Image(np.transpose(emphysema001, axes=[1, 0]), affine=None)
                    nib.save(nii, os.path.join(output_dir,'pred_empysemamask001', image_name+"_emphysema_mask001.nii.gz"))
                    
        except Exception as e:
            # Handling all kinds of error
            output_dict["success"] = False
            output_dict["message"] = "Error " + str(e)

            import sys, traceback
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            print(e)
        # Prevent memory leak, dealloc variables
        import gc
        gc.collect()

        #print_gpu_stats("before predict_snuhtb() empty_cache()")
        torch.cuda.empty_cache()
        #print_gpu_stats("after predict_snuhtb() empty_cache()")
        with open(output_dir + '\output_emphysema.csv','a') as fd:
            fd.write("\n" + image_name + ": " + str(output_dict))