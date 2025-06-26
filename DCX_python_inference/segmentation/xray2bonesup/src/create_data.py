"""
initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021
===modified: 03-September-2021
"""

import torch
import numpy as np
import numpy as np
import pydicom
import os
import glob
import nibabel as nib
import PIL
from PIL import Image

from nibabel.imageglobals import LoggingOutputSuppressor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import torchvision.transforms as transforms

class SliceData(Dataset):
    """
    Custom Pytorch Dataset
    __init__    : Define data path
    __len__     : Define total number of dataset
    __getitem__ : Get DICOM image
    """

    def __init__(self, root_A, preprocess):
        """
        root_A : path to data A
        preprocess ": pre-processing, see data.DataProcessing
        """
        self.preprocess = preprocess
        """
        examples_A : Full File Path to data A (root_A + Folder_Name_A + File_Name_A)
        example_files_A : File Name of data A (ex: 10001.dcm)
        example_folder_A : Folder Name of data A
        """
        # input
        self.examples_A = sorted(glob.glob(os.path.join(root_A, "*.gz")) + glob.glob(os.path.join(root_A, "*.dcm")) + glob.glob(os.path.join(root_A, "*.dicom"))+ glob.glob(os.path.join(root_A, "*.nii")))
        self.example_files_A = [os.path.basename(file_path) for file_path in self.examples_A]

    def get_extension(self, path):
        root, ext = os.path.splitext(path)
        if root.lower().endswith('.nii') and ext.lower() == '.gz':
            return '.nii.gz'
        elif ext.lower() == '.nii':
            return '.nii'
        else:
            return ext        
        
    def load_image(self, file_path):
        if file_path.endswith('.gz') or file_path.endswith('.nii'):
            img_arr = nib.load(file_path).get_fdata().T
        elif file_path.endswith('.dcm') or file_path.endswith('.dicom'):
            ds = pydicom.dcmread(file_path)
            img_arr = ds.pixel_array
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        #print(file_path, ' | ', img_arr.dtype)
        print(img_arr.dtype)
        return resize_to_square(np.array(img_arr, dtype=img_arr.dtype))
    


    def __len__(self):
        return len(self.examples_A)

    def __getitem__(self, i):
        #요기요기!!
        """
        ds_A : Full Dicom Information of A (Header + Image)
        img_A : DICOM image A
        file_name_A : Full File Path to data A
        folder_name_A : Folder Name of data A
        """
        
        with LoggingOutputSuppressor():
            file_path_A = self.examples_A[i]
            
            arr, height_A, width_A = self.load_image(file_path_A)
            
            file_name_A = self.example_files_A[i]

        return self.preprocess(arr, file_path_A, file_name_A, height_A, width_A)
    
    def pad_image(self, img, target_size, pad_value=-1024):
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
            ]
        )
        
class DataPreProcessing():

    def __init__(self, method='min-max'):
        """
            method : Normalization method
                - 'min-max' : min-max normalization
                - 'z-score' : z-score standardization
        """
        self.method = method

    def __call__(self, img_A, file_path_A, file_name_A, height_A, width_A):
        
        img_A = np.expand_dims(img_A, 0) # Create Channel Dimension [512,512] -> [1,512,512]
        print(img_A.shape)
        print(img_A.dtype)
        max_val_A = float(img_A.max())
        print("max_val_A", max_val_A)
        min_val_A = float(img_A.min())
        print("min_val_A", min_val_A)
        #print(max_val_A, min_val_A)
        eps = 1e-10
        if self.method == 'min-max':

            max_val = np.max(img_A)
            min_val = np.min(img_A)

            rescale_slope = max_val - min_val
            rescale_intercept = min_val

            img_A = (img_A - rescale_intercept) / rescale_slope
            
        elif self.method == 'z-score':
            # affine = np.eye(4)
            # nii_image = nib.Nifti1Image(img_A[0,:,:].T, affine)
            # nib.save(nii_image, './test4.nii.gz')
            
            min_A = np.min(img_A)
            
            if min_A < 0 :
                img_A = img_A - min_A 
            
            mean_val = np.mean(img_A)
            std_val = np.std(img_A)

            rescale_slope = std_val
            rescale_intercept = mean_val

            img_A = (img_A - rescale_intercept) / rescale_slope
            # affine = np.eye(4)
            # nii_image = nib.Nifti1Image(img_A[0,:,:].T, affine)
            # nib.save(nii_image, './test5.nii.gz')
            
        elif self.method == 'percentile':
            mean, std = np.mean(img_A), np.std(img_A)
            img_A_neg2std = np.where(img_A < mean - (2*std), mean - (2*std), img_A)
            percentile0, percentile99 = np.percentile(img_A_neg2std, 0), np.percentile(img_A_neg2std, 99)
            img_A = (img_A - percentile0) / ((percentile99 - percentile0) + eps)
            rescale_slope = ((percentile99 - percentile0) + eps)
            rescale_intercept = percentile0
            
            # affine = np.eye(4)
            # nii_image = nib.Nifti1Image(img_A[0,:,:].T, affine)
            # nib.save(nii_image, './test1.nii.gz')
            
        else:
            rescale_slope = 1
            rescale_intercept = 0

            img_A = (img_A - rescale_intercept) / rescale_slope
            
        img_A_torch = torch.from_numpy(img_A)
        print("torch shape : ", img_A_torch.shape)
        return img_A_torch, rescale_slope, rescale_intercept, file_path_A, file_name_A, height_A, width_A,max_val_A, min_val_A
    


from math import ceil, floor

def resize_keep_ratio(img, target_size):
    old_size = img.size  # old_size is in (width, height) format

    # Calculate the ratio and new size to keep the aspect ratio
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image while keeping the aspect ratio
    try:
        resized_img = img.resize(new_size, Image.LANCZOS)
    except Exception:
        resized_img = img.resize(new_size, Image.NEAREST)
        
    return resized_img

        
def resize_to_square(array, output_size=(512,512)):
    """
    Resize a 2D array to square with zero-padding and then resize to 512x512 using bilinear interpolation.

    Args:
    array (numpy.ndarray): The input 2D array.
    output_size (tuple): The desired output size (width, height).

    Returns:
    numpy.ndarray: The resized square array.
    """
    # 배열의 원래 높이와 너비 확인
    height, width = array.shape

    # 더 큰 쪽의 길이를 기준으로 정사각형 크기 결정
    square_size = max(width, height)

    # 새로운 정사각형 배열 생성 (배경은 0)
    new_array = np.zeros((square_size, square_size), dtype=array.dtype)

    # 원본 배열을 새 배열의 중앙에 복사
    new_array[(square_size - height) // 2:(square_size - height) // 2 + height,
    (square_size - width) // 2:(square_size - width) // 2 + width] = array

    # 배열 리사이징을 위한 비율 계산
    row_zoom = output_size[0] / square_size
    col_zoom = output_size[1] / square_size

    # 양선형 보간법을 사용하여 리사이징
    resized_array = zoom(new_array, (row_zoom, col_zoom), order=1)
    return resized_array, height, width

def create_datasets(args):

    val_data = SliceData(
        root_A=args.val_A_data_path,
        preprocess=DataPreProcessing(method='z-score'),
    )
    
    return val_data

def create_data_loaders(args):

    val_data = create_datasets(args)

    display_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )

    print('NUMBER OF DATASET:', len(display_loader.dataset))
    return display_loader

