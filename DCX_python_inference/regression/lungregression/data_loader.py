import os
import torch.utils.data as data
import PIL
from PIL import Image
import nibabel as nib
import pydicom
import numpy as np
import torch
import torchvision.transforms as transforms
from math import ceil, floor


def CreateDataset(opt):
    dataset = None
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset


class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.out_channels = opt.output_nc

        self.A_paths = [opt.test_input]
        self.standardization_norm = opt.standardization
        self.profnorm = opt.profnorm

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        arr, pixel_spacing, self.a_min_val, self.a_max_val, original_input_shape, sex, age, default_sex_age = self.read_input(A_path, self.opt)
        print("arr min max: ", np.min(arr), np.max(arr))
        if self.opt.input_min_max:
            self.a_min_val, self.a_max_val = self.opt.input_min_max.split(',')
            self.a_min_val, self.a_max_val = int(self.a_min_val), int(self.a_max_val)

        A = Image.fromarray(arr)
        print("A image from array min max: ", np.min(np.array(A)), np.max(np.array(A)))
        # params = get_params(self.opt, A.size)
        eps = 1e-10
        if self.opt.label_nc == 0:
            print('A.size: ', A.size)
            A = self.resize_keep_ratio(A, self.opt.loadSize) # in here the min max value of the image is changed
            print('A min max value: ', np.min(A), np.max(A))
            print("A.size after resize and keep ratio: ", A.size)
            print('loadSize: ', self.opt.loadSize)
            print('pad_value: ', int(self.a_min_val))
            img_pad = self.pad_image(A, self.opt.loadSize, pad_value=int(self.a_min_val))
            A_ = np.array(img_pad(A))
            print('A_ min max value: ', np.min(A_), np.max(A_))
            print("A_.shape: ", A_.shape)
            print("int self.a_min_val: ", int(self.a_min_val))

        if self.standardization_norm:
            # normalized_a = (a_arr - self.a_mean) / self.a_std
            normalized_a = (A_ - A_.mean()) / A_.std()
        elif self.profnorm:
            mean, std = A_.mean(), A_.std()
            print("mean and std:", mean, std)
            print(type(mean), type(std))
            A_neg2std = np.where(A_ < mean - (2*std), mean - (2*std), A_)
            print('a neg2std min, max:', A_neg2std.min(), A_neg2std.max())
            print('a neg2std type:', type(A_neg2std))
            percentile0, percentile99 = np.percentile(A_neg2std, 0), np.percentile(A_neg2std, 99)
            print('percentile0, percentile99:', percentile0, percentile99)
            normalized_a = (A_ - percentile0) / ((percentile99 - percentile0) + eps)
            print('normalized_a.min(), normalized_a.max():')
            print(normalized_a.min(), normalized_a.max())

        else:
            #print("linear_min_max")
            # normalized_a = (a_arr - a_arr.min()) / (a_arr.max() - a_arr.min())
            print("no Profnorm: self.a_min_val, self.a_max_val: ", self.a_min_val, self.a_max_val)
            print("A_.min(), A_.max(): ", A_.min(), A_.max())
            normalized_a = (A_ - self.a_min_val) / ((self.a_max_val - self.a_min_val) +eps)

            print('after norm min, max:', normalized_a.min(), normalized_a.max())

        # denorm = lambda x: (x * (self.a_max_val - self.a_min_val)) + self.a_min_val
        # denorm_a = denorm(normalized_a)
        print('normalized_a.shape', normalized_a.shape)
        save_img = normalized_a[np.newaxis, np.newaxis, ...]
        save_img = np.transpose(save_img, axes=[3, 2, 1, 0])
        print('norm_nii_np.shape', save_img.shape)
        nii = nib.Nifti1Image(save_img.astype(np.int16), affine=None)
        nii.header['pixdim'] = 0.18517382812499997
        #nib.save(nii, "denorm_img" + '.nii')


        to_tensor = transforms.ToTensor()
        normalized_a = normalized_a.astype(np.float32)
        normalized_a = to_tensor(normalized_a)

        normalized_b = inst_tensor = feat_tensor = 0

        gt_ratio_pad = []

        input_dict = {'label': normalized_a, 'inst': inst_tensor, 'image': normalized_b,
                      'feat': feat_tensor, 'path': A_path, 'gt_3channels': gt_ratio_pad,
                      'mean': A_.mean(), 'std': A_.std(), 'pixel_spacing': pixel_spacing,
                      'original_input_shape': original_input_shape, 'arr': arr, 'sex': sex,
                      'age': age, 'default_sex_age': default_sex_age}
        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

    def resize_keep_ratio(self, img, target_size):
        old_size = img.size  # old_size[0] is in (width, height) format

        ratio = float(target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        try:
            # im = img.resize(new_size, Image.LANCZOS) # this one changes the min max value of the image
            im = img.resize(new_size, Image.NEAREST)
        except:
            im = img.resize(new_size, Image.NEAREST)
        return im

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

        print("pad values: ", wl, ht, wr, hb)

        return transforms.Compose(
            [
                transforms.Pad((wl, ht, wr, hb), fill=pad_value),
                # transforms.ToTensor(),
            ]
        )

    def histogram_normalization(self, arr):
        print('calling histogram_normalization')
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

            #print(cdf)
            #print(a_norm[0:5])
            #print(arr_histnorm[0:5])


            return arr_denorm[:, :, 0]
        except:
            return arr
    def read_input(self, path, opt):
        from test import remove_private_tags
        _, file_extension = os.path.splitext(path)

        # set default sex and age
        sex = opt.sex
        age = opt.age
        pixel_spacing = opt.pixel_spacing
        default_sex_age = False

        if file_extension.lower() == '.dcm' or file_extension.lower() == '.dicom':
            # dicom data
            dicom = pydicom.dcmread(path, force=True)

            count = 0
            if sex is None or sex == '':
                try:
                    sex = dicom.PatientSex
                except:
                    sex = 'F'
                    default_sex_age = True
                if sex =='':
                    sex = 'F'
                    default_sex_age = True
            if age is None:
                try:
                    age = dicom.PatientAge
                    try:
                        if isinstance(age, list):
                            age = int(age[0].lower().replace('y', ''))
                        else:
                            age = int(age.lower().replace('y', ''))
                    except:
                        age = 50
                        default_sex_age = True
                except:
                    age = 50
                    default_sex_age = True
            # default_sex_age = False if count==2 else True

            #dicom = remove_private_tags(dicom, path)
            
            try:
                arr = dicom.pixel_array
            except:
                dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                arr = dicom.pixel_array
            # arr = np.transpose(np.array(arr, dtype=np.int16), axes=[1, 0])
            #arr = np.array(arr, dtype=np.int16)
            arr = np.array(arr, dtype=np.int32)
            if dicom.PhotometricInterpretation == 'MONOCHROME1':
                mean = np.mean(arr)
                std_dev = np.std(arr)
                threshold = mean + 2 * std_dev
                replacement_value = mean + 2 * std_dev
                arr = np.where(arr > threshold, replacement_value, arr)
                arr = arr * -1 + arr.max()
            original_input_shape = arr.shape

            if pixel_spacing is None:
                try:
                    # assert dicom.SpatialResolution == dicom.ImagerPixelSpacing[0]
                    # assert dicom.SpatialResolution == dicom.ImagerPixelSpacing[1]
                    #assert dicom.ImagerPixelSpacing[0] == dicom.ImagerPixelSpacing[1]
                    #pixel_spacing = dicom.ImagerPixelSpacing[:2]
                    #print("data_loader pixel_spacing", pixel_spacing)
                    pixel_spacing = dicom.PixelSpacing[:2]
                    assert pixel_spacing[0] == pixel_spacing[1]
                except:
                    try:
                        #assert dicom.PixelSpacing[0] == dicom.PixelSpacing[1]
                        #pixel_spacing = dicom.PixelSpacing[:2]
                        pixel_spacing = dicom.PixelSpacing[:2]
                        assert pixel_spacing[0] == pixel_spacing[1]
                    except:
                        pixel_spacing = [0.18, 0.18]

        elif file_extension.lower() == '.nii':
            nii = nib.load(path)
            nii_shape = len(np.array(nii.dataobj).shape)

            if nii_shape == 2:
                arr = np.transpose(np.array(nii.dataobj).astype(np.int16), axes=[1, 0])
            elif nii_shape == 3:
                arr = np.transpose(np.array(nii.dataobj).astype(np.int16), axes=[2, 1, 0])[0, :, :]
            elif nii_shape == 4:
                arr = np.transpose(np.array(nii.dataobj).astype(np.int16), axes=[3, 2, 1, 0])[0, 0, :, :]

            header = nii.header
            if os.path.exists(os.path.join(os.path.dirname(path), 'stat.csv')):
                import pandas as pd
                df = pd.read_csv(os.path.join(os.path.dirname(path), 'stat.csv'))
                age = (df[df['id'] == os.path.basename(path)[:-4] + '_gt1']['age'] * 100).item()
                sex = (df[df['id'] == os.path.basename(path)[:-4] + '_gt1']['sex']).item()
                sex = 'M' if sex else 'F'
            if sex is None:
                sex = 'F'
            if age is None:
                age = 50
            if pixel_spacing is None:
                pixel_spacing = header.get_zooms()

            original_input_shape = np.transpose(arr, axes=[1, 0]).shape

        else:
            try:
                im = Image.open(path).convert('L')
                arr = np.array(im)
            except FileNotFoundError:
                print("File Not Found")
                arr = np.zeros((512, 512))
            except PIL.UnidentifiedImageError:
                print("Unidentified Image Error")
                arr = np.zeros((512, 512))
            except:
                print("Unidentified Error")
                arr = np.zeros((512, 512))
            finally:
                original_input_shape = arr.shape

            if sex is None:
                sex = 'F'
            if age is None:
                age = 50
            if pixel_spacing is None:
                pixel_spacing = [0.18, 0.18]

        ## Histogram Normalization
        if opt.hn:
            print('using histogram normalization')
            arr = self.histogram_normalization(arr)
        ## Histogram Normalization
        print('arr shape', arr.shape)

        return arr, pixel_spacing, arr.min(), arr.max(), original_input_shape, sex, age, default_sex_age

