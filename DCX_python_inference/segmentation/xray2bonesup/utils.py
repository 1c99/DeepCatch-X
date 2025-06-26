import os
import numpy as np
import pydicom
import nibabel as nib
import pathlib
import glob
import cv2
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Invalid value for VR UI")
warnings.filterwarnings("ignore", message="'Bits Stored' value")

def decompress_dicom(input_dir, output_dir,handler):
    # List all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            dicom_path = os.path.join(root, file)
            if dicom_path.endswith(".nii") or dicom_path.endswith(".gz"):
                continue
            else:
                try:
                    # Read the compressed DICOM file
                    dataset = pydicom.dcmread(dicom_path)

                    if 'BitsStored' in dataset and dataset.file_meta.TransferSyntaxUID.is_compressed:
                        if dataset.BitsStored != 16:
                            print(f"Adjusting 'Bits Stored' from {dataset.BitsStored} to 16 for file: {dicom_path}")
                            dataset.BitsStored = 16
                            
                    # If the file is already uncompressed, you can skip the decompression
                    if dataset.file_meta.TransferSyntaxUID.is_compressed:
                        # Decompress the pixel data
                        dataset.decompress(handler)
                    
                    # Ensure the output directory exists
                    os.makedirs(output_dir, exist_ok=True)

                    filename = pathlib.Path(dicom_path).name
                    output_path = os.path.join(output_dir, filename)

                    # Save the decompressed DICOM file
                    dataset.save_as(output_path)
                    print(f'DICOM file saved to {filename}')
                except Exception as e:
                    print(f"Failed to process {dicom_path}: {e}")        

def convert_nii_to_dcm(org_path, ng_path, save_path):
    org_list = os.listdir(org_path)
    ng_list = os.listdir(ng_path)
    
    for i in range(len(org_list)):
        # Read the original DICOM file
        org_dicom = os.path.join(org_path, org_list[i])
        dicom = pydicom.dcmread(org_dicom)

        # Get the original image
        original_image = dicom.pixel_array

        # Check if the DICOM image is LUT inverted
        if hasattr(dicom, 'PhotometricInterpretation') and dicom.PhotometricInterpretation == 'MONOCHROME1':
            lut_inverted = True
        else:
            lut_inverted = False

        # Load the corresponding NIfTI image
        ng_file = os.path.join(ng_path, ng_list[i])
        ng_img = nib.load(ng_file).get_fdata()
        ng_img = np.array(ng_img).T

        # Get the display range of the original DICOM
        dicom_min = np.min(original_image)
        dicom_max = np.max(original_image)

        # Get the original DICOM dtype
        dicom_dtype = original_image.dtype

        # Normalize NIfTI image to match the original DICOM display range
        ng_img_min = np.min(ng_img)
        ng_img_max = np.max(ng_img)

        # Normalize to 0-1 range
        ng_img = (ng_img - ng_img_min) / (ng_img_max - ng_img_min)
        # Scale to original DICOM range
        ng_img = ng_img * (dicom_max - dicom_min) + dicom_min

        # If the original DICOM is LUT inverted, invert the NIfTI image as well
        if lut_inverted:
            ng_img = ng_img.max() - ng_img
            dicom.PhotometricInterpretation = 'MONOCHROME1'
        else:
            dicom.PhotometricInterpretation = 'MONOCHROME2'

        # Convert to the original DICOM dtype
        ng_img = ng_img.astype(dicom_dtype)

        # Ensure ng_img is a C-contiguous array
        ng_img = np.ascontiguousarray(ng_img)

        # Update the DICOM's PixelData and other attributes
        dicom.PixelData = ng_img.tobytes()
        dicom.Rows, dicom.Columns = ng_img.shape
        dicom.BitsStored = dicom[0x0028, 0x0101].value
        dicom.HighBit = dicom[0x0028, 0x0102].value
        dicom.PixelRepresentation = dicom[0x0028, 0x0103].value

        # Save the modified DICOM file
        dicom.save_as(os.path.join(save_path, org_list[i]))

def plot_dicom_fourier_transform(dicom_file_path):
    
    # Read the DICOM file
    dicom_image = pydicom.dcmread(dicom_file_path)
    image_data = dicom_image.pixel_array

    if dicom_image.PhotometricInterpretation == "MONOCHROME1":
        # Invert the image
        image_data = np.max(image_data) - image_data

    # Perform Fourier transform
    fourier_transform = np.fft.fftshift(np.fft.fft2(image_data))
    magnitude_spectrum = np.log(np.abs(fourier_transform) + 1)

    # Plot the original image and the Fourier transformed image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_data, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(magnitude_spectrum, cmap='gray')
    axes[1].set_title('Fourier Transform')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

############################################################################

def z_score_normalization(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

def process_and_compare_histograms(dicom_path, nii_path, method='zscore', lower_percentile=1, upper_percentile=99):
    # Load the DICOM image
    dicom_dataset = pydicom.dcmread(dicom_path)
    imageA = dicom_dataset.pixel_array

    # Load the NIfTI image
    nii_dataset = nib.load(nii_path)
    imageB = nii_dataset.get_fdata().T

    # Ensure both images are of the same dimensionality
    if imageA.shape != imageB.shape:
        raise ValueError("The dimensions of the DICOM and NIfTI images do not match.")
    
    if method == 'percentile':
        # Adjust histogram using percentile method
        minA, maxA = np.percentile(imageA, lower_percentile), np.percentile(imageA, upper_percentile)
        minB, maxB = np.percentile(imageB, lower_percentile), np.percentile(imageB, upper_percentile)
        B_normalized = (imageB - minB) / (maxB - minB)
        B_normalized = np.clip(B_normalized, 0, 1)
        adjusted_imageB = B_normalized * (maxA - minA) + minA
    elif method == 'zscore':
        # Adjust histogram using Z-score normalization
        adjusted_imageB = z_score_normalization(imageB) * np.std(imageA) + np.mean(imageA)
    else:
        raise ValueError("Invalid normalization method specified.")
    
    # Display histograms and images for comparison
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 3, 1)
    plt.hist(imageA.ravel(), bins=256, range=[np.min(imageA), np.max(imageA)], color='blue', alpha=0.5, label='Image A')
    plt.title('Histogram of Image A')
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.hist(imageB.ravel(), bins=256, range=[np.min(imageB), np.max(imageB)], color='green', alpha=0.5, label='Image B')
    plt.title('Histogram of Image B')
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.hist(adjusted_imageB.ravel(), bins=256, range=[np.min(imageA), np.max(imageA)], color='red', alpha=0.5, label='Adjusted Image B')
    plt.title('Histogram of Adjusted Image B')
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.imshow(imageA, cmap='gray')
    plt.title('Image A')

    plt.subplot(3, 3, 5)
    plt.imshow(imageB, cmap='gray')
    plt.title('Image B')

    plt.subplot(3, 3, 6)
    plt.imshow(adjusted_imageB, cmap='gray')
    plt.title('Adjusted Image B')

    plt.tight_layout()
    plt.show()

############################################################################

if __name__ == '__main__':
    input_dir = './datasets/input/0a425edf1164ad0a73e8b092c4cc8b3b.dicom'
    output_dir = './datasets/output/2024_06_17_09_11_39/0a425edf1164ad0a73e8b092c4cc8b3b.nii.gz'
    #handler = 'gdcm'
    #decompress_dicom(input_dir, output_dir, handler)
    #convert_nii_to_dcm(input_dir, output_dir, './datasets/output')
    #plot_dicom_fourier_transform('./datasets/decompressed_dicoms/0b98b21145a9425bf3eeea4b0de425e7.dicom')
    
    process_and_compare_histograms(input_dir, output_dir)