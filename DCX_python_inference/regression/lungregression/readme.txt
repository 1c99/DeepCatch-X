python inference_lung_volumetry.py

checkpoints = weight directory
input = input directory
output = output directory

Input filename is specified in the main function of inference_lung_reg.py
After the inference, lung image will be saved in the output directory in 2048x2048 size. Lung volumetry result will be printed on the console.

Sample:
AN_ID_20210526104509_1.dcm Lung Volume: 3.44L
AN_ID_20210526112758_1.dcm Lung Volume: 4.26L
