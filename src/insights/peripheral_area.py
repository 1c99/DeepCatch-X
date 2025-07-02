#!/usr/bin/env python
# coding: utf-8

# In[34]:

import os
os.environ['MPLBACKEND'] = 'Agg'
# Try to set matplotlib backend if available (Nuitka-safe)
try:
    import matplotlib
    if hasattr(matplotlib, 'use'):
        matplotlib.use('Agg')
except (ImportError, AttributeError):
    import matplotlib  # Import anyway if the try block failed
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import cv2
import os

def printInfo(image, name=""):
    
    print("--------- {} ---------".format(name))        
    print("{} , {}".format(type(image), image.dtype))
    print(image.shape)
    print("sum / min / max / mean / std")
    print(np.sum(image)) # 4194304
    print(np.min(image)) # -1029
    print(np.max(image)) # -878
    print(np.mean(image)) # -1021.9964039325714
    print(np.std(image)) # 13.835177613499017
    if image.ndim == 2:
        x, y = np.where(image)
        print(np.sum(x))
        print(np.sum(y))
        
    print()

# input : nifti file path
def find_contours(nii_path):
    
    print(nii_path)
    #exit(0)
    
    # 1) array from input nifti image
    img = nib.load(nii_path).get_fdata()
    print(type(img))
    print(img.shape) # (2048, 2048, 1, 1)
    img = img.reshape(img.shape[0], img.shape[1])
    print(type(img)) # numpy.memmap
    print(img.shape) # (2048, 2048)
    printInfo(img, "img - find_contours()")
    #exit(0)
    img = np.array(np.transpose(img)).astype(np.uint8)
    #img = img.astype(np.uint8)
    printInfo(img, "transpose")
    #exit(0)
    cv2.imwrite(nii_path[:-4]+'.png', img) # 17066541_20150307_fake_0.png
    #exit(0)
    #img_color = cv2.imread(path[:-4]+'.png')
    img_color = cv2.imread(nii_path[:-4]+ '.png')
    print(type(img_color)) # numpy.ndarray
    print(img_color.shape) # (2048, 2048, 3)    
    #exit(0)
    
    # 2) generate binary mask image 
    mask = np.where(img > 0,1,0).astype(np.uint8) # 0 or 1 : binary
    printInfo(mask, "mask - find_contours()")
    #exit(0)
    cv2.imwrite(nii_path[:-4]+'_mask(binary)'+'.png', mask*255) # 17066541_20150307_fake_0_binary.png
    #exit(0)
        
    # 3) find contours from binary mask - using opencv2 findContours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(type(contours)) # <class 'tuple'>
    #print(len(contours)) # 2
    #print(type(contours[0])) # <class 'numpy.ndarray'>
    #exit(0)
    
    maskTemp = mask
    #maskTemp = mask * 0 # 
    cv2.drawContours(maskTemp, contours, -1, 255)
    cv2.imwrite(nii_path[:-4]+'_contour'+'.png', maskTemp)
    printInfo(maskTemp, "maskTemp")
    #exit(0)
    
    
    # 4) sorted contours by length and select top2 
    num = len(contours)
    con = []
    for i in range(num):
        con.append(len(contours[i])) # contours[i] is set of contour pixel
    top2 = sorted(con, reverse=True)[:2]
    print(top2) # [4639, 4472]
    
    lung_contours = []
    for i in range(num):
        if con[i] in top2:
            lung_contours.append(contours[i])
   
    print(type(lung_contours)) # <class 'list'>
    print(lung_contours) # [array(), array()] = [con[0], con[1]]
    #exit(0)

    return img_color, lung_contours
    # output : image, selected contours


# In[36]:


# input : contours top2
def center_point(lung_contours):

    # define start/end(=min/max) point at y
    contour_x_st = sorted(lung_contours[0], key=lambda x: x[0][0])[0]
    contour_x_st2 = sorted(lung_contours[1], key=lambda x: x[0][0])[0]
    
    contour_x_ed = sorted(lung_contours[0], key=lambda x: x[0][0])[-1]
    contour_x_ed2 = sorted(lung_contours[1], key=lambda x: x[0][0])[-1]
    
    contour_y_st = sorted(lung_contours[0], key=lambda x: x[0][1])[0]
    contour_y_st2 = sorted(lung_contours[1], key=lambda x: x[0][1])[0]
    
    contour_y_ed = sorted(lung_contours[0], key=lambda x: x[0][1])[-1]
    contour_y_ed2 = sorted(lung_contours[1], key=lambda x: x[0][1])[-1]
    
    center_x = (min(contour_x_st[0][0],contour_x_st2[0][0]) + max(contour_x_ed[0][0],contour_x_ed2[0][0]))//2
    center_y = (min(contour_y_st[0][1],contour_y_st2[0][1]) + max(contour_y_ed[0][1],contour_y_ed2[0][1]))//2
    
    # center x = (min(contour1 min x, contour2 min x) + max(contour1 max x, contour2 max x))//2
    # center y = (min(contour1 min y, contour2 min y) + max(contour1 max y, contour2 max y))//2
    
    print("{} , {}".format(center_x, center_y)) # 1074 , 914
    #exit(0)
        
    
    return [center_x, center_y]
    # output : center point [x,y]


# In[37]:


# input : contour (length 1)
def center_point_one(lung_contours):
    
    # define start/end(=min/max) point at y
    contour_x_st = sorted(lung_contours[0], key=lambda x: x[0][0])[0]    
    contour_x_ed = sorted(lung_contours[0], key=lambda x: x[0][0])[-1]
    
    contour_y_st = sorted(lung_contours[0], key=lambda x: x[0][1])[0]    
    contour_y_ed = sorted(lung_contours[0], key=lambda x: x[0][1])[-1]
    
    center_x = (contour_x_st[0][0] + contour_x_ed[0][0])//2
    center_y = (contour_y_st[0][1] + contour_y_ed[0][1])//2
    
    # center x = (contour min x + contour max x)//2
    # center y = (contour min y + contour max y)//2
    
  
    return [center_x, center_y]
    # output : center point [x,y]


# In[38]:


# input : center point, contours
def full_mask(center, lung_contours):
    
    # 0) load center point
    center_x, center_y = center
    contour1 = []
    contour2 = []
    
    print("{} , {} - full_mask()".format(center_x, center_y))
    #exit(0)
    
    print("len(lung_contours): {}".format(len(lung_contours))) # len(lung_contours): 2
    #exit(0)    
    if len(lung_contours) == 2:
        # 1) concatenate if 2 contours
        lung_contours = np.concatenate((lung_contours[0], lung_contours[1]), axis=0)

    print("lung_contours length: {}".format(len(lung_contours))) # lung_contours length: 9111
    #exit(0)
    

    # 2) split and save 2 contours by center point x value
    for i in lung_contours:
        if i[0][0] > center_x:
            contour1.append(i[0])
        else:
            contour2.append(i[0])
    contour1 = np.array(contour1)
    contour2 = np.array(contour2)
    
    print(len(contour1)) # 4472
    print(len(contour2)) # 4639
    #exit(0)
    
    # define start/end(=min/max) point at y
    contour_y_st = sorted(contour1, key=lambda x: x[1])[0]
    contour_y_st2 = sorted(contour2, key=lambda x: x[1])[0]
    contour_y_ed = sorted(contour1, key=lambda x: x[1])[-1]
    contour_y_ed2 = sorted(contour2, key=lambda x: x[1])[-1]
    
    print("{} , {} / {} , {}".format(contour_y_st, contour_y_ed, contour_y_st2, contour_y_ed2))
    # [1318  235] , [1619 1593] / [898 247] , [ 594 1571]
    #exit(0)
   
    # 3) fill trapezoid binary mask (airway area) - using opencv FillPoly
    a,b,c = img.shape[0], img.shape[1], img.shape[2]
    print(img.shape) # (2048, 2048, 3)
    #exit(0)
    mask = np.zeros((a, b, c), dtype="uint8")
    print(mask.shape) # (2048, 2048, 3)
    #exit(0)
    
    points = np.array([contour_y_st, contour_y_st2, contour_y_ed2, contour_y_ed])
    cv2.fillPoly(mask, pts=[points], color=(255,255,255)) 
    printInfo(mask, "polygon fill");
    #exit(0) 

    # 4) fill contour binary mask - using opencv FillPoly
    cv2.fillPoly(mask, pts =[lung_contours], color=(255,255,255))
    printInfo(mask, "lung_contours fill");
    #exit(0)
    
    return mask
    # output : filled mask


# In[39]:


# input : image, filled mask, center point, percentage
def bitwise_mask(img, mask, center_point, percent):
    
    # 1) save contour of filled mask - using opencv findContours
    mask_contour, hierarchy = cv2.findContours(mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 2) for mask contour points do : 
    #        new points : int( (1-percent)*center point + percent*mask contour points )
    mask_points = mask_contour[0]
    print(len(mask_points)) # 4802
    #exit(0)
    
    new_points = []
    for i in range(len(mask_points)):
        x = center_point
        y = mask_points[i][0]
        m = percent
        n = 1-percent
        new_points.append([int((n*a+m*b)) for a,b in zip(x,y)])    
    
    print(len(new_points)) # 4802
    #exit(0)
    
    # 3) filled percent mask - using opencv FillPoly
    a,b,c = img.shape[0], img.shape[1], img.shape[2]
    mask = np.zeros((a,b,c), dtype='uint8')
    
    new_points = np.array(new_points)
    mask = cv2.fillPoly(mask, [new_points], (255,255,255))
    printInfo(mask[:,:,0], "mask[:,:,0]")
    #exit(0)
    
    # 4) copy image and generate percent masked image - using opencv bitwise_and
    masked = cv2.bitwise_and(img, mask)
    printInfo(masked[:,:,0], "masked[:,:,0]")
    #exit(0)
    
    return mask, masked
    # output : percent mask, percent masked image (lung or vascular)


# In[40]:


# input :  image path and percent masked image, save path and string percentage 
def save_mask(path, mask, save_path, idx):
    
    # 1) load nifti image, header, affine
    nii = nib.load(path)
    header = nii.header
    affine = nii.affine
    
    # 2) generate percent masked image to nifti image (applied header, affine) 
    new_nii = nib.Nifti1Image(np.transpose(mask), affine, header)
    
    # 3) save nifti image at save path
    file_name = os.path.basename(path)[:-4] + '_' + str(idx) + '.nii'
    nib.save(new_nii, save_path +'/'+ file_name)


# In[45]:


# Commented out to prevent execution when imported
if False:  # Original script execution
    #path = r'C:\Users\user\central_mask\test_control-final36'
    path = r'.\test_control-final36'
    print("path: ".format(path))
    #images = getImages(path)

    for i in os.listdir(path):
        lung_contour = []
        nii_path = path+'\\'+str(i)
        if not nii_path.endswith(".nii"):
            continue
        
        img, lung_contours = find_contours(path+'\\'+str(i))
        for c in lung_contours:
            if len(c) > 1000:
                lung_contour.append(c)
    
    '''
    #      
    #lung_contour = lung_contour[0][0]
    lung_contour = lung_contour[0]
    print(lung_contour)  
    print(len(lung_contour)) # 4639    
    exit(0)
    '''
        
                
    if len(lung_contour) == 1:
        center = center_point_one(lung_contour)
        masks = full_mask(center, lung_contour)
    else:
        center = center_point(lung_contour)
        masks = full_mask(center, lung_contour)

    cv2.imwrite(nii_path[:-4]+'_fullMask'+'.png', masks)
    printInfo(masks, 'masks')
    #exit(0)

    save_path = path + '\\' + str(i)[:-4]
    print("save_path: {}".format(save_path))    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    #exit(0)
    for p in [0.5, 0.7]: # not range
        print("p = {}".format(p))
        mask, masked = bitwise_mask(img, masks, center, p)
        print(type(mask))
        print(type(masked))
        print("save_mask: {}".format(path+'\\'+str(i)+str(p)))
        #save_mask(path+'\\'+str(i), mask[:,:,0], save_path, str(p))

    #exit(0)


# In[47]:

# Commented out to prevent execution when imported
if False:
    path = r'test_ipah_final34'
    for i in os.listdir(path):
        lung_contour = []
        img, lung_contours = find_contours(path+'\\'+str(i))
        
        for c in lung_contours:
            if len(c) > 1000:   # constant
                lung_contour.append(c)
                
        if len(lung_contour) == 1:
            center = center_point_one(lung_contour)
            masks = full_mask(center, lung_contour)
        else:
            center = center_point(lung_contour)
            masks = full_mask(center, lung_contour)

        save_path = path + '\\' + str(i)[:-4]
        os.mkdir(save_path)
        for p in [0.5, 0.7]:
            mask, masked = bitwise_mask(img, masks, center, p)
            save_mask(path+'\\'+str(i), mask[:,:,0], save_path, str(p))
            
            
    



