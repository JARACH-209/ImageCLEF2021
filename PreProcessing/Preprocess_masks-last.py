#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from skimage.util.montage import montage2d
import nibabel as nib
import pandas as pd
import numpy as np
import skimage
import time
import sys
import cv2
import os

from skimage import measure
from shutil import move
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from sklearn.model_selection import train_test_split


# In[7]:


home = os.path.expanduser('~')
directory = os.path.join('Datasets', 'ImageCLEF')
filename = '4231cdb3-af46-4674-be08-95b904a62093_TrainSet_metaData.csv'
path = os.path.join(home, directory, filename)

dataset_dir = os.path.join(home, directory, 'Testset')
mask_dir = os.path.join(home, directory, 'masks')

df = pd.read_csv(path)

fnames = df['FileName'].to_list()
labels = df['TypeOfTB'].to_list()

idxs = [i for i in range(len(labels))]

train_idxs, test_idxs = train_test_split(
                            idxs, 
                            test_size=0.2, 
                            random_state=42, 
                            shuffle=True, 
                            stratify=labels)


# In[3]:


# def img_crop_3d(self, image, msk, margin=2, ds=4, multi=False):
# def img_crop_3d(image, msk, margin=2, ds=4, multi=False):
#     if multi:
#         image = image * msk
#         print('image multiplied with msk')

#     msk_z = np.sum(msk, axis=2)
#     msk_y = np.sum(msk, axis=0)

#     img_binary = (msk_z > 0).astype(np.uint8)
#     g = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#     img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, g)
#     contours, hierarchy = cv2.findContours(img_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     area = 0
#     area_max = 0
#     minmax = []
#     for i, c in enumerate(contours):
#         #         area[i] = cv2.contourArea(c)
#         #         print('the area is %d'%area[i])
#         area = cv2.contourArea(c)
#         #         if area_max < area:
#         #             area_max = area
#         #             c_max = c
#         x_min = min(c[:, :, 0])[0] - margin
#         x_max = max(c[:, :, 0])[0] + margin
#         y_min = min(c[:, :, 1])[0] - margin
#         y_max = max(c[:, :, 1])[0] + margin

#         if area > 10:
#             minmax.append([x_min, x_max, y_min, y_max])
#     #     print(minmax)

#     x_min = min(np.array(minmax)[:, 0])
#     x_max = max(np.array(minmax)[:, 1])
#     y_min = min(np.array(minmax)[:, 2])
#     y_max = max(np.array(minmax)[:, 3])

#     msk_y = np.sum(msk, axis=0)

#     img_binary = (msk_y > 0).astype(np.uint8)
#     g = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#     img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, g)
#     contours, hierarchy = cv2.findContours(img_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#     area = 0
#     area_max = 0
#     for i, c in enumerate(contours):
#         #         area[i] = cv2.contourArea(c)
#         #         print('the area is %d'%area[i])
#         area = cv2.contourArea(c)
#         if area_max < area:
#             area_max = area
#             c_max = c

#     z_min = min(c_max[:, :, 0])[0] + margin * ds
#     z_max = max(c_max[:, :, 0])[0] - margin * ds

#     print(x_min, x_max, y_min, y_max, z_min, z_max)

#     img_crop = image[y_min:y_max, x_min:x_max, z_min:z_max]

#     return img_crop


# In[33]:


# name = 'TRN_0068'
# ct = nib.load(os.path.join(dataset_dir, f'{name}.nii.gz'))
# mask = nib.load(os.path.join(mask_dir, f'{name}_mask.nii.gz'))

# ct_np = ct.get_fdata()
# mask_np = mask.get_fdata()

# print(ct_np.shape)
# print(mask_np.shape)


# In[34]:


# img_crop = img_crop_3d(ct_np, mask_np)


# In[3]:


# img_depth = 84

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume, min, max):
    """Normalize the volume"""
    
    level = 250
    window = 50
#     max = int(level+ (window/2))
#     min = int(level- (window/2))
    
#     min = -800
#     max = -300
#     volume[volume < min] = min
#     volume[volume > max] = max
    volume = volume.clip(min, max)

#     volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    '''
    
    level = 1000
    window = 250
    max = level+ (window/2)
    max = level- (window/2)
    
    volume = volume.clip(min, max)
    volume = volume.astype("float32")
    '''
    
    return volume
'''
def intensity_seg(ct_numpy, min=-1000, max=-300):
    clipped = clip_ct(ct_numpy, min, max)
    return measure.find_contours(ct_numpy, 0.95)
'''    
def intensity_seg(ct_numpy, min=-1000, max=-300):
    clipped = ct_numpy.clip(min, max)
    clipped[clipped != max] = 1
    clipped[clipped == max] = 0
    return measure.find_contours(clipped, 0.95)
    
def contour_distance(contour):
    """
    Given a set of points that may describe a contour
     it calculates the distance between the first and the last point
     to infer if the set is closed.
    Args:
        contour: np array of x and y points

    Returns: euclidean distance of first and last point
    """
    dx = contour[0, 1] - contour[-1, 1]
    dy = contour[0, 0] - contour[-1, 0]
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))

def set_is_closed(contour):
    if contour_distance(contour) < 1:
        return True
    else:
        return False
    
def find_lungs(contours):
    """
    Chooses the contours that correspond to the lungs and the body
    First, we exclude non-closed sets-contours
    Then we assume some min area and volume to exclude small contours
    Then the body is excluded as the highest volume closed set
    The remaining areas correspond to the lungs
    Args:
        contours: all the detected contours
    Returns: contours that correspond to the lung area
    """
    body_and_lung_contours = []
    vol_contours = []
    for contour in contours:
        hull = ConvexHull(contour)
       # set some constraints for the volume
        if hull.volume > 2000 and set_is_closed(contour):
            body_and_lung_contours.append(contour)
            vol_contours.append(hull.volume)
    # Discard body contour
    if len(body_and_lung_contours) == 2:
        return body_and_lung_contours
    elif len(body_and_lung_contours) > 2:
        vol_contours, body_and_lung_contours = (list(t) for t in 
                zip(*sorted(zip(vol_contours, body_and_lung_contours))))
        body_and_lung_contours.pop(-1) # body is out!
    return body_and_lung_contours # only lungs left !!!

def create_mask_from_polygon(image, contours):
    """
    Creates a binary mask with the dimensions of the image and
    converts the list of polygon-contours to binary masks and merges them together
    Args:
        image: the image that the contours refer to
        contours: list of contours
    Returns:
    """
    lung_mask = np.array(Image.new('L', image.shape, 0))
    for contour in contours:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', image.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        lung_mask += mask
    lung_mask[lung_mask > 1] = 1  # sanity check to make 100% sure that the mask is binary
    return lung_mask.T  # transpose it to be aligned with the image dims

def create_vessel_mask(lung_mask, ct_numpy, denoise=False):
    vessels = lung_mask * ct_numpy  # isolate lung area
    vessels[vessels == 0] = -1000
    vessels[vessels >= -600] = 1
    vessels[vessels < -600] = 0
#     show_slice(vessels)
    if denoise:
        return denoise_vessels(lungs_contour, vessels)
#     show_slice(vessels)
    return vessels

def euclidean_dist(dx, dy):
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))

def denoise_vessels(lung_contour, vessels):
    vessels_coords_x, vessels_coords_y = np.nonzero(vessels)  # get non zero coordinates
    for contour in lung_contour:
        x_points, y_points = contour[:, 0], contour[:, 1]
        for (coord_x, coord_y) in zip(vessels_coords_x, vessels_coords_y):
            for (x, y) in zip(x_points, y_points):
                d = euclidean_dist(x - coord_x, y - coord_y)
                if d <= 0.1:
                    vessels[coord_x, coord_y] = 0
    return vessels

def show_slice(slice):
    """
    Function to display an image slice
    Input is a numpy 2D array
    """
    plt.figure()
    plt.imshow(slice.T, cmap="gray", origin="lower")

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    
#     print(f"Shape: {img.shape}")
    desired_depth = img_depth
    desired_width = 512
    desired_height = 512
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    
#     img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def show_contour(image, contours, name=None, save=False):
    fig, ax = plt.subplots()
    ax.imshow(image.T, cmap=plt.cm.gray)
    for contour in contours:
        ax.plot(contour[:, 0], contour[:, 1], linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        plt.savefig(name)
        plt.close(fig)
    else:
        plt.show()

def process_scan(path, min=-1000, max=-150):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    
    print(path)
    print(volume.shape)
    
    slices = volume.shape[-1]
    sl_min = int((2/9) * slices + (14/9))
    sl_max = int((7/54) * slices + (49/54))
    
    volume = volume[:, :, sl_min:-sl_max]
    
    volume = np.mean(volume, axis=2)
    
    print(f'Min {sl_min}\tMax {sl_max}')
    
    contour = intensity_seg(volume, min=-1000, max=-300)
    contour = find_lungs(contour)
    
    mask = create_mask_from_polygon(volume, contour)
    
    vessels = create_vessel_mask(mask, volume, lungs_contour=contour, denoise=False)
    
    return volume

def overlay_plot(im, mask):
    plt.figure()
    plt.imshow(im.T, 'gray', interpolation='none')
    plt.imshow(mask.T, 'jet', interpolation='none', alpha=0.5)


# In[4]:


def get_masks(img_name):
    ct_img = nib.load(os.path.join(dataset_dir, img_name))
    ct_numpy = ct_img.get_fdata()

    volume = ct_numpy

    slices = volume.shape[-1]
    sl_min = int((2/9) * slices + (14/9))
    sl_max = int((7/54) * slices + (49/54))
    
    volume = volume[:, :, sl_min:-sl_max]

    slices_arr = []

    for i in range(volume.shape[2]):
        slice_i = volume[:, :, i]
    
        try:
            contours = intensity_seg(slice_i, -1000, -300)

            lungs_contour = find_lungs(contours)
            lung_mask = create_mask_from_polygon(slice_i, lungs_contour)

            vessels_only = create_vessel_mask(lung_mask, slice_i, denoise=False)
    
            slices_arr.append(vessels_only)
        except QhullError as err:
            print(f'{err}')
    
    slices_arr = np.array(slices_arr)
#     slices_arr_mean = np.mean(slices_arr, axis=0)

#     min_t, max_t = 0.033, 0.2current
    min_t, max_t = 0.033, 0.135

#     slices_arr_mean[slices_arr_mean < min_t] = 0
#     slices_arr_mean[slices_arr_mean > max_t] = 1
    slices_arr[slices_arr < min_t] = 0
    slices_arr[slices_arr > max_t] = 1
    
    return slices_arr

# In[6]:


def crop_image(image_array):
    image = Image.fromarray(image_array)
    left, upper, right, lower = 110, 0, 512, 512
    image = image.crop((left, upper, right, lower))
#     plt.imshow
    width, height = image.size
#     print(f'Width: {width}\tHeight: {height}')
    return image


# In[6]:


save_path = os.path.join(home, directory, 'Test_Axial_Slice_Masks_300')

try:
    os.makedirs(save_path)
except FileExistsError as err:
    print(f'{err}')
    print(f'It is recommended you empty the directory before creating masks')


# In[10]:


masks_list = os.listdir(save_path)
img_names = os.listdir(dataset_dir)
img_names.sort()

# black = np.zeros(shape=(512, 100))
# (left, upper, right, lower)

import sys
lower = int(sys.argv[1].strip())
upper = int(sys.argv[2].strip())

#lower = 1
#upper = 100

for name in img_names[lower:upper]:
    name = name.split('.')
    if name[1] != 'nii' or name[2] != 'gz':
        continue
        
    sl_name = f'{name[0]}_Slice_Mask.png'
    
#     if sl_name in masks_list:
#         print(f'EXISTS: {img_save_path}')
#         continue
    
    try:
        slice_masks = get_masks(f'{name[0]}.nii.gz')
        for i in range(slice_masks.shape[0]):
            
            image = crop_image(slice_masks[i, :, :])
            
            img_arr = np.asarray(image)
            black = np.zeros(shape=(512, 100))
            img_arr = np.append(black, img_arr, axis=1)
            
            img_name = f'{name[0]}_Slice_Mask_{i+1}.png'
            img_save_path = os.path.join(save_path, img_name)
#             image = Image.fromarray(slice_masks[i, :, :])
            image = Image.fromarray(img_arr)
#             image = image.resize(size=(331, 331), resample=Image.LANCZOS)
            
            image = image.convert(mode='I')
            image.save(img_save_path)
            print(f'CREATED: {img_save_path}')            
                
#         img_arr = np.asarray(image)


        
#         img_name = f'{name[0]}_Slice_Mask.png'
#         img_save_path = os.path.join(save_path, img_name)
#         plt.imshow(img_arr, cmap='gray')
#         image = Image.fromarray(img_arr)
#         image = image.resize(size=(331, 331), resample=Image.LANCZOS)
#         image_3channel = Image.new('RGB', image.size)
#         image_3channel.paste(image)
        
#         plt.imshow(np.asarray(image_3channel))
#         image = image.convert(mode='I')
#         image.save(img_save_path)
#         print(f'CREATED: {img_save_path}')
#         break
        
    except QhullError as err:
        print(f'ERROR creating: {img_save_path} \n{err}\n')
    


# In[9]:


def move_files(idxs_list, dset_type):
    base_new = os.path.join(save_path, dset_type)
    try:
        for i in range(1, 5+1):
            os.makedirs(os.path.join(base_new, f'{i}'))
    except FileExistsError as err:
        print('Error: Directories already exist. Aborting')
#         sys.exit()
    count = 0
    
    for idx in idxs_list:
        image_class = str(labels[idx])
        
        from_dir = save_path
        to_dir = os.path.join(base_new, image_class)

        base_fname = fnames[idx].split('.')[0]
        
        for i in range(1, 200+2):
            image_name = f'{base_fname}_Slice_Mask_{i}.png'
            
            origin = os.path.join(from_dir, image_name)
            destination = os.path.join(to_dir, image_name)
            
            try:
                move(origin, destination)
                print(f'{origin} --> {destination}')
                count += 1
            except FileNotFoundError as err:
                print(f"LOG ERROR: {err}")
                
    return count

# t0 = time.monotonic()
# train_count = move_files(train_idxs, 'train')    

# test_count = move_files(test_idxs, 'test')
# t_total = time.monotonic() - t0

# print(f"\n\n{'='*125}\n")
# print(f"""
# Moved {train_count} training images and {test_count} test images in {t_total: 0.2f} seconds.
# \n\n""")

