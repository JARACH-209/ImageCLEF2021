import tensorflow as tf
import nibabel as nib
import pandas as pd
import numpy as np
import skimage
import time
import glob
import sys
import cv2
import os
import pickle
from scipy import stats
from skimage import measure
from shutil import move
from tensorflow import keras
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from sklearn.model_selection import train_test_split

model_name = sys.argv[1]

home = os.path.expanduser('~')

model_base = os.path.join(home, 'ImageCLEF2021', f'{model_name}.h5')
myModel = keras.models.load_model(model_base)

base = os.path.join(home, 'Datasets', 'ImageCLEF')

test_slices_dir = os.path.join(base, 'Test_Axial_Slice_Masks_300')

ct_predictions1 = {}
ct_predictions2 = {}

try:
    aggr = sys.argv[2]
except IndexError as err:
    pass

ldir = os.listdir(os.path.join(base, 'Testset'))
ldir.sort()

for img_name in ldir:
    img_name_split = img_name.split('.')
    base_img_name = img_name_split[0]
    if img_name_split[1] != 'nii' or img_name_split[2] != 'gz':
        continue
    
    all_slices = []
    
    slice_names = glob.glob(os.path.join(test_slices_dir, f'{base_img_name}_*'))
    
    for slice_name in slice_names:
        slice_path = os.path.join(test_slices_dir, slice_name)
        im_slice = Image.open(slice_path)
        im_slice = im_slice.resize((331, 331), resample=Image.LANCZOS)
        im_slice = np.asarray(im_slice)[..., np.newaxis]
        m_slice = np.append(im_slice, im_slice, axis=2)
        m_slice = np.append(m_slice, im_slice, axis=2)
#         print(f'{m_slice.shape}')
        all_slices.append(m_slice)
        
    all_slices = np.array(all_slices, dtype=np.float32)
    
    predictions = []
    for slice_i in all_slices:
        predictions.append(myModel(slice_i[np.newaxis, ...]))
    
#     predictions = myModel(all_slices)
    predictions = np.array(predictions)
    
    predictions = np.squeeze(predictions, axis=1)
    
    predictions1 = np.argmax(predictions, axis=1)

    num_slices = predictions.shape[0]
    sl_min = int((2/9) * num_slices + (14/9))
    sl_max = int((7/54) * num_slices + (49/54))

    predictions = predictions[sl_min:-sl_max, :]
    
    predictions2 = np.argmax(predictions, axis=1)
    
    ct_predictions1[f'{img_name}.nii.gz'] = stats.mode(predictions1).mode.item() + 1
    ct_predictions2[f'{img_name}.nii.gz'] = stats.mode(predictions2).mode.item() + 1
    
    print(f'{img_name}', end='\r')
    
    break
