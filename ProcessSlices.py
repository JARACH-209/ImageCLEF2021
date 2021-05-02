import nibabel as nib
import pandas as pd
import numpy as np
import time
import cv2
import os


home = os.path.expanduser('~')
directory = os.path.join('Datasets', 'ImageCLEF')
filename = '4231cdb3-af46-4674-be08-95b904a62093_TrainSet_metaData.csv'
path = os.path.join(home, directory, filename)

# df = pd.read_csv(path)
# df.head(10)

base_path = os.path.join(home, directory)
base_dataset = os.path.join(base_path, 'Dataset')
base_processed = os.path.join(base_path, 'Images')

image_vol_list = os.listdir(base_dataset)
total, total_time = 0, 0

for i, image in enumerate(image_vol_list):
    
    image_path = os.path.join(base_dataset, image)
    base_name = image.split('.')[0]
    
    image_vol = nib.load(image_path)
    
    try:
        vol_arr = image_vol.get_fdata()
    except OSError as err:
        print(err)
        continue
    
    t0 = time.clock()
    for slice in range(150, 350+1):
        image_slice = vol_arr[slice, :, :]
        slice_resized = cv2.resize(image_slice, 
                                   (512, 512), 
                                   interpolation=cv2.INTER_NEAREST_EXACT)
        
        new_name = f'{base_name}_{slice}.png'
        save_path = os.path.join(base_processed, new_name)
        
        cv2.imwrite(save_path, slice_resized)
        
    total += 1
    
    total_time += time.clock() - t0
    print(f"""
    Processed: {base_name}
    Total Time Elapsed: {total_time}\n""")
    
print(f"Processed {total} .nii.gz images in {total_time} seconds.")