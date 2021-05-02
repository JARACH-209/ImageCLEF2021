import nibabel as nib
import pandas as pd
import numpy as np
import time
import sys
import os

from PIL import Image

home = os.path.expanduser('~')
directory = os.path.join('Datasets', 'ImageCLEF')

filename = '4231cdb3-af46-4674-be08-95b904a62093_TrainSet_metaData.csv'
path = os.path.join(home, directory, filename)
df = pd.read_csv(path)

fnames = df['FileName'].to_list()
labels = df['TypeOfTB'].to_list()

image_label_map = dict()

for i, fname in enumerate(fnames):
    image_label_map[fname] = str(labels[i])
    

base_path = os.path.join(home, directory)
base_dataset = os.path.join(base_path, 'Dataset')
base_processed = os.path.join(base_path, 'Slices')


try:
    for i in range(1, 5+1):
        new_dir = os.path.join(base_processed, f'{i}')
        os.makedirs(new_dir)
except FileExistsError as err:
    print(f'{err}\nAborting...')
    sys.exit()
    
image_vol_list = os.listdir(base_dataset)
total, total_time = 0, 0

for i, image in enumerate(image_vol_list):
    image_path = os.path.join(base_dataset, image)
    base_name = image.split('.')[0]
    
    try:
        image_vol = nib.load(image_path)
        vol_arr = image_vol.get_fdata()
    except OSError as err:
        print(err)
        continue
    
    t0 = time.monotonic()
    for ct_slice in range(150, 350+1):
        image_slice = vol_arr[:, ct_slice, :]
        image_slice = Image.fromarray(image_slice)
        
        slice_resized = image_slice.resize(
                                (224, 224),
                                Image.LANCZOS)
        
        slice_resized = slice_resized.rotate(angle=90) # rotates by angle degress counter clockwise

        slice_resized = slice_resized.convert(mode='L')
        
        new_name = f'{base_name}_{ct_slice}.png'
        save_path = os.path.join(
                        base_processed, 
                        image_label_map[image], 
                        new_name)
        
        slice_resized.save(save_path)
        
        print(f'Created: {save_path}')
        
    total += 1
    
    total_time += time.monotonic() - t0
    print(f"""\n
    Processed: {base_name}
    Total Time Elapsed: {total_time: .2f}s\n""")

print(f"\n\n{'='*75}\n")
print(f"Processed {total} .nii.gz images in {total_time: .2f} seconds.")