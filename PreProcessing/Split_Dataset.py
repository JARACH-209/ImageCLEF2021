from sklearn.model_selection import train_test_split
from shutil import move

import pandas as pd
import time
import sys
import os

home = os.path.expanduser('~')
directory = os.path.join('Datasets', 'ImageCLEF')

filename = '4231cdb3-af46-4674-be08-95b904a62093_TrainSet_metaData.csv'
path = os.path.join(home, directory, filename)
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
  
base_path = os.path.join(home, directory)
base_dataset = os.path.join(base_path, 'Dataset')
base_processed = os.path.join(base_path, 'Slices_Crop')

try:
    for dataset_type in ['train', 'test']:
        for i in range(1, 5+1):
            new_dir = os.path.join(base_processed, dataset_type, f'{i}')
            os.makedirs(new_dir)
except FileExistsError as err:
    print(f'{err}\nAborting...')
    sys.exit()
    
total, total_time = 0, 0

def move_files(idxs_list, dset_type):
    base_new = os.path.join(base_processed, dset_type)
    
    count = 0
    
    for idx in idxs_list:
        image_class = str(labels[idx])
        
        from_dir = os.path.join(base_processed, image_class)
        to_dir = os.path.join(base_new, image_class)

        base_fname = fnames[idx].split('.')[0]
        
        for i in range(150, 350+1):
            image_name = f'{base_fname}_{i}.png'
            
            origin = os.path.join(from_dir, image_name)
            destination = os.path.join(to_dir, image_name)
            
            try:
                move(origin, destination)
                print(f'{origin} --> {destination}')
                count += 1
            except FileNotFoundError as err:
                print("LOG ERROR: {err}")
                
    return count

t0 = time.monotonic()
train_count = move_files(train_idxs, 'train')    

test_count = move_files(test_idxs, 'test')
t_total = time.monotonic() - t0

print(f"\n\n{'='*125}\n")
print(f"""
Moved {train_count} training images and {test_count} test images in {t_total: 0.2f} seconds.
\n\n""")