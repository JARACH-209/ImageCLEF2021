#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf

from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# In[2]:


# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)

# tf.config.gpu.set_per_process_memory_growth(True)


# In[3]:


model_name = '3d_image_classification_normalized'
random_state=1


# ## Directory structure:
# * **Dataset Directory**: $HOME/Datasets/ImageCLEF/
# * extracted .nii.gz files are in a Dataset subfolder in Dataset Directory
# * metadata file is in the Dataset Directory

# In[4]:


home = os.path.expanduser('~')
base = os.path.join(home, 'Datasets', 'ImageCLEF')

dataset_dir = os.path.join(base, 'Dataset')

label_path = os.path.join(base, '4231cdb3-af46-4674-be08-95b904a62093_TrainSet_metaData.csv')
df = pd.read_csv(label_path)

df.head(10)


# In[5]:


filenames = df['FileName'].tolist()
num_samples = len(filenames)

labels = df['TypeOfTB'].to_numpy() - 1
stratify = df['TypeOfTB'].to_numpy() - 1
num_classes = labels.max() + 1

labels = tf.one_hot(labels, depth=num_classes)

idxs = [i for i in range(num_samples)]

train_idxs, val_idxs = train_test_split(idxs, test_size=0.2, random_state=random_state, stratify=df['TypeOfTB'].to_numpy() - 1)

del num_classes, idxs, stratify


# In[6]:


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")

    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    
#     print(f"Shape: {img.shape}")
    desired_depth = 64
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


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


# In[7]:


def read_fn(file_names, labels, file_idxs):
    for i, idx in enumerate(file_idxs):
        img_path = os.path.join(dataset_dir, file_names[idx])
        processed = process_scan(img_path)
        
        image = tf.convert_to_tensor(processed, dtype=tf.float16)
        image = image[..., np.newaxis]
        y = labels[idx]
        
        yield image, y


# In[8]:


def train_f():
    fn = read_fn(filenames, labels, train_idxs)
    ex = next(fn)
    yield ex
    
def val_f():
    fn = read_fn(filenames, labels, val_idxs)
    ex = next(fn)
    yield ex


# In[9]:


train_batch_size = 1

train_dataset = tf.data.Dataset.from_generator(
                    train_f,
                    (tf.float16, tf.float32),
                    (tf.TensorShape([512, 512, 64, 1]), tf.TensorShape([5])))

train_dataset = train_dataset.repeat(None)
train_dataset = train_dataset.batch(train_batch_size)
train_dataset = train_dataset.prefetch(2)


val_batch_size = 1

val_dataset = tf.data.Dataset.from_generator(
                    val_f,
                    (tf.float16, tf.float32),
                    (tf.TensorShape([512, 512, 64, 1]), tf.TensorShape([5])))
val_dataset = val_dataset.repeat(None)
val_dataset = val_dataset.batch(val_batch_size)
val_dataset = val_dataset.prefetch(2)

train_steps = int(len(train_idxs) / (train_batch_size * 2))
val_steps = int(len(val_idxs) / (val_batch_size * 3))
# val_steps = 64


# In[10]:


def get_model(width=512, height=512, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, strides=(1, 1, 1), activation="relu", name='3D_64_1')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, strides=(1, 1, 1), activation="relu", name='3D_64_2')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, strides=(1, 1, 1), activation="relu", name='3D_128_1')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=128, kernel_size=3, strides=(1, 1, 1), activation="relu", name='3D_128_2')(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Conv3D(filters=64, kernel_size=3, strides=(1, 1, 1), activation="relu")(x)
#     x = layers.MaxPool3D(pool_size=2)(x)
#     x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(rate=0.8)(x)

    outputs = layers.Dense(units=5, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name=f"{model_name}")
    return model


# Build model.
model = get_model(width=512, height=512, depth=64)
model.summary()


# In[11]:


initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    f"{model_name}.h5", save_best_only=True
)

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=10)

# Train the model, doing validation at the end of each epoch
epochs = 48
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[checkpoint_cb, early_stopping_cb]
)


# In[ ]:


with open(f'{model_name}_history.pkl', 'wb') as fh:
    pickle.dump(history.history, fh)

