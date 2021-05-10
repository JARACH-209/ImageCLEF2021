#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# In[2]:


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], False)


# In[3]:


model_name = 'NASNet_Transfer'


# In[4]:


home = os.path.expanduser('~')
base = os.path.join('Datasets', 'ImageCLEF', 'Slices_Crop')

train_dir = os.path.join(home, base, 'train')
test_dir = os.path.join(home, base, 'test')


# In[5]:


seed = 42
shuffle = True
input_shape = (331, 331)
train_batch_size, val_batch_size = 8, 256

train_datagen = keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    horizontal_flip=True
)

val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape,
        batch_size=train_batch_size,
        seed=seed,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=shuffle
)

val_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape,
        batch_size=val_batch_size,
        seed=seed,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=shuffle
)


# In[ ]:


NASNet = tf.keras.applications.NASNetLarge(
    input_shape=None,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None
)
NASNet.trainable = False
NASNet.summary()


# In[ ]:


# CONv/FC -> BatchNorm -> ReLU(or other activation) -> Dropout -> CONV/FC -> ...

def get_model(base_model):
    
    inputs = keras.Input(shape=(331, 331, 3))
    x = base_model(inputs, training=False)
    
    x = layers.Conv2D(128, (3, 3), activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Flatten()(x)
    
    x = layers.Dense(units=2048, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(rate=0.5)(x)

    x = layers.Dense(units=5, activation=None)(x)
    output = layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=output, name=f'{model_name}')
    
    return model
#     model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    


# In[ ]:


model = get_model(NASNet)
model.summary()


# In[ ]:


model.compile(
    optimizer="Adam", 
    loss="categorical_crossentropy", 
    metrics=['acc']
)


# In[ ]:


checkpoint_cb = keras.callbacks.ModelCheckpoint(
    f"{model_name}.h5", save_best_only=True
)

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=10)

history = model.fit(
            train_generator,
            steps_per_epoch=2048,
            epochs=32,
            validation_data=val_generator,
            validation_steps=128,
            shuffle=False,
            callbacks=[checkpoint_cb, early_stopping_cb]
)


# In[ ]:


with open(f'{model_name}_history.pkl', 'wb') as fh:
    pickle.dump(history.history, fh)

