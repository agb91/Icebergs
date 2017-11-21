from  __future__ import division
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import math
np.random.seed(98643)
import tensorflow as tf
tf.set_random_seed(683)
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, denoise_tv_bregman, denoise_nl_means)
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.transform import resize
# Data reading and visualization
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Training part
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalAveragePooling2D, Lambda
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def resizer( images , r, c ):
    type(images)
    scale = lambda x: resize( x , (r,c) )
    images = scale(images)
    return images

# Translate data to an image format
def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        #band_1 = resize( band_1, (28, 28) )
        band_2 = np.array(row['band_2']).reshape(75, 75)
        #band_2 = resize( band_2, (28, 28) )
        angle = float( math.ceil(row['inc_angle']) ) / 100.0
        #band_3 = np.full((75, 75), angle)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = band_3


        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)

def hvhh(data):
    band_1V = []
    band_2V = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        #band_1 = resize( band_1, (28, 28) )
        band_1V.append(band_1)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        #band_2 = resize( band_2, (28, 28) )
        band_2V.append(band_2)
    return band_1V ,band_2V    

def denoise(X, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel) for item in X])

def smooth(X, sigma):
    return np.asarray([gaussian(item, sigma=sigma) for item in X])

def grayscale(X):
    return np.asarray([rgb2gray(item) for item in X])

def create_model():
    model.add(Conv2D(16, (3, 3), input_shape=(75, 75, 3)) )
    model.add(Activation('relu'))
    model.add( MaxPooling2D(pool_size=(2, 2), padding='valid', dim_ordering="th" ) )

    model.add(Conv2D(16, (3, 3), dim_ordering="th" ))
    model.add(Activation('relu'))
    model.add( MaxPooling2D(pool_size=(2, 2), padding='valid', dim_ordering="th" ) )

    model.add(Conv2D(32, (3, 3), dim_ordering="th" ))
    model.add(Activation('relu'))
    #model.add( MaxPooling2D(pool_size=(2, 2), padding='valid', dim_ordering="th" ) )



    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def run( images, labels):    
    model.fit(images, labels, epochs=13, batch_size=40, validation_split=0.3)

def evaluate(images, labels):
    score = model.evaluate(images, labels, batch_size=40)
    return score


def create_dataset(frame, labeled, smooth_rgb=0.2, smooth_gray=0.5,
                   weight_rgb=0.05, weight_gray=0.05):
    band_1, band_2 = None,None#hvhh(frame)
    images = color_composite(frame)

    #band_1 = smooth(denoise(band_1, weight_gray, False), smooth_gray)
    #band_2 = smooth(denoise(band_2, weight_gray, False), smooth_gray)
    images = smooth(denoise(images, weight_rgb, True), smooth_rgb)
    
    print('all done')
    X_angle = np.array(frame.inc_angle)
    if labeled:
        y = np.array(frame["is_iceberg"])
    else:
        y = None

    return y, X_angle, band_1, band_2, images    


model = Sequential()
    
train = pd.read_json("data/train.json")

print( train.columns.values )
#print( "len:  " + str( len(train) ) )
#print( train.describe()  )

#train = train[ 0 : 100]


train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

  
y, X_angle, band1, band2, images = create_dataset(train, True)

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.33, random_state=42)


print( "len:  " + str( len(X_train) ) )
print("\n shape:")
print( X_train[0].shape )

model = create_model()
print("we have a model")

run( X_train, y_train )
print( evaluate( X_train, y_train ) )


print( "Finished" )