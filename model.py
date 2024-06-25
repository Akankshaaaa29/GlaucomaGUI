import os
import time
import random
import pathlib
import itertools
from glob import glob
from tqdm import tqdm_notebook, tnrange

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
%matplotlib inline
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing

# # import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob
from tqdm import tqdm_notebook, tnrange

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
%matplotlib inline
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, concatenate_images

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate

import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental import preprocessing

import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import keras.backend as K

# Ignore Warnings
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, PReLU, Add, Multiply, Activation
from tensorflow.keras.models import Model


def residual_block(x, filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'):
    res = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    res = PReLU()(res)
    res = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(res)
    shortcut = Conv2D(filters, (1, 1), padding=padding, kernel_initializer=kernel_initializer)(x)
    res = Add()([res, shortcut])
    res = PReLU()(res)
    return res


def attention_block(x, g, filters):
    theta_x = Conv2D(filters, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = PReLU()(add_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(2, 2))(sigmoid_xg)
    upsample_psi = Conv2D(filters, (1, 1), padding='same')(upsample_psi)
    y = Multiply()([x, upsample_psi])
    result = Conv2D(filters, (1, 1), padding='same')(y)
    result = PReLU()(result)
    return result


def build_unet_plusresatten(inputs, ker_init, dropout):
    conv1 = residual_block(inputs, 32, kernel_initializer=ker_init)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, 64, kernel_initializer=ker_init)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_block(pool2, 128, kernel_initializer=ker_init)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_block(pool3, 256, kernel_initializer=ker_init)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = residual_block(pool4, 512, kernel_initializer=ker_init)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), attention_block(conv4, conv5, 256)], axis=3)
    conv6 = residual_block(up6, 256, kernel_initializer=ker_init)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), attention_block(conv3, conv6, 128)], axis=3)
    conv7 = residual_block(up7, 128, kernel_initializer=ker_init)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), attention_block(conv2, conv7, 64)], axis=3)
    conv8 = residual_block(up8, 64, kernel_initializer=ker_init)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), attention_block(conv1, conv8, 32)], axis=3)
    conv9 = residual_block(up9, 32, kernel_initializer=ker_init)

    conv10 = Conv2D(2, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=conv10)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, PReLU, Add, Multiply, Activation
from tensorflow.keras.models import Model


def residual_block(x, filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'):
    res = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    res = PReLU()(res)
    res = Conv2D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(res)
    shortcut = Conv2D(filters, (1, 1), padding=padding, kernel_initializer=kernel_initializer)(x)
    res = Add()([res, shortcut])
    res = PReLU()(res)
    return res


def attention_block(x, g, filters):
    theta_x = Conv2D(filters, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = Conv2D(filters, (1, 1), padding='same')(g)
    add_xg = Add()([theta_x, phi_g])
    act_xg = PReLU()(add_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(2, 2))(sigmoid_xg)
    upsample_psi = Conv2D(filters, (1, 1), padding='same')(upsample_psi)
    y = Multiply()([x, upsample_psi])
    result = Conv2D(filters, (1, 1), padding='same')(y)
    result = PReLU()(result)
    return result


def build_unet_plusresattencup(inputs, ker_init, dropout):
    conv1 = residual_block(inputs, 32, kernel_initializer=ker_init)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_block(pool1, 64, kernel_initializer=ker_init)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_block(pool2, 128, kernel_initializer=ker_init)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_block(pool3, 256, kernel_initializer=ker_init)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = residual_block(pool4, 512, kernel_initializer=ker_init)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), attention_block(conv4, conv5, 256)], axis=3)
    conv6 = residual_block(up6, 256, kernel_initializer=ker_init)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), attention_block(conv3, conv6, 128)], axis=3)
    conv7 = residual_block(up7, 128, kernel_initializer=ker_init)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), attention_block(conv2, conv7, 64)], axis=3)
    conv8 = residual_block(up8, 64, kernel_initializer=ker_init)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), attention_block(conv1, conv8, 32)], axis=3)
    conv9 = residual_block(up9, 32, kernel_initializer=ker_init)

    conv10 = Conv2D(2, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=conv10)
