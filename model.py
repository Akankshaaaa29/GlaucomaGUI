# source https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a
# import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob


# import data handling tools

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



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


# # import system libs
import os
import time
import random
import pathlib
import itertools
from glob import glob


# import data handling tools

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



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


import os
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import keras.backend as K

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, SeparableConv2D, Add, Flatten, Dropout, Dense

def custom(input_shape=(224, 224, 3), num_classes=1):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x1 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(128, (1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x1 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(256, (1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x1 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(512, (1, 1), padding='same')(x2)
    x2 = BatchNormalization()(x2)

    x = Add()([x1, x2])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


