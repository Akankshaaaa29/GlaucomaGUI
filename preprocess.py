import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

def preprocess_image(image, size=224):
    # Resize image

    height, width, _ = image.shape
    if height != width:
        size = min(width, height)
        left = (width - size)
        top = (height - size)
        image = image[500:1500, 350:1500]
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = cv2.bilateralFilter(image, 2, 50, 50)

    return image