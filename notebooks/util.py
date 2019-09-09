import PIL

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

import scipy

from config import *

# def downscale_img(img):
#     print('C', img.min(), img.max())
#     img = array_to_img(img.reshape(HEIGHT, WIDTH, 1), dtype='float32')

#     resized_img = img.resize((SCALED_WIDTH, SCALED_HEIGHT), resample=PIL.Image.BICUBIC)
#     img = img_to_array(resized_img).reshape(SCALED_HEIGHT, SCALED_WIDTH)
#     print('D', img.min(), img.max())
#     img = img / 255
#     print('E', img.min(), img.max())
#     return img

def downscale_img(img):
    return scipy.ndimage.zoom(img, (1, 0.25))

def dice_coef(y_true, y_pred, smooth=0.01):
    y_true_f = y_true
    y_pred_f = K.cast(y_pred>0.5,'float32')
    intersection = K.sum(y_true_f * y_pred_f,axis=[1,2])
    dice_vec = (2. * intersection + smooth) / (K.sum(y_true_f,axis=[1,2]) + K.sum(y_pred_f,axis=[1,2]) + smooth)
    return K.mean(dice_vec)

def dice_loss(y_true, y_pred, smooth=0.01):
    return 1. - dice_coef(y_true, y_pred, 0.01)