from pprint import pprint
import os.path as pth

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.utils import Sequence

import math

from config import *

class BaseImageLoader(Sequence):
    def __init__(self):
        self.image_cache = {}

    def get_image(self, imageid):
        if imageid not in self.image_cache:
            im = plt.imread(pth.join(TRAIN_IMG_DIR, f'{imageid}.jpg')).astype('float32')
            im /= 255
            self.image_cache[imageid] = im
            
        return self.image_cache[imageid]

class ImageLoader(BaseImageLoader):
    def __init__(self, image_names, image_groups, image_masks, batch_size):
        BaseImageLoader.__init__(self)

        self.image_names = image_names
        self.image_groups = image_groups
        self.image_masks = image_masks
        
        self.batch_size = batch_size
        
    def __len__(self):
        return math.ceil(len(self.image_names) / self.batch_size)
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        
        if end > len(self.image_names):
            end = len(self.image_names)
            
        size = end - start
        
        img = np.zeros((size, HEIGHT, WIDTH, 1), dtype='float32')
        masks = np.zeros((4, size, HEIGHT, WIDTH, 1), dtype='float32')

        for i, imageid in enumerate(self.image_names[start:end]):
            im = self.get_image(imageid)
            im /= 255
            img[i, :, :, 0] = im[:, :, 0]
            
            image_group = self.image_groups.get_group(imageid)
            for row in image_group.itertuples():
                if row.mask_present:
                    key = f'{imageid}_{row.classid}'
                    mask = self.image_masks[key].astype('float32')
                    masks[row.classid - 1, i, :, :, 0] = mask
            
        return img, [masks[0], masks[1], masks[2], masks[3]]
    
            
class BlockwiseImageLoader(ImageLoader):
    def __init__(self, image_names, image_groups, image_masks, batch_size, patch_size, stride):
        ImageLoader.__init__(self, image_names, image_groups, image_masks, batch_size)

        self.patch_size = patch_size
        self.stride = stride

        self.n_patches_x = self.compute_num_patches(WIDTH)
        self.n_patches_y = self.compute_num_patches(HEIGHT)

    def compute_num_patches(self, dim):
        return math.ceil((dim - self.patch_size) / self.stride) + 1
        
    def __len__(self):
        return math.ceil(len(self.image_names) / self.batch_size)
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        
        if end > len(self.image_names):
            end = len(self.image_names)
            
        size = end - start
        
        img = np.zeros(
            (size, self.n_patches_y, self.n_patches_x, self.patch_size, self.patch_size),
            dtype='float32')
        masks = np.zeros(
            (4, size, self.n_patches_y, self.n_patches_x, self.patch_size, self.patch_size),
            dtype='float32')

        for i, imageid in enumerate(self.image_names[start:end]):
            im = self.get_image(imageid)
            im /= 255
            self.split_image_to_patches(img[i, :, :, :, :], im[:, :, 0])
            
            image_group = self.image_groups.get_group(imageid)
            for row in image_group.itertuples():
                if row.mask_present:
                    key = f'{imageid}_{row.classid}'
                    mask = self.image_masks[key].astype('float32')
                    self.split_image_to_patches(masks[row.classid - 1, i, :, :, :, :], mask)

        img = img.reshape((size * self.n_patches_y * self.n_patches_x, self.patch_size, self.patch_size, 1))
        masks = masks.reshape((4, size * self.n_patches_y * self.n_patches_x, self.patch_size, self.patch_size, 1))
            
        return img, [masks[0], masks[1], masks[2], masks[3]]

    def split_image_to_patches(self, dst, src):
        #print(src.shape, dst.shape)
        for iy, y in enumerate(range(0, HEIGHT - self.stride, self.stride)):
            y1 = y
            y2 = y + self.patch_size
            y2 = min(y2, HEIGHT)
            ylen = y2 - y1
                
            for ix, x in enumerate(range(0, WIDTH - self.stride, self.stride)):
                x1 = x
                x2 = x + self.patch_size
                x2 = min(x2, WIDTH)
                xlen = x2 - x1

                #print(ix, iy, xlen, ylen, ' - ', x1, x2, y1, y2)
                dst[iy, ix, 0:ylen, 0:xlen] = src[y1:y2, x1:x2]

    def combine_mask_patches(self, mask_patches):
        return [self.combine_patches(mask_patch) for mask_patch in mask_patches]

    def combine_patches(self, img_patches):
        size = int(img_patches.shape[0] / (self.n_patches_x * self.n_patches_y))
        img_patches = img_patches.reshape(size, self.n_patches_y, self.n_patches_x, self.patch_size, self.patch_size)

        img = np.zeros((size, HEIGHT, WIDTH), dtype='float32')

        for i in range(size):
            for iy, y in enumerate(range(0, HEIGHT - self.stride, self.stride)):
                y1 = y
                y2 = y + self.patch_size
                y2 = min(y2, HEIGHT)
                ylen = y2 - y1
                    
                for ix, x in enumerate(range(0, WIDTH - self.stride, self.stride)):
                    x1 = x
                    x2 = x + self.patch_size
                    x2 = min(x2, WIDTH)
                    xlen = x2 - x1

                    #print(ix, iy, xlen, ylen, ' - ', x1, x2, y1, y2)
                    img[i, y1:y2, x1:x2] = img_patches[i, iy, ix, 0:ylen, 0:xlen]

        return img
