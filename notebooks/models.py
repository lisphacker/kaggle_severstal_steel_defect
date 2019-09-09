from abc import abstractmethod

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import Add, Flatten, Reshape, Concatenate
from tensorflow.keras.layers import LeakyReLU, Softmax, BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

from config import *
from util import *

class BaseModel:
    def __init__(self, num_outputs=4):
        self.model = None
        self.num_outputs = num_outputs

    @abstractmethod
    def get_io_layers(self):
        pass

    def build_model(self):
        inputs, outputs = self.get_io_layers()

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['acc'])
        #self.model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef])

    def fit(self, **args):
        return self.model.fit(**args)

    def fit_generator(self, **args):
        return self.model.fit_generator(**args)

    def predict(self, **args):
        return self.model.predict(**args)

    def predict_generator(self, **args):
        return self.model.predict_generator(**args)

    def summary(self):
        return self.model.summary()

    def plot_history(self, keys=['loss', 'val_loss']):
        if len(keys) == 0:
            return

        history = self.model.history.history
    
        n = 3
        primary = np.array(history[keys[0]])
        ymin = primary.mean() - primary.std() * n
        ymax = primary.mean() + primary.std() * n
        
        plt.figure()
        plt.ylim(ymin, ymax)
        for key in keys:
            if key in history.keys():
                #plt.plot(np.arange(len(history[key])), history[key], label=key)
                plt.plot(history[key], label=key)
            else:
                print(f'Unable to plot {key}')
        plt.legend()
        plt.show()

class SimpleAE(BaseModel):
    def __init__(self, input_size=64):
        BaseModel.__init__(self)

        self.input_size = input_size

    def make_complex_conv_layer(self, num_filters, filter_size, l,  activation='relu'):
        l = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(l)
        
        return l

    def get_io_layers(self):
        input_img = Input((SCALED_HEIGHT, SCALED_WIDTH, 1), dtype='float32')

        x = input_img

        num_filters = 8
        filter_size = 3

        x = self.make_complex_conv_layer(16, 7, x)
        x = MaxPool2D((4, 2))(x)

        x = self.make_complex_conv_layer(64, 5, x)
        x = MaxPool2D((4, 2))(x)

        x = self.make_complex_conv_layer(256, 3, x)
        x = MaxPool2D((4, 2))(x)

        o = [x] * 4

        for i in range(len(o)):
            o[i] = self.make_complex_conv_layer(64, 3, o[i])
            o[i] = UpSampling2D((4, 2))(o[i])

            o[i] = self.make_complex_conv_layer(32, 5, o[i])
            o[i] = UpSampling2D((4, 2))(o[i])

            o[i] = self.make_complex_conv_layer(16, 7, o[i])
            o[i] = UpSampling2D((4, 2))(o[i])

            o[i] = self.make_complex_conv_layer(1, 7, o[i], activation='sigmoid')

        return input_img, o

class UNet(BaseModel):
    def __init__(self, n_layers=4, num_start_filters=64, **args):
        BaseModel.__init__(self, **args)

        self.n_layers = n_layers
        self.num_start_filters = num_start_filters

    def conv_down(self, input_layer, num_filters, 
                  conv_filter_shape=(3, 3), pool_filter_shape=(2, 2), 
                  activation='relu', skip_pool=False, level_name=''):
        conv = Conv2D(num_filters, conv_filter_shape,         
                      padding='same', activation=activation,
                      name=f'{level_name}_conv1')(input_layer)
        conv = Conv2D(num_filters, conv_filter_shape,
                      padding='same', activation=activation,
                      name=f'{level_name}_conv2')(conv)

        if skip_pool:
            return conv
        else:
            pool = MaxPool2D(pool_filter_shape, name=f'{level_name}_maxpool')(conv)
            return pool, conv

    def conv_up(self, input_layer, encoder_input_layer, num_filters, 
                conv_filter_shape=3, pool_filter_shaoe=(2, 2), 
                activation='relu', level_name=''):
        num_filters = int(num_filters)

        up = UpSampling2D(name=f'{level_name}_upsample')(input_layer)
        up = Conv2D(num_filters, (2, 2), padding='same', name=f'{level_name}_conv1')(up)

        concat = Concatenate(axis=3, name=f'{level_name}_concat')([up, encoder_input_layer])
        conv = Conv2D(num_filters, conv_filter_shape,
                      padding='same', activation=activation,
                      name=f'{level_name}_conv2')(concat)
        conv = Conv2D(num_filters, conv_filter_shape, 
                      padding='same', activation=activation,
                      name=f'{level_name}_conv3')(conv)
        return conv

    def get_io_layers(self):
        input_img = Input((SCALED_HEIGHT, SCALED_WIDTH, 1), dtype='float32')

        filter_multiplier = 2
        num_filters = self.num_start_filters

        out = input_img

        saved = []

        for i in range(self.n_layers):
            out, save = self.conv_down(out, num_filters, level_name=f'down_{i}')
            saved.append(save)
            num_filters *= filter_multiplier

        mid = self.conv_down(out, num_filters, skip_pool=True, level_name='mid')

        o = [mid] * self.num_outputs

        for j in range(self.n_layers):
            num_filters /= filter_multiplier
            for i in range(len(o)):
                o[i] = self.conv_up(o[i], saved[-(j + 1)], num_filters, level_name=f'up{i}_{j}')

        for i in range(len(o)):
            o[i] = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", name=f'out{i}')(o[i])

        return input_img, o



class SimpleInception(BaseModel):
    def __init__(self, num_filters=16, input_size=64):
        BaseModel.__init__(self)

        self.num_filters = num_filters
        self.input_size = input_size

    def create_inception_pair(self, kernel_size, input_layer):
        l = Conv2D(self.num_filters, (1, 1), padding='same', activation='relu')(input_layer)
        return Conv2D(self.num_filters, (kernel_size, kernel_size), padding='same', activation='relu')(l)

    def create_inception_layer(self, input_layer):
        l_3x3 = self.create_inception_pair(3, input_layer)
        l_5x5 = self.create_inception_pair(5, input_layer)
        l_7x7 = self.create_inception_pair(7, input_layer)
        
        return Add()([l_3x3, l_5x5, l_7x7])

    def up_sampler(self, input_layer):
        l = UpSampling2D((2, 4))(input_layer)
        l = Conv2D(self.num_filters, (3, 3), padding='same', activation='relu')(l)
        
        #l = UpSampling2D((2, 4))(l)
        #l = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(l)
        
        l = UpSampling2D((2, 4))(l)
        l = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(l)
        
        return l

    def get_io_layers(self):
        input_img = Input((self.input_size, self.input_size, 1), dtype='float32')

        x = self.create_inception_layer(input_img)
        x = MaxPool2D((2, 4))(x)

        #x = create_inception_layer(16, x)
        #x = MaxPool2D((2, 4))(x)
        
        x = self.create_inception_layer(x)
        mid = MaxPool2D((2, 4))(x)

        o1 = self.up_sampler(mid)
        o2 = self.up_sampler(mid)
        o3 = self.up_sampler(mid)
        o4 = self.up_sampler(mid)

        return input_img, [o1, o2, o3, o4]


