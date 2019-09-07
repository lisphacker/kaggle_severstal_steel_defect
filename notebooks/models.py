from abc import abstractmethod

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU, Softmax, BatchNormalization

from config import *

class BaseModel:
    def __init__(self):
        self.model = None

    @abstractmethod
    def get_io_layers(self):
        pass

    def build_model(self):
        inputs, outputs = self.get_io_layers()

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

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

class SimpleAE(BaseModel):
    def __init__(self, input_size=64):
        BaseModel.__init__(self)

        self.input_size = input_size

    def make_complex_conv_layer(self, num_filters, filter_size, l,  activation='relu'):
        l = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(l)
        
        return l

    def get_io_layers(self):
        input_img = Input((HEIGHT, WIDTH, 1), dtype='float32')

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
    def __init__(self):
        BaseModel.__init__(self)

    def conv_down(self, num_filters, conv_filter_size=3, pool_filter_size=(2, 2), input_layer,  activation='relu'):
        conv = input_layer
        conv = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(conv)
        conv = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(conv)
        pool = MaxPool2D(pool_filter_size)(conv)
        return pool, conv

    def get_io_layers(self):
        input_img = Input((HEIGHT, WIDTH, 1), dtype='float32')

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


