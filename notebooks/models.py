from abc import abstractmethod

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Flatten, Reshape

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

class SimpleInception(BaseModel):
    def __init__(self, num_filters=16):
        BaseModel.__init__(self)

        self.num_filters = num_filters

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
        l = Conv2D(1, (3, 3), padding='same', activation='relu')(l)
        
        return l

    def get_io_layers(self):
        input_img = Input((HEIGHT, WIDTH, 1), dtype='float32')

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


