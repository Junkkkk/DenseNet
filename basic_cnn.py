from __future__ import print_function
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2


class Cnn:

    def __init__(self,config):
        self.config = config

        #self.input_shape = self.config.config['ingredient']['param']['input_shape']
        self.dense_blocks = self.config.config['ingredient']['param']['dense_blocks']
        self.dense_layers = self.config.config['ingredient']['param']['dense_layers']
        self.growth_rate = self.config.config['ingredient']['param']['growth_rate']
        self.nb_classes = self.config.config['ingredient']['param']['nb_classes']
        self.nb_fclayers = self.config.config['ingredient']['param']['nb_fclayers']
        self.dropout_rate = self.config.config['ingredient']['param']['dropout_rate']
        self.bottleneck = self.config.config['ingredient']['param']['bottleneck']
        self.compression = self.config.config['ingredient']['param']['compression']
        self.weight_decay = self.config.config['ingredient']['param']['weight_decay']
        self.depth = self.config.config['ingredient']['param']['depth']

        self.model = None

    def __call__(self, *args, **kwargs):
        if self.model is None:
            self.build_cnn()
        return self.model

    def build_cnn(self):

        if self.compression <= 0.0 or self.compression > 1.0:
            raise Exception(
                'Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')

        if type(self.dense_layers) is list:
            if len(self.dense_layers) != self.dense_blocks:
                raise AssertionError('Number of dense blocks have to be same length to specified layers')
        elif self.dense_layers == -1:
            if self.bottleneck:
                self.dense_layers = (self.depth - (self.dense_blocks + 1)) / self.dense_blocks // 2
            else:
                self.dense_layers = (self.depth - (self.dense_blocks + 1)) // self.dense_blocks
            self.dense_layers = [int(self.dense_layers) for _ in range(self.dense_blocks)]
        else:
            self.dense_layers = [int(self.dense_layers) for _ in range(self.dense_blocks)]

        img_input = Input(shape=(32,32,3))
        nb_channels = self.growth_rate * 2

        print('Creating DenseNet')
        print('#############################################')
        print('Dense blocks: %s' % self.dense_blocks)
        print('Layers per dense block: %s' % self.dense_layers)
        print('#############################################')

        # Initial convolution layer
        x = Conv2D(nb_channels, (3, 3), padding='same', strides=(1, 1),
                   use_bias=False, kernel_regularizer=l2(self.weight_decay))(img_input)


        x = BatchNormalization(gamma_regularizer=l2(self.weight_decay), beta_regularizer=l2(self.weight_decay))(x)
        x = Activation('relu')(x)
        x = Flatten()(x)

        x = Dense(self.nb_fclayers, activation='relu')(x)

        output = Dense(self.nb_classes, activation='softmax')(x)

        model = Model(img_input, output)
        model.summary()
        self.model = model

        return self.model