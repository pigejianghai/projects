import tensorflow as tf
import numpy as np
import netron

import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
# Model Z
class DecouplingConvNet(keras.Model):
    def __init__(self, num_classes=2):
        super(DecouplingConvNet, self).__init__()
        self.embedding = layers.Conv2D(4, kernel_size=1, strides=1, padding="valid",  
                                       activation="relu")
        self.conv_1 = layers.Conv2D(4, kernel_size=3, strides=2, padding="valid",  
                                    activation="relu", kernel_regularizer='l2')
        self.conv_2 = layers.Conv2D(2, kernel_size=3, strides=2, padding="valid",  
                                    activation="relu", kernel_regularizer='l2')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes)
        
    @tf.function
    def call(self, x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
    
class SEBlock(layers.Layer):
    def __init__(self, alpha):
        super(SEBlock, self).__init__()
        self.alpha = alpha
    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = keras.Sequential([
            ## channels // self.ratio
            layers.Dense(4, activation='relu', kernel_initializer='he_normal', use_bias=False), 
            layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False), 
            layers.Reshape((1, 1, channels))
        ])
    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation(x)
        return layers.multiply([inputs, x * self.alpha])
    # def compute_output_shape(self, input_shape):
    #     return input_shape

class Decoupling_SE_ConvNet(keras.Model):
    def __init__(self, num_classes=3, se_alpha=1):
        super(Decoupling_SE_ConvNet, self).__init__()
        # self.multi_head = layers.Conv2D(11, kernel_size=3, strides=2, )
        self.se_alpha = se_alpha
        self.embedding = layers.Conv2D(11, kernel_size=1, strides=1, padding="valid", 
                                                activation="relu")
        self.seblock = SEBlock(self.se_alpha)
        
        self.conv_1 = layers.Conv2D(4, kernel_size=3, strides=2, padding="valid", 
                                             activation="relu", kernel_regularizer='l2')
        self.conv_2 = layers.Conv2D(4, kernel_size=3, strides=2, padding="valid", 
                                             activation="relu", kernel_regularizer='l2')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes)#, activation="softmax"
        
    @tf.function
    def call(self, x):
            
        x = self.embedding(x)
        x = self.seblock(x)

        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
    
class SEResNeXt(object):
    def __init__(self, 
                 size=448, num_classes=3, depth=64, 
                 reduction_ratio=4, num_split=8, num_block=3):
        self.depth = depth              # number of channels
        self.ratio = reduction_ratio    # ratio of channel reduction in SE module
        self.num_split = num_split      # number of splitting trees for ResNeXt (so called cardinality)
        self.num_block = num_block      # number of residual blocks
        if K.image_data_format() == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3
        self.model = self.build_model(layers.Input(shape=(size,size,1)), num_classes)

    def conv_bn(self, x, filters, kernel_size, stride, padding='same'):
        x = layers.Conv2D(filters=filters, kernel_size=[kernel_size, kernel_size],
                   strides=[stride, stride], padding=padding)(x)
        x = layers.BatchNormalization()(x)
        
        return x
    
    def activation(self, x, func='relu'):
        return layers.Activation(func)(x)
    
    def channel_zeropad(self, x):
        shape = list(x.shape)
        y = K.zeros_like(x)
        
        if self.channel_axis == 3:
            y = y[:, :, :, :shape[self.channel_axis] // 2]
        else:
            y = y[:, :shape[self.channel_axis] // 2, :, :]
        
        return layers.concatenate([y, x, y], self.channel_axis)
    
    def channel_zeropad_output(self, input_shape):
        shape = list(input_shape)
        shape[self.channel_axis] *= 2

        return tuple(shape)
    
    def initial_layer(self, inputs):
        x = self.conv_bn(inputs, self.depth, 3, 1)
        x = self.activation(x)
        
        return x
    
    def transform_layer(self, x, stride):
        x = self.conv_bn(x, self.depth, 1, 1)
        x = self.activation(x)
        
        x = self.conv_bn(x, self.depth, 3, stride)
        x = self.activation(x)
        
        return x
        
    def split_layer(self, x, stride):
        splitted_branches = list()
        for i in range(self.num_split):
            branch = self.transform_layer(x, stride)
            splitted_branches.append(branch)
        
        return layers.concatenate(splitted_branches, axis=self.channel_axis)
    
    def squeeze_excitation_layer(self, x, out_dim):
        squeeze = layers.GlobalAveragePooling2D()(x)
        
        excitation = layers.Dense(units=out_dim // self.ratio)(squeeze)
        excitation = self.activation(excitation)
        excitation = layers.Dense(units=out_dim)(excitation)
        excitation = self.activation(excitation, 'sigmoid')
        excitation = layers.Reshape((1,1,out_dim))(excitation)
        
        scale = layers.multiply([x,excitation])
        
        return scale
    
    def residual_layer(self, x, out_dim):
        for i in range(self.num_block):
            input_dim = int(np.shape(x)[-1])
            
            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
            else:
                flag = False
                stride = 1
            
            subway_x = self.split_layer(x, stride)
            subway_x = self.conv_bn(subway_x, out_dim, 1, 1)
            subway_x = self.squeeze_excitation_layer(subway_x, out_dim)
            
            if flag:
                pad_x = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
                pad_x = layers.Lambda(self.channel_zeropad, output_shape=self.channel_zeropad_output)(pad_x)
            else:
                pad_x = x
            
            x = self.activation(layers.add([pad_x, subway_x]))
                
        return x
    
    def build_model(self, inputs, num_classes):
        x = self.initial_layer(inputs)
        # out_dim refers to the output_channels of resudual_layer
        x = self.residual_layer(x, out_dim=64)
        x = self.residual_layer(x, out_dim=128)
        x = self.residual_layer(x, out_dim=256)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(units=num_classes, activation='softmax')(x)
        
        return keras.models.Model(inputs, x)
    
if __name__ == '__main__':
    # resnext = SEResNeXt(size=448, num_classes=3).model
    # resnext.summary()
    # keras.utils.plot_model(resnext, to_file='/data1/jianghai/code/keras/model_structure/SEResNeXt.png', show_shapes=True)
    
    decouple_net = Decoupling_SE_ConvNet()
    # decouple_net.build(input_shape=(1, 110, 110, 11))
    keras.utils.plot_model(decouple_net, 
                           to_file='/data1/jianghai/DECT/code/tfversion/model_structure/decouple_se_net.png', 
                           show_shapes=True, dpi=64)
    # decouple_net.summary()