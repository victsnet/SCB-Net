#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:04:54 2023

@author: silva

Unet for Geographic data
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, SpatialDropout2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, ReLU, Add
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50

'''
Useful blocks to build Unet

conv - BN - Activation - conv - BN - Activation - Dropout (if enabled)

'''


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv


def repeat_elem(tensor, rep):
    
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

class Resnet_Unet:
    
    def __init__(self, dim, n_classes, dropout_rate=0.3, spatial_dropout=False, weights='imagenet'):
        
        self.dim = dim
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.spatial_dropout = spatial_dropout
        self.weights = weights
        
    def dropout_layer(self, x, spatial_dropout=False):
        
        if spatial_dropout:
            x = SpatialDropout2D(self.dropout_rate)(x)
            
        else:
            x = Dropout(self.dropout_rate)(x)
                    
        return x   
    
    # Define residual blocks with spatial dropout
    def residual_block(self, x, filters, stride=1, projection=False):
        
        shortcut = x
        x = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = self.dropout_layer(x, self.spatial_dropout)   # add spatial dropout
        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = self.dropout_layer(x, self.spatial_dropout)   # add spatial dropout
        x = Conv2D(filters * 4, kernel_size=(1, 1), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        if projection:
            shortcut = Conv2D(filters * 4, kernel_size=(1, 1), strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = ReLU()(x)
        x = self.dropout_layer(x, self.spatial_dropout)  # add spatial dropout
        return x

    def resnet50_backbone(self, inputs):
        resnet50 = ResNet50(weights=self.weights, include_top=False, input_shape=self.dim, input_tensor=inputs)
        
        return resnet50
    
    # Create the UNet model with ResNet50 backbone
    def model(self, encoder_freeze=False):
        # Create the input layer
        inputs = Input(self.dim)
        c1 = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same')(inputs)
        c2 = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same')(c1)
        compressed_inputs = Conv2D(3, kernel_size=(3, 3), strides=1, padding='same')(c2)
        resnet50 = self.resnet50_backbone(compressed_inputs)
        
        
        
        # if freeze == True > trainable = False
        trainable = True
        if encoder_freeze:
            trainable = False
        
        
        # freeze all layers in resnet50 
        for layer in resnet50.layers:
            layer.trainable = trainable
        
        # Create the ResNet50 backbone
        x = resnet50.input
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        x = self.residual_block(x, filters=32, stride=1, projection=True)
        x = self.residual_block(x, filters=32)
        x = self.residual_block(x, filters=32)
        x = self.residual_block(x, filters=64, stride=2, projection=True)
        x = self.residual_block(x, filters=64)
        x = self.residual_block(x, filters=64)
        x = self.residual_block(x, filters=128, stride=2, projection=True)
        x = self.residual_block(x, filters=128)
        x = self.residual_block(x, filters=128)
        x = self.residual_block(x, filters=128)
        x = self.residual_block(x, filters=128)
        x = self.residual_block(x, filters=256, stride=2, projection=True)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        x = self.residual_block(x, filters=256)
        x = self.residual_block(x, filters=256)
    
        # UNet decoder
        x = Conv2D(1024, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([x, resnet50.get_layer('conv4_block6_out').output], axis=3)
        x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
        x = self.dropout_layer(x, self.spatial_dropout)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([x, resnet50.get_layer('conv3_block4_out').output], axis=3)
        x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
        x = self.dropout_layer(x, self.spatial_dropout)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([x, resnet50.get_layer('conv2_block3_out').output], axis=3)
        x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([x, resnet50.get_layer('conv1_relu').output], axis=3)
        x = self.dropout_layer(x, self.spatial_dropout)
        x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
        x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = self.dropout_layer(x, self.spatial_dropout)
    
        outputs = Conv2D(self.n_classes, (1, 1), activation='softmax')(x)
    
        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        return self.model
    
    


class Unet_Resnet50:
    
    def __init__(self, n_classes, dropout_rate=0.3, spatial_dropout=False, weights='imagenet'):
        
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.spatial_dropout = spatial_dropout
        self.weights = weights
        
    def dropout_layer(self, x, spatial_dropout=False):
        
        if spatial_dropout:
            x = SpatialDropout2D(self.dropout_rate)(x)
            
        else:
            x = Dropout(self.dropout_rate)(x)
                    
        return x   

    def resnet50_backbone(self, inputs, dim):
        resnet50 = ResNet50(weights=self.weights, include_top=False, input_shape=dim, input_tensor=inputs)
        
        return resnet50
    
    # Create the UNet model with ResNet50 backbone
    def model(self, dim, encoder_freeze=False):
        # Create the input layer
        inputs = Input(dim)
        # Create the ResNet50 backbone
        resnet50 = self.resnet50_backbone(inputs, dim)
        
        # if freeze == True > trainable = False
        trainable = True
        if encoder_freeze:
            trainable = False
            
        # freeze all layers in resnet50 
        for layer in resnet50.layers:
            layer.trainable = trainable
        
        # get resnet50 layers for concatenation
        conv1_relu = resnet50.get_layer('conv1_relu').output
        conv1_relu = self.dropout_layer(conv1_relu, self.spatial_dropout)
        conv2_block3_out = resnet50.get_layer('conv2_block3_out').output
        conv2_block3_out = self.dropout_layer(conv2_block3_out, self.spatial_dropout)
        conv3_block4_out = resnet50.get_layer('conv3_block4_out').output
        conv3_block4_out = self.dropout_layer(conv3_block4_out, self.spatial_dropout)
        conv4_block6_out = resnet50.get_layer('conv4_block6_out').output
        conv4_block6_out = self.dropout_layer(conv4_block6_out, self.spatial_dropout)
        backbone_output = resnet50.get_layer('conv5_block3_out').output
        backbone_output = self.dropout_layer(backbone_output, self.spatial_dropout)
    
        # Add the first upsampling block
        upconv1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(backbone_output)
        upconv1 = concatenate([upconv1, conv4_block6_out], axis=3)
        conv1 = Conv2D(512, (3, 3), padding='same')(upconv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(512, (3, 3), padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = self.dropout_layer(conv1, self.spatial_dropout)
    
        # Add the second upsampling block
        upconv2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv1)
        upconv2 = concatenate([upconv2, conv3_block4_out], axis=3)
        conv2 = Conv2D(256, (3, 3), padding='same')(upconv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(256, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = self.dropout_layer(conv2, self.spatial_dropout)
    
        # Add the third upsampling block
        upconv3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv2)
        upconv3 = concatenate([upconv3, conv2_block3_out], axis=3)
        conv3 = Conv2D(128, (3, 3), padding='same')(upconv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = self.dropout_layer(conv3, self.spatial_dropout)
        
        # Add the fourth upsampling block
        upconv4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3)
        upconv4 = concatenate([upconv4, conv1_relu], axis=3)
        conv4 = Conv2D(64, (3, 3), padding='same')(upconv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(64, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Dropout(0.5)(conv4)
        conv4 = self.dropout_layer(conv4, self.spatial_dropout)
    
        # Add the fifth upsampling block
        upconv5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
        conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upconv5)
    
        outputs = Conv2D(self.n_classes, (1, 1), activation='softmax')(conv5)
    
        # Create the model
        self.model = Model(inputs=inputs, outputs=outputs)

        return self.model
        
    
    def backbone_layers(self, input, layer='conv2_block3_out'):
    
        '''
        layers:
            conv5_block3_out
            conv4_block6_out
            conv3_block4_out
            conv2_block3_out
            conv1_relu
        '''
        
        dim = input.shape[1:]
        inputs = Input(dim)
        
        resnet50 = ResNet50(weights=self.weights, include_top=False, input_shape=dim, input_tensor=inputs)
        output = Model(inputs=inputs, outputs=resnet50.get_layer(layer).output).predict(input)
        return output

class Unet:
    
    def __init__(self, dim, n_classes, dropout_rate=0.3):
        
        self.dim = dim
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        
    def __repr__(self):
        return 'Unet-Geo'
    
    
    def dropout_layer(self, x, spatial_dropout=False):
        
        if spatial_dropout:
            x = SpatialDropout2D(self.dropout_rate)(x)
            
        else:
            x = Dropout(self.dropout_rate)(x)
            
        
        return x

    def model(self, coords_channels, kernel_size=(3, 3), fn=[16, 32, 64, 128, 256],
              initializer='glorot_uniform', use_coordinates=True, spatial_dropout=False, activation='softmax'):
        
        
        # convert coordinates array to tensor
        input_A = Input((self.dim[0], self.dim[1], coords_channels), dtype=tf.float32)
        if use_coordinates:
            # model A - coordinates - convolutional layers
            xy = Conv2D(fn[0], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(input_A)
            xy = self.dropout_layer(xy, spatial_dropout)
            xy = Conv2D(fn[1], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(xy)
            xy = self.dropout_layer(xy, spatial_dropout)
        
        # model B - auxiliary information
        input_B = Input((self.dim[0], self.dim[1], self.dim[2]-coords_channels), dtype=tf.float32)

                # Contraction path
        c1 = Conv2D(fn[0], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(input_B)
        c1 = self.dropout_layer(c1, spatial_dropout)
        c1 = Conv2D(fn[0], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(fn[1], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(p1)
        c2 = self.dropout_layer(c2, spatial_dropout)
        c2 = Conv2D(fn[1], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(fn[2], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(p2)
        c3 = self.dropout_layer(c3, spatial_dropout)
        c3 = Conv2D(fn[2], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(fn[3], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(p3)
        c4 = self.dropout_layer(c4, spatial_dropout)
        c4 = Conv2D(fn[3], (3, 3), activation='relu', kernel_initializer=initializer, padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(fn[4], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(p4)
        c5 = self.dropout_layer(c5, spatial_dropout)
        c5 = Conv2D(fn[4], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c5)

        # Expansive path 
        u6 = Conv2DTranspose(fn[-2], (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(fn[-2], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(u6)
        c6 = self.dropout_layer(c6, spatial_dropout)
        c6 = Conv2D(fn[-2], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c6)

        u7 = Conv2DTranspose(fn[-3], (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(fn[-3], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(u7)
        c7 = self.dropout_layer(c7, spatial_dropout)
        c7 = Conv2D(fn[-3], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c7)

        u8 = Conv2DTranspose(fn[-4], (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(fn[-4], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(u8)
        c8 = self.dropout_layer(c8, spatial_dropout)
        c8 = Conv2D(fn[-4], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c8)

        u9 = Conv2DTranspose(fn[-5], (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(fn[-5], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(u9)
        c9 = self.dropout_layer(c9, spatial_dropout)
        femb = Conv2D(fn[-5], kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(c9)
    
        if use_coordinates:
            femb = concatenate([xy, femb], name="concatenated_layer")

        outputs = Conv2D(self.n_classes, (1, 1), activation=activation, kernel_initializer=initializer)(femb)

        model = Model(inputs=[input_A, input_B], outputs=[outputs])

        self.model = model
        
        return model
        

    def predict(self, X):
        
        output = self.model.predict_on_batch(X)
        return output    

        
    
class Attention_Unet:
    
    def __init__(self, dim, n_classes):
        
        self.n_classes = n_classes
        self.dim = dim
        
        
    def model(self, coords_channels, kernel_size=3, fn=64, dropout_rate=0.3, 
              initializer='glorot_uniform', use_coordinates=True, batch_norm=True):
        '''
        Attention UNet, 
        fn - number of basic filters for the first layer
        
        '''
        # network structure
    
        UP_SAMP_SIZE = 2 # size of upsampling filters
        
        # convert coordinates array to tensor
        input_A = Input((self.dim[0], self.dim[1], coords_channels), dtype=tf.float32)
        if use_coordinates:
            # model A - coordinates - convolutional layers
            xy = Conv2D(fn//2, kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(input_A)
            xy = Dropout(dropout_rate)(xy)
            xy = Conv2D(fn//2, kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(xy)
            xy = Dropout(dropout_rate)(xy)
        
        # model B - auxiliary information
        input_B = Input((self.dim[0], self.dim[1], self.dim[2]-coords_channels), dtype=tf.float32)
    
        # Downsampling layers
        # DownRes 1, convolution + pooling
        conv_128 = conv_block(input_B, kernel_size, fn, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
        # DownRes 2
        conv_64 = conv_block(pool_64, kernel_size, 2*fn, dropout_rate, batch_norm)
        pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
        # DownRes 3
        conv_32 = conv_block(pool_32, kernel_size, 4*fn, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
        # DownRes 4
        conv_16 = conv_block(pool_16, kernel_size, 8*fn, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
        # DownRes 5, convolution only
        conv_8 = conv_block(pool_8, kernel_size, 16*fn, dropout_rate, batch_norm)
    
        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        gating_16 = gating_signal(conv_8, 8*fn, batch_norm)
        att_16 = attention_block(conv_16, gating_16, 8*fn)
        up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
        up_16 = layers.concatenate([up_16, att_16], axis=3)
        up_conv_16 = conv_block(up_16, kernel_size, 8*fn, dropout_rate, batch_norm)
        # UpRes 7
        gating_32 = gating_signal(up_conv_16, 4*fn, batch_norm)
        att_32 = attention_block(conv_32, gating_32, 4*fn)
        up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
        up_32 = layers.concatenate([up_32, att_32], axis=3)
        up_conv_32 = conv_block(up_32, kernel_size, 4*fn, dropout_rate, batch_norm)
        # UpRes 8
        gating_64 = gating_signal(up_conv_32, 2*fn, batch_norm)
        att_64 = attention_block(conv_64, gating_64, 2*fn)
        up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
        up_64 = layers.concatenate([up_64, att_64], axis=3)
        up_conv_64 = conv_block(up_64, kernel_size, 2*fn, dropout_rate, batch_norm)
        # UpRes 9
        gating_128 = gating_signal(up_conv_64, fn, batch_norm)
        att_128 = attention_block(conv_128, gating_128, fn)
        up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
        up_128 = layers.concatenate([up_128, att_128], axis=3)
        up_conv_128 = conv_block(up_128, kernel_size, fn, dropout_rate, batch_norm)
        
        if use_coordinates:
            up_conv_128 = concatenate([xy, up_conv_128], name="concatenated_layer")
    
        # 1*1 convolutional layers
        conv_final = layers.Conv2D(self.n_classes, kernel_size=(1, 1))(up_conv_128)
        conv_final = layers.BatchNormalization(axis=3)(conv_final)
        outputs = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel
    
        # Model integration
        model = Model(inputs=[input_A, input_B], outputs=[outputs], name="Attention_UNet")
        return model
        
    
    def predict(self, X):
        
        output = self.model.predict_on_batch(X)
        return output   
    
    
class Attention_ResUnet:
    
    def __init__(self, dim, n_classes):
        
        self.n_classes = n_classes
        self.dim = dim
        
        
    def model(self, coords_channels, kernel_size=3, fn=64, dropout_rate=0.3, 
              initializer='glorot_uniform', use_coordinates=True, batch_norm=False):
        
        '''
        Rsidual UNet, with attention 
        
        '''
        # network structure
        UP_SAMP_SIZE = 2 # size of upsampling filters
        # input data
        # dimension of the image depth
        inputs = Input(self.dim, dtype=tf.float32)
        
        axis = 3
        # convert coordinates array to tensor
        input_A = Input((self.dim[0], self.dim[1], coords_channels), dtype=tf.float32)
        if use_coordinates:
            # model A - coordinates - convolutional layers
            xy = Conv2D(fn//2, kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(input_A)
            xy = Dropout(dropout_rate)(xy)
            xy = Conv2D(fn//2, kernel_size, activation='relu', kernel_initializer=initializer, padding='same')(xy)
            xy = Dropout(dropout_rate)(xy)
        
        # model B - auxiliary information
        input_B = Input((self.dim[0], self.dim[1], self.dim[2]-coords_channels), dtype=tf.float32)
    
        # Downsampling layers
        # DownRes 1, double residual convolution + pooling
        conv_128 = res_conv_block(input_B, kernel_size, fn, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
        # DownRes 2
        conv_64 = res_conv_block(pool_64, kernel_size, 2*fn, dropout_rate, batch_norm)
        pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
        # DownRes 3
        conv_32 = res_conv_block(pool_32, kernel_size, 4*fn, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
        # DownRes 4
        conv_16 = res_conv_block(pool_16, kernel_size, 8*fn, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
        # DownRes 5, convolution only
        conv_8 = res_conv_block(pool_8, kernel_size, 16*fn, dropout_rate, batch_norm)
    
        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        gating_16 = gating_signal(conv_8, 8*fn, batch_norm)
        att_16 = attention_block(conv_16, gating_16, 8*fn)
        up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
        up_16 = layers.concatenate([up_16, att_16], axis=axis)
        up_conv_16 = res_conv_block(up_16, kernel_size, 8*fn, dropout_rate, batch_norm)
        # UpRes 7
        gating_32 = gating_signal(up_conv_16, 4*fn, batch_norm)
        att_32 = attention_block(conv_32, gating_32, 4*fn)
        up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
        up_32 = layers.concatenate([up_32, att_32], axis=axis)
        up_conv_32 = res_conv_block(up_32, kernel_size, 4*fn, dropout_rate, batch_norm)
        # UpRes 8
        gating_64 = gating_signal(up_conv_32, 2*fn, batch_norm)
        att_64 = attention_block(conv_64, gating_64, 2*fn)
        up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
        up_64 = layers.concatenate([up_64, att_64], axis=axis)
        up_conv_64 = res_conv_block(up_64, kernel_size, 2*fn, dropout_rate, batch_norm)
        # UpRes 9
        gating_128 = gating_signal(up_conv_64, fn, batch_norm)
        att_128 = attention_block(conv_128, gating_128, fn)
        up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
        up_128 = layers.concatenate([up_128, att_128], axis=axis)
        up_conv_128 = res_conv_block(up_128, kernel_size, fn, dropout_rate, batch_norm)
    
        # 1*1 convolutional layers
        
        if use_coordinates:
            up_conv_128 = concatenate([xy, up_conv_128], name="concatenated_layer")
        
        conv_final = layers.Conv2D(self.n_classes, kernel_size=(1,1))(up_conv_128)
        conv_final = layers.BatchNormalization(axis=axis)(conv_final)
        outputs = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel
    
        # Model integration
        model = Model(inputs=[input_A, input_B], outputs=[outputs], name="Attention_ResUNet")
        return model
        
        