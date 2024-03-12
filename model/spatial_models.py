#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:04:54 2023

@author: silva

Unet for Geographic data
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, SpatialDropout2D, UpSampling2D, BatchNormalization, Add, GaussianNoise
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Activation, AveragePooling2D, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19   

def atrous_spatial_pyramid_pooling(inputs, filters=64, residual=True, batch_norm=True, activation='relu'):
    
    # Atrous convolutions with different dilation rates
    aspp1 = Conv2D(filters, (1, 1), activation='relu', padding='same')(inputs)
    aspp2 = Conv2D(filters, (3, 3), dilation_rate=6, activation='relu', padding='same')(inputs)
    aspp3 = Conv2D(filters, (3, 3), dilation_rate=12, activation='relu', padding='same')(inputs)
    aspp4 = Conv2D(filters, (3, 3), dilation_rate=18, activation='relu', padding='same')(inputs)
    aspp5 = Conv2D(filters, (3, 3), dilation_rate=24, activation='relu', padding='same')(inputs)

    # Global average pooling
    global_avg_pooling = GlobalAveragePooling2D()(inputs)
    global_avg_pooling = tf.expand_dims(tf.expand_dims(global_avg_pooling, 1), 1)
    global_avg_pooling = Conv2D(filters, (1, 1), activation='relu')(global_avg_pooling)
    global_avg_pooling = tf.image.resize(global_avg_pooling, (tf.shape(inputs)[1], tf.shape(inputs)[2]), method=tf.image.ResizeMethod.BILINEAR)

    # Concatenate the ASPP outputs and the global average pooling
    concatenated = concatenate([aspp1, aspp2, aspp3, aspp4, aspp5, global_avg_pooling])

    # Additional convolution to combine the features
    output = Conv2D(filters, (1, 1), activation=activation)(concatenated)
    
    if batch_norm:
        aspp1 = BatchNormalization()(aspp1)  
        output = BatchNormalization()(output)   
    
    if residual:
        output = Add()([aspp1, output])
        
    return output

def Adapted_ASPP(inputs, filters=256, n_classes=None, residual=True, batch_norm=True, activation='relu'):
    
    if n_classes is None:
        n_classes == filters
    
    # Atrous convolutions with different dilation rates
    aspp1 = Conv2D(filters, (3, 3), dilation_rate=6, activation='relu', padding='same')(inputs)
    aspp2 = Conv2D(filters, (3, 3), dilation_rate=12, activation='relu', padding='same')(inputs)
    aspp3 = Conv2D(filters, (3, 3), dilation_rate=18, activation='relu', padding='same')(inputs)
    aspp4 = Conv2D(filters, (3, 3), dilation_rate=24, activation='relu', padding='same')(inputs)

    # Global average pooling
    global_avg_pooling = GlobalAveragePooling2D()(inputs)
    global_avg_pooling = tf.expand_dims(tf.expand_dims(global_avg_pooling, 1), 1)
    global_avg_pooling = Conv2D(filters, (1, 1), activation='relu')(global_avg_pooling)
    global_avg_pooling = tf.image.resize(global_avg_pooling, (tf.shape(inputs)[1], tf.shape(inputs)[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # SUM the ASPP outputs and the global average pooling
    sm = Add()([aspp1, aspp2, aspp3, aspp4, global_avg_pooling])

    # Additional convolution to combine the features
    output = Conv2D(n_classes, (1, 1), activation=activation)(sm)
    
    return output

class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.2, block_size=3):
        super(DropBlock2D, self).__init__()
        self.dropout_rate = dropout_rate
        self.block_size = block_size

    def call(self, inputs):

        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)

        # Calculate the number of blocks in each dimension
        num_blocks_h = input_shape[1] // self.block_size
        num_blocks_w = input_shape[2] // self.block_size

        # Create a mask to determine which blocks to drop
        uniform_dist = tf.random.uniform([input_shape[0], num_blocks_h, num_blocks_w, 1], dtype=inputs.dtype)

        size = input_shape[1:3]  # Get spatial dimensions of the input tensor
        uniform_dist = tf.image.resize(uniform_dist, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        mask = tf.where(uniform_dist > self.dropout_rate, tf.ones_like(uniform_dist), tf.zeros_like(uniform_dist))

        # Scale the mask to maintain the expected mean of the output
        mask_scale = tf.reduce_sum(mask) / tf.reduce_sum(tf.ones_like(mask))
        mask = mask / mask_scale
        output = tf.multiply(inputs, mask)

        return output
        
class StochasticDownsampling(tf.keras.layers.Layer):
    def __init__(self, pool_size, stride, padding='same', alpha=7):
        super(StochasticDownsampling, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
        
    def __call__(self, input_tensor):
        self.input_tensor = input_tensor
        self.bsp = tf.shape(input_tensor)[0] 
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        noise_w = tf.random.uniform([1, self.bsp, 1, w], maxval=self.alpha, dtype=tf.float32)
        noise_h = tf.random.uniform([1, self.bsp, h, 1], maxval=self.alpha, dtype=tf.float32)
        noise_mat = tf.transpose(tf.matmul(noise_h, noise_w), (1, 2, 3, 0))
        noise_mat = noise_mat - tf.reduce_max(noise_mat)
        noise_mat = tf.exp(noise_mat)
        masked = tf.multiply(self.input_tensor, noise_mat)
        noise_pool = tf.keras.layers.AveragePooling2D(self.pool_size,
                                                     self.stride,
                                                     self.padding)(noise_mat)
        masked_pool = tf.keras.layers.AveragePooling2D(self.pool_size,
                                                      self.stride,
                                                      self.padding)(masked)
        return masked_pool / noise_pool

class s3pool():
    def __init__(self, pool_size=(2, 2), stride=2, padding='same', alpha=7):
        super(s3pool, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
    
    def __call__(self, input_tensor):
        l1 = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=(1, 1), padding=self.padding)(input_tensor)
        output = StochasticDownsampling(self.pool_size, self.stride, self.padding, self.alpha)(l1)
        return output

def softmax_with_temperature(x, temperature=1.3):
    return tf.exp(x/temperature)/tf.reduce_sum(tf.exp(x/temperature))

class Spatial_interp2d:
    
    def __init__(self, n_classes, dropout_rate=0.3, spatial_dropout=False, block_size=2, ASPP=False,
                 weights_path=None):
        
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.spatial_dropout = spatial_dropout
        self.block_size = block_size
        self.ASPP = ASPP
        self.weights_path = weights_path
        
    def dropout_layer(self, x, dropout_rate=0.1, spatial_dropout=False):
        
        if spatial_dropout:
            x = SpatialDropout2D(dropout_rate)(x)
        else:
            x = Dropout(dropout_rate)(x)
       
        return x   
    
    def block_distribution(self, inputs, block_size=3):
        '''
        DropBlock: A regularization method for convolutional networks
        https://proceedings.neurips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
        '''
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)
        # Calculate the number of blocks in each dimension
        num_blocks_h = input_shape[1] // block_size
        num_blocks_w = input_shape[2] // block_size
        # Create a mask to determine which blocks to drop
        uniform_dist = tf.random.uniform([input_shape[0], num_blocks_h, num_blocks_w, 1], dtype=inputs.dtype)
        size = input_shape[1:3]  # get spatial dimensions of input tensor
        uniform_dist = tf.image.resize(uniform_dist, size, method='nearest')
        
        return uniform_dist
    

    def conv_block(self, inputs, num_filters, kernel_sizes=[3, 5, 7], strides=1, padding='same', activation='relu', ASPP=False,
                   initializer='he_normal', bias_initializer='zeros', batch_norm=True, name=None, multiscale=False, residual=True, use_dropout=True):
            
        if multiscale:
            x = inputs
            for kernel_size in kernel_sizes:
                x = Conv2D(num_filters, kernel_size=kernel_size, padding=padding, kernel_initializer=initializer)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            x = Activation(activation)(x)

        else:
            if ASPP:
                x = atrous_spatial_pyramid_pooling(inputs, num_filters)
            else:
                x = Conv2D(num_filters, kernel_sizes[0], activation=activation, padding=padding, kernel_initializer=initializer)(inputs)
                x = Conv2D(num_filters, kernel_sizes[0], activation=activation, padding=padding, kernel_initializer=initializer)(x)

            if batch_norm:
                x = BatchNormalization()(x)
            x = Activation(activation)(x)
                
        if residual:    
            skip_connection = Conv2D(num_filters, (1, 1), padding=padding, strides=strides)(inputs)
            if batch_norm:
                skip_connection = BatchNormalization()(skip_connection)
            x = Add()([x, skip_connection])
            x = Activation(activation)(x)
            
        if use_dropout:
            x = DropBlock2D(self.dropout_rate, block_size=self.block_size)(x)    
        return x
    
    def attention_block(self, x, gating, filters, kernel_size=3):
        
        skip_connection = x
        # Apply additional convolutional layer to ensure compatibility
        x = Conv2D(filters, (1, 1), padding='same', strides=(2, 2))(x)
        g = Conv2D(filters, (1, 1), padding='same', strides=(1, 1))(gating)
            
        # addition 
        x_added = Add()([x, g])
        x_act = Conv2D(1, (1, 1), activation='relu')(x_added)
        sigmoid_weights = Activation('sigmoid')(x_act)
        sigmoid_weights = UpSampling2D(size=(2, 2))(sigmoid_weights)
        output = tf.multiply(skip_connection, sigmoid_weights)
        output = BatchNormalization()(output)
        
        return output

    def res_block(self, inputs, num_filters, kernel_size=3, strides=1, padding='same', activation='relu', name=None):
        
        if name is not None:
            name_conv_1 = f'{name}_conv1'  
        else:
            name_conv_1 = name
          
        skip_connection = Conv2D(num_filters, (1, 1), padding='same')(inputs)
        x = self.conv_block(inputs, num_filters, kernel_size, strides, padding, activation, name=name_conv_1, multiscale=False)
        
        x = Add()([x, skip_connection])
        x = Activation(activation)(x)
        x = BatchNormalization()(x)
        
        return x
    
    def enc_dec_block(self, features, n_filters, n_blocks=2, batch_norm=True, multiscale=False, residual=True, suffix='one', interpolation='bilinear', 
                      activation='relu', use_dropout=True, Pool_layer=MaxPooling2D):
        
        # Encoding path
        encoder_blocks = []
        pool_layers = []
        
        # Add the specified number of encoding blocks
        encoder_output = features
        num_filters = n_filters
        if self.pretrained_encoder is not None:
            num_filters *= 2 
        tf.print(f'---- {suffix} ----')
        vgg_layer_name_1 = 'block1_conv2'
        for i in range(n_blocks):            
            if i > 0:
                    num_filters = n_filters*(2**i)
                    vgg_layer_name_1 = 'block' + str(i+1) + '_conv1'
                    vgg_layer_name_2 = 'block' + str(i+1) + '_conv2'
                    
            # conv block - residual + attention
            if self.pretrained_encoder is None:
                encoder_output = self.conv_block(encoder_output, num_filters, batch_norm=batch_norm, activation=activation, 
                                               multiscale=False, use_dropout=use_dropout, residual=residual, name='mconv'+str(i+1)+suffix)
            # VGG19 encoder
            else:
                encoder_output = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(encoder_output)
                encoder_output = self.vgg.get_layer(vgg_layer_name_1)(encoder_output)
                encoder_output = BatchNormalization()(encoder_output)
                encoder_output = Activation(activation)(encoder_output)
                if i > 0:
                    encoder_output = self.vgg.get_layer(vgg_layer_name_2)(encoder_output)
                encoder_output = BatchNormalization()(encoder_output)
                encoder_output = Activation(activation)(encoder_output)
            
            if suffix == 'two':
                if (i+1) < n_blocks:
                    encoder_output = self.attention_block(encoder_output, self.decoder_layer['one'][-(i+2)], num_filters)
                
            encoder_blocks.append(encoder_output)
            encoder_output = Pool_layer(pool_size=(2, 2))(encoder_output)
            encoder_output = self.dropout_layer(encoder_output, self.dropout_rate, spatial_dropout=self.spatial_dropout)
            pool_layers.append(encoder_output)     
            tf.print(i, '-', num_filters)
            
        # Bottleneck
        num_filters = num_filters*2
        bottleneck = self.conv_block(pool_layers[-1], num_filters, batch_norm=batch_norm, name='bottleneck_'+suffix, 
                                     use_dropout=use_dropout, multiscale=False, residual=residual)
        if self.pretrained_encoder is not None:
            bottleneck = self.vgg.get_layer('block'+str(n_blocks+1)+'_conv1')(bottleneck)
            bottleneck = BatchNormalization()(bottleneck)
                        
        tf.print(i+1, '-', num_filters)
        tf.print('---')
        
        # Decoder with attention blocks
        decoder_output = bottleneck
    
        self.decoder_layer[suffix] = []
        for i in range(n_blocks-1, -1, -1):
            num_filters = num_filters//2
            attention_output = self.attention_block(encoder_blocks[i], decoder_output, filters=num_filters)
            upsampling_output = UpSampling2D(size=(2, 2), interpolation=interpolation)(decoder_output)
            decoder_input = concatenate([upsampling_output, attention_output])
            decoder_output = self.conv_block(decoder_input, num_filters, batch_norm=batch_norm, residual=residual, use_dropout=use_dropout)
            self.decoder_layer[suffix].append(decoder_output)
        last_layer = self.conv_block(decoder_input, num_filters, batch_norm=batch_norm, ASPP=False,
                                      multiscale=multiscale, activation=activation, name='last_layer_' + suffix, use_dropout=False, residual=True)
        
        return last_layer
    
        
    def spatial_constraint(self, x, y, coordinates=None,  n_filters=32, hold_out=0.1):
        
        # block dropout for cross-validation
        uniform_dist = self.block_distribution(y)
        # create block masks
        block_mask = tf.where(uniform_dist > hold_out, 1., 0.)
        mask_bool = tf.where(y > self.threshold, 1., 0.)
        mask = tf.multiply(mask_bool, block_mask)
        # supress == -9999. values
        y_hat = tf.multiply(y, mask)
        
        
        if coordinates is not None:
            # concatenate mask, coordinates
            y_hat = concatenate([coordinates, y_hat])
            
        # ground-truth and coordinates
        self.x_spatial = self.conv_block(y_hat, n_filters, kernel_sizes=(3, 3), batch_norm=True, activation='relu', 
                                            multiscale=False, use_dropout=False, residual=True, name='spatial_info') 
        input_concat = concatenate([self.x_spatial, x])
        
        interp_output = self.enc_dec_block(input_concat, n_filters, n_blocks=self.n_blocks[1], 
                                           multiscale=self.multiscale, use_dropout=False,
                                                  suffix='two', Pool_layer=AveragePooling2D)
        

        self.concat_output = concatenate([interp_output, x], name='final_embeddings')
        if self.ASPP:
            self.concat_output = atrous_spatial_pyramid_pooling(self.concat_output, n_filters*2)
        output = Conv2D(self.n_classes, (1, 1), activation=self.activation, name='softmax_layer')(self.concat_output)

        return output
    
    def unet_model(self, dim, n_filters, n_blocks=3, batch_norm=True, multiscale=False, pretrained_encoder=None,
                   encoder_freeze=False, activation='softmax'):
        
        features = Input(dim)
        coord_channels = 0
        self.pretrained_encoder = pretrained_encoder
        self.decoder_layer = {}
        
        if pretrained_encoder is not None:
            tf.print(f'Encoder: {pretrained_encoder}')
            # fixed number of filters
            n_filters = 32
            self.vgg = VGG19(weights='imagenet', include_top=False, input_shape=(dim[0], dim[1], 3))
            
            trainable = True
            if encoder_freeze:
                trainable = False
            # freeze all layers in resnet50 
            for layer in self.vgg.layers:
                layer.trainable = trainable
            
            vgg_layers = []
            k_max = dim[-1]-coord_channels
            for k in range(0, k_max, 3):
                if tf.abs(k_max-k) >= 3:
                    vgg_layers.append(self.vgg.get_layer('block1_conv1')(features[:, :, :, k:k+3]))
                else:
                    vgg_layers.append(self.vgg.get_layer('block1_conv1')(features[:, :, :,-3:]))

            x = concatenate(vgg_layers)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = self.dropout_layer(x, spatial_dropout=self.spatial_dropout)
            x = Conv2D(n_filters, (1, 1))(x)
            x = BatchNormalization()(x)
            features = Activation('relu')(x)
        
        
        last_conv = self.enc_dec_block(features, n_filters, n_blocks=n_blocks, batch_norm=batch_norm,
                                       multiscale=multiscale, activation='leaky_relu', Pool_layer=AveragePooling2D)
        
        
        outputs = Conv2D(self.n_classes, (1, 1), activation=activation)(last_conv)
        model = Model(inputs=features, outputs=[outputs, last_conv])
        self.not_spatial_model = model
        
        return model
    
    # Define the Spatially Constrained model
    def bayesian_constrained_model(self, dim_1, dim_2, kernel_size=3, n_filters=32, n_blocks=(2, 3),
                                    threshold=0., hold_out=0.1,  multiscale=True, coord_channels=3, 
                                    batch_norm=True, pretrained_encoder=None, encoder_freeze=False, activation='softmax'):
        
        '''
        For each convolutional layer in the model, the code sets the kernel and bias prior distributions 
        using the make_default_prior function, which creates a prior distribution with a specified dtype,
        shape, and name. The code also sets the kernel and bias posterior distributions using the default_mean_field_normal_fn 
        function from TensoprFlow Probability (TFP), which creates a Gaussian distribution with learnable mean and standard deviation
        parameters.

        By setting the prior and posterior distributions for each layer in the model, 
        this code allows for probabilistic inference and uncertainty quantification during model 
        training and evaluation.
        
        '''

        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.pretrained_encoder = pretrained_encoder
        self.activation = activation
        self.threshold = threshold
        self.multiscale = multiscale
        self.n_blocks = n_blocks
        self.decoder_layer = {}
        
        input_X = Input(dim_1)
        input_Y = Input(dim_2)
        
        if coord_channels > 0:
            features = input_X[:, :, :, coord_channels:]
            coords = input_X[:, :, :, :coord_channels]        
        else:
            features = input_X
            coords = None
        
        if pretrained_encoder is not None:
            tf.print(f'Encoder: {pretrained_encoder}')
            # fixed number of filters
            n_filters = 32
            self.vgg = VGG19(weights='imagenet', include_top=False, input_shape=(self.dim_1[0], self.dim_1[1], 3))
            
            trainable = True
            if encoder_freeze:
                trainable = False
            # freeze all layers in resnet50 
            for layer in self.vgg.layers:
                layer.trainable = trainable            
        
        # first u-net (secondary variables)
        #noisy_features = GaussianNoise(stddev)(features)
        embeddings = self.enc_dec_block(features, n_filters, n_blocks=self.n_blocks[0], 
                                        batch_norm=batch_norm, multiscale=False, 
                                        activation='relu', Pool_layer=MaxPooling2D)
         
        # second u-net (secondary variables + coordinates + ground-truth)
        class_output = self.spatial_constraint(embeddings, input_Y, coords, n_filters, hold_out=hold_out)
        self.spatial_model = Model(inputs=[input_X, input_Y], outputs=[class_output, self.concat_output])
          
        return  self.spatial_model 
    
    def pretrained_model(self, weights_path=None, n_classes=None):
        
        if weights_path is None:
            weights_path = self.weights_path
            
        # load weights
        self.spatial_model.load_weights(weights_path)
        
        # Set the trainable status of original model's layers to True
        for layer in self.spatial_model.layers:
            layer.trainable = True
    
        # get penultimate layer
        intermediate_model = Model(inputs=self.spatial_model.input, outputs=self.spatial_model.get_layer('final_embeddings').output)
        embeddings = intermediate_model([self.spatial_model.input])
        #embeddings= concatenate(embeddings, axis=-1)
        outputs = Conv2D(n_classes, (1, 1), activation=self.activation)(embeddings)
        self.spatial_model = Model(inputs=self.spatial_model.input, outputs=[outputs, embeddings])
        
        return  self.spatial_model
    
    
    def predict(self, X, verbose=0, spatial_model=True):
        
        if spatial_model:
            return self.spatial_model.predict(X, verbose=verbose)
        else:
            return self.not_spatial_model.predict(X, verbose=verbose)
                      