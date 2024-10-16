#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:04:54 2023

@author: silva

Unet for Geospatial data
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, SpatialDropout2D, UpSampling2D, BatchNormalization, Add, Concatenate
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Activation, Multiply, AveragePooling2D 

class Msk_Average_Pooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, strides=2, mask=None, padding='SAME', **kwargs):
        super(Msk_Average_Pooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.mask = mask

    def call(self, inputs, threshold=0.0, eps=1e-7):
        if self.mask is None:
            mask = tf.where(inputs > threshold, 1.0, 0.0)
        else:
            mask = self.mask

        # apply mask
        inputs = inputs * mask
        # Compute avg pooling
        avg_output = tf.nn.avg_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding)
        avg_mask = tf.nn.avg_pool2d(mask, ksize=self.pool_size, strides=self.strides, padding=self.padding)
        return avg_output/(avg_mask + eps)

class TSoftmax(tf.keras.layers.Layer):
    def __init__(self, temperature=1.0):
        super(TSoftmax, self).__init__()
        self.temperature = temperature

    def call(self, logits):
        scaled_logits = logits / self.temperature
        return tf.nn.softmax(scaled_logits)
    
def ms_attention(inputs, n_filters, rates=[4, 8, 16], kernel_initializer='glorot_uniform'):
    # downsample the input
    down_input = MaxPooling2D(pool_size=(2, 2))(inputs)
    attention_maps = []
    for rate in rates:
        x = DepthwiseConv2D((3, 3), padding="same", dilation_rate=rate)(down_input)
        attention_maps.append(x)
    
    concat = concatenate(attention_maps)
    redim = Conv2D(n_filters, 1, padding="same", kernel_initializer=kernel_initializer)(concat)
    rescale = UpSampling2D(size=(2, 2), interpolation='bilinear')(redim)
    weights = Activation('sigmoid')(rescale)
    output = Multiply()([inputs, weights])
    return output

class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.3, block_size=3, training_only=True):
        super(DropBlock2D, self).__init__()
        self.dropout_rate = dropout_rate
        self.block_size = block_size
        self.training_only = training_only

    def call(self, inputs, training=None):
        if self.training_only and not training:
            return inputs

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
        mask_scale = tf.reduce_sum(mask) / (tf.reduce_sum(tf.ones_like(mask)) + 1e-6)
        mask = mask / mask_scale
        output = tf.multiply(inputs, mask)

        return output
    
class Spatial_interp2d:

    def __init__(self, n_classes, dropout_rate=0.3, spatial_dropout=True, block_size=5, weights_path=None):
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.spatial_dropout = spatial_dropout
        self.block_size = block_size
        self.weights_path = weights_path
    
    def dropout_layer(self, x, dropout_rate=0.3, dropblock=True, spatial_dropout=True, training_only=False):
        if dropblock:
            x = DropBlock2D(dropout_rate, block_size=self.block_size, training_only=training_only)(x)
        if spatial_dropout:
            x = SpatialDropout2D(dropout_rate)(x)
        return x   
    
    def attention_gate(self, x, gating, filters):
        skip_connection = x
        # Apply additional convolutional layer to ensure compatibility
        x = Conv2D(filters, (1, 1), padding='same', strides=(2, 2))(x)
        g = Conv2D(filters, (1, 1), padding='same', strides=(1, 1))(gating)
        
        # addition 
        x_added = Add()([x, g])
        x_act = Activation('relu')(x_added)
        x_act = Conv2D(1, (1, 1), padding='same')(x_act)
        x_act = UpSampling2D(size=(2, 2), interpolation='bilinear')(x_act)
        sigmoid_weights = Activation('sigmoid')(x_act)
        output = Multiply()([skip_connection, sigmoid_weights])
        output = BatchNormalization()(output)
        
        return output

    def normalize(self, x, instance_norm=False):
        if instance_norm:
            x = BatchNormalization(axis=(1, 2))(x)
        else:
            x = BatchNormalization()(x)
        return x
    
    def depthwise_conv_block(self, inputs, num_filters, activation='relu', residual=True, instance_norm=False, dropout=False, dropblock=False, kernel_initializer='glorot_uniform'):
        
        # First convolution
        x = DepthwiseConv2D((7, 7), padding="same", kernel_initializer=kernel_initializer)(inputs)
        x = self.normalize(x, instance_norm)
        if dropout:
            x = self.dropout_layer(x, self.dropout_rate, False, self.spatial_dropout)
        x = Activation(activation)(x)
        
        # Second convolution
        x = Conv2D(num_filters, (1, 1), padding="same", kernel_initializer=kernel_initializer)(x)
        x = self.normalize(x, instance_norm)
        if dropout:
            x = self.dropout_layer(x, self.dropout_rate, False, self.spatial_dropout)
        x = Activation(activation)(x)
        
        if residual:
            # Skip connection with a 1x1 convolution
            skip_connection = Conv2D(num_filters, (1, 1), kernel_initializer=kernel_initializer)(inputs)
            # Add the first skip connection
            x = Add()([x, skip_connection])

        return x
    
    def regular_conv_block(self, inputs, num_filters, activation='relu', residual=True, dilation_rate=1, instance_norm=False,
                        dropout=False, dropblock=False, kernel_initializer='glorot_uniform', training_only=False):
        
        # First convolution
        x = Conv2D(num_filters, 3, padding="same", dilation_rate=dilation_rate, 
                   kernel_initializer=kernel_initializer)(inputs)
        x = self.normalize(x, instance_norm)
        if dropout:
            x = self.dropout_layer(x, self.dropout_rate, dropblock, self.spatial_dropout, training_only)
        x = Activation(activation)(x)

        # Second convolution
        x = Conv2D(num_filters, 3, padding="same", dilation_rate=dilation_rate,
                   kernel_initializer=kernel_initializer)(x)
        x = self.normalize(x, instance_norm)
        if dropout:
            x = self.dropout_layer(x, self.dropout_rate, False, self.spatial_dropout, training_only)
        x = Activation(activation)(x)
        
        if residual:
            # Skip connection with a 1x1 convolution
            residual = Conv2D(num_filters, (1, 1), kernel_initializer=kernel_initializer)(inputs)
            residual = self.normalize(residual)
            
            # Gating mechanism: apply a gate to the residual connection
            gate = Conv2D(num_filters, (1, 1), kernel_initializer=kernel_initializer, activation='sigmoid')(inputs)
            gated_residual = Multiply()([residual, gate])
            
            # Add the gated residual to the output of the second convolution
            x = Add()([x, gated_residual])
        
        return x
    
    def ms_conv(self, inputs, n_filters, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform'):
        
        skip = Conv2D(n_filters, 1, kernel_initializer=kernel_initializer)(inputs)
        skip = BatchNormalization()(skip)
        x = Conv2D(n_filters, kernel_size, padding="same", kernel_initializer=kernel_initializer)(inputs)
        x = BatchNormalization()(x)
        x = Add()([x, skip])
        x = Activation(activation)(x)

        return x
    
    def conv_block(self, inputs, num_filters, activation='relu', residual=True, dilation_rate=1, instance_norm=False, 
                   dropout=False, dropblock=False, use_depthwise_conv=False, training_only=False, name=None):
        if use_depthwise_conv:
            x = self.depthwise_conv_block(inputs, num_filters, activation=activation)
        else:
            x = self.regular_conv_block(inputs, num_filters, activation=activation, residual=residual, dilation_rate=dilation_rate, 
                                        instance_norm=instance_norm, dropout=dropout, dropblock=dropblock, training_only=training_only)
            
        if name is not None:
            x = Activation(activation, name=name)(x)
        else:
            x = Activation(activation)(x)
        return x
    
    def skip_connection_upsampling(self, x, skip_connection, n_filters, activation='relu', interpolation='bilinear'):

        x = Conv2D(n_filters, (1, 1))(x)
        x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
        x = self.normalize(x)
        x = Activation(activation)(x)
        x = concatenate([x, skip_connection])
        return x

    def enc_dec_block(self, features, n_filters, n_blocks=3, spatial_enc=None, suffix='one', interpolation='bilinear', activation='glorot_uniform',
                      use_dropout=True, dropblock=False, instance_norm=False, use_depthwise_conv=False, constrained=False):
        
        # Encoding path
        encoder_blocks = []
        pool_layers = []
        
        # Add the specified number of encoding blocks
        encoder_output = features
        num_filters = n_filters

        tf.print(f'---- {suffix} -----')
        for i in range(n_blocks):          
            if i > 0:
                if suffix == 'two':
                    encoder_output = concatenate([encoder_output, self.decoder_layers[::-1][i]])
                num_filters = n_filters*(2**i)

            # conv block
            encoder_output = self.conv_block(encoder_output, num_filters, activation=activation, instance_norm=instance_norm,
                                              dropout=use_dropout, dropblock=dropblock, use_depthwise_conv=use_depthwise_conv)
                    
            # downsampling conv block
            max_pool = MaxPooling2D(pool_size=(2, 2))(encoder_output)
            if constrained:
                avg_pool = Msk_Average_Pooling2D(pool_size=2)(encoder_output)
                concat = concatenate([max_pool, avg_pool])
                encoder_blocks.append(self.skip_connection_upsampling(concat, encoder_output, num_filters//2, activation=activation, interpolation=interpolation))
                encoder_output = concat
            else:
                encoder_blocks.append(encoder_output)
                encoder_output = max_pool
            pool_layers.append(encoder_output)

            # print layer information 
            tf.print('layer: ', i+1, '-', num_filters, ' filters')
        
        # Bottleneck
        num_filters = num_filters*2
        encoder_output = self.conv_block(pool_layers[-1], num_filters, activation=activation, instance_norm=instance_norm,
                                       dropout=use_dropout, dropblock=dropblock, use_depthwise_conv=use_depthwise_conv, training_only=True)
        
        # assign bottleneck layer
        decoder_output = encoder_output

        tf.print('layer: ', i+2, '-', num_filters, ' filters')
        tf.print('---')

        # store decoder layers
        decoder_layers = []

        for i in range(n_blocks-1, -1, -1):
            num_filters = num_filters//2

            # concatenate attention and upsampling output
            if constrained:
                concat = concatenate([encoder_blocks[i], spatial_enc[i]])
                attention_input = self.ms_conv(concat, num_filters, activation=activation)
            else:
                attention_input = encoder_blocks[i]

            attention_output = self.attention_gate(attention_input, decoder_output, num_filters)
            upsampling_output = UpSampling2D(size=(2, 2), interpolation=interpolation)(decoder_output)

            # concatenate attention output and upsampling output from the current decoder output
            decoder_input = concatenate([upsampling_output, attention_output])
            
            # conv block
            decoder_output = self.conv_block(decoder_input, num_filters, activation=activation, instance_norm=instance_norm,
                                             dropout=use_dropout, dropblock=dropblock, use_depthwise_conv=use_depthwise_conv, training_only=True)
            
            # store decoder layers
            decoder_layers.append(decoder_output)
        
        return decoder_output, decoder_layers
    
    def decoder_upsampling(self, decoder_layers, x, n_filters=32, activation='leaky_relu', use_dropout=True, interpolation='bilinear'):

        outputs = []
        scale = (2**self.n_blocks[0])//2        
        for x in decoder_layers:
            x = self.conv_block(x, n_filters, activation=activation, dropout=use_dropout, dropblock=use_dropout, training_only=True) 
            if scale > 1:
                x = UpSampling2D(size=(scale, scale), interpolation=interpolation)(x)
                scale //= 2
                
            outputs.append(x)

        return concatenate(outputs) 
    
    def spatial_constraint_block(self, y_hat, coordinates=None, n_filters=32, activation='leaky_relu', training_only=False):
        
        # apply threshold using relu
        y_hat = Activation('relu')(y_hat)
        self.ground_truth = y_hat

        # concatenate mask, coordinates
        if coordinates is not None:
            y_hat = concatenate([coordinates, y_hat])

        # spatial embeddings
        spatial_enc = []
        num_filters = n_filters
        x = y_hat
        for _ in range(self.n_blocks[0]+1):
            x = self.conv_block(x, num_filters, activation=activation,  dropout=True, dropblock=True, use_depthwise_conv=False, training_only=training_only)
            max_pool = MaxPooling2D(pool_size=(2, 2))(x)
            avg_pool = Msk_Average_Pooling2D(pool_size=2)(x)
            concat = concatenate([max_pool, avg_pool])
            spatial_enc.append(self.skip_connection_upsampling(concat, x, num_filters//2, activation=activation, interpolation='bilinear'))
            x = concat
            num_filters *= 2

        return spatial_enc
    
    def add_ground_truth(self, x, n_classes=None):
        max_value = tf.reduce_max(x)
        gt = max_value * self.ground_truth
        if n_classes is not None:
            gt = gt[:, :, :, :n_classes]

        # add ground truth only during inference
        def apply_inference_phase():
            return Add()([x, gt])

        # Use identity (no-op) in training phase
        return tf.keras.backend.in_train_phase(x, apply_inference_phase)
    
    def unet_model(self, dim, n_filters, n_blocks=4, activation='softmax', temperature=1.0):
        
        inputs = Input(dim)
        self.decoder_layer = {}
        self.threshold = 0.0
        self.activation = activation
        self.temperature = temperature

        # u-net model
        embeddings, _ = self.enc_dec_block(inputs, n_filters, n_blocks=n_blocks, use_dropout=True,
                                            dropblock=True, activation='leaky_relu', instance_norm=False)

        # classification layer
        k_outputs = Conv2D(self.n_classes, (1, 1))(embeddings)
        if activation is not None:
            k_outputs = TSoftmax(temperature=self.temperature)(k_outputs)
        model = Model(inputs=inputs, outputs=[k_outputs, embeddings])
        self.non_spatial_model = model
        
        return self.non_spatial_model
    
    # Define the Spatially Constrained model
    def bayesian_constrained_model(self, dim_1, dim_2, n_filters=32, n_blocks=(4, 4),
                                    threshold=0., n_features=None, coord_channels=0, pretrained_encoder=None,
                                    activation='softmax', temperature=1.0, brute_force=False):
        
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.pretrained_encoder = pretrained_encoder
        self.activation = activation
        self.threshold = threshold
        self.n_blocks = n_blocks
        self.temperature = temperature
        
        input_X = Input(dim_1)
        input_Y = Input(dim_2)
        
        if coord_channels > 0:
            features = input_X[:, :, :, coord_channels:]
            coords = input_X[:, :, :, :coord_channels]        
        else:
            features = input_X
            coords = None 

        # spatial constraint
        spatial_enc = self.spatial_constraint_block(input_Y, coords, n_filters, activation='leaky_relu', training_only=False)
        
        # first u-net 
        embeddings_m, decoder_layers = self.enc_dec_block(features, n_filters, n_blocks=self.n_blocks[0], spatial_enc=spatial_enc, use_dropout=True, dropblock=True,
                                          activation='leaky_relu', suffix='one', use_depthwise_conv=False, constrained=True) 

        #final stage 
        embeddings_m = self.decoder_upsampling(decoder_layers, n_filters, activation='leaky_relu', use_dropout=True)
        embeddings_l = self.conv_block(embeddings_m, 2*n_filters, activation='leaky_relu', residual=True, dropout=True, 
                                        dropblock=True, use_depthwise_conv=False, training_only=True, name='final_layer')

        
        k_outputs = Conv2D(self.n_classes, (1, 1), 
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='zeros')(embeddings_l)
        
        if brute_force:
            k_outputs = self.add_ground_truth(k_outputs)

        if activation is not None:
            k_outputs = TSoftmax(temperature=self.temperature)(k_outputs)

        self.spatial_model = Model(inputs=[input_X, input_Y], outputs=[k_outputs, embeddings_l])
          
        return  self.spatial_model 
    
    def pretrained_model(self, weights_path=None, n_classes=None, brute_force=False):
        
        if weights_path is None:
            weights_path = self.weights_path
            
        # load weights
        self.spatial_model.load_weights(weights_path)
        
        # Set the trainable status of original model's layers to True
        for layer in self.spatial_model.layers:
            layer.trainable = True
    
        # get penultimate layer
        intermediate_model = Model(inputs=self.spatial_model.input, 
                                   outputs=self.spatial_model.get_layer('final_layer').output)
        embeddings = intermediate_model([self.spatial_model.input])
        k_outputs = Conv2D(n_classes, (1, 1), 
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='zeros')(embeddings)
        
        if brute_force:
            k_outputs = self.add_ground_truth(k_outputs, n_classes=n_classes)

        if self.activation is not None:
            k_outputs = TSoftmax(temperature=self.temperature)(k_outputs)
        self.spatial_model = Model(inputs=self.spatial_model.input, outputs=[k_outputs, embeddings])
        
        return  self.spatial_model
    
    def predict(self, X, verbose=0, spatial_model=True):
        
        if spatial_model:
            predictions, embeddings = self.spatial_model.predict(X, verbose=verbose)[:2]
            if self.activation is None:
                predictions = TSoftmax(temperature=self.temperature)(predictions)
        else:
            k_outputs, embeddings = self.non_spatial_model.predict(X, verbose=verbose)[:2]
            if self.activation is None:
                predictions = TSoftmax(temperature=self.temperature)(k_outputs)
        return predictions, embeddings
                      