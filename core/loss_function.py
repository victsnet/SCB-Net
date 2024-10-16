#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:51:51 2023

@author: silva
"""
import tensorflow as tf


class TSoftmax(tf.keras.layers.Layer):
    def __init__(self, temperature=1.0):
        super(TSoftmax, self).__init__()
        self.temperature = temperature

    def call(self, logits):
        scaled_logits = logits / self.temperature
        return tf.nn.softmax(scaled_logits)

def dilation2d(inputs, kernel_size):
            
    inputs = tf.convert_to_tensor(inputs)
    kernel = tf.ones((kernel_size, kernel_size, inputs.shape[-1]), dtype=inputs.dtype)
    output = tf.nn.dilation2d(inputs, filters=kernel, strides=(1, 1, 1, 1), data_format='NHWC', dilations=(1, 1, 1, 1), padding="SAME")
    output = output - tf.ones_like(output)
            
    return output    

class Masked_Average_Pooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, strides=2, mask=None, padding='SAME', **kwargs):
        super(Masked_Average_Pooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.mask = mask

    def call(self, inputs, threshold=0.0, eps=1e-7):
        if self.mask is None:
            mask = tf.where(inputs > threshold, 1.0, 0.0)
        else:
            mask = self.mask
        # Apply the mask to the inputs
        inputs = inputs * mask
        # Compute avg pooling
        avg_output = tf.nn.avg_pool2d(inputs, ksize=self.pool_size, strides=self.strides, padding=self.padding)
        avg_mask = tf.nn.avg_pool2d(mask, ksize=self.pool_size, strides=self.strides, padding=self.padding)
        return avg_output/(avg_mask + eps)
    

class spatial_losses:

    def __init__(self, dim, ths=0.5, fw=[0.5, 0.2, 0.15, 0.15], q=0.7,
                 fs=[1, 3, 5, 11], fname='dilation', interp_type='classification', temperature=1.5):
        self.dim = dim
        self.ths = ths
        self.interp_type = interp_type
        self.fs = fs
        self.fw = fw
        self.fname = fname
        self.temperature = temperature
        self.q = q

    def gauss_kernel(self, channels, kernel_size, sigma=1.5):

        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        
        return kernel
    
    def gaussian_blur(self, img, kernel_size=11, sigma=1.5):

        gaussian_kernel = self.gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
        gaussian_kernel = gaussian_kernel[..., tf.newaxis]
        output = tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                          padding='SAME', data_format='NHWC')        
        return output
        
    def dilation2d(self, inputs, kernel_size, strides=1):
            
        inputs = tf.convert_to_tensor(inputs)
        kernel = tf.ones((kernel_size, kernel_size, inputs.shape[-1]), dtype=inputs.dtype)
        output = tf.nn.dilation2d(inputs, filters=kernel, strides=(1, strides, strides, 1), data_format='NHWC', dilations=(1, 1, 1, 1), padding="SAME")
        output = output - tf.ones_like(output)
            
        return output
    
    def mse_loss(self, y_true, y_pred):

        # distance between the predictions and simulation probabilities
        squared_diff = tf.square(y_true-y_pred)
        loss = tf.reduce_mean(squared_diff)
        return loss
    
    def calculate_weights(self, y_true):
        # Sum up occurrences of each class across all dimensions except for the last one (class dimension)
        class_totals = tf.reduce_sum(y_true, axis=(0, 1, 2), keepdims=True)

        # total number of pixels
        total_pixels = tf.reduce_sum(y_true)
        
        # Calculate the weights as the inverse of the class totals
        weights = total_pixels / (class_totals + 1e-7)
        
        return weights
    

    def accuracy(self, y_true, logits, rescale=None, epsilon=1e-7, epoch=None):
        """
        Calculate the accuracy of predictions.

        Args:
            y_true (tf.Tensor): Ground truth labels.
            y_pred (tf.Tensor): Predicted labels.
            rescale (Optional[int]): Rescaling factor for average pooling.
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            tf.Tensor: Accuracy score.
        """
        # Ensure inputs are float32 tensors
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)

        if tf.reduce_min(logits) < 0.0 or tf.reduce_max(logits) > 1.0:
            y_pred = TSoftmax(self.temperature)(logits)
        else:
            y_pred = logits

        # get k value
        if epoch is not None:
            k = 0.1
            truncation_mask = tf.where(y_pred > k, 1.0, 0.0)
            y_true = y_true * truncation_mask

        # Get the predicted indices and one-hot encode them
        indices = tf.argmax(logits, -1)
        n_classes = logits.shape[-1]
        y_pred = tf.one_hot(indices, n_classes)

        # Apply rescaling if specified
        if rescale is not None:
            y_true = self.avg_pool2d(y_true, rescale, masked=True, threshold=0.0)
            y_pred = self.avg_pool2d(y_pred, rescale)

        # Binarize y_true
        y_true = tf.where(y_true > 0.5, 1.0, 0.0)
        mask = tf.where(y_true > 0.5, 1.0, 0.0)

        # Apply mask to y_pred
        y_pred = y_pred * mask

        # Calculate sample weights
        sample_weight = tf.reduce_sum(mask) / (tf.reduce_sum(mask, axis=(0, 1, 2)) + 1e-6)

        # Calculate accuracy
        bool_mask = tf.cast(tf.equal(y_true, y_pred), tf.float32) * mask
        acc = tf.reduce_sum(bool_mask, axis=(0, 1, 2)) / (tf.reduce_sum(mask, axis=(0, 1, 2)) + epsilon)

        # Apply sample weights to accuracy
        acc = tf.reduce_sum(acc * sample_weight) / tf.reduce_sum(sample_weight)
        
        return acc
    
    def get_n_samples(self, y_true):
        mask = tf.where(y_true > 0.5, 1.0, 0.0)
        n_samples = tf.reduce_sum(mask)
        n_samples = tf.cast(n_samples, tf.int32)
        return n_samples 
    
    def variance_filter(self, inputs, filter_size=3, filter_sigma=1.5, eps=1e-7, padding='SAME'):
        # gaussian filter
        kernel = self.gauss_kernel(tf.shape(inputs)[-1], filter_size, filter_sigma)
        kernel = kernel[..., tf.newaxis]

        # compute mean
        mu = tf.nn.depthwise_conv2d(inputs, kernel, strides=[1,1,1,1], padding=padding, data_format='NHWC') 

        # compute variance
        var = tf.nn.depthwise_conv2d(tf.square(inputs), kernel, strides=[1,1,1,1], padding=padding, data_format='NHWC') - tf.square(mu)

        return var + eps
        
    def avg_filter(self, inputs, filter_size=11, filter_sigma=1.5, strides=1):
        # gaussian filter
        kernel = self.gauss_kernel(tf.shape(inputs)[-1], filter_size, filter_sigma)
        kernel = kernel[..., tf.newaxis]

        # compute mean
        mu = tf.nn.depthwise_conv2d(inputs, kernel, strides=[1,strides,strides,1],
                                    padding='SAME', data_format='NHWC') 
        
        return mu
    
    def scharr_filter(self, img, eps=1e-7, norm=False):
        scharr_x = tf.constant([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=tf.float32)
        scharr_x = scharr_x[:, :, tf.newaxis, tf.newaxis]
        scharr_x = tf.tile(scharr_x, [1, 1, tf.shape(img)[-1], 1])

        scharr_y = tf.constant([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=tf.float32)
        scharr_y = scharr_y[:, :, tf.newaxis, tf.newaxis]
        scharr_y = tf.tile(scharr_y, [1, 1, tf.shape(img)[-1], 1])

        grad_x = tf.nn.depthwise_conv2d(img, scharr_x, [1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.depthwise_conv2d(img, scharr_y, [1, 1, 1, 1], padding='SAME')

        grad = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))
        if norm:
            grad /= tf.reduce_max(grad, axis=(1, 2), keepdims=True) 
        return grad + eps    
    
    def ms_cce_loss(self, y_true, logits, mask, gamma=2.0, alpha=0.25, epsilon=1e-7):

        y_pred = TSoftmax(temperature=self.temperature)(logits)

        # Cast mask to float32
        mask = tf.cast(mask, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.multiply(y_true, mask)

        # Initialize weights
        weights = tf.reduce_sum(mask)/(tf.reduce_sum(mask, axis=(0, 1, 2), keepdims=True) + epsilon)
                
        # Compute MS-CCE for each scale
        cce_list = []
        
        # Avoid division by zero
        eps = 1e-5
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        # Compute focal weights
        focal_weights = alpha * tf.pow(1 - y_pred, gamma) 
    
        filtered_y_pred = y_pred
        
        for filter_weight, filter_size in zip(self.fw, self.fs):

            if filter_size > 1:                
                # filter the images for the next scale
                if self.fname == 'dilation':
                    filtered_y_pred = self.dilation2d(y_pred, filter_size)

                elif self.fname == 'gaussian':
                    filtered_y_pred = self.gaussian_blur(y_pred, filter_size, sigma=1.5)
                    
                elif self.fname == 'avg' or self.fname == 'average':
                    filtered_y_pred = tf.nn.avg_pool(y_pred, ksize=[1, filter_size, filter_size, 1], 
                                                    strides=[1, 1, 1, 1], padding='SAME')

            # calculate cross-entropy loss where there are samples
            cce = focal_weights * y_true * tf.math.log(filtered_y_pred + epsilon)
            cce *= weights
            cce_sum = -tf.reduce_sum(cce, axis=-1)
            cce_mean = tf.reduce_sum(cce_sum)/(tf.reduce_sum(mask) + epsilon)
            cce_list.append(cce_mean * filter_weight)
        
        # Compute final loss
        fcce = tf.reduce_sum(cce_list)
        
        return fcce

    def spatial_loss(self, y_true, y_pred, training_set=True):
        
        if training_set is False:
            self.hold_out = 0.0
        
        if self.interp_type == 'regression':
            # calculate the loss only where there are samples
            mask = tf.where(y_true > self.ths, 1.0, 0.)
            if self.hold_out > 0.0:
                mask = self.dropblock(mask, self.block_size, self.hold_out)
            loss = self.ms_reg_loss(y_true, y_pred, mask)

        elif self.interp_type == 'classification':
            # calculate the loss only where there are samples
            mask = tf.where(y_true > self.ths, 1.0, 0.)
              
            if self.hold_out > 0.0:
                mask = self.dropblock(mask, self.block_size, self.hold_out)
            loss = self.ms_cce_loss(y_true, y_pred, mask)

        else:
            raise ValueError('Wrong interp_type!')

        return loss
    
    def avg_pool2d(self, inputs, filter_size=2, masked=False, threshold=0.0, one_hot_mask=False):
        if masked:
            if one_hot_mask:
                # Get the predicted indices and one-hot encode them
                indices = tf.argmax(inputs, -1)
                n_classes = inputs.shape[-1]
                mask = tf.one_hot(indices, n_classes)
            else:
                mask = tf.where(inputs > threshold, 1.0, 0.0)

            # mask inputs
            inputs = inputs * mask
            # Compute avg pooling
            avg_output = tf.nn.avg_pool2d(inputs, ksize=filter_size, strides=filter_size, padding='VALID')
            avg_mask = tf.nn.avg_pool2d(mask, ksize=filter_size, strides=filter_size, padding='VALID')
            outputs = avg_output/(avg_mask + 1e-7)
        else:
            # Compute avg pooling
            outputs = tf.nn.avg_pool2d(inputs, ksize=filter_size, strides=filter_size, padding='VALID')
        return outputs
    
    def max_pool2d(self, inputs, filter_size=2):
        # Compute max pooling
        outputs = tf.nn.max_pool2d(inputs, ksize=filter_size, strides=filter_size, padding='VALID')
        return outputs
        
    def downsample_image(self, image, scale):
        """
        Downsample an image by a given scale factor.
        """
        return tf.image.resize(image, [tf.shape(image)[1] // scale, tf.shape(image)[2] // scale], method='bilinear')
    
    def ms_ssim(self, original, y_pred, max_val=1.0, filter_size=11, filter_sigma=1.5, weights=[0.25, 0.25, 0.25, 0.25]):

        reconstructed = y_pred

        # Compute SSIM for each scale
        ssim = 0.0; i = 0; pool_size = 2
        for ws, recons in zip(weights, reconstructed):
            recons = self.normalize_inputs(recons)
            if i > 0:
                original = self.downsample_image(original, pool_size)
            original = self.normalize_inputs(original)
            ssim += ws * tf.image.ssim(original, recons, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma)
            i += 1
        
        # SSIM loss is defined as 1 - SSIM score
        loss = 1.0 - ssim
        loss = tf.reduce_mean(loss)
        return loss

    def scharr_filter_xy(self, img, eps=1e-7):
        scharr_x = tf.constant([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=tf.float32)
        scharr_x = scharr_x[:, :, tf.newaxis, tf.newaxis]
        scharr_x = tf.tile(scharr_x, [1, 1, tf.shape(img)[-1], 1])

        scharr_y = tf.constant([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=tf.float32)
        scharr_y = scharr_y[:, :, tf.newaxis, tf.newaxis]
        scharr_y = tf.tile(scharr_y, [1, 1, tf.shape(img)[-1], 1])

        grad_x = tf.nn.depthwise_conv2d(img, scharr_x, [1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.depthwise_conv2d(img, scharr_y, [1, 1, 1, 1], padding='SAME')
        
        grad_x = tf.square(grad_x)
        grad_x /= (tf.reduce_max(grad_x, axis=(1, 2), keepdims=True) + eps)
        
        grad_y = tf.square(grad_y)
        grad_y /= (tf.reduce_max(grad_y, axis=(1, 2), keepdims=True) + eps)
        
        return grad_x, grad_y
    
    
    def norm_gradients(self, inputs, sigma=4.0):

        grad_x, grad_y = self.scharr_filter_xy(inputs)
        grad_x_norm = tf.exp(-tf.reduce_sum(grad_x, axis=-1, keepdims=True) / sigma)
        grad_y_norm = tf.exp(-tf.reduce_sum(grad_y, axis=-1, keepdims=True) / sigma)
        return grad_x_norm, grad_y_norm

    def gradient_similarity(self, logits, image, sigma=4.0, filter_size=11, filter_sigma=1.5):

        # compute gradients
        im_grad_x, im_grad_y = self.norm_gradients(image, sigma=sigma)
        lg_grad_x, lg_grad_y = self.norm_gradients(logits, sigma=sigma)
        
        # Compute SSIM score
        ssim_score_x = tf.image.ssim(im_grad_x, lg_grad_x, max_val=1.0, filter_size=filter_size, filter_sigma=filter_sigma)
        
        # Compute SSIM score
        ssim_score_y = tf.image.ssim(im_grad_y, lg_grad_y, max_val=1.0, filter_size=filter_size, filter_sigma=filter_sigma)


        # SSIM loss is defined as 1 - SSIM score
        loss = 1.0 - (ssim_score_x + ssim_score_y)/2.0
        loss = tf.reduce_mean(loss)

        return loss
    
    def ms_gradient_similarity(self, logits, image, sigma=4.0, filter_size=11, filter_sigma=1.5, pad=4, weights=[0.1, 0.25, 0.35, 0.3]):
        
        loss = 0.0
        for i, ws in enumerate(weights):
            if i > 0:
                logits = self.downsample_image(logits, 2)
                image = self.downsample_image(image, 2)
            loss += ws * self.gradient_similarity(logits, image, sigma=sigma, filter_size=filter_size, filter_sigma=filter_sigma)

        return loss
    
    def cross_entropy(self, y_true, logits, epsilon=1e-5):
        
        y_pred = logits
        if tf.reduce_min(logits) < 0.0 or tf.reduce_max(logits) > 1.0:
            y_pred = TSoftmax(self.temperature)(logits)
        weights = tf.reduce_sum(y_true)/(tf.reduce_sum(y_true, axis=(0, 1, 2), keepdims=True) + epsilon)

        # Avoid division by zero
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # create a mask for pixels above the threshold
        counts = tf.reduce_sum(tf.where(y_true > 0.5, 1.0, 0.0))

        # calculate cross-entropy loss where there are samples
        cce = y_true * tf.math.log(y_pred)
        cce = cce * weights
        cce_sum = -tf.reduce_sum(cce, axis=-1)/tf.reduce_sum(weights)
        cce_mean = tf.reduce_sum(cce_sum)/(counts + epsilon)

        return cce_mean
    
    def smooth_k(self, epoch, initial_value=0.0, final_value=0.5, r=0.005):
        """
        Smoothly transitions from `initial_value` to `final_value` as epoch increases.
        
        Args:
            epoch (int): Current epoch number.
            initial_value (float): Starting value at epoch 0.
            final_value (float): Value to approach as epoch increases (e.g., 0.5).
            r (float): Rate of change. Larger values make it converge faster.
        
        Returns:
            k (float): The value of k at the current epoch.
        """
        k = final_value - (final_value - initial_value) * tf.exp(-r * epoch)
        return k

    def generalized_cross_entropy(self, y_true, logits, epsilon=1e-5, k=0.05, pad=3, epoch=None):

        '''
        Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
        paper: https://arxiv.org/abs/1805.07836

        '''
        # Cast mask to float32
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)
        y_true = y_true[:, pad:-pad, pad:-pad, :]
        logits = logits[:, pad:-pad, pad:-pad, :]

        # Calculate weights based on true labels (optional, depending on your use case)
        weights = tf.reduce_sum(y_true, keepdims=True) / (tf.reduce_sum(y_true, axis=(0, 1, 2), keepdims=True) + epsilon)

        # Apply softmax if logits are not in probability range
        y_pred = logits
        y_true = tf.where(y_true > 0.0, y_true, 0.0)

        if tf.reduce_min(logits) < 0.0 or tf.reduce_max(logits) > 1.0:
            y_pred = TSoftmax(self.temperature)(logits)

        # Avoid division by zero
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Create a mask for pixels above the threshold (optional)
        samples_mask = tf.where(y_true > 0.5, 1.0, 0.0)
        counts = tf.reduce_sum(samples_mask)

        # Calculate the GCE loss (L_q(f(x), e_j))
        if self.q > 0.0:
            lq_loss = (1 - tf.pow(y_pred, self.q)) / self.q
        else:
            lq_loss = -tf.math.log(y_pred)

        gce = tf.reduce_sum(weights * y_true * lq_loss, axis=-1)
        gce_mean = tf.reduce_sum(gce) / (counts + epsilon)

        return gce_mean 

    def softargmax(self, x, beta=1e10, keepdims=False): 
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype) 
        return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1, keepdims=keepdims)
    

    def dice_loss(self, y_true, logits, smooth=1e-6):
        """
        Compute the Dice loss for batched inputs.
        
        Args:
            y_true: Ground truth binary mask, shape (batch_size, ny, nx, n_channels).
            logits: Predicted logits, shape (batch_size, ny, nx, n_channels).
            smooth: A small constant to avoid division by zero.
        
        Returns:
            Dice loss for each sample in the batch, shape (batch_size,).
        """

        # # Get the predicted indices and one-hot encode them
        # indices = self.softargmax(logits, -1)
        # indices = tf.cast(indices, tf.int64)
        # n_classes = logits.shape[-1]
        # y_pred = tf.one_hot(indices, n_classes)
        # y_pred = tf.cast(y_pred, tf.float32)

        y_pred = logits
        if tf.reduce_min(logits) < 0.0 or tf.reduce_max(logits) > 1.0:
            y_pred = TSoftmax(self.temperature)(logits)
            
        mask = tf.where(y_true > 0.5, 1.0, 0.0)
        # apply mask to y_true and y_pred
        y_true = tf.multiply(y_true, mask)
        y_pred = tf.multiply(y_pred, mask)
         
        # Compute intersection
        intersection = tf.reduce_sum(y_true * y_pred)
        
        # Compute Dice coefficient
        dice_coeff = (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
        
        # Dice loss
        dice_loss_value = 1 - dice_coeff
        return dice_loss_value
    
    def tversky(self, y_true, logits, smooth=1, alpha=0.7):
        y_pred = logits
        if tf.reduce_min(logits) < 0.0 or tf.reduce_max(logits) > 1.0:
            y_pred = TSoftmax(self.temperature)(logits)

        mask = tf.where(y_true > 0.0, 1.0, 0.0)
        true_pos = tf.reduce_sum(y_true * y_pred * mask)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred) * mask)
        false_pos = tf.reduce_sum((1 - y_true) * y_pred * mask)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky(y_true, y_pred)

    def multiscale_cce_loss(self, y_true, logits, epsilon=1e-6, weights=[0.4, 0.3, 0.15, 0.15], pad=4):

        # Cast mask to float32
        y_true = tf.cast(y_true, tf.float32)
        logits = tf.cast(logits, tf.float32)
        y_true = y_true[:, pad:-pad, pad:-pad, :]
        logits = logits[:, pad:-pad, pad:-pad, :]

        if self.fw is not None:
            weights = self.fw

        # Compute MS-CCE for each scale
        cce_list = []
        mask = tf.where(y_true > 0.0, 1.0, 0.0)
        y_true = y_true * mask
        # Avoid division by zero
        y_true = tf.clip_by_value(y_true, epsilon, 1.0-epsilon)

        # create copy of y_true and y_pred
        y_true_scaled = y_true
        logits_scaled = logits

        pool_size = 2
        for i, ws in enumerate(weights):
            if i > 1:
                # Compute pooling
                y_true_scaled = Masked_Average_Pooling2D(pool_size=pool_size, strides=pool_size)(y_true, threshold=0.0) 
                logits_scaled = self.avg_pool2d(logits, pool_size, masked=True, one_hot_mask=True) 
                pool_size *= 2

            # calculate cross-entropy loss where there are samples
            cce_mean = self.generalized_cross_entropy(y_true_scaled, logits_scaled)  
            cce_list.append(cce_mean * ws)
            
        # Compute final loss
        loss = tf.reduce_sum(cce_list)
            
        return loss
    
    def normalize_inputs(self, inputs):
        # normlize inputs between 0 and 1
        num = inputs - tf.reduce_min(inputs, axis=[0, 1, 2], keepdims=True)
        denom = tf.reduce_max(inputs, axis=[0, 1, 2], keepdims=True) - tf.reduce_min(inputs, axis=[0, 1, 2], keepdims=True) + 1e-7
        outputs = num/denom

        return outputs
    

    def multiscale_ssim(self, y_true, y_pred,  pool_size=2, max_val=1.0, filter_size=11, filter_sigma=1.5,
                         weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]):

        original = y_true
        reconstructed = y_pred

        # Compute SSIM for each scale
        ssim = 0.0
        for i, ws in enumerate(weights):
            if i > 0:
                original = self.downsample_image(original, pool_size)
                reconstructed = self.downsample_image(reconstructed, pool_size)

            if original.shape[1] <= filter_size:
                filter_size = original.shape[1]//2 + 1
            ssim += ws * tf.image.ssim(original, reconstructed, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma)

        return ssim 

    def ssim_loss(self, y_true, y_pred, filter_size=11, filter_sigma=1.5, pad=3):
        """
        Computes the SSIM loss between the true and predicted images.
        
        Args:
        y_true: Tensor of true images.
        y_pred: Tensor of predicted images.
        
        Returns:
        A tensor representing the SSIM loss.
        """

        # pad the images
        y_true = y_true[:, pad:-pad, pad:-pad, :]
        y_pred = y_pred[:, pad:-pad, pad:-pad, :]

        # normalize inputs between 0 and 1
        y_true = self.normalize_inputs(y_true)
        #y_pred = self.normalize_inputs(y_pred)


        if y_true.shape[1] < 128:
            # Compute SSIM score
            ssim_score = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=filter_size, filter_sigma=filter_sigma)
        else:
            # Compute SSIM score
            ssim_score = self.multiscale_ssim(y_true, y_pred, max_val=1.0, filter_size=filter_size, filter_sigma=filter_sigma)

        # SSIM loss is defined as 1 - SSIM score
        loss = 1.0 - ssim_score
        loss = tf.reduce_mean(loss)
 
        # Return the mean SSIM loss over the batch
        return loss
    
    def l1_loss(self, y_true, y_pred):

        # distance between the predictions and simulation probabilities
        squared_diff = tf.abs(y_true-y_pred)
        loss = tf.reduce_mean(squared_diff)
        return loss
    
    def reconstruction_loss(self, y_true, y_pred, alpha=0.0, beta=1.0):
        
        # Compute the L1 loss
        l1 = 0.0
        if alpha > 0.0:
            l1 = self.l1_loss(y_true, y_pred)

        # Compute the SSIM loss
        ssim = self.ssim_loss(y_true, y_pred)

        # Compute the reconstruction loss
        loss = alpha * l1 + beta * ssim
        
        return loss
    
    def smoothness(self, logits, image, sigma=8.0, apply_gaussian_filter=False, pad=3):
        """
        Smoothness loss defined in eq. (3)
        
        Args:
            logits: tf.Tensor
                A Tensor of shape (n_batches, ny, nx, n_channels)
            image: tf.Tensor
                A Tensor of shape (n_batches, ny, nx, n_channels)
        link: https://openaccess.thecvf.com/content_cvpr_2017/html/Godard_Unsupervised_Monocular_Depth_CVPR_2017_paper.html
        """
        
        if apply_gaussian_filter:
            # gaussian filter
            #logits = self.gaussian_blur(logits, kernel_size=3, sigma=1.5)
            image = self.gaussian_blur(image, kernel_size=3, sigma=1.5)

        # pad the images
        logits = logits[:, pad:-pad, pad:-pad, :]
        image = image[:, pad:-pad, pad:-pad, :]

        # Apply softmax operation to logits
        prob = TSoftmax(self.temperature)(logits)
        
        # Compute the differences along x and y axes for probabilities and images
        dp_dx = prob[..., :-1, :] - prob[..., 1:, :]
        dp_dy = prob[..., :-1, :, :] - prob[..., 1:, :, :]
        di_dx = image[..., :-1, :] - image[..., 1:, :]
        di_dy = image[..., :-1, :, :] - image[..., 1:, :, :]

        # Calculate smoothness loss for x and y directions
        smoothness_x = tf.reduce_mean(
            tf.reduce_sum(tf.abs(dp_dx), axis=-1) * tf.exp(-tf.reduce_sum(tf.pow(di_dx, 2), axis=-1) / sigma)
        )
        smoothness_y = tf.reduce_mean(
            tf.reduce_sum(tf.abs(dp_dy), axis=-1) * tf.exp(-tf.reduce_sum(tf.pow(di_dy, 2), axis=-1) / sigma)
        )

        # Combine losses in x and y directions
        return smoothness_x + smoothness_y
           
    def mutual_information(self, logits, coeff=2.5, temperature=1.5, pad=3):
            """
            Mutual information defined in eq. (2)

            Args:
                logits: tf.Tensor
                    A Tensor of shape (batch_size, ny, nx, n_channels) representing the predicted logits.
                coeff: float
                    A coefficient corresponding to lambda in eq. (2), controlling regularization.

            Returns:
                float: Mutual information approximation based on pixel-wise and marginal entropies.
            """
            # Add a small constant to logits for numerical stability
            logits /= temperature
            logits = logits[:, pad:-pad, pad:-pad, :]

            # Calculate softmax probabilities along the last dimension (channels)
            prob = tf.nn.softmax(logits, axis=-1)

            # Compute pixel-wise entropy
            pixel_wise_ent = -tf.reduce_sum(prob * tf.nn.log_softmax(logits, axis=-1), axis=-1)
            pixel_wise_ent = tf.reduce_mean(pixel_wise_ent, axis=[1, 2])

            # Compute marginal probabilities by averaging over spatial dimensions
            marginal_prob = tf.reduce_mean(prob, axis=[1, 2])

            # Compute marginal entropy
            # Add a small constant to marginal probabilities for numerical stability
            marginal_ent = -tf.reduce_sum(marginal_prob * tf.math.log(marginal_prob + 1e-16), axis=-1)

            # Approximate mutual information as the difference between pixel-wise and marginal entropies
            mutual_info = pixel_wise_ent - coeff * marginal_ent
            mutual_info = 1.0 + mutual_info/15.0
            loss = tf.reduce_mean(mutual_info)

            return loss