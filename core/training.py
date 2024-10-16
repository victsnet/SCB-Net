#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:26:20 2023

@author: silva
"""
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import os
import time
from datetime import date
import pandas as pd
import numpy as np

def add_gaussian_noise(images, stddev=0.1):
    noise = tf.random.normal(shape=tf.shape(images), mean=tf.reduce_mean(images), stddev=stddev)
    return images + noise

class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.3, block_size=3):
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
        uniform_dist = tf.random.uniform([input_shape[0], num_blocks_h, num_blocks_w, input_shape[-1]], dtype=inputs.dtype)

        size = input_shape[1:3]  # Get spatial dimensions of the input tensor
        uniform_dist = tf.image.resize(uniform_dist, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        mask = tf.where(uniform_dist > self.dropout_rate, 1.0, 0.0)
        output = tf.multiply(inputs, mask)

        return output

class HoldOut2d(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.3, block_size=3):
        super(HoldOut2d, self).__init__()
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

        mask = tf.where(uniform_dist > self.dropout_rate, 1.0, 0.0)

        return mask
    

class fit():
    def __init__(self, loss_func, metrics, lr=1e-4, spatial_constraint=False, n_coords=0, hold_out=0.5, block_size=5, batch_size=16, n_steps=1):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_func = loss_func
        self.metrics = metrics
        self.val_mask = None
        self.spatial_constraint = spatial_constraint
        self.n_coords = n_coords
        self.hold_out = hold_out
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_steps = n_steps        

    @tf.function(reduce_retracing=True)
    def grad_step(self, model, features, ground_truth, target, epoch):

        self.model = model
        if not hasattr(self, 'mean_train_loss'):
            self.mean_train_loss = Mean()
            self.mean_train_metrics = {}
            for metric in self.metrics:
                self.mean_train_metrics[metric] = Mean()
            
        else:
            self.mean_train_loss.reset_states()
            for metric in self.metrics:
                self.mean_train_metrics[metric].reset_states()
            
        with tf.GradientTape() as tape:
            # cross-validation mask
            block_size = tf.random.shuffle([1, 5, 10, 15])[0]
            cv_mask = HoldOut2d(dropout_rate=self.hold_out, block_size=block_size)(target)
        
            if self.spatial_constraint:
                ground_truth = tf.where(ground_truth > 0.0, ground_truth, 0.0)
                ground_truth = tf.multiply(ground_truth, 1.0-cv_mask)
                block_size = tf.random.shuffle([5, 7, 10, 15])[0]
                hold_out_mask = HoldOut2d(dropout_rate=0.2, block_size=block_size)(features)
                masked_features = tf.multiply(features, hold_out_mask)
                predictions, _ = model([masked_features, ground_truth])  
            else:
                predictions, _ = model(features)

            # calculate loss
            n_classes = predictions.shape[-1]
            # cross-validation mask
            target = tf.multiply(target, cv_mask)
            # GCE loss
            loss = self.loss_func.generalized_cross_entropy(target[:, :, :, :n_classes], predictions, epoch=epoch)    
            dice_loss = self.loss_func.dice_loss(target[:, :, :, :n_classes], predictions)
            recons_loss = 0.0

            # calculate metrics
            metrics_list = []
            for metric in self.metrics:
                if metric == 'acc':
                    acc = metric_value = self.loss_func.accuracy(target[:, :, :, :n_classes], predictions) 
                elif metric == 'acc-rescale-2':
                    metric_value = self.loss_func.accuracy(target[:, :, :, :n_classes], predictions, rescale=2)
                elif metric == 'acc-rescale-4':
                    metric_value = self.loss_func.accuracy(target[:, :, :, :n_classes], predictions, rescale=4)
                elif 'n_samples' in metric:
                    metric_value = self.loss_func.get_n_samples(target[:, :, :, :n_classes])
                elif 'recons' in metric:
                    metric_value = recons_loss
                elif 'acc_recons' in metric:
                    metric_value = 0.9*acc + 0.1*(1.0-recons_loss)
                elif 'dice' in metric:
                    metric_value = dice_loss
                else:
                    pass
                metrics_list.append(self.mean_train_metrics[metric](metric_value))

            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Return the mean train loss and metrics
        return self.mean_train_loss(loss), metrics_list
    
    @tf.function(reduce_retracing=True)
    def val_step(self, model, features, ground_truth, target, epoch):
        
        if not hasattr(self, 'mean_val_loss'):
            self.mean_val_loss = Mean()
            self.mean_val_metrics = {}
            for metric in self.metrics:
                self.mean_val_metrics[metric] = Mean()
        else:
            self.mean_val_loss.reset_states()
            for metric in self.metrics:
                self.mean_val_metrics[metric].reset_states()
        
        if self.spatial_constraint:
            ground_truth = tf.where(ground_truth > 0.0, ground_truth, 0.0)
            predictions, _ = model([features, ground_truth])                     
        else:
            predictions, _ = model(features)

        # calculate validation loss
        n_classes = predictions.shape[-1]
        val_loss = self.loss_func.generalized_cross_entropy(target[:, :, :, :n_classes], predictions, epoch=epoch)
        dice_loss = self.loss_func.dice_loss(target[:, :, :, :n_classes], predictions)
        recons_loss = 0.0
        # Validation metrics
        metrics_list = []
        for metric in self.metrics:
            if metric == 'acc':
                val_acc = metric_value = self.loss_func.accuracy(target[:, :, :, :n_classes], predictions) 
            elif metric == 'acc-rescale-2':
                metric_value = self.loss_func.accuracy(target[:, :, :, :n_classes], predictions, rescale=2)
            elif metric == 'acc-rescale-4':
                metric_value = self.loss_func.accuracy(target[:, :, :, :n_classes], predictions, rescale=4)
            elif 'n_samples' in metric:
                metric_value = self.loss_func.get_n_samples(target[:, :, :, :n_classes])
            elif 'recons' in metric:
                metric_value = recons_loss
            elif 'acc_recons' in metric:
                metric_value = 0.9*val_acc + 0.1*(1.0-recons_loss)
            elif 'dice' in metric:
                metric_value = dice_loss
            else:
                pass
            metrics_list.append(self.mean_val_metrics[metric](metric_value))
            
        return self.mean_val_loss(val_loss), metrics_list
    

class gen_training:
    
    def __init__(self, model, generate_tiles, epochs, n_patches, batch_size, n_steps=1, loss_func=None, n_classes=None, n_coords=0, hold_out=0.5, block_size=5, metrics=['acc'], lr=1e-4, decay={'epochs':1000, 'new_lr':1e-5}, 
                          min_delta=1e-4, patience=20, monitor='loss',
                           spatial_constraint=False, check_point=None, model_name=None):
        self.model = model
        self.generate_tiles = generate_tiles
        self.epochs = epochs
        self.n_patches = n_patches
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_coords = n_coords
        self.hold_out = hold_out
        self.block_size = block_size
        self.loss_func = loss_func
        self.metrics = metrics
        self.n_steps = n_steps
        self.lr = lr
        self.decay = decay
        self.min_delta = min_delta
        self.patience = patience
        self.spatial_constraint = spatial_constraint
        self.monitor = monitor
        self.check_point = check_point
        self.model_name = model_name
        
    def add_text(self, text, skip=False, screen=True):
        print(text, file=self.f)
        self.f.flush()
        if skip:
            print(' ', file=self.f)
            self.f.flush()
            
        if screen:
            print(text)
            self.f.flush()
            
    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' +  directory)
            
            
    def save_csv(self, directory, filename, df, epoch, history):
        csv_file_path = os.path.join(directory, filename)
    
        if epoch == 0:
            if os.path.exists(csv_file_path):
                # If the CSV file exists, read it and merge with the new dataframe
                existing_dataframe = pd.read_csv(csv_file_path)
                merged_dataframe = pd.concat([existing_dataframe, df], ignore_index=True)
                history['loss'] += list(merged_dataframe['loss'])
                history['val_loss'] += list(merged_dataframe['val_loss'])
                for metric in self.metrics:
                    history[f'{metric}'] += list(merged_dataframe[f'{metric}'])
                    history[f'val_{metric}'] += list(merged_dataframe[f'val_{metric}'])
            else:
                # If the CSV file doesn't exist, create a new dataframe
                merged_dataframe = df
        else:
            merged_dataframe = df
    
        # Save the merged DataFrame as a CSV file
        merged_dataframe.to_csv(csv_file_path, sep=',', index=False, index_label=False)
        
        return history
    
    def plot_history(self, history):
        return history                   
    
    def fit(self):
        
        if 'acc' or 'accuracy' in self.metrics:
            calc_metrics = self.loss_func.accuracy
        elif 'mse' in self.metrics:
            calc_metrics = self.loss_func.mse
            
        else:
            raise ValueError('The metric provided is missing.')
        
        history = {'loss': [], 
                   'val_loss': []}
        for metric in self.metrics:
            history[f'{metric}'] = []
            history[f'val_{metric}'] = []

        # define fit object 
        train = fit(self.loss_func, self.metrics, self.lr, spatial_constraint=self.spatial_constraint, n_coords=self.n_coords,
                    hold_out=self.hold_out, block_size=self.block_size, batch_size=self.batch_size, n_steps=self.n_steps)
        
        # register computation time
        tic = time.time()
        
        # get today's date
        today = str(date.today()).replace('-', '_')
        
        # initialize report 
        self.createFolder(os.path.join(self.check_point, self.model_name))
        self.createFolder(os.path.join(self.check_point, self.model_name, 'reports'))
        
        path = os.path.join(self.check_point, self.model_name, f'reports/report_model_training_{self.epochs}_{self.batch_size}_{today}.txt')
        self.f = open(path, 'w')
        print('====================== MODEL TRAINING - SPATIAL INTERPOLATION ================= \n', file=self.f, flush=True)

        # Define the training loop
        for epoch in range(self.epochs):
            if epoch > self.decay['epochs']:
                self.lr = self.decay['new_lr']

            epoch_tensor = tf.convert_to_tensor(epoch, dtype=tf.float32)  # Ensure epoch is a tensor
            self.n_patches = self.batch_size*(self.n_patches)//self.batch_size

            # generate training patches
            X_train, Y_train = self.generate_tiles.training_patches(batch_size=self.n_patches)

            # generate validation patches
            X_val, Y_gt, Y_val = self.generate_tiles.validation_patches(batch_size=None) 
            val_patches = self.batch_size*(X_val.shape[0])//self.batch_size
                
            losses = []
            metrics_dict = {}
            for metric in self.metrics:
                metrics_dict[metric] = []

            for i in range(0, self.n_patches, self.batch_size):
                #Training step 
                loss, metric_value =\
                    train.grad_step(self.model, X_train[i:i+self.batch_size], Y_train[i:i+self.batch_size],
                                     Y_train[i:i+self.batch_size], epoch=epoch_tensor)
                    
                losses.append(loss.numpy())
                for m, metric in enumerate(self.metrics):
                    metrics_dict[metric].append(metric_value[m].numpy())
                
            val_losses = []
            val_metrics_dict = {}
            for metric in self.metrics:
                val_metrics_dict[metric] = []

            for i in range(0, val_patches, self.batch_size):   
                # validation step
                val_loss, val_metric_value =\
                train.val_step(self.model, X_val[i:i+self.batch_size], Y_gt[i:i+self.batch_size], Y_val[i:i+self.batch_size], epoch=epoch_tensor)
                val_losses.append(val_loss.numpy())
                for m, metric in enumerate(self.metrics):
                    val_metrics_dict[metric].append(val_metric_value[m].numpy())
                
            # training loss
            history['loss'].append(np.nanmean(losses))
            
            # validation loss
            history['val_loss'].append(np.nanmean(val_losses))
            
            # train metric
            for metric in self.metrics:
                history[f'{metric}'].append(np.nanmean(metrics_dict[metric]))

            # val metrics
            for metric in self.metrics:
                history[f'val_{metric}'].append(np.nanmean(val_metrics_dict[metric]))
            
            self.add_text(f'Epoch {epoch + 1}: \n'
                  f'Loss: {np.nanmean(losses):.4f} | Val_loss: {np.nanmean(val_losses):.4f} |')
            
            for metric in self.metrics:
                self.add_text(f'{metric}: {np.nanmean(metrics_dict[metric]):.4f} | Val_{metric}: {np.nanmean(val_metrics_dict[metric]):.4f}.')
            
            toc = time.time()
            self.add_text(f'Learning rate: {self.lr:.2e}.')
            self.add_text(f'Batch size: {self.batch_size}.')
            self.add_text(f'Running time: {(toc-tic)/60.:.2f}min.')
                 
            
            # weights_path 
            best_weights_path = os.path.join(self.check_point, self.model_name, f'best_model_{self.batch_size}_{today}.h5')
            second_weights_path = os.path.join(self.check_point, self.model_name, f'second_model_{self.batch_size}_{today}.h5')
            if epoch > 0:
                if self.monitor == 'loss':
                    min_loss = np.min(history['loss'][:-1])
                    if history['loss'][-1] < min_loss:
                        self.model.save_weights(best_weights_path)
                        print(f"The model was saved ({self.monitor}): {history['loss'][-1]:.4f} < {min_loss:.4f}.")
                    else:
                        print(f"The model was not saved ({self.monitor}): {history['loss'][-1]:.4f} > {min_loss:.4f}.")
                        
                elif self.monitor == 'val_loss':
                    min_val_loss = np.min(history['val_loss'][:-1])
                    if history['val_loss'][-1] < min_val_loss:
                        self.model.save_weights(best_weights_path)
                        print(f"The model was saved ({self.monitor}): {history[self.monitor][-1]:.4f} < {min_val_loss:.4f}.")                      
                    else:
                        print(f"The model was not saved ({self.monitor}): {history[self.monitor][-1]:.4f} > {min_val_loss:.4f}.") 
                        
                        
                elif self.monitor == 'acc':
                    max_acc = np.max(history['acc'][:-1])
                    if history['acc'][-1] >= max_acc:
                       self.model.save_weights(best_weights_path)
                       print(f"The model was saved ({self.monitor}): {history[self.monitor][-1]:.4f} > {max_acc:.4f}.")                      
                    else:
                        print(f"The model was not saved ({self.monitor}): {history[self.monitor][-1]:.4f} < {max_acc:.4f}.") 
                        
                elif self.monitor == 'val_acc':
                    # Exclude the latest value to find the previous best and second-best
                    previous_val_acc = history['val_acc'][:-1]
                    
                    if len(previous_val_acc) > 0:
                        # Sort in ascending order
                        sorted_val_acc = np.sort(previous_val_acc)
                        
                        # Find the maximum (best) and second-best accuracy values
                        max_val_acc = sorted_val_acc[-1]
                        snc_val_acc = sorted_val_acc[-2] if len(sorted_val_acc) > 1 else -np.inf
                    else:
                        max_val_acc = -np.inf
                        snc_val_acc = -np.inf

                    # Check if the latest validation accuracy is greater or equal to the best
                    current_val_acc = history['val_acc'][-1]
                    
                    if current_val_acc >= max_val_acc:
                        # Save the best model
                        self.model.save_weights(best_weights_path)
                        print(f"The model was saved ({self.monitor}): {current_val_acc:.4f} >= {max_val_acc:.4f}.")
                    elif current_val_acc >= snc_val_acc:
                        # Save the second-best model
                        self.model.save_weights(second_weights_path)
                        print(f"The 2nd-best-model was saved ({self.monitor}): {current_val_acc:.4f} >= {snc_val_acc:.4f}.")
                    else:
                        # Do nothing if neither best nor second-best
                        print(f"The model was not saved ({self.monitor}): {current_val_acc:.4f} < {max_val_acc:.4f}.")

                elif self.monitor == 'val_acc_recons':
                    max_val_acc_recons = np.max(history['val_acc_recons'][:-1])
                    if history['val_acc_recons'][-1] > max_val_acc_recons:
                       self.model.save_weights(best_weights_path)
                       print(f"The model was saved ({self.monitor}): {history[self.monitor][-1]:.4f} > {max_val_acc_recons:.4f}.")                                                    
                    else:
                        print(f"The model was not saved ({self.monitor}): {history[self.monitor][-1]:.4f} < {max_val_acc_recons:.4f}.")
               
            # save dataframe as csv
            df = pd.DataFrame(history)
            directory = os.path.join(self.check_point, self.model_name)
            csv_file_name = f'history_{self.batch_size}_{today}.csv'
            history = self.save_csv(directory, csv_file_name, df, epoch, history)
            
            # early stopping -- monitor 
            delta = np.max(history[self.monitor])
            if epoch > self.patience:
                delta = np.max(history[self.monitor][-self.patience:]) - np.max(history[self.monitor][:-self.patience]) 
                if delta <= self.min_delta:
                    break
                self.add_text(f'Early stopping ({self.monitor} delta): {delta:.2e} > {self.min_delta:.2e}.', skip=True)
            self.add_text('-------------------------------------------------------------------', skip=True)

        # return best model
        self.model.load_weights(best_weights_path)
            
        return history, self.model
    
    