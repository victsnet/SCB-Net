#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:26:20 2023

@author: silva
"""
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from keras import backend as K
import time
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calc_gradients(images):
    # Convert images to float32
    images = tf.cast(images, tf.float32)
    
    # Apply Sobel filters for edge detection
    gradients = tf.image.sobel_edges(images)
    gradients_y, gradients_x = gradients[:, :, :, :, 0], gradients[:, :, :, :, 1]
    
    # Calculate magnitude of gradients
    magnitude = tf.sqrt(tf.square(gradients_x) + tf.square(gradients_y))
    #magnitude = tf.keras.layers.Activation('sigmoid')(magnitude)
    
    return magnitude

class fit_both():
    def __init__(self, loss_func_1, loss_func_2, metric_1, metric_2, lr1=1e-4, lr2=1e-3, cross_validation=False):
        self.optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr1)
        self.optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr2)
        self.loss_func_1 = loss_func_1
        self.loss_func_2 = loss_func_2
        self.metric_1 = metric_1
        self.metric_2 = metric_2
        self.val_mask_1 = None
        self.val_mask_2 = None
        self.cross_validation = cross_validation
    
    @tf.function
    def grad_step(self, model1, model2, features_1, features_2, labels, block_mask_1=None, block_mask_2=None):
        if not hasattr(self, 'mean_train_loss1'):
            self.mean_train_loss1 = Mean()
            self.mean_train_loss2 = Mean()
            self.mean_train_mse1 = Mean()
            self.mean_train_mse2 = Mean()
            
        else:
            self.mean_train_loss1.reset_states()
            self.mean_train_loss2.reset_states()
            self.mean_train_mse1.reset_states()
            self.mean_train_mse2.reset_states()
            
        with tf.GradientTape(persistent=True) as tape:
            predictions_1 = model1(features_1)
            predictions_2 = model2(features_2)
            
            # calculate losses
            if self.cross_validation:
                loss1, block_mask_1 = self.loss_func_1.total_loss(labels, predictions_1)
                loss2, block_mask_2 = self.loss_func_2.total_loss(labels, predictions_2)
            else:
                loss1 = self.loss_func_1.total_loss(labels, predictions_1)
                loss2 = self.loss_func_2.total_loss(labels, predictions_2)

            # calculate mse metrics
            mse1 = self.metric_1(labels, predictions_1)
            mse2 = self.metric_2(labels, predictions_2)
    
        gradients1 = tape.gradient(loss1, model1.trainable_variables)
        gradients2 = tape.gradient(loss2, model2.trainable_variables)
        self.optimizer_1.apply_gradients(zip(gradients1, model1.trainable_variables))
        self.optimizer_2.apply_gradients(zip(gradients2, model2.trainable_variables))
        
        losses = (self.mean_train_loss1(loss1), self.mean_train_loss2(loss2))
        metrics = (self.mean_train_mse1(mse1), self.mean_train_mse2(mse2))
        
        self.model1 = model1
        self.model2 = model2
        
        return losses, metrics, (block_mask_1, block_mask_2)
    
    @tf.function
    def val_step(self, features_1, features_2, labels, block_mask_1=None, block_mask_2=None):
        
        if not hasattr(self, 'val_mean_loss1'):
            self.val_mean_loss1 = Mean()
            self.val_mean_loss2 = Mean()
            self.val_mean_loss_combination = Mean()
            self.val_mean_mse1 = Mean()
            self.val_mean_mse2 = Mean()
            self.val_mse_combination = Mean()
    
        else:
            self.val_mean_loss1.reset_states()
            self.val_mean_loss2.reset_states()
            self.val_mean_loss_combination.reset_states()
            self.val_mean_mse1.reset_states()
            self.val_mean_mse2.reset_states()
            self.val_mse_combination.reset_states()
            
        predictions_1 = self.model1(features_1)
        predictions_2 = self.model2(features_2)
        
        
        if self.cross_validation:
            n_batches = tf.shape(labels)[0]
            val_loss1 = self.loss_func_1.spatial_loss(block_mask_1[:n_batches], predictions_1)
            val_loss2 = self.loss_func_2.spatial_loss(block_mask_2[:n_batches], predictions_2)

        else:
            val_loss1 = self.loss_func_1.total_loss(labels, predictions_1)
            val_loss2 = self.loss_func_2.total_loss(labels, predictions_2)
        
        
        # calculate mse metrics
        mse1 = self.metric_1(labels, predictions_1)
        mse2 = self.metric_2(labels, predictions_2)
        
        val_losses = (self.val_mean_loss1(val_loss1), self.val_mean_loss2(val_loss2))
        
        val_metrics = (self.val_mean_mse1(mse1), self.val_mean_mse2(mse2))
        return val_losses, val_metrics


def parallel_training(model1, model2, features, coords, target, epochs, batch_size, loss_func_1, loss_func_2,
                      metric_1, metric_2, lr1=1e-4, lr2=1e-4, cross_validation=False):
    
    train_features, val_features = features
    train_coords, val_coords = coords
    train_target, val_target = target
    
    history = {'loss_1': [], 'loss_2': [], 
               'val_loss_1': [],  'val_loss_2': [], 
               'mse_1': [], 'mse_2': [],
               'val_mse_1': [],  'val_mse_2': []}
    
    n_train_batches = train_features.shape[0]
    n_val_batches = val_features.shape[0]
    train = fit_both(loss_func_1, loss_func_2,
                          metric_1, metric_2, lr1, lr2, cross_validation=cross_validation)
    # Define the training loop
    for epoch in range(epochs):
        for i in range(0, n_train_batches, batch_size):
            
            # Train both models 
            (loss1, loss2), (mse1, mse2), (block_mask_1, block_mask_2) =\
                train.grad_step(model1, model2, train_features[i:i+batch_size],
                                         train_coords[i:i+batch_size], train_target[i:i+batch_size])
            
        for i in range(0, n_val_batches, batch_size):

            # validation
            (val_loss1, val_loss2), (val_mse1, val_mse2) =\
                train.val_step(val_features[i:i+batch_size], val_coords[i:i+batch_size], 
                               val_target[i:i+batch_size], block_mask_1, block_mask_2)
            
            
        # train losses
        history['loss_1'].append(loss1)
        history['loss_2'].append(loss2)
        
        # train metrics
        history['mse_1'].append(mse1)
        history['mse_2'].append(mse2)
        
        # val losses
        history['val_loss_1'].append(val_loss1)
        history['val_loss_2'].append(val_loss2)
        
        # val metrics
        history['val_mse_1'].append(val_mse1)
        history['val_mse_2'].append(val_mse2)
        
        print(f'Epoch {epoch + 1}: '
              f'Loss_1: {loss1:.4f} | Loss_2: {loss2:.4f}. '
              f'Val_loss_1: {val_loss1:.4f} | Val_loss_2: {val_loss2:.4f}.')
        
        print(f'                   '
              f'MSE-1: {mse1:.4f} | MSE-2: {mse2:.4f}. '
              f'Val_mse_1: {val_mse1:.4f} | Val_mse_2: {val_mse2:.4f}.')
        
        print('---------------------------------------------------------------------------------------------')
        
    return history


class fit():
    def __init__(self, loss_func, metrics, lr=1e-4, spatial_constraint=False):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_func = loss_func
        self.metrics = metrics
        self.val_mask = None
        self.spatial_constraint = spatial_constraint
    
    @tf.function
    def grad_step(self, model, features, ground_truth, target):
        if not hasattr(self, 'mean_train_loss'):
            self.mean_train_loss = Mean()
            self.mean_train_metrics = {}
            for metric in self.metrics:
                self.mean_train_metrics[metric] = Mean()
            
        else:
            self.mean_train_loss.reset_states()
            for metric in self.metrics:
                self.mean_train_metrics[metric].reset_states()
            
        with tf.GradientTape(persistent=False) as tape:
        
            if self.spatial_constraint:
                predictions, embeddings = model([features, ground_truth])
            else:
                predictions, _ = model(features)
                
            # training loss
            loss = self.loss_func.spatial_loss(target, predictions)

            # calculate metrics
            metrics_list = []
            for metric in self.metrics:
                if metric == 'acc':
                    metric_value = self.loss_func.accuracy(target, predictions)
                elif metric == 'ssim':
                    metric_value = self.loss_func.masked_ssim(target, predictions)
                else:
                    pass
                metrics_list.append(self.mean_train_metrics[metric](metric_value))
            
            # compute gradients and update params
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        self.model = model
        
        return self.mean_train_loss(loss), metrics_list
    
    @tf.function
    def val_step(self, model, features, ground_truth, target):
        
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
            predictions, _ = model([features, ground_truth])                     
        else:
            predictions, _ = model(features)
        
        # calculate validation loss
        val_loss = self.loss_func.spatial_loss(target, predictions, training_set=False)
        
        # Validation metrics
        metrics_list = []
        for metric in self.metrics:
            if metric == 'acc':
                metric_value = self.loss_func.accuracy(target, predictions)
            elif metric == 'ssim':
                metric_value = self.loss_func.masked_ssim(target, predictions)
            else:
                pass
            metrics_list.append(self.mean_val_metrics[metric](metric_value))
            
        return self.mean_val_loss(val_loss), metrics_list

class model_training:
    
    def __init__(self, model, features, ground_truth, target, epochs, batch_size, loss_func, n_classes=None,
                          metrics=['acc'], lr=1e-4, decay={'epochs':1000, 'new_lr':10e-5}, 
                          min_delta=1e-4, patience=20, monitor='loss',
                           spatial_constraint=False, check_point=None, model_name=None):
        self.model = model
        self.features = features
        self.ground_truth = ground_truth
        self.target = target
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.loss_func = loss_func
        self.metrics = metrics
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
        
        # unpack features, targets
        train_features, val_features = self.features
        train_ground_truth, val_ground_truth = self.ground_truth
        train_target, val_target = self.target
        
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
        
        n_classes = train_target.shape[-1]
        if self.n_classes is not None:
            n_classes = self.n_classes
        
        dim = train_features.shape
        n_train_batches = (train_features.shape[0]//self.batch_size)*self.batch_size
        n_val_batches = (val_features.shape[0]//self.batch_size)*self.batch_size
        
        # define fit object 
        train = fit(self.loss_func, self.metrics, self.lr, spatial_constraint=self.spatial_constraint)
        
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
            
            # Use the random indices to shuffle the array
            random_indices = np.random.permutation(train_features.shape[0])
            train_features = train_features[random_indices]
            train_ground_truth = train_ground_truth[random_indices]
            train_target = train_target[random_indices]
                
            losses = []
            metrics_dict = {}
            for metric in self.metrics:
                metrics_dict[metric] = []
            for i in range(0, n_train_batches, self.batch_size):                

                #Train both models 
                loss, metric_value =\
                    train.grad_step(self.model, train_features[i:i+self.batch_size], train_ground_truth[i:i+self.batch_size], 
                                    train_target[i:i+self.batch_size, :, :, :n_classes])
                    
                losses.append(loss.numpy())
                for m, metric in enumerate(self.metrics):
                    metrics_dict[metric].append(metric_value[m].numpy())
                
            val_losses = []
            val_metrics_dict = {}
            for metric in self.metrics:
                val_metrics_dict[metric] = []
            for i in range(0, n_val_batches, self.batch_size):
                
                # Use the random indices to shuffle the array
                random_indices = np.random.permutation(val_features.shape[0])
                val_features = val_features[random_indices]
                val_ground_truth = val_ground_truth[random_indices]
                val_target = val_target[random_indices]
                
                # validation
                val_loss, val_metric_value =\
                train.val_step(self.model, val_features[i:i+self.batch_size], val_ground_truth[i:i+self.batch_size], 
                               val_target[i:i+self.batch_size, :, :, :n_classes])
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
            weights_path = os.path.join(self.check_point, self.model_name, f'model_{self.batch_size}_{dim[0]}_{today}.h5')
            if epoch > 0:
                if self.monitor == 'loss':
                    min_loss = np.min(history['loss'][:-1])
                    if history['loss'][-1] < min_loss:
                        self.model.save_weights(weights_path)
                        print(f"The model was saved ({self.monitor}): {history['loss'][-1]:.4f} < {min_loss:.4f}.")
                    else:
                        print(f"The model was not saved ({self.monitor}): {history['loss'][-1]:.4f} > {min_loss:.4f}.")
                        
                elif self.monitor == 'val_loss':
                    min_val_loss = np.min(history['val_loss'][:-1])
                    if history['val_loss'][-1] < min_val_loss:
                        self.model.save_weights(weights_path)
                        print(f"The model was saved ({self.monitor}): {history[self.monitor][-1]:.4f} < {min_val_loss:.4f}.")                      
                    else:
                        print(f"The model was not saved ({self.monitor}): {history[self.monitor][-1]:.4f} > {min_val_loss:.4f}.") 
                        
                        
                elif self.monitor == 'acc':
                    max_acc = np.max(history['acc'][:-1])
                    if history['acc'][-1] > max_acc:
                       self.model.save_weights(weights_path)
                       print(f"The model was saved ({self.monitor}): {history[self.monitor][-1]:.4f} > {max_acc:.4f}.")                      
                    else:
                        print(f"The model was not saved ({self.monitor}): {history[self.monitor][-1]:.4f} < {max_acc:.4f}.") 
                        
                     
                elif self.monitor == 'val_acc':
                    max_val_acc = np.max(history['val_acc'][:-1])
                    if history['val_acc'][-1] > max_val_acc:
                       self.model.save_weights(weights_path)
                       print(f"The model was saved ({self.monitor}): {history[self.monitor][-1]:.4f} > {max_val_acc:.4f}.")                                                    
                    else:
                        print(f"The model was not saved ({self.monitor}): {history[self.monitor][-1]:.4f} < {max_val_acc:.4f}.") 
               
            # save dataframe as csv
            df = pd.DataFrame(history)
            directory = os.path.join(self.check_point, self.model_name)
            csv_file_name = f'history_{self.batch_size}_{dim[1]}_{today}.csv'
            history = self.save_csv(directory, csv_file_name, df, epoch, history)
            
            # early stopping -- monitor 
            delta = np.max(history[self.monitor])
            if epoch > self.patience:
                delta = np.max(history[self.monitor][-self.patience:]) - np.max(history[self.monitor][:-self.patience]) 
                if delta <= self.min_delta:
                    break
                self.add_text(f'Early stopping ({self.monitor} delta): {delta:.2e} > {self.min_delta:.2e}.', skip=True)
            self.add_text('-------------------------------------------------------------------', skip=True)
            
        return history