# Description: This script is used to train a spatially constrained Bayesian network (SCB-Net) for lithological mapping # -*- coding: utf-8 -*-

#%% set working directory
import os
os.chdir('/home/silva/Codes-Notebooks/PREDQC/Spyder')

# %% import modules, functions and libraries
import tensorflow as tf
from core import class_weights, loss_function, preprocessing, postprocessing
from core.training import gen_training
from utils.tools import cmap_from_labels, reinterp_xarrays
from utils.tools import adjust_rgb, hillshade, plot_cfm
from model import spatial_models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date
import rioxarray
import numpy as np
from sklearn import metrics
from os.path import join
import cv2

# get today's date
today = str(date.today()).replace('-', '_')
#%% check the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% CMAP SCALE
'''
    I1B - Granite 11669
    I1P - Granite à hypersthène (charnockite) 1700
    I2J - Diorite 1421
    I3A - Gabbro 13610
    I4I - Péridotite 1422
    M1 - Gneiss 12050
    M12 - Quartzite 755
    M16 - Amphibolite 1656
    M8 - Schiste 1607
    S1 - Grès 2450
    S3 - Wacke 1077
    S4 - Conglomérat 610
    S6 - Mudrock 3743
    S8 - Dolomie 2141
    S9 - Formation de fer 1090
    V3 - Roches volcaniques mafiques 9242
'''

labels = [
    "Granite",
    "Charnockite",
    "Diorite",
    "Gabbro",
    "Peridotite",
    "Gneiss",
    "Quartzite",
    "Amphibolite",
    "Schist",
    "Sandstone",
    "Wacke",
    "Conglomerate",
    "Mudrock",
    "Dolomite",
    "Iron Formation",
    "Basalt",
]

cmap = cmap_from_labels(labels, 'data/colors_catalog/catlog.csv', from_lang='en', to_lang='en') 

#%%
def remove_outliers_iqr(data):
    """
    Removes outliers from the dataset using the Interquartile Range (IQR).
    
    Parameters:
    data (list or numpy array): The dataset from which to remove outliers.
    
    Returns:
    numpy array: The dataset with outliers removed.
    """
    data = np.array(data)
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1

    # Define the outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter the data to exclude outliers
    data[data < lower_bound] = lower_bound
    data[data > upper_bound] = upper_bound
    
    return data

#%% import satellite data
scaling = 1 # set rescaling factor
spatial_block_size = 10

# pick sensors
use_radar = True
use_sentinel_2 = True
use_alos = True
use_gamma = False
use_mag = True
use_pcs = False

paths = []
path_ext = ''
subpath = 'data/east_qc/'
ref_path = subpath+f'prob_masks/train_prob_mask_bs10_400_code_r3.tif'

if use_alos:
    paths.append(subpath+'ALOS/alos_elev_east_qc_100m.tif')
    path_ext += 'alos_'
    
if use_radar:
    paths.append(subpath+'ALOS/ALOS_PALSAR_RADAR_MOSAIC_QC_100m.tif')
    path_ext += 'sar_'
    
if use_sentinel_2:
    paths.append(subpath+'SENTINEL2/sentinel2_multispec_east_qc_100m.tif')
    path_ext += 's2_'
    
if use_gamma:
    paths.append(subpath+'gamma/gamma_kperc_east.tif')
    paths.append(subpath+'gamma/gamma_th_east.tif')
    path_ext += 'gamma_'
    
if use_mag:
    paths.append(subpath+'mag_fed/MAG_QC_LOWRES_RESMAG_4269_epsg.tif')
    paths.append(subpath+'mag_fed/MAGRES_QC_LOWRES_AS_4269_epsg.tif')
    paths.append(subpath+'mag_fed/MAGRES_QC_LOWRES_DV1_4269_epsg.tif')
    path_ext += 'mag_'
    
if use_pcs:
    paths.append(subpath+'SAT_PCS/NORTHEAST_SAT_PCS_4269_epsg.tif')
    path_ext += 'sat_pcs_'
    
if path_ext.endswith('_'):
    path_ext = path_ext[:-1]
# merge datasets
x, y, sat_grid = reinterp_xarrays(ref_path, paths, scaling=scaling, method='linear')

for n in range(sat_grid.shape[0]):
    sat_grid[n] = remove_outliers_iqr(sat_grid[n])
    sat_grid[n] = cv2.bilateralFilter(np.float32(sat_grid[n]), d=3, sigmaColor=25, sigmaSpace=15)

_, _, pc1_radar = reinterp_xarrays(ref_path, [subpath+'SENTINEL1/pc1_radar.tif'], scaling=scaling)
# create water mask
pc1_radar = cv2.bilateralFilter(pc1_radar[0], d=3, sigmaColor=35, sigmaSpace=35)
_, wmask = cv2.threshold(pc1_radar, -9., 1, cv2.THRESH_BINARY)
wmask = np.expand_dims(wmask, -1)

if use_gamma:
    gamma_mask = np.where(sat_grid[-2:].swapaxes(0, 1).swapaxes(1, 2) == 254.0, 0., 1.)[:, :, :1]
    
# hillshade
hshade = hillshade(sat_grid[0], azimuth=225, angle_altitude=30)

# %% check on the RS variables
for rsim in sat_grid:
    plt.figure(figsize=(12, 12))
    plt.imshow(rsim, vmin=np.nanpercentile(rsim, 2),
               vmax=np.nanpercentile(rsim, 98))
    plt.axis('off')
    plt.axis('scaled')
    plt.show()
    
#%% plot water mask
plt.figure(figsize=(10, 9))
plt.imshow(1.0-wmask, cmap='Blues')
plt.axis('off')
plt.axis('scaled')
plt.colorbar(label=u'Binary water mask', orientation='horizontal')
plt.show()

#%% import prob-masks
add_wmask = False
test_set = False

train_dmask = rioxarray.open_rasterio(join(subpath, f'prob_masks/train_prob_mask_bs10_400_code_r3.tif' ))
prob_mask_training = train_dmask.data.swapaxes(0, 1).swapaxes(1, 2)

if test_set:
    test_dmask = rioxarray.open_rasterio(join(subpath,f'prob_masks/test_prob_mask_10_400_code_r3.tif' ))
    prob_mask_testing = test_dmask.data.swapaxes(0, 1).swapaxes(1, 2)

val_dmask = rioxarray.open_rasterio(join(subpath,f'prob_masks/val_prob_mask_bs10_400_code_r3.tif' ))
prob_mask_validation = val_dmask.data.swapaxes(0, 1).swapaxes(1, 2)

prob_mask_training *= wmask
prob_mask_validation *= wmask 
if test_set:
    prob_mask_testing *= wmask

if add_wmask:
    water_mask_train = tf.where(tf.random.uniform(wmask.shape) > 0.99, 1.0, 0.0) * (1.0 - wmask)
    water_mask_train = water_mask_train.numpy()
    
    if test_set:
        water_mask_test = tf.where(tf.random.uniform(wmask.shape) > 0.995, 1.0, 0.0) *  (1.0 - wmask)
        water_mask_test = water_mask_test.numpy()
    
    water_mask_val = tf.where(tf.random.uniform(wmask.shape) > 0.995, 1.0, 0.0) *  (1.0 - wmask)
    water_mask_val = water_mask_val.numpy()
    
    # add water mask
    prob_mask_training = np.concatenate((prob_mask_training, water_mask_train), -1)
    if test_set:
        prob_mask_testing = np.concatenate((prob_mask_testing, water_mask_train), -1)
    prob_mask_validation = np.concatenate((prob_mask_validation, water_mask_val), -1)

if use_gamma:
    prob_mask_training *= gamma_mask
    prob_mask_validation *= gamma_mask 
    if test_set:
        prob_mask_testing *= gamma_mask
    
prob_mask_training = np.float32(prob_mask_training)
prob_mask_validation = np.float32(prob_mask_validation)
if test_set:
    prob_mask_testing = np.float32(prob_mask_testing)
    prob_mask_testing[prob_mask_testing < 0] = 0.0
prob_mask_validation[prob_mask_validation < 0] = 0.0
prob_mask_training[prob_mask_training < 0] = 0.0

cmap_lito = cmap_from_labels(labels, 'data/colors_catalog/catlog.csv', from_lang='en', to_lang='en') 

#%% standardize the gridded-data
concat = True
use_coords = True
               
if concat:
    data = sat_grid.copy()
    data = preprocessing.standardize_grid(data)
    
if use_coords:
    #xn, yn, zn = preprocessing.geodetic_to_cartesian(lat=y, lon=x, elevation=sat_grid[0])
    #xnorm, ynorm, zn = preprocessing.normalize_coordinates(x, y, np.zeros_like(y))
    coords = np.array([x, y])
    coords = preprocessing.standardize_grid(coords)
    data = np.concatenate((coords, data), axis=0)

data = data.swapaxes(0, 1).swapaxes(1, 2)
data = np.nan_to_num(data, nan=0.0)
data = np.float32(data)

if use_gamma:
    data *= gamma_mask

#%% plots
for n in range(0, data.shape[-1]-1, 3):
    plt.figure(figsize=(10, 10))
    plt.imshow(adjust_rgb(data[:, :, n:n+3], 10, 90), cmap='Greys')
    #plt.axis('off')
    plt.axis('scaled')
    plt.show()
    
# %% visualize masks
plt.figure(figsize=(10, 10))
for n in range(prob_mask_training.shape[-1]):
    sy, sx = np.where(prob_mask_training[:, :, n] > 0.0)[:2]
    plt.scatter(x[sy,sx], y[sy, sx], label=labels[n], s=5)
    plt.axis('scaled')
    print(f'n_samples - {labels[n]}: {sy.size}')
# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make room for the legend
if use_gamma:
    data *= gamma_mask
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# %% visualize masks
plt.figure(figsize=(10, 10))
for n in range(prob_mask_validation.shape[-1]):
    sy, sx = np.where(prob_mask_validation[:, :, n] > 0.0)[:2]
    plt.scatter(x[sy,sx], y[sy, sx], s=5, label=labels[n])
    plt.axis('scaled')
    print(f'n_samples - {labels[n]}: {sy.size}')
# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


# %% create image tiles
patch_num = 6                                              # Number of patches
patch_size = 256                                           # Patch size
dim = (patch_size, patch_size, data.shape[-1])             # tile dimensions
generate_tiles = preprocessing.ImageTiling(data, prob_mask_training,
                                           prob_mask_validation, dim, min_dist=patch_size//5,
                                           add_padding=True, seed=None, 
                                           eps=1.2, min_samples=2, n_train_iter=16, n_val_iter=25)
# get the number of classes
n_classes = np.min(prob_mask_training.shape)

X_train, Y_train = generate_tiles.training_patches(batch_size=patch_num)
X_val, Y_gt, Y_val = generate_tiles.validation_patches(batch_size=patch_num)


#%% Plot training tiles
for _ in range(5):
    i = np.random.choice(np.arange(X_train.shape[0]), size=1, replace=False)[0]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(adjust_rgb(X_train[i, :, :, -3:]))
    ax[1].imshow(adjust_rgb(X_train[i, :, :, -3:]))
    im1 = ax[1].imshow(np.argmax(Y_train[i], -1), cmap='Dark2', alpha=1.0)
    ax[0].axis('off')
    ax[1].axis('off')
    print(Y_train[i][Y_train[i] > 0.].sum())

# Plot validation tiles
for _ in range(5):    
    i = np.random.choice(np.arange(X_val.shape[0]), size=1, replace=False)[0]
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(adjust_rgb(X_val[i, :, :, -3:]))
    ax[1].imshow(adjust_rgb(X_val[i, :, :, -3:]))
    im1 = ax[1].imshow(np.argmax(Y_val[i], -1), cmap='Dark2', alpha=0.7)
    im1 = ax[2].imshow(np.argmax(Y_gt[i], -1), cmap='Dark2', alpha=0.7)
    ax[0].axis('off')
    ax[1].axis('off')
    print(Y_val[i][Y_val[i] > 0.].sum())
    
# %%  model compilation & hyperparams
dim_1 = X_train.shape[1:]
dim_2 = Y_train.shape[1:]
# if True, the model will be spatially constrained
spatial_constraint = True
coord_channels = 0
if use_coords:
    # number of coordinate channels
    coord_channels = coords.shape[0]
block_size = 5                       # block-size for cross-validation and dropout
n_filters = 32                       # number of filters
threshold = 0.0                      # threshold used on the target mask
temperature = 1.0                    # controls softmax temperature

# (A, B) number of blocks in each U-net model
n_blocks = (4, 4)

# call model
dropout_rate = 0.5
spatial_dropout = True
encoder = None
encoder_freeze = False
activation = None                  # activation used in the last layer
epochs = 1000

sm = spatial_models.Spatial_interp2d(n_classes, dropout_rate=dropout_rate,
                                     spatial_dropout=spatial_dropout, block_size=block_size)

if spatial_constraint:
    model = sm.bayesian_constrained_model(dim_1, dim_2, n_filters=n_filters, n_blocks=n_blocks, n_features=sat_grid.shape[0],
                                          coord_channels=coord_channels, pretrained_encoder=encoder,
                                          threshold=threshold, activation=activation, temperature=temperature, brute_force=False)
    
    fig_ext = 'scbnet'
else:
    model = sm.unet_model(dim_1, n_filters=n_filters, n_blocks=n_blocks[0], activation=activation)
    fig_ext = 'unet'



 #%% load weights
load_wts = True
check_point = 'save_models'
model_name = 'U_NET_4_128_alos_sar_s2_mag_2024_10_01'
weights_path = os.path.join(check_point, model_name, 'second_model_16_2024_10_01.h5')
if load_wts:
    import pandas as pd
    model.load_weights(weights_path)
    history_unet = pd.read_csv(os.path.join(check_point, model_name, 'history_16_2024_10_01.csv'))
    
 #%% load weights
load_wts = True
check_point = 'save_models'
model_name = 'SCB_NET_4_128_alos_sar_s2_mag_None_2024_09_30_scbnet_plus'
weights_path = os.path.join(check_point, model_name, 'second_model_16_2024_10_05.h5')
if load_wts:
    import pandas as pd
    model.load_weights(weights_path)
    history = pd.read_csv(os.path.join(check_point, model_name, 'history_16_2024_09_30.csv'))

# %% model training
training = True
lr = 5e-5                           # learning rate
batch_size = 16                     # number of batches
n_patches = 1800
n_steps = 1
hold_out = 0.25                      # percentage of the mask hold out at each epoch
fw = [1.0]                          # filter weights
fs = [1]                            # filter-size
q = 0.7

fname = 'dilation'                  # filter name

# ----- Early stopping -------
patience = 200                      # number of epochs to wait for the given delta
min_delta = 1e-3                    # min accepted variation

custom_loss = loss_function.spatial_losses(dim, threshold, q=q, fs=fs, fw=fw,
                                           fname=fname, temperature=temperature)

if training:
    # first step - only training set
    features = (X_train, X_val)
    ground_truth = (Y_train, Y_val)
    target = (Y_train, Y_val)

    # train models in parallel
    check_point = 'models/'
    if load_wts is False:
        if spatial_constraint:
            model_name = f'SCB_NET_{n_blocks[0]}_{dim[0]}_{path_ext}_{encoder}_{today}_scbnet_plus'
        else:
            model_name = f'U_NET_{n_blocks[0]}_{dim[0]}_{path_ext}_{today}'

    history, model = gen_training(model, generate_tiles, epochs, n_patches, batch_size, n_steps, custom_loss, 
                                  n_coords=coord_channels, hold_out=hold_out, block_size=block_size//2,
                                  metrics=['acc', 'acc-rescale-2', 'acc-rescale-4', 'dice'], lr=lr,
                                  spatial_constraint=spatial_constraint, monitor='val_acc',
                                  min_delta=min_delta, patience=patience, check_point=check_point, model_name=model_name).fit()

# %% visualize loss and val_loss
view_history = True
savefigs = False
if view_history:
    
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label='Loss')
    plt.plot(history['val_loss'], label='test-loss')

    plt.legend()
    plt.ylabel('Loss', fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    #plt.xlim(0, 210)
    plt.grid(linewidth=0.2, axis='x', alpha=0.9)
    if savefigs:
        plt.savefig(f'plots/east_qc/metrics_validation/history_loss_{path_ext}_{model_name}_{fname}.png',
                    dpi=300, bbox_inches='tight')
    plt.show()

    # visualize mse and val_mse

    plt.figure(figsize=(8, 6))
    plt.plot(history['acc'], label='Accuracy')
    plt.plot(history['val_acc'], label='Val-accuracy')

    plt.legend()
    plt.ylabel('Overall accuracy', fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    plt.grid(linewidth=0.2, axis='x', alpha=0.9)
    if savefigs:
        plt.savefig(f'plots/east_qc/metrics_validation/history_metrics_{path_ext}_{model_name}_{fname}.png',
                    dpi=300, bbox_inches='tight')
    
# %% local prediction
im = np.random.choice(np.arange(X_train.shape[0]), size=1, replace=False)[0]

if Y_train[im][Y_train[im] > 0].any():
    if spatial_constraint:
        local_predictions, recons = model.predict([X_train[im:im+1], Y_train[im:im+1]], verbose=0)[:2]
        
        local_predictions = tf.nn.softmax(local_predictions/1.0)
        
    else:
        local_predictions, recons = model.predict(
            X_train[im:im+1], verbose=0)

    for n in range(0, 1):
        vmin = np.nanmean(
            local_predictions[0, :, :, n]) - 1.75 * np.nanstd(local_predictions[0, :, :, n])
        vmax = np.nanmean(
            local_predictions[0, :, :, n]) + 1.75 * np.nanstd(local_predictions[0, :, :, n])
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5))
        
        ax1.imshow(adjust_rgb(X_train[im, :, :, -3:], 10, 90))
        ax2.imshow(adjust_rgb(X_train[im, :, :, -3:], 10, 90), zorder=0)
        ax2.imshow(np.argmax(local_predictions[0, :, :, :], -1),
                   interpolation='bilinear', cmap='tab20b', zorder=1, alpha=0.9)
        ax3.imshow(adjust_rgb(
            local_predictions[0, :, :, :3]), vmin=vmin, vmax=vmax)

        ax4.imshow(adjust_rgb(recons[0, :, :, -3:]))
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax3.axis('off')
        plt.show()
                      
# %% predict on study area
batches_num = 4
overlap_ratio = 0.5
pad = dim[0]//4
reflect = False
add_padding = True
output = 0 # 0 - predictions / 1 - embeddings
msk = prob_mask_training 
if spatial_constraint is False:
    # predict on full area - model
    n_layers = local_predictions.shape[-1]
    if output == 1:
        n_layers = recons.shape[-1]
        
    pred_tile = postprocessing.predict_tiles(
        sm, merge_func=np.nanmean, add_padding=add_padding, reflect=reflect, pad=pad)
    pred_tile.create_batches(
        data, dim_1, overlap_ratio=overlap_ratio, n_classes=n_layers)
    pred_tile.predict(batches_num=batches_num,
                      extra_channels=0)
    pred_grid = pred_tile.merge()

else:
    n_layers = local_predictions.shape[-1]
    if output == 1:
        n_layers = recons.shape[-1]
    # predict on full area - model
    pred_tile = postprocessing.predict_tiles(
        sm, merge_func=np.nanmean, add_padding=add_padding, reflect=reflect, pad=pad)
    pred_tile.create_batches(np.concatenate([data, msk], -1), (dim[0], dim[1], dim_1[-1]+dim_2[-1]),
                             overlap_ratio=overlap_ratio, n_classes=n_layers)
    pred_tile.predict(batches_num=batches_num,
                      extra_channels=dim_1[-1], output=output)

    pred_grid = pred_tile.merge()
    if output == 1:
        embeddings = pred_grid.copy()
        embeddings = np.nan_to_num(embeddings, 0)

cond = np.isnan(pred_grid) == True
pred_grid = np.nan_to_num(pred_grid, 0) 

        
#%% plot prediction
for c in range(pred_grid.shape[-1]):
    fig, ax = plt.subplots(figsize=(10, 9))
    img = ax.imshow(pred_grid[:, :, c], vmin=np.nanpercentile(pred_grid[:, :, c], 1),
                    vmax=np.nanpercentile(pred_grid[:, :, c], 99), cmap='rainbow')
    ax.axis('off')
    plt.colorbar(img, label=u'Probability', orientation='horizontal')
    plt.axis('scaled')
    plt.show()
    
#%% plot predictive map
ths = 0.5
maxprob = np.nanmax(pred_grid, -1)
igy, igx = np.where(maxprob < ths)[:2]
cat = np.float32(np.argmax(pred_grid, -1))
max_prob_mask = np.where(maxprob > ths, np.nan, 1.)
cat[igy, igx] = np.nan


# Creating custom legend handles based on unique IDs in cat and corresponding labels
unique_ids = np.unique(cat[~np.isnan(cat)])  # Get unique ids excluding NaNs
handles = [mpatches.Patch(color=cmap(i / n_classes), label=labels[int(i)]) for i in unique_ids]

fig, ax = plt.subplots(figsize=(16, 14))
ax.pcolormesh(x, y, hshade, shading='auto', cmap='Greys')
im = ax.pcolormesh(x, y, cat, vmin=0.0, vmax=n_classes, shading='auto', cmap=cmap, alpha=0.8)
ax.contour(x, y, cat, levels=25, colors='k', linewidths=0.07, zorder=4, alpha=0.5, antialiased=True)
ax.pcolormesh(x, y, max_prob_mask, shading='auto', cmap='Greys_r', vmin=1.0, vmax=1.)

# Add the legend
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

ax.axis('scaled')
ax.set_xlabel('Longitude', fontsize=13)
ax.set_ylabel('Latitude', fontsize=13)
ax.set_xlim([x.min() + 0.05, x.max() - 0.05])
ax.set_ylim([y.min() + 0.1, y.max() - 0.05])
plt.show()

# %% validation -- Confusion matrix - one pred
min_samples = 0
threshold = 0.5
use_cv = False
pad = 32

for scale in [1]:
    
    train_mask = prob_mask_training.copy()
    val_mask = prob_mask_validation.copy()
    train_mask[igy, igx] = 0.0
    val_mask[igy, igx] = 0.0
    predictions_grid = pred_grid.copy()
    #predictions_grid[igy, igx] = 0.0
    if scale > 1:
        train_mask = tf.nn.max_pool2d(train_mask[None], ksize=scale, strides=scale, padding='VALID')[0].numpy()
        val_mask = tf.nn.max_pool2d(val_mask[None], ksize=scale, strides=scale, padding='VALID')[0].numpy()
        predictions_grid = tf.nn.avg_pool2d(predictions_grid[None], ksize=scale, strides=scale, padding='VALID')[0].numpy()

    nth = 1
    for plot_train_set in ['train', 'val']:
        if plot_train_set == 'train':
            set_ext = 'train'
            msk = np.max(train_mask, axis=(0, 1)) == 1.0
            idy, idx = np.where(np.max(train_mask[pad:-pad,pad:-pad], -1) > threshold)
            ground_truth = np.argmax(train_mask[pad:-pad,pad:-pad], -nth)[idy, idx]
    
        else:
            set_ext = 'val'
            msk = np.max(val_mask, axis=(0, 1)) == 1.0
            idy, idx = np.where(np.max(val_mask[pad:-pad,pad:-pad], -1) > threshold)
            ground_truth = np.argmax(val_mask[pad:-pad,pad:-pad], -nth)[idy, idx]
    
        # predictions at sampled locations
        predictions = np.argmax(predictions_grid[pad:-pad,pad:-pad], -1)[idy, idx]
    
        print(np.unique(ground_truth).size)
        cfm_path = f'figs/{fig_ext}/cfm_{set_ext}_{fig_ext}.png'
        confusion_matrix = metrics.confusion_matrix(ground_truth, predictions)
        
        plot_cfm(confusion_matrix, labels,
                 norm=True, savefig=False, path=cfm_path,
                 cmap='coolwarm', figsize=(13, 11))
        

# %% bayesian prediction
n_samples = 30
batches_num = 2
overlap = 0.5
pad = dim[0]//4
reflect = False
add_padding = True
msk = prob_mask_training

if spatial_constraint:
    # predict on full area - model 1
    pred_tile = postprocessing.predict_tiles(
        sm, merge_func=np.nanmean, add_padding=add_padding, reflect=reflect, pad=pad)
    pred_tile.create_batches(np.concatenate([data, msk], -1), (dim[0], dim[1], dim_1[-1]+dim_2[-1]),
                             overlap_ratio=overlap, n_classes=n_classes)
    pred_tile.bayesian_prediction(
        batches_num=batches_num, extra_channels=dim_1[-1], n_samples=n_samples)
    pred_grid_mean, pred_grid_var = pred_tile.merge()

else:
    # predict on full area - model 2
    pred_tile = postprocessing.predict_tiles(
        model, merge_func=np.nanmean, add_padding=add_padding, reflect=reflect, pad=pad)
    pred_tile.create_batches(
        data, dim, overlap_ratio=overlap, n_classes=n_classes)
    pred_tile.bayesian_prediction(
        batches_num=batches_num, extra_channels=0, n_samples=n_samples)
    pred_grid_mean, pred_grid_var = pred_tile.merge()

pred_grid_mean = np.nan_to_num(pred_grid_mean, 0) 
pred_grid_var = np.nan_to_num(pred_grid_var, 0) 

#%% plot prediction - MEAN & std
import matplotlib.colors as mcolors
fig, ax = plt.subplots(n_classes//2, 4, figsize=(17, 18))
xlim = [16, pred_grid_mean.shape[1]-16]
ylim = [pred_grid_mean.shape[0]-16, 16]

# Create a power normalization
gamma = 0.2  # Adjust gamma (>1 means more detail near 1, <1 means more detail near 0)
norm = mcolors.PowerNorm(gamma=0.1, vmin=0.0, vmax=1.0)
norm_std = mcolors.PowerNorm(gamma=0.08, vmin=0.0, vmax=0.22)

for c in range((2*n_classes)//2):
    if c < n_classes//2:
        im1 = ax[c, 0].pcolormesh(x, y, pred_grid_mean[:, :, c], cmap='nipy_spectral', norm=norm)
        #plt.colorbar(im1, label=u'Mean', orientation='horizontal')
        im2 = ax[c, 1].pcolormesh(x, y, np.sqrt(pred_grid_var[:, :, c]), cmap='Spectral_r', norm=norm_std)
        ax[c, 0].axis('off'); ax[c, 1].axis('off')
        ax[c, 0].axis('scaled'); ax[c, 1].axis('scaled')
        if c == 0:
            # Adding the colorbar
            cbaxes1 = fig.add_axes([1.01, 0.52, 0.018, 0.13])
            cbaxes2 = fig.add_axes([1.01, 0.35, 0.018, 0.13])

            # position for the colorbar
            plt.colorbar(im1, cax=cbaxes1, ticks=[0.0, 0.05, 0.2, 0.5, 1.0], label=u'Mean')
            plt.colorbar(im2, cax=cbaxes2, ticks=[0.0, 0.02, 0.08, 0.20], label=u'Standard Deviation')

        ax[c, 0].set_title(labels[c] + ' (mean)', fontsize=13);
        ax[c, 1].set_title(labels[c] + ' (std)', fontsize=13)


    else:
        im1 = ax[c-n_classes//2, 2].pcolormesh(x, y, pred_grid_mean[:, :, c], cmap='nipy_spectral', norm=norm)
        im2 = ax[c-n_classes//2, 3].pcolormesh(x, y,np.sqrt(pred_grid_var[:, :, c]), cmap='Spectral_r', norm=norm_std)
        ax[c-n_classes//2, 2].set_title(labels[c] + ' (mean)', fontsize=13);
        ax[c-n_classes//2, 3].set_title(labels[c] + ' (std)', fontsize=13)
        ax[c-n_classes//2, 2].axis('off'); ax[c-n_classes//2, 3].axis('off')
        ax[c-n_classes//2, 2].axis('scaled'); ax[c-n_classes//2, 3].axis('scaled')

plt.tight_layout()
fig.savefig(f'figs/{fig_ext}/litmaps_mean_std.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
norm_std = mcolors.PowerNorm(gamma=0.22, vmin=0.0, vmax=0.28)
plt.figure(figsize=(10, 10))
plt.pcolormesh(x, y, pred_grid_var[:, :, 3], cmap='nipy_spectral', norm=norm_std)
plt.axis('scaled')
plt.show()


#%% plot predictive map
ths = 0.6
maxprob = np.nanmax(pred_grid_mean, -1)
igy, igx = np.where(maxprob < ths)[:2]
cat = np.float32(np.argmax(pred_grid_mean, -1))
max_prob_mask = np.where(maxprob > ths, np.nan, 1.)
cat[igy, igx] = np.nan


# Creating custom legend handles based on unique IDs in cat and corresponding labels
unique_ids = np.unique(cat[~np.isnan(cat)])  # Get unique ids excluding NaNs
handles = [mpatches.Patch(color=cmap(i / n_classes), label=labels[int(i)]) for i in unique_ids]

fig, ax = plt.subplots(figsize=(16, 14))
ax.pcolormesh(x, y, hshade, shading='auto', cmap='Greys')

im = ax.pcolormesh(x, y, cat, vmin=0.0, vmax=n_classes, shading='auto', cmap=cmap, alpha=0.8)
ax.contour(x, y, cat, levels=25, colors='k', linewidths=0.07, zorder=4, alpha=0.5, antialiased=True)
ax.pcolormesh(x, y, max_prob_mask, shading='auto', cmap='Greys_r', vmin=1.0, vmax=1.)

# Add the legend
ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

ax.axis('scaled')
ax.set_xlabel('Longitude', fontsize=13)
ax.set_ylabel('Latitude', fontsize=13)
ax.set_xlim([x.min() + 0.05, x.max() - 0.05])
ax.set_ylim([y.min() + 0.1, y.max() - 0.05])
plt.show()
    
# %% validation -- Confusion matrix - one pred
min_samples = 0
threshold = 0.9
use_cv = False
pad = 32

for scale in [1]:    
    train_mask = prob_mask_training.copy()
    val_mask = prob_mask_validation.copy()
    train_mask[igy, igx] = 0.0
    val_mask[igy, igx] = 0.0
    predictions_grid = pred_grid_mean.copy()
    #predictions_grid[igy, igx] = 0.0
    if scale > 1:
        train_mask = tf.nn.max_pool2d(train_mask[None], ksize=scale, strides=scale, padding='VALID')[0].numpy()
        val_mask = tf.nn.max_pool2d(val_mask[None], ksize=scale, strides=scale, padding='VALID')[0].numpy()
        predictions_grid = tf.nn.avg_pool2d(predictions_grid[None], ksize=scale, strides=scale, padding='VALID')[0].numpy()

    nth = 1
    for plot_train_set in ['val']:
        if plot_train_set == 'train':
            set_ext = 'train'
            msk = np.max(train_mask, axis=(0, 1)) == 1.0
            idy, idx = np.where(np.max(train_mask[pad:-pad,pad:-pad], -1) > threshold)
            ground_truth = np.argmax(train_mask[pad:-pad,pad:-pad], -nth)[idy, idx]
    
        else:
            set_ext = 'val'
            msk = np.max(val_mask, axis=(0, 1)) == 1.0
            idy, idx = np.where(np.max(val_mask[pad:-pad,pad:-pad], -1) > threshold)
            ground_truth = np.argmax(val_mask[pad:-pad,pad:-pad], -nth)[idy, idx]
    
        # predictions at sampled locations
        predictions = np.argmax(predictions_grid[pad:-pad,pad:-pad], -1)[idy, idx]
    
        print(np.unique(ground_truth).size)
        cfm_path = f'figs/{fig_ext}/cfm_avg_{set_ext}_{fig_ext}.png'
        confusion_matrix = metrics.confusion_matrix(ground_truth, predictions)
        
        plot_cfm(confusion_matrix, labels, display_counts=True,
                 norm=True, savefig=False, path=cfm_path,
                 cmap='coolwarm', figsize=(15, 13))
