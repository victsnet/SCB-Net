# SCB-Net
Spatially Constrained Bayesian Network (SCB-Net): An Approach to Obtaining Field Data-Constrained Predictive Maps with Uncertainty Assessment. In this project, we have developed an innovative approach to ensure that predicted lithologies accurately reflect the samples collected in the field. Moreover, our model leverages multisource remote sensing data to make predictions in areas where no direct samples are available. Our study focuses on the Churchill Province, located in Quebec, Canada.


## Predictive Lithological Map displaying 16 lithologic units 

![output](https://github.com/victsnet/SCB-Net/assets/53713685/81b74534-f222-4854-8d4e-ff265c06011d)
### (instability = uncertain predictions)

Authors: Victor S. Santos (INRS & NRCan), Erwan Gloaguen (INRS), and Shiva Tirdad (NRCan).

[Link to Preprint - ArXiv](https://arxiv.org/abs/2403.20195)

INRS: Institut National de la Recherche Scientifique

NRCan: Natural Resources Canada

## Requirements
- python>=3.7
- numpy>=1.22
- pandas>=1.4
- xarray>=2022.03
- rioxarray>=0.10
- rasterio>=1.3
- geopandas>=0.11
- shapely>=1.8
- scipy>=1.8
- tqdm>=4.64
- matplotlib>=3.5
- scikit-learn>=1.1
- textdistance>=4.2
- translate>=3.6
- tensorflow>=2.8
- opencv-python>=4.5
- pyproj>=3.3

[Link to remotely sensed data, probability masks, and weights of the models.](https://drive.google.com/drive/folders/1XKIUqlInuHOdva_IWbaQ_Ynezm4p_Fcd?usp=drive_link)

## Inputs
### Northeast area
#### Probability masks
- training mask: train_prob_mask_bs10_400_code_r3.tif
- validation mask: val_prob_mask_bs10_400_code_r3.tif

#### Remote sensing layers
- Multispectral: sentinel2_multispec_east_qc_100m.tif
- RADAR: ALOS_PALSAR_RADAR_MOSAIC_QC_100m.tif
- Magnetic: MAG_QC_LOWRES_RESMAG_4269_epsg.tif; MAGRES_QC_LOWRES_AS_4269_epsg.tif; MAGRES_QC_LOWRES_DV1_4269_epsg.tif
- DEM: alos_elev_east_qc_100m.tif

### North area
#### Probability masks
- training mask: train_prob_mask_bs10_400_code_r2_north.tif
- validation mask: val_prob_mask_bs10_400_code_r2_north.tif

#### Remote sensing layers
- Multispectral: sentinel2_multispec_east_qc_100m.tif
- RADAR: ALOS_PALSAR_RADAR_MOSAIC_QC_100m.tif
- Magnetic: MAG_QC_LOWRES_RESMAG_4269_epsg.tif; MAGRES_QC_LOWRES_AS_4269_epsg.tif; MAGRES_QC_LOWRES_DV1_4269_epsg.tif
- DEM: alos_elev_east_qc_100m.tif

## Outputs


### License

This project is licensed under the [Creative Commons Attribution 4.0 International License](LICENSE) - see the [LICENSE](LICENSE) file for details.
