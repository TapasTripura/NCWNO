# NCWNO
Neural Combinatorial Wavelet Neural Operator for catastrophic forgetting free in-context operator learning of multiple partial differential equations

## What are we trying to do?
![WNO](/media/WNN_Neurips_INAE_Objective.png)

## NCWNO architecture in a glimpse.
![WNO](/media/ncwno.jpg)

## Simultaneously Predicting solutions of multiple PDEs using a Pre-Trained NCWNO.
  > 1D problems:
  ![Pre-training is done on a 256 spatial grid.](/media/Animation_NCWNO_256_1d.gif)
  
  > 2D problems:
  ![Pre-training is done on 64 x 64 spatial grid](/media/Animation_NCWNO_64_2d.gif)

## Files
A short despcription on the files are provided below for ease of readers. For `time-dependent` problems, please implement the autoregressive schemes provided in `Version 2.0.0`.
```
  + `wno1d_Burgers_v3.py`: For 1D Burger's equation (time-independent problem).
  + `wno2d_Darcy_cwt_v3.py`: For 2D Darcy equation using Slim Continuous Wavelet Transform (time-independent problem).
  + `wno2d_Darcy_dwt_v3.py`: For 2D Darcy equation using Discrete wavelet transform (time-independent problem).
  + `wno3d_NS_dwt_v3.py`: For 2D Navier-Stokes equation using 3D WNO (as a time-independent problem).

  + `Test_wno_1d_Burgers.py`: An example of Testing on new data.
  
  + `utils.py` contains some useful functions for data handling (improvised from [FNO paper](https://github.com/zongyi-li/fourier_neural_operator)).
  + `wavelet_convolution_v3.py` contains functions for 1D, 2D, and 3D convolution in wavelet domain.
```
📂 Allen_Cahn                     # Contains files of the Allen Cahn equation.
  |_📂 data                       # Folder for storing DATA and generating data.
    |_📄 Allen_cahn_Init.m                         # Generates random initial conditions for the Allen Cahn equation.
    |_📄 RandField_Matern.m                        # Generates random fields using Mattern kernel.
    |_📄 stationary_Gaussian_process.m             # Contains functions for constructing kernels.
  |_📁 model                      # Folder for storing trained models.
  |_📁 results                    # Folder for storing analysis results post-training.

## Essential Python Libraries
Following packages are required to be installed to run the above codes:
  + [PyTorch](https://pytorch.org/)
  + [PyWavelets - Wavelet Transforms in Python](https://pywavelets.readthedocs.io/en/latest/)
  + [Wavelet Transforms in Pytorch](https://github.com/fbcotter/pytorch_wavelets)
  + [Wavelet Transform Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)
  + [Xarray-Grib reader (To read ERA5 data in section 5)](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html?highlight=install)

## Dataset
  + The training and testing datasets for the (i) Burgers equation with discontinuity in the solution field (section 4.1), (ii) 2-D Allen-Cahn equation (section 4.5), and (iii) Weakly-monthly mean 2m air temperature (section 5) are available in the following link:
    > [Dataset-1](https://drive.google.com/drive/folders/1scfrpChQ1wqFu8VAyieoSrdgHYCbrT6T?usp=sharing) \
The dataset for the Weakly and monthly mean 2m air temperature are downloaded from 'European Centre for Medium-Range Weather Forecasts (ECMEF)' database. For more information on the dataset one can browse the link 
    [ECMEF](https://www.ecmwf.int/en/forecasts/datasets/browse-reanalysis-datasets).
  + The datasets for (i) 1-D Burgers equation ('burgers_data_R10.zip'), (ii) 2-D Darcy flow equation in a rectangular domain ('Darcy_421.zip'), (iii) 2-D time-dependent Navier-Stokes equation ('ns_V1e-3_N5000_T50.zip'), are taken from the following link:
    > [Dataset-2](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
  + The datasets for 2-D Darcy flow equation with a notch in triangular domain ('Darcy_Triangular_FNO.mat') and 1-D time-dependent wave advection equation are taken from the following link:
    > [Dataset-3](https://github.com/lu-group/deeponet-fno/tree/main/data)
    
