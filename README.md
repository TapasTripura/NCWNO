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

## File description
  📂 1d continual learning          # Contains files of the 1d continual learning.
    |_📂 data                       # Folder for storing DATA and generating data.
      |_📁 model                    # Folder for storing trained models.
        |_📄 'pre-trained model'               # location of model.
      |_📄 'data file'                         # location of data.
      |_📄 ...                     
    |_📁 results                               # location of results.
  📂 1d multiple learning           # Contains files of the 1d multiple physics training.
    |_📂 data                       
      |_📁 model                    
    |_📁 results                    
  📂 2d continual learning          # Contains files of the 2d continual learning.
    |_📂 data                       
      |_📁 model                    
    |_📁 results                    
  📂 2d multiple learning           # Contains files of the 2d multiple physics training.
    |_📂 data                       
      |_📁 model                    
    |_📁 results                    

## Essential Python Libraries
Following packages are required to be installed to run the above codes:
  + Numpy
  + Matplotlib
  + [PyTorch](https://pytorch.org/)
  + [PyWavelets - Wavelet Transforms in Python](https://pywavelets.readthedocs.io/en/latest/)
  + [Wavelet Transforms in Pytorch](https://github.com/fbcotter/pytorch_wavelets)
  + [Wavelet Transform Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)

## Dataset
  + The testing datasets are available at the following link:
    > [Dataset](https://csciitd-my.sharepoint.com/:f:/g/personal/amz218313_iitd_ac_in/Ek5VwNvhbKhIjWR6L4SJvKUBqaPpQAj7akhWhpqt6Ucq-w?e=2Lwm1S)
  `The datasets must be placed inside the corresponding data folder`

## Dataset
  + The testing datasets are available at the following link:
    > [Dataset](https://csciitd-my.sharepoint.com/:f:/g/personal/amz218313_iitd_ac_in/Ek5VwNvhbKhIjWR6L4SJvKUBqaPpQAj7akhWhpqt6Ucq-w?e=2Lwm1S)
  `The models must be placed inside the corresponding data/model/ folder`
