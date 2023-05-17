# ECE-GY-6123-OpticalFlowEstimation

## Overview

This repo contains the final project of Dehit Trivedi and Jesse Inouye for ECE-6123 Image and Video Processing and NYU.

The overarching goal of the project was to compare different methods of optical flow estimation, and the code is split into two parts:
1. Iterative algorithms (e.g. Lucas-Kanade)
2. PWC-Net


## Iterative algorithms

To run the script for iterative algorithms, upload `iterative_algorithms.ipynb` to Google Colab. You will likely need to download the SINTEL dataset (or any prefered optical flow dataset) and upload your test images to your Google Drive account, changing the path in the script to match the path of your images.

## PWC-Net

To run the PWC-Net code, the following is needed:
1. PyTorch (preferrably version 2.0 - your PyTorch install must match your version of CUDA)
2. A GPU with a CUDA toolkit installed that matches your PyTorch install
3. Visual Studio for the necessary C++ utilties

You will need to compile the correlation_package for your system.

To do so, in ECE-GY-6123-OpticalFlowEstimation/model/correlation_package/setup.py, make sure your GPUs architecture is represented.

More info can be found here: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

Once the correlation package is compiled, you can test the PWC-Net model with:

```
python test_model.py
```
