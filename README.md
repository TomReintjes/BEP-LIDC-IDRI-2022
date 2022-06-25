# BEP-LIDC-IDRI-2022
Bachelor End Project - Tom Reintjes
This repository contains code to use for the LIDC-IDRI dataset in Multitask Learning. The folder 'Code' contains python scirpts for read data, preprocessing data and use the data in the Baseline model and Multitask model in 3D ResNet50.

**SETUP**

PYTHON_VERSION = '3.9.7'
KERAS_VERSION = '2.9.0'
TENSOR_FLOW_GPU = '2.9.1'

Download the Pylidc library via pip install pylidc

More information about the library and its classes go to https://pylidc.github.io/

Pylidc is used to easily read the .dcm and .xml files to convert to NumPy. It contains a Scan, Annotation and a Consesus class.
![t](https://github.com/TomReintjes/BEP-LIDC-IDRI-2022/blob/main/Figure%202022-06-09%20222958.png)


Also install the sci-kit-image library via pip install sci-kit image

**CODE OVERVIEW**

The convert data file reads the LIDC-IDRI dataset, creates a CSV file and convert all scans to NumPy arrays. Which also are stacked on top of eachother to create a voxel as input for the 3D ResNet.

Baseline model imports traintestsplit.py , reportresults.py and constants.py

In the constants.py file you can adjust the constants used in the model. For example; Epochs, Inputshape, annotation choice and True/False statements
The multitask model is almost the same as the Baseline model, but an annotation is added with  sigmoid and linear as activation 

For improve the speed of the model, a keras.mixed_precision.LossScaleOptimizer is used. Also the EarlyStopping function is used to prevent overfitting.

ALL files contain short description as explaination created functions.
