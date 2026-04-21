fp_data.py

This file implements utilities related to the dataset and formatting, such as the values typically used for segmentation masks in literature.

Of specific note are the AI4Mars_DataSet and AI4Mars_SubSet classes. Both extract images from the AI4Mars dataset folder.

Since the dataset is randomly split into training and validation subsets from images in the same folder, the subset wrapper is used to provide different transform pipelines for the augmentation of the training subset.

fp_metrics.py

This script implements the loss functions and standardized assessment functions.

While it produces confusion matrices and validation metrics, the standalone Dice Loss implementation currently has a known issue within the training loop.

However, the Log Cosh Dice Loss works correctly, even though it references the Dice loss class. This is likely due to how temporary instances of the loss handle tensor sizes.

train_resnet.py and pretrain_resnet.py

These files contain the implementation of the training and pre-training loops on the M2020 and MSL datasets, respectively.

They include comprehensive command-line argument parsing to allow for the control of hyperparameters and settings.

This was designed to facilitate job requests on the server cluster used for the majority of the training process.