## All projects and assignments for EAS586–Medical Imaging Analysis

### In U-Net_CNN.ipynb:
#### Left Ventricle Segmentation in Cardiac MRI with U-Net
This project implements a Convolutional Neural Network using a U-Net architecture to perform semantic segmentation of the left ventricle in cardiac cine-MRI images.
I trained the model using a dataset of expert-annotated contours and evaluate performance using the DICE coefficient as both a loss function and evaluation metric. The model is then applied to previously unseen images to generate automated left ventricle segmentations.

* Built with Keras and TensorFlow
* Trained on expert-labeled cardiac MRI contours
* Applied to a test set of unseen cine-MRI images

<img width="624" height="361" alt="Screenshot 2025-08-02 at 11 54 29 AM" src="https://github.com/user-attachments/assets/7c8d4bb2-2473-456b-b088-97004351c0fe" />

* Average DICE score: 0.86

Data citation:
Radau P, Lu Y, Connelly K, Paul G, Dick AJ, Wright GA. “Evaluation Framework for Algorithms Segmenting Short Axis Cardiac MRI.” The MIDAS Journal – Cardiac MR Left Ventricle Segmentation Challenge, http://hdl.handle.net/10380/3070

### In CNN_pneumonia_cls.ipynb:
#### Pneumonia Detection from Chest X-Rays using CNNs
This project applies convolutional neural networks (CNNs) to classify chest X-ray images as either Normal (0) or Pneumonia (1). I did the following:

* Built a CNN model in Keras to classify lung X-ray images
* Used ImageDataGenerator for data augmentation
* Added Dropout layers to reduce overfitting
* Trained the model on a labeled dataset of X-rays and evaluated its accuracy on a validation set
* Accuracy: 0.92
* Precision: 0.91
* Sensitivity: 0.97


### In pneumonia_class_imbalance.ipynb:
#### Handling Class Imbalance
This notebook focuses on improving model performance by addressing class imbalance in the pneumonia dataset. I compared the effectiveness of these methods:

* Random oversampling and undersampling
* SMOTE-CNN (Synthetic Minority Over-sampling Technique)
* Binary Focal Crossentropy loss to prioritize harder examples
* Evaluation of sensitivity, precision, and accuracy across all methods

