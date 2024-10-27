# Pneumonia Detection Model

This project is developed as part of a student practical course on *Computer Vision and Artificial Intelligence*. The model is based on the VGG16 convolutional neural network architecture and is designed for automatic classification of chest X-ray images to detect pneumonia.

## Features
- Data preprocessing (resizing, normalization, and augmentation)
- Model architecture based on VGG16 with additional classification layers
- Code execution for model training and evaluation with metric visualization
- Achieved test set accuracy: **88.62%**

## Installation
1. Clone the repository.
2. Install the required libraries:
   - TensorFlow
   - Keras
   - Pandas
   - NumPy
   - Matplotlib
   - Seaborn
   - Pillow
   - Albumentations
   - scikit-learn

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in the `./dataset` folder in the root directory of the project.

## Running the Project
1. Run the `main.py` file to train and test the model.
2. The project was tested with Python version 3.12.0.

For a detailed explanation of functions and model architecture, refer to the comments in the code.





