# COVID-19 Image Classification

A deep learning project that uses chest X-ray images to classify cases into **COVID-19** and **Normal** categories using Convolutional Neural Networks (CNNs).

---

## 📌 Project Overview

The project aims to develop an AI-based image classification system that can analyze chest X-ray images and identify patterns associated with COVID-19.

The project includes:

- Dataset loading
- Data exploration
- Exploratory Data Analysis (EDA)
- Image preprocessing
- Image normalization
- CNN model development
- Model training
- Model evaluation
- Model comparison
- Final model selection
- Test dataset evaluation

The project is developed using Python, TensorFlow/Keras, NumPy, Pandas, OpenCV, Matplotlib, Seaborn, and Scikit-learn.

> **Disclaimer:** This project is intended for educational and research purposes only. It should not be used as a medical diagnostic system or as a replacement for professional medical advice.

---

## 🎯 Problem Statement

COVID-19 diagnosis can require significant time and resources, especially during periods of high demand.

This project investigates whether deep learning and chest X-ray images can be used to classify patients into COVID-19 and Normal categories.

The objective is to build a CNN-based image classification model that can identify COVID-19 patterns from chest X-ray images.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze the provided COVID-19 image dataset.
- Understand the structure and distribution of the dataset.
- Perform Exploratory Data Analysis.
- Preprocess the image data.
- Normalize image pixel values.
- Split the dataset into training, validation, and testing datasets.
- Build CNN-based classification models.
- Train the CNN models.
- Evaluate model performance.
- Compare different CNN architectures.
- Select the better-performing model.
- Evaluate the selected model on the test dataset.

---

## 📂 Dataset

The project uses two main dataset files:

```text
CovidImages.npy
CovidLabels.csv
```

### CovidImages.npy

This file contains the image data stored in NumPy array format.

### CovidLabels.csv

This file contains the corresponding labels for the images.

The dataset contains two classes:

```text
0 → Normal
1 → COVID
```

---

## 📊 Dataset Description

The image dataset has the following shape:

```text
(251, 128, 128, 3)
```

This represents:

- **251** images
- **128 × 128** image dimensions
- **3** color channels (RGB)

Therefore, the dataset contains 251 RGB images with a resolution of 128 × 128 pixels.

---

## 🛠️ Technologies Used

The project is implemented using Python.

### Programming Language

- Python 3

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- TensorFlow
- Keras
- Scikit-learn

### Important Modules

```python
numpy
pandas
matplotlib
seaborn
cv2
tensorflow
keras
sklearn
```

### Machine Learning Components

The following components are used:

- `Conv2D`
- `MaxPooling2D`
- `Flatten`
- `Dense`
- `Input`
- `Adam`
- `train_test_split`
- `LabelEncoder`
- `classification_report`
- `confusion_matrix`

---

## 🔄 Project Workflow

The overall workflow of the project is:

```text
                    COVID-19 Image Dataset
                              |
                              ↓
                    Load Images and Labels
                              |
                              ↓
                  Data Overview and Analysis
                              |
                              ↓
                  Exploratory Data Analysis
                              |
                              ↓
                     Data Preprocessing
                              |
                              ↓
                    Train/Validation/Test
                              |
                              ↓
                       Normalization
                              |
                              ↓
                    Build CNN Model 1
                              |
                              ↓
                     Train and Evaluate
                              |
                              ↓
                    Build CNN Model 2
                              |
                              ↓
                     Train and Evaluate
                              |
                              ↓
                    Compare Both Models
                              |
                              ↓
                     Select Final Model
                              |
                              ↓
                     Test Set Evaluation
```

---

# 🔍 Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the dataset before model development.

The analysis includes:

- Dataset shape
- Label distribution
- Image visualization
- Understanding image dimensions
- Checking the data structure
- Understanding COVID and Normal class distribution

Visualization libraries such as Matplotlib and Seaborn are used for data exploration.

---

# ⚙️ Data Preprocessing

The dataset is divided into training, validation, and testing datasets.

The notebook uses a stratified split to maintain the class distribution.

### Dataset Split

```text
Training Data   → 70%
Validation Data → 15%
Test Data       → 15%
```

The first split divides the dataset into:

```text
70% Training
30% Temporary
```

The temporary dataset is then divided equally into:

```text
15% Validation
15% Test
```

The split uses:

```python
random_state=42
```

and stratification based on the labels.

---

# 🔢 Data Normalization

The image pixel values are normalized by dividing them by `255.0`.

```python
X_train_normalized = X_train / 255.0
X_val_normalized = X_val / 255.0
X_test_normalized = X_test / 255.0
```

---

# 🧠 Model Development

Two different Convolutional Neural Network architectures are developed and compared.

---

# 🧠 CNN Model 1

The first model contains three convolutional and pooling combinations.

### Architecture

```text
Input Layer
     ↓
Conv2D - 32 Filters
     ↓
MaxPooling2D
     ↓
Conv2D - 64 Filters
     ↓
MaxPooling2D
     ↓
Conv2D - 128 Filters
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense - 128 Neurons
     ↓
Output - 1 Neuron
     ↓
Sigmoid
```

### Model Configuration

- Input shape: `128 × 128 × 3`
- Convolution layers: 3
- Pooling layers: 3
- Hidden dense layer: 128 neurons
- Activation function: ReLU
- Output activation: Sigmoid
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Metric: Accuracy

---

# 📈 CNN Model 1 Training

The model is trained using:

```text
Epochs     : 10
Batch Size : 8
Optimizer  : Adam
Loss       : Binary Crossentropy
```

The notebook records the training time during model training.

---

# 📊 CNN Model 1 Evaluation

The model is evaluated using:

- Training Accuracy
- Validation Accuracy
- Training COVID Recall
- Validation COVID Recall

The notebook reports:

```text
Training Accuracy       : 100%
Validation Accuracy     : 97.37%
Training COVID Recall   : 100%
Validation COVID Recall : 100%
```

The perfect training accuracy may indicate that the model is fitting the training data very closely and may have some overfitting.

---

# 🧠 CNN Model 2

The second CNN model uses fewer convolutional layers compared with Model 1.

The purpose is to create a simpler architecture and improve generalization.

### Architecture

```text
Input Layer
     ↓
Conv2D - 32 Filters
     ↓
MaxPooling2D
     ↓
Conv2D - 64 Filters
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense - 128 Neurons
     ↓
Dense - 64 Neurons
     ↓
Output - 1 Neuron
     ↓
Sigmoid
```

### Model Configuration

- Input shape: `128 × 128 × 3`
- Convolution layers: 2
- Pooling layers: 2
- Dense layer: 128 neurons
- Dense layer: 64 neurons
- Activation function: ReLU
- Output activation: Sigmoid
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Learning rate: `0.0001`
- Metric: Accuracy


---

# 📈 CNN Model 2 Training

CNN Model 2 is trained using:

```text
Epochs        : 10
Batch Size    : 8
Optimizer     : Adam
Learning Rate : 0.0001
Loss          : Binary Crossentropy
```

The training time is also recorded in the notebook.

---

# 📊 CNN Model 2 Evaluation

The model is evaluated using:

- Training Accuracy
- Validation Accuracy
- Training COVID Recall
- Validation COVID Recall

The notebook reports:

```text
Training Accuracy       : 98.29%
Validation Accuracy     : 100%
Training COVID Recall   : 96.10%
Validation COVID Recall : 100%
```

---

# 📋 Model Performance Comparison

The two models are compared based on accuracy and COVID recall.

| Model | Train Accuracy | Validation Accuracy | Train COVID Recall | Validation COVID Recall |
|---|---:|---:|---:|---:|
| CNN Model 1 - 3 Conv Layers | 100.00% | 97.37% | 100.00% | 100.00% |
| CNN Model 2 - 2 Conv Layers | 98.29% | 100.00% | 96.10% | 100.00% |

---

# 🏆 Final Model Selection

Based on the notebook's model comparison, **CNN Model 2** is selected as the final model.

The reasons include:

- Simpler architecture
- Fewer convolutional layers
- Lower training accuracy compared with Model 1
- Higher validation accuracy
- Strong COVID recall
- Better indication of generalization compared with the deeper model

Therefore, CNN Model 2 is used for the final test evaluation.

---

# 📝 Conclusion

This project demonstrates the development of a CNN-based COVID-19 image classification system using chest X-ray images.

Two CNN architectures were developed and evaluated. The first model uses three convolutional layers, while the second model uses two convolutional layers with additional dense layers.

Based on the notebook's model comparison, **CNN Model 2** was selected as the final model because it achieved strong validation performance while using a simpler convolutional architecture.

The executed test evaluation reports:

```text
Test Accuracy       : 100%
COVID Recall        : 100%
```

However, because another section of the notebook reports different test values, the final test performance should be verified by rerunning the notebook.

Overall, the project demonstrates how deep learning and CNN architectures can be applied to image classification tasks involving COVID-19 chest X-ray images.

---



