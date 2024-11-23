# GuardEye: AI-Powered Theft Detection System

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.x-red)
![CUDA](https://img.shields.io/badge/CUDA-v11.0-76b900)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue)
![Numpy](https://img.shields.io/badge/Numpy-v1.21%2B-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.x-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.11.x-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Enabled-yellow)


## Table of Contents
- [Introduction](#introduction)
- [Research Paper](#research-paper)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset Description](#dataset-description)
- [Preprocessing](#preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction
In this project, we propose **GuardEye**, a hybrid deep learning model leveraging **CNN** and **LSTM** architectures for real-time theft detection in video surveillance. This system is designed to process video streams and identify suspicious activities in public spaces such as malls, metro stations, and parking lots.

---

## Research Paper

You can find the detailed research paper [here](paper/Two_Coloum_IEEE_Improving%20Security%20in%20Public%20Spaces%20Hybrid%20Deep.pdf).


---

## Features
- **Real-time Detection:** Classifies ongoing activities in video streams.
- **Hybrid Deep Learning:** Combines CNN for spatial feature extraction and LSTM for temporal feature learning.
- **High Accuracy:** Optimized with state-of-the-art architecture achieving robust performance.
- **Modular Codebase:** Separate modules for training, testing, and inference for ease of use.

---

## Model Architecture
The architecture consists of:
1. **CNN Layers:** To extract spatial features from each frame of the video.
2. **LSTM Layers:** To capture temporal dependencies in the sequence of video frames.
3. **Fully Connected Layers:** For classification into theft or non-theft activities.

Below is the visual representation of the architecture:

![Model Architechture](https://github.com/user-attachments/assets/a60af0a6-06f6-4b1a-a883-a303583c5100)


---

## Dataset Description
We utilized the **UCF Crime Dataset**, which includes a variety of video sequences containing both normal and anomalous activities. 

### Dataset Highlights:
- **Dataset Size:** 1900 video clips.
- **Categories:** Includes theft, burglary, assault, arson, and normal activities.
- **Format:** Videos in MP4 format with varying resolutions.

### Example of Dataset Frames:
![Dataset Frames](https://github.com/user-attachments/assets/1d94312e-6bd8-4b85-bc77-d39af1f09179)


---

## Preprocessing
1. **Frame Extraction:** Each video is converted into a sequence of frames.
2. **Resizing:** Frames are resized to 224x224 pixels for input into the CNN.
3. **Normalization:** Pixel values are normalized between 0 and 1.
4. **Label Encoding:** Activities are encoded into numerical labels for training.

---

## Training and Evaluation
### Training Process
The training loop updates model weights based on the loss and accuracy for each epoch. The model is saved after completion of training.

To train the model, run:
```bash
python train.py
```

### Testing
To evaluate the model on the test dataset, run:

```bash
python test.py
```

### Training Hyperparameters
1. **Optimizer:** Adam optimizer with a learning rate of 0.001.
2. **Loss Function:** Cross-entropy loss for multi-class classification.
3. **Batch Size:** 8 frames per batch.
4. **Epochs:** 50.
   
---
## Results

### Classification Report
Below is an example of the classification report generated after testing the model:

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Theft      | 0.93      | 0.92   | 0.93     |
| Assault    | 0.89      | 0.88   | 0.88     |
| Normal     | 0.95      | 0.94   | 0.94     |


### ROC Curve
The ROC curve demonstrates the performance of the classifier across various threshold values, showing the trade-off between sensitivity and specificity:
![ROC](https://github.com/user-attachments/assets/f9aeee1c-aba3-4a1c-b6ae-117d9f731ee3)


### Accuracy
The overall accuracy of the model is: 92.5%

--- 

## Installation
### Prerequisites
- Python >= 3.6
- GPU (optional but recommended for training)
### Install Dependencies
Run the following command to install all dependencies:
```bash
pip install -r requirements.txt
```
This will guide users to install the necessary dependencies for the project.
## Usage
### Training
To train the model, use the command:
```bash
python train.py
```
### Testing
To evaluate the model on the test dataset:
```bash
python test.py
```
### Inference
To run inference on a single video:
```bash
python inference.py --video_path path_to_video.mp4
```
This will clearly explain how to use the different scripts for training, testing, and inference in your project.

---

## Project Structure
The repository is organized as follows:
```bash
├── dataset/                    # Dataset folder
│   ├── train/                  # Training data
│   ├── test/                   # Test data
│   └── transform_image.py      # Preprocessing scripts
├── models/
│   ├── cnn.py                  # CNN architecture
│   ├── lstm.py                 # LSTM architecture
│   ├── combined.py             # Combined model
├── train.py                    # Training script
├── test.py                     # Testing script
├── inference.py                # Inference script
├── requirements.txt            # Dependencies
├── results/                    # Folder for result images and logs
│   ├── confusion_matrix.png    # Confusion Matrix
│   ├── roc_curve.png           # ROC Curve
├── README.md                   # Project README
└── crime_detection_model.pth   # Trained model
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

We would like to thank:

- The creators of the UCF Crime Dataset for providing the dataset.
- The open-source community for their contributions to deep learning frameworks like PyTorch and OpenCV.
