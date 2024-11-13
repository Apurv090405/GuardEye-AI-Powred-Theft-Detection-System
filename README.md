# GuardEye: AI Powred Theft Detection System

**A research project focusing on enhancing security in public spaces using a hybrid deep learning approach for real-time theft detection and surveillance.**

![Python](https://img.shields.io/badge/Python-3.6%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.x-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-v4.x-blue)

## Table of Contents
- [Introduction](#introduction)
- [Research Paper](#research-paper)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset Description](#dataset-description)
- [Preprocessing](#preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project explores the application of a hybrid deep learning model, combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), for enhancing security in public spaces. The model is trained to detect theft and other suspicious activities in real-time from surveillance footage, providing an advanced tool for public safety.

## Research Paper
The full research paper detailing this approach, methodology, and results can be found in the `docs/` folder:
- [Improving Security in Public Spaces: A Hybrid Deep Learning Approach](docs/YourResearchPaper.pdf)

## Features
- **Hybrid Model Architecture**: Utilizes CNN for feature extraction and RNN for sequence modeling, making it ideal for processing video data for anomaly detection.
- **Real-time Threat Detection**: Capable of processing live video feeds to detect potential security threats in real-time.
- **Scalability**: Deployable across a wide range of public spaces and adaptable to different security scenarios.
- **Alert System**: Generates automated alerts upon detecting suspicious activities.
- **Performance Tracking**: Tracks metrics like accuracy, detection rate, and latency to ensure system reliability.

## Model Architecture
The hybrid model combines:
- **Convolutional Neural Network (CNN)**: Extracts spatial features from video frames, highlighting patterns indicative of suspicious activities.
- **Recurrent Neural Network (RNN)**: Processes sequential data, allowing the model to understand temporal relationships across frames.

## Dataset Description
The model is trained on a custom dataset containing labeled instances of various activities, such as:
- **Normal Activity**
- **Theft and Robbery**
- **Suspicious Behaviors**

The dataset is split into training, validation, and test sets to ensure robust model performance. Each video is preprocessed and labeled according to activity type.

## Preprocessing
1. **Frame Extraction**: Each video is broken down into individual frames.
2. **Normalization**: Pixel values are normalized for faster processing.
3. **Resizing**: Frames are resized to a consistent shape for model input.
4. **Label Encoding**: Activities are encoded into numerical labels for classification.

## Training and Evaluation
- **Training Process**: The model is trained on labeled frames using a supervised learning approach, optimizing for accuracy and minimizing false positives.
- **Evaluation Metrics**:
  - **Accuracy**: Measures overall detection accuracy.
  - **Precision and Recall**: Evaluate the model’s ability to detect theft accurately without missing any instances.
  - **F1 Score**: Provides a balance between precision and recall.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
