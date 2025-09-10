# Human Activity Recognition using Deep Learning

This project benchmarks four different deep learning models for the task of Human Activity Recognition (HAR) using the raw inertial signals from the UCI HAR Dataset. The implementation is done using PyTorch.

## Overview

The goal of this project is to implement, train, and evaluate the following models on the UCI HAR dataset:

* Multi-Layer Perceptron (MLP)
* Convolutional Neural Network (CNN)
* Long Short-Term Memory Network (LSTM)
* A hybrid CNN-LSTM Network

The performance of each model is measured by its Accuracy, Precision, Recall, and F1-score, providing a comprehensive comparison of their effectiveness for this task.

## Results and Analysis

The final performance metrics for each model are summarized in the table below. The reported Precision, Recall, and F1-Score are macro-averaged across all six activity classes.

### Comparative Performance Metrics

| Model      | Accuracy | Precision | Recall   | F1-Score |
| :--------- | :------- | :-------- | :------- | :------- |
| **MLP** | 0.8823   | 0.8837    | 0.8844   | 0.8821   |
| **CNN** | 0.8968   | 0.8983    | 0.8999   | 0.8973   |
| **LSTM** | 0.8860   | 0.8871    | 0.8887   | 0.8859   |
| **CNN-LSTM** | 0.8996   | 0.8993    | 0.9024   | 0.9000   |

### Analysis

The evaluation demonstrates that the CNN-based models outperformed the others, with the hybrid **CNN-LSTM model achieving the highest performance** across all metrics.

This superior performance is likely attributable to its architecture. The initial Convolutional Neural Network (CNN) layers are adept at extracting salient spatial features from the raw inertial signals within each time window. Subsequently, the Long Short-Term Memory (LSTM) layers model the temporal dependencies between these extracted features, effectively capturing the dynamics of the different human activities.

The standard CNN also performed exceptionally well, indicating the strong predictive power of the spatial features alone. The MLP and standalone LSTM models, were also effective, just slightly lower performance.
