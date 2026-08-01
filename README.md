#  Wildlife Image Classification (DeepNet - Custom CNN)

An end-to-end deep learning pipeline for multi-class camera trap wildlife image classification, originally built for the **DrivenData ConserVision** competition. 

##  Project Overview

This repository contains the foundational approach to classifying wildlife images into 8 distinct categories. Rather than relying on pre-trained weights or object detection pipelines, this iteration of the project implements a custom Convolutional Neural Network (CNN) built from scratch to extract spatial features directly from raw camera trap images.

---

## Implemented Pipeline & Methods

### 1. Model Architecture (Custom CNN)
* **Backbone**: A custom-designed Convolutional Neural Network built entirely in PyTorch.
* **Structure**: Features a series of stacked convolutional layers and max-pooling operations to learn hierarchical spatial features from the dataset, followed by fully connected layers mapping to the 8 target species categories.

### 2. Data Management & Path Routing
* **Strict Path Execution**: All file traversals, directory structures, and image path listings are strictly handled using Python's `pathlib` module to guarantee robust, cross-platform compatibility and clean path construction.
* **Preprocessing**: Standard image resizing and RGB conversion are applied before loading images into PyTorch tensors.

### 3. Inference Workflow
* **Core Evaluation Function**: The final testing loop utilizes the exact `file_to_confidence(test_image_path, image_id)` function.
* **Pipeline**: This function ingests the raw image path, applies the necessary transformations, passes it through the custom CNN, and returns a formatted pandas DataFrame containing the softmax probability distribution for the competition submission.
---
