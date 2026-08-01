# ConserVision: Multi-Class Camera Trap Wildlife Image Classification

An end-to-end deep learning pipeline for multi-class camera trap wildlife image classification built for the DrivenData ConserVision competition. This repository contains the data preparation, normalization, model architecture, class imbalance handling, and evaluation scripts built using PyTorch and Torchvision.

## Project Overview

Camera trap image datasets often suffer from low resolution, occlusion, severe class imbalance, and a high frequency of "blank" (no animal) frames. This project implements a transfer learning approach leveraging a ResNet50 backbone modified with a custom classification head to classify wildlife images into 8 distinct categories.

## Target Classes
The model classifies images across 8 species/categories:
- antelope_duiker
- bird
- blank (no animal present)
- civet_genet
- hog
- leopard
- monkey_prosimian
- rodent

## Implemented Pipeline & Methods

1. Data Pipeline & Directory Structuring 
   - Extraction & Mapping: Unzips raw feature archives and parses train_labels.csv to map unique image IDs to their target class labels.
   - Directory Sorting: Dynamically creates subdirectories for each of the 8 classes and moves feature images into an organized_features/ folder structure compatible with standard dataset loaders.

2. Preprocessing & Custom Normalization
   - Channel Consistency: Applied a custom ConvertToRGB transform to handle non-RGB image modes across the dataset.
   - Resizing: Standardized all input images to 224 x 224 pixels.
   - Dynamic Normalization: Implemented a channel-wise mean and standard deviation extraction function (get_mean_std) over the training dataset to center features around zero before training.

3. Model Architecture (ResNet50 Transfer Learning)
   - Backbone: Pre-trained ResNet50 from torchvision.models.
   - Feature Extraction: Initially froze the convolutional backbone parameters (requires_grad = False).
   - Custom Classifier Head: Replaced the original 1000-node output layer (fc) with a custom Sequential classification head:

```text
ResNet50 (Frozen Convolutional Layers)
└── AdaptiveAvgPool2d (2048 features)
    └── Sequential Custom Head:
        ├── Linear(in_features=2048, out_features=500)
        ├── ReLU()
        ├── Dropout(p=0.5)
        └── Linear(in_features=500, out_features=8)
