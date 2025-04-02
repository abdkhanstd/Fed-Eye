# Fed-Eye: Privacy-Preserving Federated Learning for Ophthalmological Disease Classification

Welcome to the official repository for **Fed-Eye**, a federated learning framework designed for ophthalmological disease classification across multiple hospitals while preserving patient privacy. This project leverages a Vision Transformer (ViT) model and federated learning techniques to address data heterogeneity and domain shift challenges in medical imaging. For more details, refer to our paper: [Fed-Eye: Privacy-Preserving Federated Learning for Ophthalmological Disease Classification](#).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [Download Links](#download-links)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Testing the Model](#testing-the-model)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
Fed-Eye enables collaborative training of a deep learning model across multiple institutions without sharing raw patient data. It uses a pre-trained ViT-Large model fine-tuned on diverse ophthalmological datasets, including diabetic retinopathy, glaucoma, and mixed pathologies. The framework ensures privacy by exchanging only model parameters, adhering to regulations like HIPAA and GDPR.

## Requirements
To run this project, ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch 1.12+ with CUDA support (if using GPU)
- torchvision
- timm
- numpy
- scikit-learn
- tqdm
- PIL (Pillow)
- argparse
- logging

Install the dependencies using:
```bash
pip install -r requirements.txt
```

Dataset Setup
The code supports eight ophthalmological datasets, which must be organized in a specific structure under the datasets/ directory. Each dataset should have train/, val/, and test/ subdirectories containing class-labeled images.

Supported Datasets
JSIEC: Multi-disease fundus photographs
IDRiD: Diabetic retinopathy with severity grades
APTOS2019: Diabetic retinopathy dataset
MESSIDOR2: Diabetic retinopathy fundus images
PAPILA: Glaucoma dataset
Glaucoma_fundus: Glaucoma detection dataset
OCTID: OCT images for multiple pathologies
Retina: Mixed pathology dataset
Directory Structure
Organize your datasets as follows:


## Directory Structure
```
datasets/
├── JSIEC/
│   ├── train/
│   │   ├── 0.0.Normal/
│   │   ├── 0.3.DR1/
│   │   └── ...
│   ├── val/
│   └── test/
├── IDRiD/
│   ├── train/
│   │   ├── anoDR/
│   │   ├── bmildDR/
│   │   └── ...
│   ├── val/
│   └── test/
├── APTOS2019/
│   ├── train/
│   ├── val/
│   └── test/
└── ...
```


## Download Links

### Datasets
To download the datasets used in this project, refer to the benchmark documentation:  
[Download Datasets](https://github.com/rmaphoh/RETFound_MAE/blob/main/BENCHMARK.md)

### Pretrained Weights
To download the pretrained weights (`best_model.pth`) for initializing the model:  
[Download Pretrained Weights](https://github.com/abdkhanstd/ATLASS)  
*Note*: Place the downloaded `best_model.pth` in the root directory of the project.

### Best Federated Model
To test the results using the best federated model trained in this project:  
[Download Best Federated Model](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/EtMOmkgrT99LtGqAjXRnl84BNKo_VSUQJUYGhRH9_Rgncg?e=NP8Mvo)  
*Note*: Download `best_federated_model_final.pth` and place it in the `conf_results/` directory. This link assumes the model is uploaded to the releases section of this repository.
