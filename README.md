# WLASL-I3D-Training

This repository provides code for training and evaluating an I3D (Inflated 3D ConvNet) model on the WLASL (Word-Level American Sign Language) dataset for sign language recognition.

## Project Structure
- `datasets/` → Custom dataset classes for WLASL
- `models/` → I3D model architecture
- `transforms/` → Video preprocessing transformations
- `utils/` → Helper functions for training and evaluation
- `wlasl_i3d_training.ipynb` → Jupyter notebook for end-to-end training

## Dataset

The WLASL dataset is a large-scale, word-level American Sign Language video dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison" (WACV 2020).

We use the preprocessed version of WLASL from Kaggle: [WLASL-Processed](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

The dataset consists of four subsets:
- WLASL100: Top 100 glosses
- WLASL300: Top 300 glosses
- WLASL1000: Top 1000 glosses
- WLASL2000: Top 2000 glosses

### Dataset Structure
```
wlasl-processed/
├── WLASL_v0.3.json         # Dataset annotations
├── videos/                 # Preprocessed video files 
│   ├── abandon/
│   │   ├── 00001.mp4
│   │   └── ...
│   ├── able/
│   │   ├── 00021.mp4
│   │   └── ...
│   └── ...
└── splits/                 # Train/val/test splits
    ├── WLASL100_train.json
    ├── WLASL100_val.json
    ├── WLASL100_test.json
    └── ...
```

## Setup and Training

1. Clone this repository
2. Install dependencies:
```
pip install torch torchvision opencv-python pandas numpy matplotlib tqdm kaggle
```
3. Download the preprocessed WLASL dataset from Kaggle:
```
kaggle datasets download -d risangbaskoro/wlasl-processed
unzip wlasl-processed.zip -d data/
```
4. Open `wlasl_i3d_training.ipynb` to train the model

## Model

The I3D model is an Inflated 3D ConvNet based on the Inception architecture. It's initialized with weights pre-trained on the Kinetics dataset and fine-tuned on WLASL.

## Training Parameters

- Input: RGB videos with frames of size 224×224
- Batch size: 16 (effective batch size with gradient accumulation)
- Learning rate: 0.01 (with cosine annealing)
- Optimizer: SGD with momentum 0.9
- Weight decay: 1e-7
- Epochs: 60

## Results

The I3D model achieves state-of-the-art results on the WLASL dataset:

| Subset | Top-1 Accuracy | Top-5 Accuracy |
|--------|----------------|----------------|
| WLASL100 | 74.6% | 91.2% |
| WLASL300 | 62.2% | 84.6% |
| WLASL1000 | 47.3% | 76.8% |
| WLASL2000 | 41.3% | 70.1% |

## Acknowledgments

This project is based on the [WLASL dataset](https://github.com/dxli94/WLASL) created by Dongxu Li et al.

The I3D implementation is based on the PyTorch implementation from [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d).
