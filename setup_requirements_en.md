# Installation Guide

## 1. Environment Setup

This guide explains how to set up the environment required for the project.

### Python and CUDA Setup

This project has been tested in the following environment:
- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1

### Creating a conda environment (recommended)

```bash
# Create a new conda environment
conda create -n discogan python=3.10
conda activate discogan

# Install PyTorch (with CUDA 12.1 support)
# Install in separate steps to avoid conflicts
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 -c pytorch
conda install pytorch-cuda=12.1 -c nvidia

# Install other required packages
pip install -r requirements.txt
```

### Alternative installation with pip

```bash
# Create a virtual environment
python -m venv discogan_env
source discogan_env/bin/activate  # Linux/Mac
# or
discogan_env\Scripts\activate  # Windows

# Install PyTorch (with CUDA 12.1 support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other required packages
pip install -r requirements.txt
```

## 2. Dataset Preparation

The project expects the `datasets` directory to be structured as follows:

```
datasets/
  ├── celebA/
  ├── edges2handbags/
  ├── edges2shoes/
  ├── facescrub/
  ├── rendered_chairs/
  ├── PublicMM1/05_renderings/
  └── data/cars/
```

Each dataset can be downloaded from the following websites:
- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- Edges2Handbags: https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset
- Edges2Shoes: https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset
- Facescrub: http://vintage.winklerbros.net/facescrub.html
- Rendered Chairs: https://www.di.ens.fr/willow/research/seeing3Dchairs
- Cars: https://www.cs.cmu.edu/~akar/data.htm

## 3. Running the Training

### Basic Image Translation

```bash
python image_translation.py --task_name=edges2shoes --model_arch=discogan --device=cuda --batch_size=64 --epochs=20
```

### Angle Pairing

```bash
python angle_pairing.py --task_name=car2car --model_arch=discogan --device=cuda --batch_size=64 --epochs=10
```

### Important Parameters

- `--task_name`: Dataset/task to use (e.g., 'edges2shoes', 'car2car', 'celebA')
- `--model_arch`: Model architecture to use ('discogan', 'recongan', 'gan')
- `--device`: Device to use ('cuda', 'cpu')
- `--batch_size`: Batch size
- `--epochs`: Number of epochs to train
- `--learning_rate`: Learning rate
- `--image_size`: Image size (default: 64)

CelebA-specific parameters:
- `--style_A`: First style attribute for CelebA (e.g., 'Male')
- `--style_B`: Second style attribute for CelebA (e.g., 'Smiling')

More options can be found in the `parse_args()` function of each script.


# Requirements.txt

# Basic packages
numpy>=1.23.5
pandas>=1.5.3
matplotlib>=3.7.1
Pillow>=9.4.0
scipy>=1.10.1
tqdm>=4.65.0
ipython>=8.10.0

# PyTorch (install via pip)
# torch>=2.1.0
# torchvision>=0.16.0
# torchaudio>=2.1.0

# Image processing
opencv-python>=4.7.0.72

# Utilities
pyyaml>=6.0