# DiscoGAN: Discovering Cross-Domain Relations with Generative Adversarial Networks

[English](#english) | [한국어](#korean)

---

<a name="english"></a>
## English

### Overview

This repository contains a PyTorch implementation of DiscoGAN (Discovering Cross-Domain Relations with Generative Adversarial Networks). DiscoGAN is designed to learn cross-domain relations with unpaired data, allowing it to translate images from one domain to another without explicit pairing information.

### Features

- Multiple translation tasks support:
  - Edge-to-photo translation (shoes, handbags)
  - Face attribute translation (gender, expression)
  - Angle pairing (cars, chairs, faces)
  - Custom domain translations (e.g., tops to hanbok)
- Distributed training support for multi-GPU environments
- Hyperparameter tuning utilities
- Modern PyTorch implementation with improved usability

### Installation Requirements

This project has been tested with:
- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1

#### Creating a conda environment (recommended)

```bash
# Create a new conda environment
conda create -n discogan python=3.10
conda activate discogan

# Install PyTorch (with CUDA 12.1 support)
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 -c pytorch
conda install pytorch-cuda=12.1 -c nvidia

# Install other required packages
pip install -r requirements.txt
```

#### Alternative installation with pip

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

### Dataset Preparation

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

### Usage Examples

> **Important Note**: When using the `tops2hanbok` task, you must set `--image_size=512` to avoid errors.

#### Image-to-Image Translation

```bash
# Edges to shoes translation
python image_translation.py --task_name=edges2shoes --model_arch=discogan --batch_size=64 --epochs=50

# CelebA attribute translation
python image_translation.py --task_name=celebA --style_A=Male --style_B=Smiling --model_arch=discogan --batch_size=64
```

#### Angle Pairing Tasks

```bash
# Car angle pairing
python angle_pairing.py --task_name=car2car --model_arch=discogan --batch_size=64 --epochs=30

# Chair angle pairing
python angle_pairing.py --task_name=chair2chair --model_arch=discogan --batch_size=64 --epochs=30
```

#### Distributed Training

```bash
# Using distributed_training.sh script
bash distributed_training.sh --task_name=celebA --style_A=Male --style_B=Smiling --batch_size=64 --epochs=50

# Single GPU distributed training with specific device
export CUDA_VISIBLE_DEVICES=0
python distributed_image_translation.py --task_name=tops2hanbok --model_arch=discogan --batch_size=32 --image_size=512 --epochs=500 --learning_rate=0.0001  # image_size 반드시 512로 설정  # image_size must be 512 for tops2hanbok
```

#### Inference with Trained Models

```bash
# Inference on single image
python inference.py --model_path=./models/edges2shoes/discogan/20240325_120000/ --input_path=./test_images/test_shoe.jpg --direction=AtoB

# Inference on a directory of images with high-resolution model
python inference.py \
  --model_path ./models/tops2hanbok/discogan/20250325_224905_rank0 \
  --input_path ./datasets/custom/tops/test \
  --image_size 512  # tops2hanbok 모델에 필수  # Required for tops2hanbok models
```

### Important Parameters

- `--task_name`: Dataset/task to use (e.g., 'edges2shoes', 'car2car', 'celebA', 'tops2hanbok')
- `--model_arch`: Model architecture to use ('discogan', 'recongan', 'gan')
- `--device`: Device to use ('cuda', 'cpu')
- `--batch_size`: Batch size
- `--epochs`: Number of epochs to train
- `--learning_rate`: Learning rate (default: 0.0002)
- `--image_size`: Image size (default: 64, **must use 512 for tops2hanbok task to avoid errors**)

CelebA-specific parameters:
- `--style_A`: First style attribute for CelebA (e.g., 'Male')
- `--style_B`: Second style attribute for CelebA (e.g., 'Smiling')

### Monitoring and Utility Scripts

- `gpu_monitor.py`: Monitors GPU usage and running tasks
- `batch_size_optimization.py`: Helps find optimal batch size based on GPU memory
- `hyperparameter_search.py`: Runs hyperparameter search across multiple GPUs

### Reference

This implementation is based on the DiscoGAN paper:
- [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)

---

<a name="korean"></a>
## 한국어

### 개요

이 저장소는 DiscoGAN(Discovering Cross-Domain Relations with Generative Adversarial Networks)의 PyTorch 구현을 포함하고 있습니다. DiscoGAN은 쌍을 이루지 않은 데이터로 도메인 간 관계를 학습하도록 설계되어 명시적인 페어링 정보 없이도 한 도메인에서 다른 도메인으로 이미지를 변환할 수 있습니다.

### 특징

- 다양한 변환 작업 지원:
  - 윤곽선-사진 변환 (신발, 핸드백)
  - 얼굴 속성 변환 (성별, 표정)
  - 각도 페어링 (자동차, 의자, 얼굴)
  - 사용자 정의 도메인 변환 (예: 상의에서 한복으로)
- 다중 GPU 환경을 위한 분산 학습 지원
- 하이퍼파라미터 튜닝 유틸리티
- 사용성이 향상된 현대적인 PyTorch 구현

### 설치 요구사항

이 프로젝트는 다음 환경에서 테스트되었습니다:
- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1

#### conda 환경 생성 (권장)

```bash
# 새 conda 환경 생성
conda create -n discogan python=3.10
conda activate discogan

# PyTorch 설치 (CUDA 12.1 지원)
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 -c pytorch
conda install pytorch-cuda=12.1 -c nvidia

# 기타 필요한 패키지 설치
pip install -r requirements.txt
```

#### pip로 대체 설치

```bash
# 가상 환경 생성
python -m venv discogan_env
source discogan_env/bin/activate  # Linux/Mac
# 또는
discogan_env\Scripts\activate  # Windows

# PyTorch 설치 (CUDA 12.1 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 기타 필요한 패키지 설치
pip install -r requirements.txt
```

### 데이터셋 준비

프로젝트는 `datasets` 디렉토리가 다음과 같은 구조로 구성되어 있어야 합니다:

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

### 사용 예시

> **중요 참고사항**: `tops2hanbok` 작업을 사용할 때는 오류를 방지하기 위해 반드시 `--image_size=512`로 설정해야 합니다.

#### 이미지-이미지 변환

```bash
# 윤곽선에서 신발 이미지 변환
python image_translation.py --task_name=edges2shoes --model_arch=discogan --batch_size=64 --epochs=50

# CelebA 속성 변환
python image_translation.py --task_name=celebA --style_A=Male --style_B=Smiling --model_arch=discogan --batch_size=64
```

#### 각도 페어링 작업

```bash
# 자동차 각도 페어링
python angle_pairing.py --task_name=car2car --model_arch=discogan --batch_size=64 --epochs=30

# 의자 각도 페어링
python angle_pairing.py --task_name=chair2chair --model_arch=discogan --batch_size=64 --epochs=30
```

#### 분산 학습

```bash
# distributed_training.sh 스크립트 사용
bash distributed_training.sh --task_name=celebA --style_A=Male --style_B=Smiling --batch_size=64 --epochs=50

# 단일 GPU 분산 학습 (특정 장치 지정)
export CUDA_VISIBLE_DEVICES=0
python distributed_image_translation.py --task_name=tops2hanbok --model_arch=discogan --batch_size=32 --image_size=512 --epochs=500 --learning_rate=0.0001
```

#### 학습된 모델로 추론

```bash
# 단일 이미지에 대한 추론
python inference.py --model_path=./models/edges2shoes/discogan/20240325_120000/ --input_path=./test_images/test_shoe.jpg --direction=AtoB

# 고해상도 모델로 디렉토리의 모든 이미지에 대한 추론
python inference.py \
  --model_path ./models/tops2hanbok/discogan/20250325_224905_rank0 \
  --input_path ./datasets/custom/tops/test \
  --image_size 512
```

### 중요 매개변수

- `--task_name`: 사용할 데이터셋/작업 (예: 'edges2shoes', 'car2car', 'celebA', 'tops2hanbok')
- `--model_arch`: 사용할 모델 구조 ('discogan', 'recongan', 'gan')
- `--device`: 사용할 디바이스 ('cuda', 'cpu')
- `--batch_size`: 배치 크기
- `--epochs`: 학습할 에포크 수
- `--learning_rate`: 학습률 (기본값: 0.0002)
- `--image_size`: 이미지 크기 (기본값: 64, **tops2hanbok 작업은 오류 방지를 위해 반드시 512 사용**)

CelebA 특정 매개변수:
- `--style_A`: CelebA 첫 번째 스타일 속성 (예: 'Male')
- `--style_B`: CelebA 두 번째 스타일 속성 (예: 'Smiling')

### 모니터링 및 유틸리티 스크립트

- `gpu_monitor.py`: GPU 사용량 및 실행 중인 작업 모니터링
- `batch_size_optimization.py`: GPU 메모리에 기반한 최적의 배치 크기 찾기
- `hyperparameter_search.py`: 여러 GPU에서 하이퍼파라미터 검색 실행

### 참고 문헌

이 구현은 DiscoGAN 논문을 기반으로 합니다:
- [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)
