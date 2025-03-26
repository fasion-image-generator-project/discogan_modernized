#!/bin/bash
# DiscoGAN 환경 설정 스크립트
# 사용법: bash setup_environment.sh

set -e  # 오류 발생 시 스크립트 중단

# 색상 설정
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 환경 이름 설정
ENV_NAME="discogan"
PYTHON_VERSION="3.10"

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  DiscoGAN 환경 설정 스크립트  ${NC}"
echo -e "${BLUE}=======================================${NC}"

# conda가 설치되어 있는지 확인
if ! command -v conda &> /dev/null; then
    echo -e "${RED}[오류] conda가 설치되어 있지 않습니다.${NC}"
    echo -e "${YELLOW}Miniconda 또는 Anaconda를 설치한 후 다시 시도하세요.${NC}"
    echo "설치 링크: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 기존 환경이 있는지 확인
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}[경고] '$ENV_NAME' 환경이 이미 존재합니다.${NC}"
    read -p "환경을 삭제하고 다시 만드시겠습니까? (y/n): " RECREATE
    if [[ "$RECREATE" == "y" ]]; then
        echo -e "${YELLOW}기존 환경을 삭제합니다...${NC}"
        conda deactivate
        conda env remove -n $ENV_NAME
    else
        echo -e "${GREEN}기존 환경을 유지합니다. 패키지 업데이트를 진행합니다.${NC}"
    fi
fi

# 새 환경 생성 (존재하지 않거나 재생성하는 경우)
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${GREEN}새로운 '$ENV_NAME' 환경을 생성합니다...${NC}"
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

# 환경 활성화
echo -e "${GREEN}'$ENV_NAME' 환경을 활성화합니다...${NC}"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 시스템 및 CUDA 확인
echo -e "${BLUE}시스템 및 CUDA 환경을 확인합니다...${NC}"
CUDA_AVAILABLE=false
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}CUDA $CUDA_VERSION이 감지되었습니다.${NC}"
    CUDA_AVAILABLE=true
else
    echo -e "${YELLOW}CUDA가 감지되지 않았습니다. CPU 버전 PyTorch를 설치합니다.${NC}"
fi

# PyTorch 설치
echo -e "${BLUE}PyTorch 라이브러리를 설치합니다...${NC}"
if [ "$CUDA_AVAILABLE" = true ]; then
    # CUDA 버전에 따른 설치 방법 선택
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [ $CUDA_MAJOR -ge 12 ]; then
        echo -e "${GREEN}CUDA 12.x 감지: PyTorch 2.1.0과 CUDA 12.1 지원을 설치합니다.${NC}"
        conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 -c pytorch -y
        conda install pytorch-cuda=12.1 -c pytorch -c nvidia -y
    elif [ $CUDA_MAJOR -ge 11 ] && [ $CUDA_MINOR -ge 7 ]; then
        echo -e "${GREEN}CUDA 11.7+ 감지: PyTorch 2.1.0과 CUDA 11.8 지원을 설치합니다.${NC}"
        conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo -e "${YELLOW}CUDA $CUDA_VERSION 감지: 최신 호환 PyTorch를 설치합니다.${NC}"
        conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
    fi
else
    echo -e "${YELLOW}CPU 버전 PyTorch를 설치합니다.${NC}"
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# 기타 필수 라이브러리 설치
echo -e "${BLUE}필수 라이브러리를 설치합니다...${NC}"
conda install -y numpy pandas matplotlib pillow scipy tqdm ipython -c conda-forge
pip install opencv-python pyyaml

# 설치 확인
echo -e "${BLUE}설치 환경을 확인합니다...${NC}"

# Python 버전 확인
PYTHON_INSTALLED=$(python --version)
echo -e "${GREEN}$PYTHON_INSTALLED 설치됨${NC}"

# PyTorch 버전 확인
echo -e "${GREEN}PyTorch $(python -c "import torch; print(torch.__version__)") 설치됨${NC}"

# CUDA 사용 가능 여부 확인
if [ "$CUDA_AVAILABLE" = true ]; then
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo -e "${GREEN}PyTorch CUDA 사용 가능: $(python -c "import torch; print(torch.version.cuda)")${NC}"
        echo -e "${GREEN}사용 가능한 GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')")${NC}"
    else
        echo -e "${YELLOW}경고: CUDA가 시스템에 설치되어 있지만 PyTorch에서 사용할 수 없습니다.${NC}"
        echo -e "${YELLOW}드라이버 또는 PyTorch CUDA 버전 호환성 문제일 수 있습니다.${NC}"
    fi
else
    echo -e "${YELLOW}PyTorch CPU 버전이 설치되었습니다.${NC}"
fi

# 추가 안내 메시지
echo -e "${BLUE}=======================================${NC}"
echo -e "${GREEN}DiscoGAN 환경 설정이 완료되었습니다!${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "${YELLOW}환경 활성화 방법:${NC} conda activate $ENV_NAME"
echo -e "${YELLOW}훈련 실행 예시:${NC} python image_translation.py --task_name=edges2shoes --model_arch=discogan --batch_size=64"
echo -e "${BLUE}=======================================${NC}"

# 윈도우용 배치 파일 생성
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo -e "${BLUE}윈도우 환경을 위한 배치 파일을 생성합니다...${NC}"
    cat > setup_environment.bat << EOF
@echo off
echo DiscoGAN 환경 설정 스크립트
echo =======================================

rem conda가 설치되어 있는지 확인
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [오류] conda가 설치되어 있지 않습니다.
    echo Miniconda 또는 Anaconda를 설치한 후 다시 시도하세요.
    echo 설치 링크: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

rem 환경 생성
call conda create -n $ENV_NAME python=$PYTHON_VERSION -y
if %ERRORLEVEL% neq 0 (
    echo [오류] 환경 생성에 실패했습니다.
    exit /b 1
)

rem 환경 활성화
call conda activate $ENV_NAME
if %ERRORLEVEL% neq 0 (
    echo [오류] 환경 활성화에 실패했습니다.
    exit /b 1
)

rem CUDA 확인 및 PyTorch 설치
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

rem 기타 라이브러리 설치
call conda install -y numpy pandas matplotlib pillow scipy tqdm ipython -c conda-forge
call pip install opencv-python pyyaml

echo =======================================
echo DiscoGAN 환경 설정이 완료되었습니다!
echo =======================================
echo 환경 활성화 방법: conda activate $ENV_NAME
echo 훈련 실행 예시: python image_translation.py --task_name=edges2shoes --model_arch=discogan --batch_size=64
echo =======================================

pause
EOF
    echo -e "${GREEN}setup_environment.bat 파일이 생성되었습니다.${NC}"
fi

echo -e "${GREEN}설정 완료!${NC}"
