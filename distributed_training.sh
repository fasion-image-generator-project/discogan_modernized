#!/bin/bash
# 여러 GPU에서 분산 학습을 실행하는 스크립트

# 사용할 GPU 목록 (사용 가능한 모든 GPU를 지정)
GPUS="0,1,4,5,6"  # 시스템에 맞게 사용 가능한 GPU 지정
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
WORLD_SIZE=${#GPU_ARRAY[@]}  # GPU 개수

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 기본 인자
BATCH_SIZE=64  # GPU당 배치 크기
EPOCHS=50
LEARNING_RATE=0.0002
IMAGE_SIZE=64
TASK_NAME="edges2shoes"  # 학습할 작업 (edges2shoes, celebA, facescrub 등)
MODEL_ARCH="discogan"    # 모델 아키텍처 (discogan, recongan, gan)

# CelebA 작업에 필요한 추가 인자
STYLE_A=""
STYLE_B=""
STYLE_ARGS=""

# 명령줄 인자 파싱
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --task_name=*)
        TASK_NAME="${key#*=}"
        shift
        ;;
        --model_arch=*)
        MODEL_ARCH="${key#*=}"
        shift
        ;;
        --batch_size=*)
        BATCH_SIZE="${key#*=}"
        shift
        ;;
        --epochs=*)
        EPOCHS="${key#*=}"
        shift
        ;;
        --learning_rate=*)
        LEARNING_RATE="${key#*=}"
        shift
        ;;
        --image_size=*)
        IMAGE_SIZE="${key#*=}"
        shift
        ;;
        --style_A=*)
        STYLE_A="${key#*=}"
        STYLE_ARGS="$STYLE_ARGS --style_A=${STYLE_A}"
        shift
        ;;
        --style_B=*)
        STYLE_B="${key#*=}"
        STYLE_ARGS="$STYLE_ARGS --style_B=${STYLE_B}"
        shift
        ;;
        --gpus=*)
        GPUS="${key#*=}"
        # GPU 개수 계산
        IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
        WORLD_SIZE=${#GPU_ARRAY[@]}
        shift
        ;;
        *)
        echo "알 수 없는 인자: $key"
        exit 1
        ;;
    esac
done

# 결과 및 모델 디렉토리
RESULTS_DIR="./results/distributed_${TASK_NAME}_${MODEL_ARCH}_${TIMESTAMP}"
MODELS_DIR="./models/distributed_${TASK_NAME}_${MODEL_ARCH}_${TIMESTAMP}"
LOG_DIR="./logs/distributed_${TIMESTAMP}"

mkdir -p $LOG_DIR

echo "분산 학습 시작:"
echo "  작업: $TASK_NAME"
echo "  모델: $MODEL_ARCH"
echo "  GPU: $GPUS (총 $WORLD_SIZE 개)"
echo "  배치 크기: $BATCH_SIZE (GPU당)"
echo "  에포크: $EPOCHS"
echo "  학습률: $LEARNING_RATE"
echo "  이미지 크기: $IMAGE_SIZE"
echo "  결과 디렉토리: $RESULTS_DIR"
echo "  모델 디렉토리: $MODELS_DIR"
echo "  로그 디렉토리: $LOG_DIR"

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=$GPUS

# 분산 학습 실행
python -m torch.distributed.launch \
    --nproc_per_node=$WORLD_SIZE \
    distributed_image_translation.py \
    --distributed \
    --world_size=$WORLD_SIZE \
    --task_name=$TASK_NAME \
    --model_arch=$MODEL_ARCH \
    --batch_size=$BATCH_SIZE \
    --epochs=$EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --image_size=$IMAGE_SIZE \
    --results_dir=$RESULTS_DIR \
    --models_dir=$MODELS_DIR \
    $STYLE_ARGS \
    > $LOG_DIR/train.log 2>&1

echo "분산 학습이 완료되었습니다."
echo "로그: $LOG_DIR/train.log"
