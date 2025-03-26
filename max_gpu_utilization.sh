#!/bin/bash
# 모든 가용 GPU를 최대한 활용하는 스크립트

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/$TIMESTAMP"
mkdir -p $LOG_DIR

# 사용 가능한 GPU 확인
echo "사용 가능한 GPU 확인 중..."
AVAILABLE_GPUS=(0 1 4 5 6)  # 기본 GPU 목록 (nvidia-smi 출력에서 2, 3번은 사용 중)
USED_GPUS=()
MAX_MEMORY_USAGE=500  # 500MB 이하 사용 중인 GPU는 '사용 가능'으로 간주

# 각 GPU의 메모리 사용량 확인
for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
    MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)
    if [ "$MEMORY_USED" -gt "$MAX_MEMORY_USAGE" ]; then
        echo "GPU $GPU_ID: 이미 사용 중 (${MEMORY_USED}MB)"
        USED_GPUS+=($GPU_ID)
    else
        echo "GPU $GPU_ID: 사용 가능 (${MEMORY_USED}MB)"
    fi
done

# 사용 가능한 GPU 필터링
for GPU_ID in "${USED_GPUS[@]}"; do
    AVAILABLE_GPUS=(${AVAILABLE_GPUS[@]/$GPU_ID/})
done

# 사용 가능한 GPU가 있는지 확인
if [ ${#AVAILABLE_GPUS[@]} -eq 0 ]; then
    echo "사용 가능한 GPU가 없습니다."
    exit 1
fi

echo "사용 가능한 GPU: ${AVAILABLE_GPUS[@]}"

# 작업 설정
echo "각 GPU에서 실행할 작업 설정:"
echo "1) 모든 GPU에서 동일한 작업 실행"
echo "2) 다양한 작업 분산 실행"
echo "3) 하이퍼파라미터 탐색 실행"
echo "4) 분산 학습 실행 (여러 GPU 하나의 작업)"
read -p "선택: " TASK_MODE

case $TASK_MODE in
    1)  # 모든 GPU에서 동일한 작업 실행
        read -p "실행할 작업 (edges2shoes/celebA/facescrub/car2car): " TASK_NAME
        read -p "모델 아키텍처 (discogan/recongan/gan): " MODEL_ARCH
        read -p "배치 크기: " BATCH_SIZE
        read -p "에포크 수: " EPOCHS
        
        ADDITIONAL_ARGS=""
        if [ "$TASK_NAME" == "celebA" ]; then
            read -p "스타일 A (예: Male): " STYLE_A
            read -p "스타일 B (예: Smiling): " STYLE_B
            if [ -n "$STYLE_A" ]; then
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_A=$STYLE_A"
            fi
            if [ -n "$STYLE_B" ]; then
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_B=$STYLE_B"
            fi
        fi
        
        # 스크립트 선택
        if [[ "$TASK_NAME" == "car2car" || "$TASK_NAME" == "chair2chair" || "$TASK_NAME" == "face2face" ]]; then
            SCRIPT="angle_pairing.py"
        else
            SCRIPT="image_translation.py"
        fi
        
        # 각 GPU에서 실행
        for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
            echo "GPU $GPU_ID에서 $TASK_NAME 실행 중..."
            RESULTS_DIR="./results/${TASK_NAME}_gpu${GPU_ID}_${TIMESTAMP}"
            MODELS_DIR="./models/${TASK_NAME}_gpu${GPU_ID}_${TIMESTAMP}"
            
            CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
                --task_name=$TASK_NAME \
                --model_arch=$MODEL_ARCH \
                --batch_size=$BATCH_SIZE \
                --epochs=$EPOCHS \
                --results_dir=$RESULTS_DIR \
                --models_dir=$MODELS_DIR \
                --device=cuda \
                $ADDITIONAL_ARGS \
                > $LOG_DIR/gpu${GPU_ID}_${TASK_NAME}.log 2>&1 &
            
            echo "작업 시작됨: GPU $GPU_ID, PID $!, 로그: $LOG_DIR/gpu${GPU_ID}_${TASK_NAME}.log"
        done
        ;;
        
    2)  # 다양한 작업 분산 실행
        # 각 GPU에 다른 작업 할당
        declare -A TASKS
        
        echo "각 GPU에 실행할 작업 설정:"
        for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
            echo "GPU $GPU_ID 설정:"
            read -p "  작업 (edges2shoes/celebA/facescrub/car2car): " TASK_NAME
            read -p "  모델 아키텍처 (discogan/recongan/gan): " MODEL_ARCH
            read -p "  배치 크기: " BATCH_SIZE
            read -p "  에포크 수: " EPOCHS
            
            ADDITIONAL_ARGS=""
            if [ "$TASK_NAME" == "celebA" ]; then
                read -p "  스타일 A (예: Male): " STYLE_A
                read -p "  스타일 B (예: Smiling): " STYLE_B
                if [ -n "$STYLE_A" ]; then
                    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_A=$STYLE_A"
                fi
                if [ -n "$STYLE_B" ]; then
                    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_B=$STYLE_B"
                fi
            fi
            
            TASKS[$GPU_ID]="--task_name=$TASK_NAME --model_arch=$MODEL_ARCH --batch_size=$BATCH_SIZE --epochs=$EPOCHS $ADDITIONAL_ARGS"
        done
        
        # 각 GPU에서 작업 실행
        for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
            TASK_ARGS=${TASKS[$GPU_ID]}
            TASK_NAME=$(echo $TASK_ARGS | grep -o "task_name=[^ ]*" | cut -d'=' -f2)
            
            # 스크립트 선택
            if [[ "$TASK_NAME" == "car2car" || "$TASK_NAME" == "chair2chair" || "$TASK_NAME" == "face2face" ]]; then
                SCRIPT="angle_pairing.py"
            else
                SCRIPT="image_translation.py"
            fi
            
            echo "GPU $GPU_ID에서 $TASK_NAME 실행 중..."
            RESULTS_DIR="./results/${TASK_NAME}_gpu${GPU_ID}_${TIMESTAMP}"
            MODELS_DIR="./models/${TASK_NAME}_gpu${GPU_ID}_${TIMESTAMP}"
            
            CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
                $TASK_ARGS \
                --results_dir=$RESULTS_DIR \
                --models_dir=$MODELS_DIR \
                --device=cuda \
                > $LOG_DIR/gpu${GPU_ID}_${TASK_NAME}.log 2>&1 &
            
            echo "작업 시작됨: GPU $GPU_ID, PID $!, 로그: $LOG_DIR/gpu${GPU_ID}_${TASK_NAME}.log"
            
            # 각 작업 시작 사이에 짧은 대기 시간 추가
            sleep 2
        done
        ;;
        
    3)  # 하이퍼파라미터 탐색 실행
        read -p "작업 (edges2shoes/celebA/facescrub/car2car): " TASK_NAME
        read -p "모델 아키텍처 (discogan/recongan/gan): " MODEL_ARCH
        read -p "실험 횟수: " TRIALS
        read -p "기본 에포크 수: " EPOCHS
        read -p "배치 크기: " BATCH_SIZE
        
        ADDITIONAL_ARGS=""
        if [ "$TASK_NAME" == "celebA" ]; then
            read -p "스타일 A (예: Male): " STYLE_A
            read -p "스타일 B (예: Smiling): " STYLE_B
            if [ -n "$STYLE_A" ]; then
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_A=$STYLE_A"
            fi
            if [ -n "$STYLE_B" ]; then
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_B=$STYLE_B"
            fi
        fi
        
        # GPU 목록을 쉼표로 구분된 문자열로 변환
        GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
        
        # 하이퍼파라미터 탐색 스크립트 실행
        python hyperparameter_search.py \
            --task_name=$TASK_NAME \
            --model_arch=$MODEL_ARCH \
            --gpus=$GPU_LIST \
            --trials=$TRIALS \
            --base_epochs=$EPOCHS \
            --batch_size=$BATCH_SIZE \
            --early_stopping \
            $ADDITIONAL_ARGS \
            > $LOG_DIR/hp_search_${TASK_NAME}.log 2>&1 &
        
        echo "하이퍼파라미터 탐색 시작됨: PID $!, 로그: $LOG_DIR/hp_search_${TASK_NAME}.log"
        ;;
        
    4)  # 분산 학습 실행
        read -p "작업 (edges2shoes/celebA/facescrub/car2car): " TASK_NAME
        read -p "모델 아키텍처 (discogan/recongan/gan): " MODEL_ARCH
        read -p "GPU당 배치 크기: " BATCH_SIZE
        read -p "에포크 수: " EPOCHS
        
        ADDITIONAL_ARGS=""
        if [ "$TASK_NAME" == "celebA" ]; then
            read -p "스타일 A (예: Male): " STYLE_A
            read -p "스타일 B (예: Smiling): " STYLE_B
            if [ -n "$STYLE_A" ]; then
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_A=$STYLE_A"
            fi
            if [ -n "$STYLE_B" ]; then
                ADDITIONAL_ARGS="$ADDITIONAL_ARGS --style_B=$STYLE_B"
            fi
        fi
        
        # GPU 목록을 쉼표로 구분된 문자열로 변환
        GPU_LIST=$(IFS=,; echo "${AVAILABLE_GPUS[*]}")
        
        # 분산 학습 스크립트 실행
        bash distributed_training.sh \
            --task_name=$TASK_NAME \
            --model_arch=$MODEL_ARCH \
            --batch_size=$BATCH_SIZE \
            --epochs=$EPOCHS \
            $ADDITIONAL_ARGS \
            > $LOG_DIR/distributed_${TASK_NAME}.log 2>&1 &
        
        echo "분산 학습 시작됨: PID $!, 로그: $LOG_DIR/distributed_${TASK_NAME}.log"
        ;;
        
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac

echo "모든 작업이 백그라운드에서 실행 중입니다."
echo "로그를 확인하려면 다음 명령어를 사용하세요:"
echo "  tail -f $LOG_DIR/*.log"

# 모니터링 스크립트 실행
read -p "GPU 모니터링을 실행하시겠습니까? (y/n): " RUN_MONITOR
if [ "$RUN_MONITOR" == "y" ]; then
    python gpu_monitor.py --interval=10 --output=$LOG_DIR
fi
