#!/bin/bash
# 여러 GPU에서 다양한 DiscoGAN 작업을 병렬로 실행하는 스크립트

# 사용 가능한 GPU 목록 (사용 중인 2번과 3번을 제외)
AVAILABLE_GPUS=(0 1 4 5 6)

# 각 GPU에서 실행할 작업 정의
declare -A TASKS
TASKS[0]="--task_name=edges2shoes --model_arch=discogan --batch_size=128 --epochs=50"
TASKS[1]="--task_name=edges2handbags --model_arch=discogan --batch_size=128 --epochs=50"
TASKS[4]="--task_name=celebA --model_arch=discogan --style_A=Male --style_B=Smiling --batch_size=128 --epochs=50"
TASKS[5]="--task_name=facescrub --model_arch=discogan --batch_size=128 --epochs=50"
TASKS[6]="--task_name=car2car --model_arch=discogan --batch_size=128 --epochs=30"

# 타임스탬프 설정 (모든 작업에 동일한 타임스탬프 적용)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="parallel_training_${TIMESTAMP}"
RESULTS_DIR="./results/${BASE_DIR}"
MODELS_DIR="./models/${BASE_DIR}"

# 기본 로그 디렉토리 생성
mkdir -p logs/${BASE_DIR}

# 각 GPU마다 작업 실행
for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
    TASK_ARGS=${TASKS[$GPU_ID]}
    
    if [ -n "$TASK_ARGS" ]; then
        # 작업 이름 추출 (로그 파일 이름에 사용)
        TASK_NAME=$(echo $TASK_ARGS | grep -o "task_name=[^ ]*" | cut -d'=' -f2)
        LOG_FILE="logs/${BASE_DIR}/gpu${GPU_ID}_${TASK_NAME}.log"
        
        echo "GPU $GPU_ID에서 작업 실행 중: $TASK_NAME"
        echo "로그 파일: $LOG_FILE"
        
        # 작업이 edges2shoes 또는 edges2handbags인 경우 image_translation.py 실행
        if [[ "$TASK_NAME" == "edges2shoes" || "$TASK_NAME" == "edges2handbags" || "$TASK_NAME" == "celebA" || "$TASK_NAME" == "facescrub" ]]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python image_translation.py $TASK_ARGS \
                --results_dir="${RESULTS_DIR}" \
                --models_dir="${MODELS_DIR}" \
                --device=cuda \
                > $LOG_FILE 2>&1 &
        
        # 작업이 car2car 또는 다른 각도 페어링 작업인 경우 angle_pairing.py 실행
        elif [[ "$TASK_NAME" == "car2car" || "$TASK_NAME" == "chair2chair" || "$TASK_NAME" == "face2face" ]]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python angle_pairing.py $TASK_ARGS \
                --results_dir="${RESULTS_DIR}" \
                --models_dir="${MODELS_DIR}" \
                --device=cuda \
                > $LOG_FILE 2>&1 &
        fi
        
        # 작업이 성공적으로 시작되었는지 확인
        if [ $? -eq 0 ]; then
            echo "GPU $GPU_ID에서 작업이 성공적으로 시작되었습니다."
        else
            echo "오류: GPU $GPU_ID에서 작업 시작에 실패했습니다."
        fi
        
        # 각 작업 시작 사이에 짧은 대기 시간을 추가하여 메모리 할당 충돌 방지
        sleep 5
    fi
done

echo "모든 작업이 백그라운드에서 실행 중입니다."
echo "로그를 확인하려면 다음 명령어를 사용하세요:"
echo "  tail -f logs/${BASE_DIR}/*.log"

# 모든 작업의 상태를 확인하기 위한 함수
check_tasks() {
    echo "실행 중인 작업 상태:"
    for GPU_ID in "${AVAILABLE_GPUS[@]}"; do
        if [ -n "${TASKS[$GPU_ID]}" ]; then
            TASK_NAME=$(echo ${TASKS[$GPU_ID]} | grep -o "task_name=[^ ]*" | cut -d'=' -f2)
            PID=$(ps aux | grep "python [a-z_]*\.py.*task_name=$TASK_NAME" | grep -v grep | awk '{print $2}')
            
            if [ -n "$PID" ]; then
                echo "GPU $GPU_ID ($TASK_NAME): 실행 중 (PID: $PID)"
            else
                echo "GPU $GPU_ID ($TASK_NAME): 종료됨"
            fi
        fi
    done
}

# 사용자에게 작업 상태 확인 옵션 제공
echo "작업 상태를 확인하려면 아무 키나 누르세요 (q를 누르면 종료)..."
read -n 1 -s INPUT

while [ "$INPUT" != "q" ]; do
    check_tasks
    echo "작업 상태를 확인하려면 아무 키나 누르세요 (q를 누르면 종료)..."
    read -n 1 -s INPUT
done

echo "스크립트를 종료합니다. 작업은 계속 백그라운드에서 실행됩니다."
