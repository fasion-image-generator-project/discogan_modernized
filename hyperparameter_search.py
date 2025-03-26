#!/usr/bin/env python3
"""
DiscoGAN 하이퍼파라미터 탐색 스크립트

여러 GPU에서 병렬로 다양한 하이퍼파라미터 조합을 테스트하고 최적의 설정을 찾습니다.
"""

import argparse
import subprocess
import time
from datetime import datetime
import os
import json
import random
import itertools
from pathlib import Path
import numpy as np
import re
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='DiscoGAN 하이퍼파라미터 탐색')
    parser.add_argument('--task_name', type=str, default='edges2shoes',
                      help='실험할 작업 (edges2shoes, celebA, facescrub, car2car 등)')
    parser.add_argument('--model_arch', type=str, default='discogan',
                      help='모델 아키텍처 (discogan, recongan, gan)')
    parser.add_argument('--gpus', type=str, default='0,1,4,5,6',
                      help='사용할 GPU ID 목록 (쉼표로 구분)')
    parser.add_argument('--trials', type=int, default=20,
                      help='총 실험 횟수')
    parser.add_argument('--base_epochs', type=int, default=20,
                      help='각 실험의 기본 에포크 수')
    parser.add_argument('--style_A', type=str, default=None,
                      help='CelebA 스타일 A (CelebA 작업일 경우)')
    parser.add_argument('--style_B', type=str, default=None,
                      help='CelebA 스타일 B (CelebA 작업일 경우)')
    parser.add_argument('--output_dir', type=str, default='./hp_search',
                      help='결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='기본 배치 크기')
    parser.add_argument('--early_stopping', action='store_true',
                      help='조기 종료 활성화')
    parser.add_argument('--patience', type=int, default=5,
                      help='조기 종료 인내심 값')
    return parser.parse_args()

def generate_hyperparameters():
    """하이퍼파라미터 조합을 생성합니다."""
    
    # 탐색할 하이퍼파라미터 범위 정의
    param_ranges = {
        'learning_rate': [0.0001, 0.0002, 0.0003, 0.0005],
        'beta1': [0.5, 0.7, 0.9],
        'beta2': [0.9, 0.99, 0.999],
        'starting_rate': [0.01, 0.05, 0.1, 0.2],
        'default_rate': [0.3, 0.5, 0.7, 0.9],
        'gan_curriculum': [5000, 10000, 15000, 20000],
        'update_interval': [1, 2, 3, 5]
    }
    
    # 모든 가능한 조합 생성
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    
    all_combinations = list(itertools.product(*values))
    
    # 조합과 해당 매개변수 이름을 연결하여 딕셔너리 목록 생성
    hp_combinations = []
    for combo in all_combinations:
        hp_dict = {keys[i]: combo[i] for i in range(len(keys))}
        hp_combinations.append(hp_dict)
    
    return hp_combinations

def sample_hyperparameters(num_samples=10):
    """가능한 하이퍼파라미터 공간에서 무작위로 샘플링합니다."""
    
    # 전체 조합 대신 범위에서 무작위 샘플링
    samples = []
    
    for _ in range(num_samples):
        # 무작위 하이퍼파라미터 값 선택
        hp = {
            'learning_rate': random.choice([0.0001, 0.0002, 0.0003, 0.0005, 0.0008]),
            'beta1': random.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            'beta2': random.choice([0.9, 0.95, 0.99, 0.999]),
            'starting_rate': random.choice([0.01, 0.05, 0.1, 0.2, 0.3]),
            'default_rate': random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.9]),
            'gan_curriculum': random.choice([5000, 8000, 10000, 15000, 20000]),
            'update_interval': random.choice([1, 2, 3, 5])
        }
        samples.append(hp)
    
    return samples

def get_available_gpus(gpu_list):
    """사용 가능한 GPU 목록을 반환합니다."""
    gpus = [int(gpu.strip()) for gpu in gpu_list.split(',')]
    
    # nvidia-smi로 GPU 메모리 사용량 확인
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'])
        output = output.decode('utf-8').strip().split('\n')
        
        gpu_usage = {}
        for line in output:
            idx, mem_used = map(int, line.split(','))
            gpu_usage[idx] = mem_used
        
        # 500MB 미만 사용 중인 GPU만 사용
        available_gpus = [gpu for gpu in gpus if gpu_usage.get(gpu, 0) < 500]
        
        return available_gpus
    except:
        # nvidia-smi 실행 실패 시 모든 GPU 사용
        return gpus

def run_trial(hp, args, gpu_id, trial_id):
    """주어진 하이퍼파라미터로 실험을 실행합니다."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 결과 디렉토리 생성
    result_base = Path(args.output_dir) / args.task_name / args.model_arch
    result_dir = result_base / f"trial_{trial_id}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 하이퍼파라미터 저장
    with open(result_dir / "hyperparameters.json", 'w') as f:
        json.dump(hp, f, indent=2)
    
    # 사용할 스크립트 결정
    if args.task_name in ['car2car', 'chair2chair', 'face2face']:
        script = "angle_pairing.py"
    else:
        script = "image_translation.py"
    
    # 명령 구성
    cmd = [
        "python", script,
        f"--task_name={args.task_name}",
        f"--model_arch={args.model_arch}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.base_epochs}",
        f"--results_dir={result_dir/'results'}",
        f"--models_dir={result_dir/'models'}",
        f"--learning_rate={hp['learning_rate']}",
        f"--beta1={hp['beta1']}",
        f"--beta2={hp['beta2']}",
        f"--starting_rate={hp['starting_rate']}",
        f"--default_rate={hp['default_rate']}",
        f"--gan_curriculum={hp['gan_curriculum']}",
        f"--update_interval={hp['update_interval']}",
        f"--device=cuda"
    ]
    
    # CelebA 작업일 경우 스타일 인자 추가
    if args.task_name == 'celebA':
        if args.style_A:
            cmd.append(f"--style_A={args.style_A}")
        if args.style_B:
            cmd.append(f"--style_B={args.style_B}")
    
    # 환경 변수 설정 (GPU 선택)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 로그 파일 설정
    log_file = result_dir / "train.log"
    
    print(f"시작: 실험 {trial_id}, GPU {gpu_id}")
    print(f"하이퍼파라미터: {hp}")
    print(f"명령: {' '.join(cmd)}")
    print(f"로그: {log_file}")
    
    # 프로세스 시작
    start_time = time.time()
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env
        )
    
    # 실행 정보 저장
    trial_info = {
        "trial_id": trial_id,
        "gpu_id": gpu_id,
        "hyperparameters": hp,
        "command": " ".join(cmd),
        "log_file": str(log_file),
        "start_time": timestamp,
        "pid": process.pid,
        "status": "running"
    }
    
    with open(result_dir / "trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=2)
    
    return process, result_dir, trial_info

def monitor_process(process, result_dir, trial_info, args):
    """프로세스를 모니터링하고 필요한 경우 조기 종료합니다."""
    log_file = Path(trial_info["log_file"])
    best_recon_loss = float('inf')
    no_improvement_count = 0
    
    while process.poll() is None:  # 프로세스가 실행 중인 동안
        time.sleep(30)  # 30초마다 확인
        
        # 조기 종료가 활성화된 경우
        if args.early_stopping and log_file.exists():
            try:
                # 로그 파일에서 재구성 손실 추출
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                # 정규 표현식으로 재구성 손실 추출
                recon_losses = re.findall(r'RECON: (\d+\.\d+)/(\d+\.\d+)', log_content)
                
                if recon_losses:
                    # 가장 최근 손실 추출
                    latest_losses = recon_losses[-1]
                    avg_recon_loss = (float(latest_losses[0]) + float(latest_losses[1])) / 2
                    
                    # 개선 여부 확인
                    if avg_recon_loss < best_recon_loss:
                        best_recon_loss = avg_recon_loss
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # 지정된 에포크 동안 개선이 없으면 조기 종료
                    if no_improvement_count >= args.patience:
                        print(f"조기 종료: 실험 {trial_info['trial_id']}, {args.patience}회 이상 성능 개선 없음")
                        process.terminate()
                        break
            except Exception as e:
                print(f"로그 파싱 오류: {e}")
    
    # 프로세스 종료 후 정보 업데이트
    trial_info["status"] = "completed"
    trial_info["end_time"] = datetime.now().strftime('%Y%m%d_%H%M%S')
    trial_info["duration"] = time.time() - start_time
    
    with open(result_dir / "trial_info.json", 'w') as f:
        json.dump(trial_info, f, indent=2)
    
    print(f"완료: 실험 {trial_info['trial_id']}, GPU {trial_info['gpu_id']}")
    
    return extract_metrics(log_file)

def extract_metrics(log_file):
    """로그 파일에서 성능 지표를 추출합니다."""
    metrics = {
        "final_gen_loss_A": None,
        "final_gen_loss_B": None,
        "final_recon_loss_A": None,
        "final_recon_loss_B": None,
        "final_dis_loss_A": None,
        "final_dis_loss_B": None
    }
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # 마지막 손실 값 추출
        gen_matches = re.findall(r'GEN: (\d+\.\d+)/(\d+\.\d+)', log_content)
        recon_matches = re.findall(r'RECON: (\d+\.\d+)/(\d+\.\d+)', log_content)
        dis_matches = re.findall(r'DIS: (\d+\.\d+)/(\d+\.\d+)', log_content)
        
        if gen_matches:
            metrics["final_gen_loss_A"] = float(gen_matches[-1][0])
            metrics["final_gen_loss_B"] = float(gen_matches[-1][1])
        
        if recon_matches:
            metrics["final_recon_loss_A"] = float(recon_matches[-1][0])
            metrics["final_recon_loss_B"] = float(recon_matches[-1][1])
        
        if dis_matches:
            metrics["final_dis_loss_A"] = float(dis_matches[-1][0])
            metrics["final_dis_loss_B"] = float(dis_matches[-1][1])
        
        # 평균 재구성 손실 계산
        if metrics["final_recon_loss_A"] is not None and metrics["final_recon_loss_B"] is not None:
            metrics["avg_recon_loss"] = (metrics["final_recon_loss_A"] + metrics["final_recon_loss_B"]) / 2
    
    except Exception as e:
        print(f"메트릭 추출 오류: {e}")
    
    return metrics

def analyze_results(output_dir, task_name, model_arch):
    """모든 실험 결과를 분석하고 최적의 하이퍼파라미터를 찾습니다."""
    result_base = Path(output_dir) / task_name / model_arch
    
    # 모든 실험 디렉토리 수집
    trial_dirs = [d for d in result_base.glob("trial_*") if d.is_dir()]
    
    if not trial_dirs:
        print("분석할 실험 결과가 없습니다.")
        return
    
    all_trials = []
    
    for trial_dir in trial_dirs:
        info_file = trial_dir / "trial_info.json"
        hp_file = trial_dir / "hyperparameters.json"
        
        if info_file.exists() and hp_file.exists():
            try:
                with open(info_file, 'r') as f:
                    trial_info = json.load(f)
                
                with open(hp_file, 'r') as f:
                    hyperparameters = json.load(f)
                
                # 메트릭 추출
                metrics = extract_metrics(Path(trial_info["log_file"]))
                
                # 실험 정보 병합
                trial_data = {
                    "trial_id": trial_info["trial_id"],
                    "status": trial_info.get("status", "unknown"),
                    "hyperparameters": hyperparameters,
                    "metrics": metrics,
                    "result_dir": str(trial_dir)
                }
                
                all_trials.append(trial_data)
            
            except Exception as e:
                print(f"실험 {trial_dir} 데이터 로드 오류: {e}")
    
    if not all_trials:
        print("분석할 유효한 실험 결과가 없습니다.")
        return
    
    # 완료된 실험만 필터링
    completed_trials = [t for t in all_trials if t["status"] == "completed"]
    
    if not completed_trials:
        print("완료된 실험이 없습니다.")
        return
    
    # 재구성 손실 기준으로 정렬
    sorted_trials = sorted(completed_trials, 
                          key=lambda x: x["metrics"].get("avg_recon_loss", float('inf')))
    
    # 결과 요약
    print("\n===== 실험 결과 요약 =====")
    print(f"총 실험 수: {len(all_trials)}")
    print(f"완료된 실험 수: {len(completed_trials)}")
    
    # 최적의 하이퍼파라미터 출력
    if sorted_trials:
        best_trial = sorted_trials[0]
        print("\n===== 최적의 하이퍼파라미터 =====")
        print(f"실험 ID: {best_trial['trial_id']}")
        print(f"평균 재구성 손실: