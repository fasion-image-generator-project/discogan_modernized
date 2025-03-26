#!/usr/bin/env python3
"""
GPU 사용량을 모니터링하고 학습 작업을 관리하는 스크립트
"""

import subprocess
import time
import os
import re
import argparse
import signal
import sys
from datetime import datetime
import json
import threading
from pathlib import Path

# 인자 파싱
parser = argparse.ArgumentParser(description='GPU 모니터링 및 작업 관리')
parser.add_argument('--interval', type=int, default=10, help='모니터링 간격(초)')
parser.add_argument('--output', type=str, default='./', help='로그 저장 디렉토리')
parser.add_argument('--auto-restart', action='store_true', help='실패한 작업 자동 재시작')
parser.add_argument('--max-utilization', type=float, default=95.0, help='최대 GPU 메모리 사용률(%)')
parser.add_argument('--alert-threshold', type=float, default=85.0, help='경고 임계값(%)')
args = parser.parse_args()

# 로그 디렉토리 생성
log_dir = Path(args.output) / 'gpu_monitor'
log_dir.mkdir(parents=True, exist_ok=True)

# 실행 중인 작업 목록
running_jobs = {}

# 종료 신호 핸들러
def signal_handler(sig, frame):
    print('\n프로그램을 종료합니다...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_gpu_info():
    """nvidia-smi 명령을 실행하여 GPU 정보를 가져옵니다."""
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'])
        nvidia_smi = nvidia_smi.decode('utf-8').strip()
        
        gpu_info = []
        for line in nvidia_smi.split('\n'):
            values = [value.strip() for value in line.split(',')]
            if len(values) == 8:
                gpu = {
                    'index': int(values[0]),
                    'name': values[1],
                    'temperature': int(values[2]),
                    'gpu_util': float(values[3]),
                    'memory_util': float(values[4]),
                    'memory_total': float(values[5]),
                    'memory_used': float(values[6]),
                    'memory_free': float(values[7])
                }
                gpu_info.append(gpu)
        
        return gpu_info
    except Exception as e:
        print(f"GPU 정보를 가져오는 중 오류 발생: {e}")
        return []

def get_processes():
    """실행 중인 GPU 프로세스 정보를 가져옵니다."""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_name,gpu_bus_id,used_memory', '--format=csv,noheader,nounits'])
        result = result.decode('utf-8').strip()
        
        processes = []
        for line in result.split('\n'):
            if not line:
                continue
            
            values = [value.strip() for value in line.split(',')]
            if len(values) >= 4:
                process = {
                    'pid': int(values[0]),
                    'gpu_name': values[1],
                    'gpu_bus_id': values[2],
                    'used_memory': float(values[3]),
                }
                
                # 추가 프로세스 정보 가져오기
                try:
                    cmd = subprocess.check_output(['ps', '-p', str(process['pid']), '-o', 'cmd='])
                    process['command'] = cmd.decode('utf-8').strip()
                    
                    # Python 스크립트 이름 추출
                    if 'python' in process['command']:
                        match = re.search(r'python\s+([^\s]+\.py)', process['command'])
                        if match:
                            process['script'] = match.group(1)
                        else:
                            process['script'] = 'unknown'
                            
                    # 작업 이름 추출 (--task_name 인자)
                    task_match = re.search(r'--task_name=([^\s]+)', process['command'])
                    if task_match:
                        process['task_name'] = task_match.group(1)
                    else:
                        process['task_name'] = 'unknown'
                except:
                    process['command'] = 'unknown'
                    process['script'] = 'unknown'
                    process['task_name'] = 'unknown'
                
                processes.append(process)
        
        return processes
    except Exception as e:
        print(f"프로세스 정보를 가져오는 중 오류 발생: {e}")
        return []

def monitor_gpu():
    """GPU 상태를 모니터링하고 로그를 기록합니다."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = log_dir / f'gpu_monitor_{timestamp}.log'
    json_file = log_dir / f'gpu_monitor_{timestamp}.json'
    
    print(f"GPU 모니터링 시작 (간격: {args.interval}초)")
    print(f"로그 파일: {log_file}")
    
    with open(log_file, 'w') as f:
        f.write(f"GPU 모니터링 시작: {timestamp}\n")
        f.write(f"간격: {args.interval}초\n")
        f.write("-" * 80 + "\n")
    
    all_data = []
    
    while True:
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            gpu_info = get_gpu_info()
            processes = get_processes()
            
            # 콘솔에 출력
            print(f"\n===== {current_time} =====")
            print("GPU 상태:")
            for gpu in gpu_info:
                memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                alert = ""
                if memory_percent >= args.max_utilization:
                    alert = " [경고: 메모리 초과]"
                elif memory_percent >= args.alert_threshold:
                    alert = " [주의: 높은 메모리 사용]"
                
                print(f"GPU {gpu['index']} ({gpu['name']}): "
                      f"온도 {gpu['temperature']}°C, "
                      f"GPU 사용 {gpu['gpu_util']:.1f}%, "
                      f"메모리 {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB "
                      f"({memory_percent:.1f}%){alert}")
            
            print("\nGPU 프로세스:")
            for proc in processes:
                print(f"PID {proc['pid']} (GPU: {proc['gpu_name']}): "
                      f"메모리 {proc['used_memory']:.0f} MB, "
                      f"작업: {proc['task_name']}, "
                      f"스크립트: {proc['script']}")
            
            # 파일에 기록
            with open(log_file, 'a') as f:
                f.write(f"\n===== {current_time} =====\n")
                f.write("GPU 상태:\n")
                for gpu in gpu_info:
                    memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                    f.write(f"GPU {gpu['index']} ({gpu['name']}): "
                           f"온도 {gpu['temperature']}°C, "
                           f"GPU 사용 {gpu['gpu_util']:.1f}%, "
                           f"메모리 {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB "
                           f"({memory_percent:.1f}%)\n")
                
                f.write("\nGPU 프로세스:\n")
                for proc in processes:
                    f.write(f"PID {proc['pid']} (GPU: {proc['gpu_name']}): "
                           f"메모리 {proc['used_memory']:.0f} MB, "
                           f"작업: {proc['task_name']}, "
                           f"스크립트: {proc['script']}\n")
                           
            # JSON 데이터 저장
            data_point = {
                'timestamp': current_time,
                'gpu_info': gpu_info,
                'processes': processes
            }
            all_data.append(data_point)
            
            with open(json_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            # 자동 재시작 기능
            if args.auto_restart:
                check_and_restart_jobs(processes)
                
            # 메모리 사용량 경고
            for gpu in gpu_info:
                memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                if memory_percent >= args.max_utilization:
                    print(f"\n[경고] GPU {gpu['index']} 메모리 사용량이 {memory_percent:.1f}%로 매우 높습니다!")
                    # 여기에 메모리 사용량이 높을 때 조치를 추가할 수 있습니다
            
            time.sleep(args.interval)
            
        except Exception as e:
            print(f"모니터링 중 오류 발생: {e}")
            time.sleep(args.interval)

def check_and_restart_jobs(current_processes):
    """작업이 실패했는지 확인하고 필요한 경우 재시작합니다."""
    global running_jobs
    
    # 현재 실행 중인 PID 목록
    current_pids = {proc['pid'] for proc in current_processes}
    
    # 종료된 작업 확인
    terminated_jobs = {}
    for pid, job_info in list(running_jobs.items()):
        if pid not in current_pids:
            print(f"작업이 종료됨: PID {pid} ({job_info['task_name']})")
            terminated_jobs[pid] = job_info
            del running_jobs[pid]
    
    # 종료된 작업 재시작
    for pid, job_info in terminated_jobs.items():
        # 정상 종료됐을 수 있으므로 추가 확인 필요
        # 학습 중 에러로 인한 종료인지 확인하는 로직을 추가할 수 있습니다
        
        if job_info.get('restart_count', 0) < 3:  # 최대 3번까지만 재시작
            print(f"작업 재시작 중: {job_info['command']}")
            try:
                # 환경 변수 설정
                env = os.environ.copy()
                if 'gpu_id' in job_info:
                    env['CUDA_VISIBLE_DEVICES'] = str(job_info['gpu_id'])
                
                # 작업 재시작
                process = subprocess.Popen(
                    job_info['command'].split(),
                    stdout=open(f"logs/restart_{job_info['task_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", 'w'),
                    stderr=subprocess.STDOUT,
                    env=env
                )
                
                # 작업 정보 업데이트
                job_info['restart_count'] = job_info.get('restart_count', 0) + 1
                job_info['restart_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                running_jobs[process.pid] = job_info
                
                print(f"작업이 재시작됨: 새 PID {process.pid}, 재시작 횟수: {job_info['restart_count']}")
                
            except Exception as e:
                print(f"작업 재시작 실패: {e}")
    
    # 새 작업 추가
    for proc in current_processes:
        if proc['pid'] not in running_jobs:
            # 새 DiscoGAN 작업 감지
            if 'python' in proc.get('command', '') and ('image_translation.py' in proc.get('command', '') or 'angle_pairing.py' in proc.get('command', '')):
                running_jobs[proc['pid']] = {
                    'pid': proc['pid'],
                    'command': proc['command'],
                    'task_name': proc.get('task_name', 'unknown'),
                    'script': proc.get('script', 'unknown'),
                    'gpu_id': proc.get('gpu_bus_id', 'unknown'),
                    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'restart_count': 0
                }
                print(f"새 작업 감지: PID {proc['pid']} ({proc.get('task_name', 'unknown')})")

def execute_command(command):
    """명령을 실행하고 결과를 반환합니다."""
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"오류 (코드 {e.returncode}): {e.output.decode('utf-8')}"

def interactive_mode():
    """대화형 모드로 GPU를 관리합니다."""
    while True:
        print("\n===== 대화형 모드 =====")
        print("1. GPU 상태 조회")
        print("2. 프로세스 목록 조회")
        print("3. 프로세스 종료 (PID 지정)")
        print("4. 새 DiscoGAN 작업 시작")
        print("5. 변경된 코드 적용 (git pull)")
        print("0. 종료")
        
        choice = input("\n선택: ")
        
        if choice == '1':
            print(execute_command("nvidia-smi"))
        
        elif choice == '2':
            processes = get_processes()
            print("\nGPU 프로세스 목록:")
            for proc in processes:
                print(f"PID {proc['pid']} (GPU: {proc['gpu_name']}): "
                     f"메모리 {proc['used_memory']:.0f} MB, "
                     f"작업: {proc.get('task_name', 'unknown')}, "
                     f"스크립트: {proc.get('script', 'unknown')}")
        
        elif choice == '3':
            pid = input("종료할 프로세스 PID: ")
            try:
                print(execute_command(f"kill -9 {pid}"))
                print(f"PID {pid} 프로세스가 종료되었습니다.")
            except Exception as e:
                print(f"프로세스 종료 실패: {e}")
        
        elif choice == '4':
            gpu_id = input("사용할 GPU ID: ")
            task_name = input("작업 이름 (edges2shoes, celebA, facescrub, car2car 등): ")
            batch_size = input("배치 크기 (기본값: 64): ") or "64"
            epochs = input("에포크 수 (기본값: 50): ") or "50"
            
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python "
            
            if task_name in ['car2car', 'chair2chair', 'face2face']:
                cmd += f"angle_pairing.py --task_name={task_name} --model_arch=discogan --batch_size={batch_size} --epochs={epochs}"
            else:
                cmd += f"image_translation.py --task_name={task_name} --model_arch=discogan --batch_size={batch_size} --epochs={epochs}"
                
                if task_name == 'celebA':
                    style_A = input("스타일 A (예: Male): ")
                    style_B = input("스타일 B (예: Smiling): ")
                    if style_A:
                        cmd += f" --style_A={style_A}"
                    if style_B:
                        cmd += f" --style_B={style_B}"
            
            log_file = f"logs/{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            cmd += f" > {log_file} 2>&1"
            
            print(f"명령 실행: {cmd}")
            try:
                subprocess.Popen(cmd, shell=True)
                print(f"작업이 시작되었습니다. 로그: {log_file}")
            except Exception as e:
                print(f"작업 시작 실패: {e}")
        
        elif choice == '5':
            print("git pull로 코드 업데이트 중...")
            print(execute_command("git pull"))
        
        elif choice == '0':
            print("프로그램을 종료합니다.")
            sys.exit(0)
        
        else:
            print("잘못된 선택입니다. 다시 시도하세요.")

def main():
    """메인 함수"""
    # 모니터링 스레드 시작
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    
    # 대화형 모드 시작
    interactive_mode()

if __name__ == "__main__":
    main()
