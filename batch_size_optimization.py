#!/usr/bin/env python3
"""
메모리 사용량에 따라 최적의 배치 크기를 자동으로 찾는 스크립트
이 스크립트는 GPU 메모리를 최대한 활용할 수 있는 배치 크기를 찾습니다.
"""

import argparse
import subprocess
import time
import torch
import numpy as np
from model import Generator, Discriminator
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description='GPU 메모리에 최적화된 배치 크기 찾기')
    parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU ID')
    parser.add_argument('--model_arch', type=str, default='discogan', choices=['discogan', 'recongan', 'gan'],
                       help='모델 아키텍처')
    parser.add_argument('--image_size', type=int, default=64, help='이미지 크기')
    parser.add_argument('--min_batch', type=int, default=16, help='최소 배치 크기')
    parser.add_argument('--max_batch', type=int, default=512, help='최대 배치 크기')
    parser.add_argument('--step', type=int, default=16, help='배치 크기 증가 단계')
    parser.add_argument('--target_memory', type=float, default=0.85, 
                       help='목표 메모리 사용률 (0.0-1.0)')
    parser.add_argument('--extra_layers', action='store_true', help='Generator에 추가 레이어 사용')
    parser.add_argument('--safety_margin', type=float, default=0.9, 
                       help='결과 배치 크기에 적용할 안전 마진 (0.0-1.0)')
    parser.add_argument('--output', type=str, default='batch_size_results.json',
                       help='결과 저장 파일')
    return parser.parse_args()

def get_gpu_memory(gpu_id):
    """지정된 GPU의 총 메모리 및 사용 가능한 메모리를 반환합니다."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            encoding='utf-8')
        total, free = map(int, output.strip().split(','))
        return total, free
    except:
        print(f"오류: GPU {gpu_id}에 접근할 수 없습니다.")
        return 0, 0

def test_batch_size(gpu_id, batch_size, image_size, extra_layers=False):
    """지정된 배치 크기로 모델을 생성하고 메모리 사용량을 측정합니다."""
    torch.cuda.empty_cache()
    device = torch.device(f'cuda:{gpu_id}')
    
    try:
        # 메모리 사용량 측정 전
        _, free_before = get_gpu_memory(gpu_id)
        
        # 모델 생성
        generator_A = Generator(extra_layers=extra_layers).to(device)
        generator_B = Generator(extra_layers=extra_layers).to(device)
        discriminator_A = Discriminator().to(device)
        discriminator_B = Discriminator().to(device)
        
        # 랜덤 배치 생성
        batch_A = torch.randn(batch_size, 3, image_size, image_size).to(device)
        batch_B = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        # 전방 전파 실행
        AB = generator_B(batch_A)
        BA = generator_A(batch_B)
        ABA = generator_A(AB)
        BAB = generator_B(BA)
        
        # 판별자 출력
        A_dis_real, A_feats_real = discriminator_A(batch_A)
        A_dis_fake, A_feats_fake = discriminator_A(BA)
        B_dis_real, B_feats_real = discriminator_B(batch_B)
        B_dis_fake, B_feats_fake = discriminator_B(AB)
        
        # 메모리 사용량 측정 후
        torch.cuda.synchronize()
        _, free_after = get_gpu_memory(gpu_id)
        
        # 메모리 사용량 계산 (사용된 메모리 = 이전 여유 메모리 - 현재 여유 메모리)
        memory_used = free_before - free_after
        
        # 모델과 텐서 해제
        del generator_A, generator_B, discriminator_A, discriminator_B
        del batch_A, batch_B, AB, BA, ABA, BAB
        del A_dis_real, A_feats_real, A_dis_fake, A_feats_fake
        del B_dis_real, B_feats_real, B_dis_fake, B_feats_fake
        torch.cuda.empty_cache()
        
        return memory_used, True
    
    except torch.cuda.OutOfMemoryError:
        # 메모리 부족 오류 처리
        torch.cuda.empty_cache()
        return 0, False
    
    except Exception as e:
        print(f"오류 발생: {e}")
        torch.cuda.empty_cache()
        return 0, False

def find_optimal_batch_size(args):
    """최적의 배치 크기를 찾습니다."""
    print(f"GPU {args.gpu}에서 최적의 배치 크기 찾는 중...")
    
    # GPU 메모리 확인
    total_memory, free_memory = get_gpu_memory(args.gpu)
    if total_memory == 0:
        print("GPU를 찾을 수 없습니다.")
        return None
    
    print(f"GPU {args.gpu}: 총 메모리 {total_memory}MB, 사용 가능 메모리 {free_memory}MB")
    
    # 배치 크기 목록 생성
    batch_sizes = list(range(args.min_batch, args.max_batch + args.step, args.step))
    
    # 이진 검색으로 최적 배치 크기 찾기
    left, right = 0, len(batch_sizes) - 1
    optimal_batch = args.min_batch
    memory_usages = {}
    
    while left <= right:
        mid = (left + right) // 2
        batch_size = batch_sizes[mid]
        
        print(f"배치 크기 {batch_size} 테스트 중...")
        memory_used, success = test_batch_size(args.gpu, batch_size, args.image_size, args.extra_layers)
        
        if success:
            memory_usages[batch_size] = memory_used
            memory_utilization = memory_used / total_memory
            print(f"  성공: {memory_used}MB 사용 (전체 메모리의 {memory_utilization:.1%})")
            
            if memory_utilization <= args.target_memory:
                # 목표 메모리 사용률보다 낮으면 더 큰 배치 크기 시도
                optimal_batch = batch_size
                left = mid + 1
            else:
                # 목표 메모리 사용률보다 높으면 더 작은 배치 크기로 돌아감
                right = mid - 1
        else:
            print(f"  실패: 메모리 부족")
            right = mid - 1
    
    # 안전 마진 적용
    safe_batch_size = int(optimal_batch * args.safety_margin)
    safe_batch_size = max(args.min_batch, safe_batch_size)
    
    # 배치 크기가 step의 배수가 되도록 조정
    safe_batch_size = (safe_batch_size // args.step) * args.step
    
    # 결과 정리
    results = {
        "gpu_id": args.gpu,
        "total_memory_mb": total_memory,
        "image_size": args.image_size,
        "model_arch": args.model_arch,
        "extra_layers": args.extra_layers,
        "optimal_batch_size": optimal_batch,
        "safe_batch_size": safe_batch_size,
        "safety_margin": args.safety_margin,
        "memory_usages": memory_usages
    }
    
    return results

def main():
    args = parse_args()
    
    # 결과 파일 경로
    output_path = Path(args.output)
    
    # 최적 배치 크기 찾기
    results = find_optimal_batch_size(args)
    
    if results:
        optimal_batch = results["optimal_batch_size"]
        safe_batch = results["safe_batch_size"]
        total_memory = results["total_memory_mb"]
        memory_used = results["memory_usages"].get(optimal_batch, 0)
        
        print("\n===== 결과 =====")
        print(f"최적 배치 크기: {optimal_batch} (메모리 사용: {memory_used}MB, {memory_used/total_memory:.1%})")
        print(f"안전 배치 크기: {safe_batch} (안전 마진: {args.safety_margin:.1%} 적용)")
        print(f"모델: {args.model_arch}, 이미지 크기: {args.image_size}x{args.image_size}")
        print(f"추가 레이어: {'사용' if args.extra_layers else '미사용'}")
        
        # 결과 저장
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n결과가 {output_path}에 저장되었습니다.")
        
        # 실행 예시 출력
        if args.model_arch == 'discogan' and args.image_size == 64:
            task_examples = {
                'edges2shoes': 'image_translation.py',
                'celebA': 'image_translation.py --style_A=Male --style_B=Smiling',
                'car2car': 'angle_pairing.py',
            }
            
            print("\n===== 실행 예시 =====")
            for task, script in task_examples.items():
                print(f"python {script} --task_name={task} --model_arch={args.model_arch} "
                     f"--batch_size={safe_batch} --device=cuda:{args.gpu}")
    else:
        print("배치 크기 최적화에 실패했습니다.")

if __name__ == "__main__":
    main()
