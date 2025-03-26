import os
import argparse
from datetime import datetime
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# 자체 모듈 가져오기
from model import Generator, Discriminator
from dataset import (
    get_cars, get_chairs, get_faces_3d, 
    read_images, shuffle_data
)

def parse_args():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN for angle pairing')
    
    # 기본 설정
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--task_name', type=str, default='car2car', 
                        help='Set data name (car2car/face2face/chair2chair/chair2car/...)')
    parser.add_argument('--results_dir', type=str, default='./results/', 
                        help='Directory to save the results')
    parser.add_argument('--models_dir', type=str, default='./models/', 
                        help='Directory to save models')
    parser.add_argument('--model_arch', type=str, default='discogan', 
                        help='Model architecture: gan/recongan/discogan')
    
    # 학습 관련 인수
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, 
                        help='Learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, 
                        help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, 
                        help='Beta2 for Adam optimizer')
    parser.add_argument('--image_size', type=int, default=64, 
                        help='Image size')
    
    # GAN 관련 인수
    parser.add_argument('--gan_curriculum', type=int, default=10000, 
                        help='Number of steps with strong GAN loss')
    parser.add_argument('--starting_rate', type=float, default=0.9, 
                        help='Initial lambda weight between GAN and Recon loss')
    parser.add_argument('--default_rate', type=float, default=0.9, 
                        help='Lambda weight between GAN and Recon loss after curriculum')
    
    # 기타 인수
    parser.add_argument('--n_test', type=int, default=200, 
                        help='Number of test images')
    parser.add_argument('--update_interval', type=int, default=3, 
                        help='Interval for discriminator updates')
    parser.add_argument('--log_interval', type=int, default=50, 
                        help='Print loss interval')
    parser.add_argument('--image_save_interval', type=int, default=500, 
                        help='Image save interval')
    parser.add_argument('--model_save_interval', type=int, default=10000, 
                        help='Model save interval')
    
    return parser.parse_args()

def get_data(args):
    """지정된 작업에 대한 데이터셋을 가져옵니다."""
    if args.task_name == 'car2car':
        data_A = get_cars(test=False, ver=180, half='first', image_size=args.image_size)
        data_B = get_cars(test=False, ver=180, half='last', image_size=args.image_size)
        test_A = test_B = get_cars(test=True, ver=180, image_size=args.image_size)

    elif args.task_name == 'face2face':
        data_A = get_faces_3d(test=False, half='first')
        data_B = get_faces_3d(test=False, half='last')
        test_A = test_B = get_faces_3d(test=True)

    elif args.task_name == 'chair2chair':
        data_A = get_chairs(test=False, half='first', ver=360)
        data_B = get_chairs(test=False, half='last', ver=360)
        test_A = test_B = get_chairs(test=True, ver=360)

    elif args.task_name == 'chair2car':
        data_A = get_chairs(test=False, half=None, ver=180)
        data_B = get_cars(test=False, half=None, ver=180)
        test_A = get_chairs(test=True, ver=180)
        test_B = get_cars(test=True, ver=180)

    elif args.task_name == 'chair2face':
        data_A = get_chairs(test=False, half=None, ver=180)
        data_B = get_faces_3d(test=False, half=None)
        test_A = get_chairs(test=True, ver=180)
        test_B = get_faces_3d(test=True)

    elif args.task_name == 'car2face':
        data_A = get_cars(test=False, ver=180, half=None)
        data_B = get_faces_3d(test=False, half=None)
        test_A = get_cars(test=True, ver=180)
        test_B = get_faces_3d(test=True)

    return data_A, data_B, test_A, test_B

def get_fm_loss(real_feats, fake_feats, criterion, device):
    """Feature matching 손실을 계산합니다."""
    losses = 0
    # 첫 번째 피처는 건너뛰고 나머지 피처만 사용
    for real_feat, fake_feat in zip(real_feats[1:], fake_feats[1:]):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion(l2, torch.ones(l2.size()).to(device))
        losses += loss
    
    return losses

def get_gan_loss(dis_real, dis_fake, criterion, device):
    """GAN 손실을 계산합니다."""
    batch_size = dis_real.size(0)
    
    # 레이블 생성
    labels_dis_real = torch.ones(batch_size, 1).to(device)
    labels_dis_fake = torch.zeros(batch_size, 1).to(device)
    labels_gen = torch.ones(batch_size, 1).to(device)
    
    # 판별자 손실 계산
    dis_loss = (criterion(dis_real, labels_dis_real) + 
                criterion(dis_fake, labels_dis_fake)) * 0.5
    
    # 생성자 손실 계산
    gen_loss = criterion(dis_fake, labels_gen)
    
    return dis_loss, gen_loss

def save_sample_images(test_A, test_B, generator_A, generator_B, save_dir, iteration, n_samples=5):
    """샘플 이미지를 저장합니다."""
    with torch.no_grad():
        AB = generator_B(test_A)
        BA = generator_A(test_B)
        ABA = generator_A(AB)
        BAB = generator_B(BA)
    
    # 이미지 저장 경로 생성
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # matplotlib로 결과 시각화
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3 * n_samples))
    
    for i in range(n_samples):
        images = [
            test_A[i].cpu(), test_B[i].cpu(),
            AB[i].cpu(), BA[i].cpu(),
            ABA[i].cpu(), BAB[i].cpu()
        ]
        
        titles = ['A', 'B', 'A->B', 'B->A', 'A->B->A', 'B->A->B']
        
        for j, (img, title) in enumerate(zip(images, titles)):
            img_np = img.numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
            img_np = np.clip(img_np, 0, 1)
            
            if n_samples > 1:
                axes[i, j].imshow(img_np)
                axes[i, j].set_title(title)
                axes[i, j].axis('off')
            else:
                axes[j].imshow(img_np)
                axes[j].set_title(title)
                axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'samples_iter_{iteration}.png')
    plt.close()

def main():
    # 인수 파싱
    args = parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 경로 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = Path(args.results_dir) / args.task_name / args.model_arch / timestamp
    model_path = Path(args.models_dir) / args.task_name / args.model_arch / timestamp
    
    # 데이터 가져오기
    data_style_A, data_style_B, test_style_A, test_style_B = get_data(args)
    
    # 테스트 데이터 처리
    if args.task_name.startswith('car') and args.task_name.endswith('car'):
        test_A = test_style_A
        test_B = test_style_B
    elif args.task_name.startswith('car') and not args.task_name.endswith('car'):
        test_A = test_style_A
        test_B = read_images(test_style_B, None, args.image_size)
    else:
        test_A = read_images(test_style_A, None, args.image_size)
        test_B = read_images(test_style_B, None, args.image_size)
    
    # 텐서로 변환
    test_A = torch.FloatTensor(test_A).to(device)
    test_B = torch.FloatTensor(test_B).to(device)
    
    # 디렉토리 생성
    result_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 초기화 - 각도 페어링에는 추가 레이어 사용
    generator_A = Generator(extra_layers=True).to(device)
    generator_B = Generator(extra_layers=True).to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)
    
    # 손실 함수
    recon_criterion = nn.MSELoss()
    gan_criterion = nn.BCELoss()
    feat_criterion = nn.HingeEmbeddingLoss()
    
    # 옵티마이저
    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())
    
    optim_gen = optim.Adam(
        gen_params, 
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=0.00001
    )
    
    optim_dis = optim.Adam(
        dis_params,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=0.00001
    )
    
    # 학습 반복 횟수
    data_size = min(len(data_style_A), len(data_style_B))
    n_batches = data_size // args.batch_size
    total_iterations = args.epochs * n_batches
    
    # 학습 로그
    log_file = result_path / "training_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Training started at {timestamp}\n")
        f.write(f"Task: {args.task_name}, Model: {args.model_arch}\n")
        f.write(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}\n\n")
    
    print(f"Total iterations: {total_iterations}")
    print(f"Saving results to: {result_path}")
    print(f"Saving models to: {model_path}")
    
    # 학습 시작
    iters = 0
    
    for epoch in range(args.epochs):
        # 데이터 셔플
        data_style_A, data_style_B = shuffle_data(data_style_A, data_style_B)
        
        # 에포크 진행률 막대
        progress_bar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i in progress_bar:
            # 배치 데이터 가져오기
            A_path = data_style_A[i * args.batch_size: (i + 1) * args.batch_size]
            B_path = data_style_B[i * args.batch_size: (i + 1) * args.batch_size]
            
            # 이미지 로드 및 전처리
            if args.task_name.startswith('car') and args.task_name.endswith('car'):
                A = A_path
                B = B_path
            elif args.task_name.startswith('car') and not args.task_name.endswith('car'):
                A = A_path
                B = read_images(B_path, None, args.image_size)
            else:
                A = read_images(A_path, None, args.image_size)
                B = read_images(B_path, None, args.image_size)
            
            # 텐서로 변환
            A = torch.FloatTensor(A).to(device)
            B = torch.FloatTensor(B).to(device)
            
            # 그래디언트 초기화
            generator_A.zero_grad()
            generator_B.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()
            
            # 생성자 출력
            AB = generator_B(A)  # A -> B
            BA = generator_A(B)  # B -> A
            
            ABA = generator_A(AB)  # A -> B -> A
            BAB = generator_B(BA)  # B -> A -> B
            
            # 재구성 손실
            recon_loss_A = recon_criterion(ABA, A)
            recon_loss_B = recon_criterion(BAB, B)
            
            # 판별자 A 손실
            A_dis_real, A_feats_real = discriminator_A(A)
            A_dis_fake, A_feats_fake = discriminator_A(BA)
            
            dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_criterion, device)
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion, device)
            
            # 판별자 B 손실
            B_dis_real, B_feats_real = discriminator_B(B)
            B_dis_fake, B_feats_fake = discriminator_B(AB)
            
            dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, gan_criterion, device)
            fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_criterion, device)
            
            # GAN vs 재구성 손실 비율 조정
            rate = args.starting_rate if iters < args.gan_curriculum else args.default_rate
            
            # 생성자 손실 계산
            gen_loss_A_total = (fm_loss_B * 0.9 + gen_loss_B * 0.1) * (1 - rate) + recon_loss_A * rate
            gen_loss_B_total = (fm_loss_A * 0.9 + gen_loss_A * 0.1) * (1 - rate) + recon_loss_B * rate
            
            # 아키텍처에 따른 최종 손실 계산
            if args.model_arch == 'discogan':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B
            elif args.model_arch == 'recongan':
                gen_loss = gen_loss_A_total
                dis_loss = dis_loss_B
            elif args.model_arch == 'gan':
                gen_loss = gen_loss_B * 0.1 + fm_loss_B * 0.9
                dis_loss = dis_loss_B
            
            # 학습 단계 실행
            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()
            
            # 로그 출력
            if iters % args.log_interval == 0:
                log_message = (f"Iter [{iters}/{total_iterations}] "
                              f"GEN: {gen_loss_A.item():.4f}/{gen_loss_B.item():.4f}, "
                              f"RECON: {recon_loss_A.item():.4f}/{recon_loss_B.item():.4f}, "
                              f"DIS: {dis_loss_A.item():.4f}/{dis_loss_B.item():.4f}")
                
                print(log_message)
                with open(log_file, "a") as f:
                    f.write(log_message + "\n")
                
                # 진행률 막대 업데이트
                progress_bar.set_postfix({
                    'D_loss': f"{dis_loss.item():.4f}",
                    'G_loss': f"{gen_loss.item():.4f}"
                })
            
            # 이미지 저장
            if iters % args.image_save_interval == 0:
                with torch.no_grad():
                    AB = generator_B(test_A)
                    BA = generator_A(test_B)
                    ABA = generator_A(AB)
                    BAB = generator_B(BA)
                
                n_testset = min(test_A.size(0), test_B.size(0))
                subdir_path = result_path / str(iters // args.image_save_interval)
                subdir_path.mkdir(parents=True, exist_ok=True)
                
                for im_idx in range(min(n_testset, args.n_test)):
                    A_val = test_A[im_idx].cpu().numpy().transpose(1, 2, 0) * 255.
                    B_val = test_B[im_idx].cpu().numpy().transpose(1, 2, 0) * 255.
                    BA_val = BA[im_idx].cpu().numpy().transpose(1, 2, 0) * 255.
                    ABA_val = ABA[im_idx].cpu().numpy().transpose(1, 2, 0) * 255.
                    AB_val = AB[im_idx].cpu().numpy().transpose(1, 2, 0) * 255.
                    BAB_val = BAB[im_idx].cpu().numpy().transpose(1, 2, 0) * 255.
                    
                    # matplotlib으로 저장
                    filename_prefix = subdir_path / f"{im_idx}"
                    
                    # 각 이미지 개별 저장
                    plt.figure(figsize=(6, 6))
                    plt.imshow(A_val.astype(np.uint8))
                    plt.axis('off')
                    plt.savefig(f"{filename_prefix}.A.jpg", bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(B_val.astype(np.uint8))
                    plt.axis('off')
                    plt.savefig(f"{filename_prefix}.B.jpg", bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(BA_val.astype(np.uint8))
                    plt.axis('off')
                    plt.savefig(f"{filename_prefix}.BA.jpg", bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(AB_val.astype(np.uint8))
                    plt.axis('off')
                    plt.savefig(f"{filename_prefix}.AB.jpg", bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(ABA_val.astype(np.uint8))
                    plt.axis('off')
                    plt.savefig(f"{filename_prefix}.ABA.jpg", bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(BAB_val.astype(np.uint8))
                    plt.axis('off')
                    plt.savefig(f"{filename_prefix}.BAB.jpg", bbox_inches='tight', pad_inches=0)
                    plt.close()
                
                # 샘플 그리드 이미지도 저장
                save_sample_images(
                    test_A[:5], test_B[:5],
                    generator_A, generator_B,
                    result_path / 'sample_grids',
                    iters
                )
            
            # 모델 저장
            if iters % args.model_save_interval == 0:
                torch.save(generator_A.state_dict(), model_path / f'gen_A_{iters}.pth')
                torch.save(generator_B.state_dict(), model_path / f'gen_B_{iters}.pth')
                torch.save(discriminator_A.state_dict(), model_path / f'dis_A_{iters}.pth')
                torch.save(discriminator_B.state_dict(), model_path / f'dis_B_{iters}.pth')
            
            iters += 1
    
    # 최종 모델 저장
    torch.save(generator_A.state_dict(), model_path / 'gen_A_final.pth')
    torch.save(generator_B.state_dict(), model_path / 'gen_B_final.pth')
    torch.save(discriminator_A.state_dict(), model_path / 'dis_A_final.pth')
    torch.save(discriminator_B.state_dict(), model_path / 'dis_B_final.pth')
    
    print(f"Training completed. Final models saved to {model_path}")
    print(f"Results and logs saved to {result_path}")

if __name__ == "__main__":
    main()