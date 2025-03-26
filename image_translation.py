import os
import argparse
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# 자체 모듈 가져오기
from model import Generator, Discriminator
from dataset import (
    get_celebA_files, get_edge2photo_files, get_facescrub_files,
    shuffle_data, read_images, DiscoGANDataset, get_custom_data
)

def parse_args():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
    
    # 기본 설정
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--task_name', type=str, default='facescrub', 
                        help='Set data name (facescrub/celebA/edges2shoes/...)')
    parser.add_argument('--results_dir', type=str, default='./results/', 
                        help='Directory to save the results')
    parser.add_argument('--models_dir', type=str, default='./models/', 
                        help='Directory to save models')
    parser.add_argument('--model_arch', type=str, default='discogan', 
                        help='Model architecture: gan/recongan/discogan')
    
    # 학습 관련 인수
    parser.add_argument('--epochs', type=int, default=100, 
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
    parser.add_argument('--starting_rate', type=float, default=0.01, 
                        help='Initial lambda weight between GAN and Recon loss')
    parser.add_argument('--default_rate', type=float, default=0.5, 
                        help='Lambda weight between GAN and Recon loss after curriculum')
    
    # CelebA 관련 인수
    parser.add_argument('--style_A', type=str, default=None, 
                        help='Style A for CelebA (e.g., Male, Young)')
    parser.add_argument('--style_B', type=str, default=None, 
                        help='Style B for CelebA')
    parser.add_argument('--constraint', type=str, default=None, 
                        help='Constraint for CelebA')
    parser.add_argument('--constraint_type', type=str, default=None, 
                        help='Constraint type (1/-1) for CelebA')
    
    # 기타 인수
    parser.add_argument('--n_test', type=int, default=200, 
                        help='Number of test images')
    parser.add_argument('--update_interval', type=int, default=3, 
                        help='Interval for discriminator updates')
    parser.add_argument('--log_interval', type=int, default=50, 
                        help='Print loss interval')
    parser.add_argument('--image_save_interval', type=int, default=1000, 
                        help='Image save interval')
    parser.add_argument('--model_save_interval', type=int, default=10000, 
                        help='Model save interval')
    
    return parser.parse_args()

def get_data(args):
    """지정된 작업에 대한 데이터셋을 가져옵니다."""
    if args.task_name == 'facescrub':
        data_A, data_B = get_facescrub_files(test=False, n_test=args.n_test)
        test_A, test_B = get_facescrub_files(test=True, n_test=args.n_test)
    
    elif args.task_name == 'celebA':
        data_A, data_B = get_celebA_files(
            style_A=args.style_A, style_B=args.style_B,
            constraint=args.constraint, constraint_type=args.constraint_type,
            test=False, n_test=args.n_test
        )
        test_A, test_B = get_celebA_files(
            style_A=args.style_A, style_B=args.style_B,
            constraint=args.constraint, constraint_type=args.constraint_type,
            test=True, n_test=args.n_test
        )
    
    elif args.task_name == 'edges2shoes':
        data_A, data_B = get_edge2photo_files(item='edges2shoes', test=False)
        test_A, test_B = get_edge2photo_files(item='edges2shoes', test=True)
    
    elif args.task_name == 'edges2handbags':
        data_A, data_B = get_edge2photo_files(item='edges2handbags', test=False)
        test_A, test_B = get_edge2photo_files(item='edges2handbags', test=True)
    
    elif args.task_name == 'handbags2shoes':
        data_A_1, data_A_2 = get_edge2photo_files(item='edges2handbags', test=False)
        test_A_1, test_A_2 = get_edge2photo_files(item='edges2handbags', test=True)
        
        data_A = np.hstack([data_A_1, data_A_2])
        test_A = np.hstack([test_A_1, test_A_2])
        
        data_B_1, data_B_2 = get_edge2photo_files(item='edges2shoes', test=False)
        test_B_1, test_B_2 = get_edge2photo_files(item='edges2shoes', test=True)
        
        data_B = np.hstack([data_B_1, data_B_2])
        test_B = np.hstack([test_B_1, test_B_2])

    elif args.task_name == 'tops2hanbok' or args.task_name == 'hanbok2tops':
        # 데이터셋 순서 결정
        if args.task_name == 'tops2hanbok':
            item_a, item_b = 'tops', 'hanbok'
        else:
            item_a, item_b = 'hanbok', 'tops'
            
        data_A, data_B = get_custom_data(item_a=item_a, item_b=item_b, 
                                         test=False, image_size=args.image_size)
        test_A, test_B = get_custom_data(item_a=item_a, item_b=item_b, 
                                         test=True, image_size=args.image_size)
    
    return data_A, data_B, test_A, test_B

def get_fm_loss(real_feats, fake_feats, criterion, device):
    """Feature matching 손실을 계산합니다."""
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion(l2, torch.ones(l2.size()).to(device))
        losses += loss
    
    return losses

def get_gan_loss(dis_real, dis_fake, criterion, device):
    """GAN 손실을 계산합니다."""
    batch_size = dis_real.size(0)
    
    # 판별자 출력 텐서 크기 조정 (필요한 경우)
    if len(dis_real.size()) > 2:  # [batch, 1, 1, 1] 형태인 경우
        dis_real = dis_real.view(batch_size, -1)  # [batch, 1]로 변환
    if len(dis_fake.size()) > 2:  # [batch, 1, 1, 1] 형태인 경우
        dis_fake = dis_fake.view(batch_size, -1)  # [batch, 1]로 변환
    
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
    
    # 디바이스 설정 (더 자세한 정보 출력)
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
        if args.device == 'cuda':
            print("CUDA 사용이 요청되었지만, 사용할 수 없습니다. CUDA 설치를 확인하세요.")
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")
    
    # 경로 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = Path(args.results_dir) / args.task_name
    if args.style_A:
        result_path = result_path / args.style_A
    result_path = result_path / args.model_arch / timestamp
    
    model_path = Path(args.models_dir) / args.task_name
    if args.style_A:
        model_path = model_path / args.style_A
    model_path = model_path / args.model_arch / timestamp
    
    # 데이터 가져오기
    data_A, data_B, test_A, test_B = get_data(args)
    
    # 테스트 데이터 전처리
    if args.task_name.startswith('edges2'):
        test_A_processed = read_images(test_A, 'A', args.image_size)
        test_B_processed = read_images(test_B, 'B', args.image_size)
    elif args.task_name in ['handbags2shoes', 'shoes2handbags']:
        test_A_processed = read_images(test_A, 'B', args.image_size)
        test_B_processed = read_images(test_B, 'B', args.image_size)
    else:
        test_A_processed = read_images(test_A, None, args.image_size)
        test_B_processed = read_images(test_B, None, args.image_size)
    
    # 텐서로 변환
    test_A_tensor = torch.FloatTensor(test_A_processed).to(device)
    test_B_tensor = torch.FloatTensor(test_B_processed).to(device)
    
    # 디렉토리 생성
    result_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # 모델 초기화
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
    data_size = min(len(data_A), len(data_B))
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
        data_A_shuffled, data_B_shuffled = shuffle_data(data_A, data_B)
        
        # 에포크 진행률 막대
        progress_bar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i in progress_bar:
            # 배치 데이터 가져오기
            A_paths = data_A_shuffled[i * args.batch_size: (i + 1) * args.batch_size]
            B_paths = data_B_shuffled[i * args.batch_size: (i + 1) * args.batch_size]
            
            # 이미지 로드 및 전처리
            if args.task_name.startswith('edges2'):
                A = read_images(A_paths, 'A', args.image_size)
                B = read_images(B_paths, 'B', args.image_size)
            elif args.task_name in ['handbags2shoes', 'shoes2handbags']:
                A = read_images(A_paths, 'B', args.image_size)
                B = read_images(B_paths, 'B', args.image_size)
            else:
                A = read_images(A_paths, None, args.image_size)
                B = read_images(B_paths, None, args.image_size)
            
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
                              f"FM: {fm_loss_A.item():.4f}/{fm_loss_B.item():.4f}, "
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
                save_sample_images(
                    test_A_tensor, test_B_tensor,
                    generator_A, generator_B,
                    result_path / 'samples',
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