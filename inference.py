import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 자체 모듈 가져오기
from model import Generator

def parse_args():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN inference')
    
    # 기본 설정
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model weights')
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Path to input image or directory with images')
    parser.add_argument('--output_dir', type=str, default='./inference_results', 
                        help='Directory to save the inference results')
    parser.add_argument('--image_size', type=int, default=64, 
                        help='Image size')
    parser.add_argument('--direction', type=str, default='AtoB', choices=['AtoB', 'BtoA'],
                        help='Translation direction: AtoB or BtoA')
    parser.add_argument('--use_extra_layers', action='store_true',
                        help='Use extra layers in Generator (for angle pairing)')
    parser.add_argument('--dataset_type', type=str, default=None, 
                        choices=['edges2handbags', 'edges2shoes', 'handbags2shoes', 'celebA', None],
                        help='Dataset type for domain-specific preprocessing')
    parser.add_argument('--domain', type=str, default=None, choices=['A', 'B', None],
                        help='Domain for preprocessing (A: edge/sketch, B: real image)')
    
    return parser.parse_args()

def load_image(image_path, image_size=64, domain=None, dataset_type=None):
    """이미지를 로드하고 전처리합니다."""
    try:
        # PIL로 이미지 로드
        image = Image.open(image_path).convert('RGB')
        # numpy 배열로 변환
        image = np.array(image)
        
        # 데이터셋 유형에 따른 처리
        if dataset_type == 'edges2handbags' or dataset_type == 'edges2shoes':
            if domain == 'A':
                # 왼쪽 절반(윤곽선)
                kernel = np.ones((3, 3), np.uint8)
                image = image[:, :256, :]
                image = 255. - image
                image = cv2.dilate(image, kernel, iterations=1)
                image = 255. - image
            elif domain == 'B':
                # 오른쪽 절반(실제 이미지)
                image = image[:, 256:, :]
        
        # 이미지 크기 조정
        image = cv2.resize(image, (image_size, image_size))
        # 정규화 및 채널 재배치 (PyTorch 형식: C, H, W)
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        return image
    except Exception as e:
        print(f"이미지 로딩 실패: {image_path}, 오류: {e}")
        return None

def save_images(input_image, generated_image, reverse_generated_image, save_path):
    """입력, 생성, 재구성 이미지를 저장합니다."""
    # 이미지 형식 변환 (C,H,W -> H,W,C)
    input_np = input_image.cpu().numpy().transpose(1, 2, 0)
    generated_np = generated_image.cpu().numpy().transpose(1, 2, 0)
    if reverse_generated_image is not None:
        reverse_np = reverse_generated_image.cpu().numpy().transpose(1, 2, 0)
    
    # 값 범위 조정
    input_np = np.clip(input_np, 0, 1)
    generated_np = np.clip(generated_np, 0, 1)
    if reverse_generated_image is not None:
        reverse_np = np.clip(reverse_np, 0, 1)
    
    # 결과 시각화
    if reverse_generated_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(input_np)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        axes[1].imshow(generated_np)
        axes[1].set_title('Generated')
        axes[1].axis('off')
        
        axes[2].imshow(reverse_np)
        axes[2].set_title('Reconstructed')
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(input_np)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        axes[1].imshow(generated_np)
        axes[1].set_title('Generated')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 인수 파싱
    args = parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    if args.direction == 'AtoB':
        generator = Generator(extra_layers=args.use_extra_layers).to(device)
        model_path = Path(args.model_path) / 'gen_B_final.pth'  # A -> B 변환
    else:  # BtoA
        generator = Generator(extra_layers=args.use_extra_layers).to(device)
        model_path = Path(args.model_path) / 'gen_A_final.pth'  # B -> A 변환
    
    # 가중치 로드
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("다음 파일이 있는지 확인하세요:")
        try:
            for file in Path(args.model_path).glob('*.pth'):
                print(f" - {file}")
        except Exception:
            pass
        return
    
    # 평가 모드로 설정
    generator.eval()
    
    # 입력 경로가 디렉토리인지 파일인지 확인
    input_path = Path(args.input_path)
    if input_path.is_dir():
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    else:
        image_files = [input_path]
    
    # 각 이미지에 대해 추론 수행
    for img_file in image_files:
        print(f"Processing: {img_file}")
        
        # 이미지 로드
        img = load_image(img_file, args.image_size, args.domain, args.dataset_type)
        if img is None:
            continue
        
        # 텐서로 변환
        img_tensor = torch.FloatTensor(img).unsqueeze(0).to(device)
        
        # 추론 (생성)
        with torch.no_grad():
            generated = generator(img_tensor)
            
            # 선택적으로 A->B->A 또는 B->A->B 재구성을 위한 추론
            reverse_generator_path = None
            if args.direction == 'AtoB':
                reverse_generator_path = Path(args.model_path) / 'gen_A_final.pth'
            else:  # BtoA
                reverse_generator_path = Path(args.model_path) / 'gen_B_final.pth'
            
            # 역방향 생성기가 있는 경우 재구성 이미지도 생성
            reconstructed = None
            if reverse_generator_path.exists():
                reverse_generator = Generator(extra_layers=args.use_extra_layers).to(device)
                reverse_generator.load_state_dict(torch.load(reverse_generator_path, map_location=device))
                reverse_generator.eval()
                reconstructed = reverse_generator(generated)
        
        # 결과 저장
        output_filename = output_dir / f"{img_file.stem}_result.png"
        save_images(img_tensor[0], generated[0], reconstructed[0] if reconstructed is not None else None, output_filename)
        
        print(f"저장 완료: {output_filename}")
    
    print(f"모든 이미지 처리 완료. 결과는 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    main()
