import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io
from typing import List, Tuple, Optional, Dict, Union, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 경로 설정 - 환경에 맞게 조정 필요
dataset_path = Path('./datasets')
celebA_path = dataset_path / 'celebA'
handbag_path = dataset_path / 'edges2handbags'
shoe_path = dataset_path / 'edges2shoes'
facescrub_path = dataset_path / 'facescrub'
chair_path = dataset_path / 'rendered_chairs'
face_3d_path = dataset_path / 'PublicMM1' / '05_renderings'
face_real_path = dataset_path / 'real_face'
car_path = dataset_path / 'data' / 'cars'

def shuffle_data(da: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """두 데이터셋을 독립적으로 셔플합니다."""
    a_idx = np.arange(len(da))
    np.random.shuffle(a_idx)

    b_idx = np.arange(len(db))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[a_idx]
    shuffled_db = np.array(db)[b_idx]

    return shuffled_da, shuffled_db

def read_images(filenames: List[str], domain: Optional[str] = None, image_size: int = 64) -> np.ndarray:
    """이미지 파일 목록을 읽어 처리합니다."""
    images = []
    for fn in filenames:
        # OpenCV 대신 PIL 사용 (더 나은 호환성)
        try:
            image = Image.open(fn).convert('RGB')
        except Exception as e:
            print(f"이미지 로딩 실패: {fn}, 오류: {e}")
            continue

        # PIL 이미지를 numpy 배열로 변환
        image = np.array(image)
        
        # 도메인에 따른 이미지 처리
        if domain == 'A':
            kernel = np.ones((3, 3), np.uint8)
            image = image[:, :256, :]
            image = 255. - image
            image = cv2.dilate(image, kernel, iterations=1)
            image = 255. - image
        elif domain == 'B':
            image = image[:, 256:, :]

        # 이미지 크기 조정
        image = cv2.resize(image, (image_size, image_size))
        
        # 정규화 및 채널 재배치 (PyTorch 형식: C, H, W)
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(image)

    if not images:
        raise ValueError("유효한 이미지가 없습니다.")
        
    images_array = np.stack(images)
    return images_array

def read_attr_file(attr_path: str, image_dir: str) -> pd.DataFrame:
    """CelebA 속성 파일을 읽어 데이터프레임으로 변환합니다."""
    with open(attr_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        
    columns = ['image_path'] + lines[1].split()
    items = [line.split() for line in lines[2:]]
    
    df = pd.DataFrame(items, columns=columns)
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(image_dir, x))
    
    return df

def get_celebA_files(
    style_A: str, 
    style_B: Optional[str], 
    constraint: Optional[str], 
    constraint_type: Optional[str], 
    test: bool = False, 
    n_test: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """CelebA 데이터셋에서 지정된 스타일에 맞는 이미지 파일 경로를 가져옵니다."""
    attr_file = os.path.join(celebA_path, 'list_attr_celeba.txt')
    image_dir = os.path.join(celebA_path, 'img_align_celeba')
    image_data = read_attr_file(attr_file, image_dir)

    if constraint:
        image_data = image_data[image_data[constraint] == constraint_type]

    style_A_data = image_data[image_data[style_A] == '1']['image_path'].values
    
    if style_B:
        style_B_data = image_data[image_data[style_B] == '1']['image_path'].values
    else:
        style_B_data = image_data[image_data[style_A] == '-1']['image_path'].values

    if not test:
        return style_A_data[:-n_test], style_B_data[:-n_test]
    else:
        return style_A_data[-n_test:], style_B_data[-n_test:]
    
def get_edge2photo_files(item='edges2shoes', test=False):
    """edges2shoes 또는 edges2handbags 데이터셋의 파일 경로를 가져옵니다."""
    
    if item == 'edges2shoes':
        data_path = shoe_path
    elif item == 'edges2handbags':
        data_path = handbag_path
    else:
        raise ValueError(f'지원되지 않는 아이템: {item}')
    
    # 훈련 또는 테스트 데이터 경로 설정
    data_dir = 'test' if test else 'train'
    path = Path(data_path) / data_dir
    
    # 모든 이미지 파일 경로 가져오기
    if path.exists():
        files = sorted(list(path.glob('*.jpg')))
        files = [str(f) for f in files]
    else:
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")
    
    if not files:
        raise ValueError(f"'{path}' 경로에서 이미지 파일을 찾을 수 없습니다.")
    
    # A(edge)와 B(photo) 데이터 분리 (같은 파일을 반환, 나중에 로딩 시 처리)
    return files, files

def get_facescrub_files(test=False, n_test=200):
    """FaceScrub 데이터셋의 파일 경로를 가져옵니다."""
    
    # 데이터셋 경로 확인
    if not facescrub_path.exists():
        raise FileNotFoundError(f"FaceScrub 데이터셋 경로를 찾을 수 없습니다: {facescrub_path}")
    
    # 모든 이미지 파일 경로 가져오기 (재귀적으로 모든 하위 디렉토리 포함)
    all_images = []
    for ext in ['*.jpg', '*.png']:
        all_images.extend(list(facescrub_path.glob(f'**/{ext}')))
    
    all_images = sorted([str(img) for img in all_images])
    
    if len(all_images) == 0:
        raise ValueError(f"'{facescrub_path}' 경로에서 이미지 파일을 찾을 수 없습니다.")
    
    # 남성/여성 이미지 분리 (디렉토리 구조에 따라 조정 필요)
    # 예시: 폴더 이름에 'actors'와 'actresses'가 있다고 가정
    male_images = [img for img in all_images if 'actors' in img.lower()]
    female_images = [img for img in all_images if 'actresses' in img.lower()]
    
    # 테스트/학습 데이터 분리
    if test:
        return male_images[-n_test:], female_images[-n_test:]
    else:
        return male_images[:-n_test], female_images[:-n_test]

def get_custom_data(item_a='tops', item_b='hanbok', test=False, image_size=512):
    """커스텀 데이터셋을 로드합니다."""
    custom_path = dataset_path / 'custom'
    
    # 경로 설정
    if test:
        data_A_path = custom_path / item_a / 'test'
        data_B_path = custom_path / item_b / 'test'
    else:
        data_A_path = custom_path / item_a / 'train'
        data_B_path = custom_path / item_b / 'train'
    
    # 이미지 파일 경로 가져오기
    data_A = [str(f) for f in data_A_path.glob('*.jpg')] + [str(f) for f in data_A_path.glob('*.png')]
    data_B = [str(f) for f in data_B_path.glob('*.jpg')] + [str(f) for f in data_B_path.glob('*.png')]
    
    if not data_A or not data_B:
        raise ValueError(f"데이터셋을 찾을 수 없습니다: {data_A_path} 또는 {data_B_path}")
    
    print(f"데이터셋 로드 완료: A({len(data_A)}개), B({len(data_B)}개)")
    return np.array(data_A), np.array(data_B)

# DiscoGAN 데이터셋 클래스 추가 (PyTorch의 Dataset 상속)
class DiscoGANDataset(Dataset):
    def __init__(
        self, 
        domain_A_paths: List[str],
        domain_B_paths: List[str],
        domain_A_type: Optional[str] = None,
        domain_B_type: Optional[str] = None,
        image_size: int = 64,
        transform: Optional[transforms.Compose] = None
    ):
        self.domain_A_paths = domain_A_paths
        self.domain_B_paths = domain_B_paths
        self.domain_A_type = domain_A_type
        self.domain_B_type = domain_B_type
        self.image_size = image_size
        self.transform = transform
        self.length = min(len(domain_A_paths), len(domain_B_paths))
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 도메인 A 이미지 로드
        a_path = self.domain_A_paths[index % len(self.domain_A_paths)]
        a_img = self._load_and_process_image(a_path, self.domain_A_type)
        
        # 도메인 B 이미지 로드
        b_path = self.domain_B_paths[index % len(self.domain_B_paths)]
        b_img = self._load_and_process_image(b_path, self.domain_B_type)
        
        # 텐서로 변환
        a_tensor = torch.FloatTensor(a_img)
        b_tensor = torch.FloatTensor(b_img)
        
        # 추가 변환 적용 (옵션)
        if self.transform:
            a_tensor = self.transform(a_tensor)
            b_tensor = self.transform(b_tensor)
            
        return a_tensor, b_tensor
    
    def _load_and_process_image(self, img_path: str, domain_type: Optional[str]) -> np.ndarray:
        """단일 이미지를 로드하고 전처리합니다."""
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # 도메인 처리
            if domain_type == 'A':
                kernel = np.ones((3, 3), np.uint8)
                image = image[:, :256, :]
                image = 255. - image
                image = cv2.dilate(image, kernel, iterations=1)
                image = 255. - image
            elif domain_type == 'B':
                image = image[:, 256:, :]
            
            # 크기 조정 및 정규화
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.
            image = image.transpose(2, 0, 1)  # C, H, W 포맷으로 변환
            
            return image
            
        except Exception as e:
            print(f"이미지 로딩 실패: {img_path}, 오류: {e}")
            # 에러 발생 시 랜덤 이미지 반환
            return np.random.rand(3, self.image_size, self.image_size).astype(np.float32)
        

# 나머지 데이터 로딩 함수들...
# (get_edge2photo_files, get_facescrub_files, get_chairs, get_cars, get_faces_3d)
# 필요한 경우 이런 함수들도 현대화하여 업데이트할 수 있습니다.
