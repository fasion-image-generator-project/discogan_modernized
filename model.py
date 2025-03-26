import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)              # 512 -> 256
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)         # 256 -> 128
        self.bn2 = nn.BatchNorm2d(64 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)     # 128 -> 64
        self.bn3 = nn.BatchNorm2d(64 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)     # 64 -> 32
        self.bn4 = nn.BatchNorm2d(64 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False)    # 32 -> 16
        self.bn5 = nn.BatchNorm2d(64 * 16)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv6 = nn.Conv2d(64 * 16, 64 * 32, 4, 2, 1, bias=False)   # 16 -> 8
        self.bn6 = nn.BatchNorm2d(64 * 32)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv7 = nn.Conv2d(64 * 32, 64 * 32, 4, 2, 1, bias=False)   # 8 -> 4
        self.bn7 = nn.BatchNorm2d(64 * 32)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)

        self.conv8 = nn.Conv2d(64 * 32, 1, 4, 1, 0, bias=False)         # 4 -> 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        relu1 = self.relu1(conv1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)

        conv4 = self.conv4(relu3)
        bn4 = self.bn4(conv4)
        relu4 = self.relu4(bn4)
        
        conv5 = self.conv5(relu4)
        bn5 = self.bn5(conv5)
        relu5 = self.relu5(bn5)
        
        conv6 = self.conv6(relu5)
        bn6 = self.bn6(conv6)
        relu6 = self.relu6(bn6)
        
        conv7 = self.conv7(relu6)
        bn7 = self.bn7(conv7)
        relu7 = self.relu7(bn7)

        conv8 = self.conv8(relu7)
        output = self.sigmoid(conv8)

        return output, [relu2, relu3, relu4, relu5, relu6, relu7]


class Generator(nn.Module):
    def __init__(self, extra_layers=False):
        super(Generator, self).__init__()

        # 기본 인코더 (extra_layers 옵션과 관계없이 항상 정의)
        if extra_layers:
            # 512x512 이미지를 위한 더 깊은 인코더
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),          # 512 -> 256
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),     # 256 -> 128
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False), # 128 -> 64
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False), # 64 -> 32
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False), # 32 -> 16
                nn.BatchNorm2d(64 * 16),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 16, 64 * 32, 4, 2, 1, bias=False), # 16 -> 8
                nn.BatchNorm2d(64 * 32),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 32, 64 * 32, 4, 2, 1, bias=False), # 8 -> 4
                nn.BatchNorm2d(64 * 32),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 32, 100, 4, 1, 0, bias=False),   # 4 -> 1
                nn.BatchNorm2d(100),
                nn.LeakyReLU(0.2, inplace=True),
            )
            
            # 512x512 이미지를 위한 더 깊은 디코더
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(100, 64 * 32, 4, 1, 0, bias=False),  # 1 -> 4
                nn.BatchNorm2d(64 * 32),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 32, 64 * 32, 4, 2, 1, bias=False), # 4 -> 8
                nn.BatchNorm2d(64 * 32),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 32, 64 * 16, 4, 2, 1, bias=False), # 8 -> 16
                nn.BatchNorm2d(64 * 16),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1, bias=False), # 16 -> 32
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False), # 32 -> 64
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False), # 64 -> 128
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),     # 128 -> 256
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),          # 256 -> 512
                nn.Sigmoid()
            )
        else:
            # 기본 인코더 - 표준 이미지 크기 (512x512)용
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),          # 512 -> 256
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),     # 256 -> 128
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False), # 128 -> 64
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False), # 64 -> 32
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False), # 32 -> 16
                nn.BatchNorm2d(64 * 16),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 16, 64 * 32, 4, 2, 1, bias=False), # 16 -> 8
                nn.BatchNorm2d(64 * 32),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 32, 64 * 32, 4, 2, 1, bias=False), # 8 -> 4
                nn.BatchNorm2d(64 * 32),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(64 * 32, 100, 4, 1, 0, bias=False),    # 4 -> 1
                nn.BatchNorm2d(100),
                nn.LeakyReLU(0.2, inplace=True),
            )
            
            # 기본 디코더 - 표준 이미지 크기 (512x512)용
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(100, 64 * 32, 4, 1, 0, bias=False),   # 1 -> 4
                nn.BatchNorm2d(64 * 32),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 32, 64 * 32, 4, 2, 1, bias=False), # 4 -> 8
                nn.BatchNorm2d(64 * 32),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 32, 64 * 16, 4, 2, 1, bias=False), # 8 -> 16
                nn.BatchNorm2d(64 * 16),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1, bias=False), # 16 -> 32
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False), # 32 -> 64
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False), # 64 -> 128
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),     # 128 -> 256
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),          # 256 -> 512
                nn.Sigmoid()
            )

        # 이전 버전과의 호환성을 위한 main 속성 (사용되지 않음)
        self.main = None

    def forward(self, input_tensor):
        if self.main is not None:
            # 이전 버전과의 호환성 유지
            return self.main(input_tensor)
        else:
            # 인코더-디코더 아키텍처 사용
            features = self.encoder(input_tensor)
            output = self.decoder(features)
            return output
