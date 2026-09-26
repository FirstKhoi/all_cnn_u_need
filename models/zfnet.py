"""
BÀI 3 — ZFNet. Zeiler & Fergus, 2013, "Visualizing and Understanding Convolutional Networks".
Thắng ILSVRC 2013 (top-5 11.7%).

Đóng góp chính KHÔNG phải kiến trúc, mà là cách NHÌN vào bên trong mạng:
  - Dùng deconvnet chiếu activation ngược về không gian pixel → thấy mỗi layer học gì
    (layer 1: cạnh/màu, layer 2: góc/texture, layer sâu: bộ phận vật thể).
  - Nhờ visualize, phát hiện conv1 của AlexNet (11x11 s4) có nhiều filter "chết" và
    thiếu tần số trung bình; layer 2 bị aliasing do stride 4 quá lớn.
Sửa AlexNet:
  1. conv1: 11x11 s4 → 7x7 s2.
  2. conv2: thêm stride 2 để bù kích thước.
  3. (Bản lớn hơn trong paper) conv3,4,5: 384,384,256 → 512,1024,512.

Kiến trúc gốc (224x224):
  conv 96, 7x7 s2 → ReLU → maxpool 3 s2
  conv 256, 5x5 s2 → ReLU → maxpool 3 s2
  conv 384, 3x3 p1 → conv 384, 3x3 p1 → conv 256, 3x3 p1 → maxpool 3 s2
  fc 4096 → fc 4096 → fc num_classes (dropout như AlexNet)

Bản CIFAR: đối xứng với AlexNet-CIFAR, chỉ làm conv1 "nhẹ tay" hơn (kernel nhỏ, stride 1):
  conv 64, 3x3 s1 p1      → ReLU → maxpool 3 s2 p1    32 → 16
  conv 192, 5x5 p2        → ReLU → maxpool 3 s2 p1    16 → 8
  conv 384 → conv 384 → conv 256 (3x3 p1, ReLU)  → maxpool 3 s2 p1    8 → 4
  flatten 256*4*4 → dropout → fc 1024 → dropout → fc 1024 → fc num_classes

Thí nghiệm (đây là bài học của ZFNet):
  - `python visualize.py filters alexnet` vs `python visualize.py filters zfnet`.
  - `python visualize.py features zfnet --idx 3`: xem feature map nông → sâu.
  - `python visualize.py compare`: stride nhỏ ở conv1 đổi được bao nhiêu accuracy?
"""
import torch.nn as nn
import torch

class ZFNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # (3, 32, 32) -> (64, 32, 32)
          nn.ReLU(inplace=True),
          nn.MaxPool2d(3, stride=2, padding=1) # -> (64, 16, 16)
        )
        
        self.block2 = nn.Sequential(
          nn.Conv2d(64, 192, kernel_size=5, padding=2), # (64, 16, 16) -> (192, 16, 16)
          nn.ReLU(inplace=True),
          nn.MaxPool2d(3, stride=2, padding=1) # -> (192, 8, 8)
        )
        
        self.block3 = nn.Sequential(
          nn.Conv2d(192, 384, kernel_size=3, padding=1), # -> (384, 8, 8)
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 384, kernel_size=3, padding=1), # -> (384, 8, 8)
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=3, padding=1), # -> (256, 8, 8)
          nn.ReLU(inplace=True),
          nn.MaxPool2d(3, stride=2, padding=1) # -> (256, 4, 4)
        )
        
        self.outputs = nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(256 * 4 * 4, 1024),
          nn.ReLU(inplace=True),
          nn.Dropout(0.5),
          nn.Linear(1024, 1024),
          nn.ReLU(inplace=True),
          nn.Linear(1024, num_classes)
        )

    def forward(self, x):
      x = self.block1(x)
      x = self.block2(x)
      x = self.block3(x)
      x = torch.flatten(x, start_dim=1) # (256, 4, 4) -> 4096
      x = self.outputs(x)
      return x
