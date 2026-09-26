"""
BÀI 2 — AlexNet. Krizhevsky, Sutskever, Hinton, 2012, "ImageNet Classification with Deep CNNs".
Thắng ILSVRC 2012: top-5 error 15.3% (hạng 2: 26.2%). Mở màn kỷ nguyên deep learning.

Cải tiến so với LeNet:
  1. ReLU thay tanh → không bão hoà gradient, hội tụ nhanh ~6x.
  2. Dropout 0.5 ở các lớp FC → chống overfit (60M params).
  3. Overlapping max pooling (k=3, s=2) thay avg pool.
  4. Sâu và rộng hơn nhiều: 5 conv + 3 FC.
  5. Data augmentation + train trên GPU.
  6. LRN (Local Response Normalization): sau này bị BatchNorm thay thế, bỏ qua được.

Kiến trúc gốc (input 224x224):
  conv 96, 11x11 s4       → ReLU → maxpool 3 s2
  conv 256, 5x5 p2        → ReLU → maxpool 3 s2
  conv 384, 3x3 p1        → ReLU
  conv 384, 3x3 p1        → ReLU
  conv 256, 3x3 p1        → ReLU → maxpool 3 s2       → 256 x 6 x 6
  dropout → fc 4096 → ReLU → dropout → fc 4096 → ReLU → fc num_classes

Bản CIFAR (khuyên dùng). 11x11 s4 trên ảnh 32x32 sẽ phá hết thông tin, nên thu nhỏ
nhưng giữ "tinh thần" conv1 kernel lớn + stride lớn:
  conv 64, 5x5 s2 p2      → ReLU → maxpool 3 s2 p1    32 → 16 → 8
  conv 192, 5x5 p2        → ReLU → maxpool 3 s2 p1    8 → 4
  conv 384, 3x3 p1 → ReLU → conv 256, 3x3 p1 → ReLU → conv 256, 3x3 p1 → ReLU
                                 → maxpool 3 s2 p1    4 → 2
  flatten 256*2*2 → dropout → fc 1024 → ReLU → dropout → fc 1024 → ReLU → fc num_classes
Muốn bản gốc: viết đúng kiến trúc 224, thêm nn.AdaptiveAvgPool2d((6, 6)) trước FC,
rồi `python train.py --model alexnet --img-size 224` (chậm hơn nhiều).

Thí nghiệm:
  - Bỏ dropout (--tag alexnet_nodrop) → so khoảng cách train acc vs test acc.
  - ReLU → Tanh → so tốc độ giảm loss trong `visualize.py curves`.
  - `python visualize.py filters alexnet`: filter conv1 sau khi train có dạng cạnh/màu không?
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),  # (3, 32, 32) -> (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # -> (64, 8, 8)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),          # (64, 8, 8) -> (192, 8, 8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # -> (192, 4, 4)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),         # -> (384, 4, 4)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),         # -> (256, 4, 4)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),         # -> (256, 4, 4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),      # -> (256, 2, 2)
        )
        self.outputs = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)                          # (256, 2, 2) -> 1024
        x = self.outputs(x)
        return x
