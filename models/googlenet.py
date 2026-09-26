"""
BÀI 5 — GoogLeNet (Inception v1). Szegedy et al., 2014, "Going Deeper with Convolutions".
Thắng ILSVRC 2014 (top-5 6.7%), chỉ ~7M params (AlexNet 60M, VGG16 138M).

VGG đi sâu bằng cách chồng thật nhiều. GoogLeNet đi sâu VÀ rộng nhưng rẻ:
  1. Inception module: 4 nhánh song song, concat theo channel → mạng tự chọn scale phù hợp.
       nhánh 1: 1x1
       nhánh 2: 1x1 (giảm) → 3x3
       nhánh 3: 1x1 (giảm) → 5x5
       nhánh 4: maxpool 3x3 s1 p1 → 1x1
  2. Conv 1x1 làm bottleneck giảm channel trước 3x3/5x5 (ý tưởng từ Network-in-Network).
     VD: 192 → 32 channel bằng 5x5 trên 28x28:
       trực tiếp:  28*28 * 32 * 5*5 * 192           ≈ 120M phép nhân
       qua 1x1→16: 28*28*16*192 + 28*28*32*5*5*16   ≈ 12.4M   → rẻ hơn ~10x
  3. Global Average Pooling thay FC lớn → gần như không params ở classifier, ít overfit.
  4. Auxiliary classifiers giữa mạng để chống vanishing gradient. Bỏ qua ở đây (BN thay vai trò đó).

Bảng channel (paper, Table 1). Output module = c1 + c3 + c5 + pool_proj:
  name  c1   c3_reduce c3   c5_reduce c5   pool_proj   out
  3a    64   96        128  16        32   32          256
  3b    128  128       192  32        96   64          480
  --- maxpool 3 s2 p1 ---
  4a    192  96        208  16        48   64          512
  4b    160  112       224  24        64   64          512
  4c    128  128       256  24        64   64          512
  4d    112  144       288  32        64   64          528
  4e    256  160       320  32        128  128         832
  --- maxpool 3 s2 p1 ---
  5a    256  160       320  32        128  128         832
  5b    384  192       384  48        128  128         1024

Bản CIFAR: stem = conv 3x3 → 192 channel (thay 7x7 s2 + pool của bản gốc), không downsample sớm.
  stem (32x32) → 3a, 3b → pool (16x16) → 4a..4e → pool (8x8) → 5a, 5b
  → AdaptiveAvgPool2d(1) → flatten → dropout 0.4 → fc 1024 → num_classes
Gợi ý: mọi conv = Conv → BatchNorm → ReLU (thêm BN = Inception v2) để train từ đầu dễ hơn.
Chậm trên máy yếu → chia đôi toàn bộ channel.

Thí nghiệm: `python visualize.py summary googlenet` → đếm params so với VGG16.
"""
import torch
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch, kernel_size, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),  # BN already adds a bias
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Inception(nn.Module):
    def __init__(self, in_ch, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
        super().__init__()
        # 4 parallel branches; each keeps H x W so outputs can be concatenated on channels
        self.branch1 = conv_bn_relu(in_ch, c1, 1)
        self.branch2 = nn.Sequential(
            conv_bn_relu(in_ch, c3_reduce, 1),           # 1x1 bottleneck: cut channels first
            conv_bn_relu(c3_reduce, c3, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            conv_bn_relu(in_ch, c5_reduce, 1),           # 1x1 bottleneck
            conv_bn_relu(c5_reduce, c5, 5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_bn_relu(in_ch, pool_proj, 1),
        )

    def forward(self, x):
        # out channels = c1 + c3 + c5 + pool_proj
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = conv_bn_relu(3, 192, 3, padding=1)        # (3, 32, 32) -> (192, 32, 32)

        self.block3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),          # 3a -> (256, 32, 32)
            Inception(256, 128, 128, 192, 32, 96, 64),        # 3b -> (480, 32, 32)
            nn.MaxPool2d(3, stride=2, padding=1),             # -> (480, 16, 16)
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),         # 4a -> (512, 16, 16)
            Inception(512, 160, 112, 224, 24, 64, 64),        # 4b -> (512, 16, 16)
            Inception(512, 128, 128, 256, 24, 64, 64),        # 4c -> (512, 16, 16)
            Inception(512, 112, 144, 288, 32, 64, 64),        # 4d -> (528, 16, 16)
            Inception(528, 256, 160, 320, 32, 128, 128),      # 4e -> (832, 16, 16)
            nn.MaxPool2d(3, stride=2, padding=1),             # -> (832, 8, 8)
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),      # 5a -> (832, 8, 8)
            Inception(832, 384, 192, 384, 48, 128, 128),      # 5b -> (1024, 8, 8)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)                    # global average pool -> (1024, 1, 1)
        self.outputs = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)                     # (1024, 1, 1) -> 1024
        x = self.outputs(x)
        return x
