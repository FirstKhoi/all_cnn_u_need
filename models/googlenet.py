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


class Inception(nn.Module):
    def __init__(self, in_ch, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
        super().__init__()
        
        raise NotImplementedError("TODO: 4 nhánh")

    def forward(self, x):
        raise NotImplementedError("TODO: torch.cat([...], dim=1)")


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        raise NotImplementedError("TODO")

    def forward(self, x):
        raise NotImplementedError("TODO")
