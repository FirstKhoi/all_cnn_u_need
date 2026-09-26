"""
BÀI 7 — MobileNet v1. Howard et al., 2017, "MobileNets: Efficient CNNs for Mobile Vision Applications".

Đổi mục tiêu: không phải accuracy cao nhất, mà accuracy / chi phí tốt nhất (điện thoại, embedded).

Cải tiến: depthwise separable convolution. Tách conv 3x3 chuẩn thành 2 bước:
  1. depthwise 3x3: mỗi channel một filter riêng, không trộn channel
       nn.Conv2d(C_in, C_in, 3, stride, padding=1, groups=C_in) → BN → ReLU
  2. pointwise 1x1: trộn channel
       nn.Conv2d(C_in, C_out, 1) → BN → ReLU
Chi phí (phép nhân, feature map H x W):
  chuẩn:  H*W * C_in * C_out * 9
  tách:   H*W * C_in * 9  +  H*W * C_in * C_out
  tỉ lệ = 1/C_out + 1/9  → rẻ hơn ~8-9x.
Siêu tham số: width multiplier α (nhân mọi số channel, VD 0.5), resolution multiplier ρ.

Kiến trúc (bản CIFAR: stem stride 1 thay vì 2):
  conv 3x3, 32 ch, s1 → BN → ReLU
  DS blocks (out_ch, stride):
    (64,1) (128,2) (128,1) (256,2) (256,1) (512,2) (512,1)x5 (1024,2) (1024,1)
    → 32 → 16 → 8 → 4 → 2
  AdaptiveAvgPool2d(1) → fc 1024 → num_classes

Thí nghiệm:
  - `visualize.py compare`: params và giây/epoch so với VGG16, ResNet18.
    Chú ý: ít FLOPs chưa chắc chạy nhanh hơn trên GPU/MPS (depthwise conv kém tối ưu). Vì sao?
  - Thử width=0.5 (--tag mobilenet_w05): mất bao nhiêu accuracy?
Mở rộng: MobileNetV2 = inverted residual + linear bottleneck (block của nó dẫn tới ConvNeXt).
"""
import torch.nn as nn

CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
       (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        raise NotImplementedError("TODO")

    def forward(self, x):
        raise NotImplementedError("TODO")


class MobileNet(nn.Module):
    def __init__(self, num_classes=10, width=1.0):
        super().__init__()
        raise NotImplementedError("TODO: nhân mọi số channel với width")

    def forward(self, x):
        raise NotImplementedError("TODO")
