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
import torch
import torch.nn as nn

CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
       (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(                                      # 1 filter per channel, no mixing
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(                                      # 1x1 conv mixes channels
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobileNet(nn.Module):
    def __init__(self, num_classes=10, width=1.0):
        super().__init__()
        in_ch = int(32 * width)                                              # width multiplier α scales every layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, padding=1, bias=False),                   # (3, 32, 32) -> (32, 32, 32)
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        layers = []
        for out_ch, stride in CFG:                                           # 32 -> 16 -> 8 -> 4 -> 2
            out_ch = int(out_ch * width)
            layers.append(DepthwiseSeparable(in_ch, out_ch, stride))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)                               # -> (1024, 2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.outputs = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)                                    # (1024, 1, 1) -> 1024
        x = self.outputs(x)
        return x
