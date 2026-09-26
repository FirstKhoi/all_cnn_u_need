"""
BÀI 8 — ConvNeXt. Liu et al., 2022, "A ConvNet for the 2020s".

Câu hỏi của paper: Vision Transformer thắng CNN nhờ attention, hay nhờ các lựa chọn thiết kế
và công thức huấn luyện đi kèm? → Hiện đại hoá ResNet-50 từng bước theo Swin Transformer,
KHÔNG dùng attention, vẫn ngang/vượt Swin. Mỗi bước là 1 cải tiến đo được:
  1. Recipe huấn luyện hiện đại: AdamW, 300 epoch, Mixup/CutMix, RandAugment, label smoothing.
  2. Tỉ lệ stage (3,3,9,3) thay (3,4,6,3).
  3. "Patchify" stem: conv 4x4 stride 4 (giống patch embedding của ViT).
  4. Depthwise conv (từ MobileNet) + tăng width.
  5. Inverted bottleneck: hẹp → rộng 4x → hẹp (giống MLP của Transformer, MobileNetV2).
  6. Kernel lớn 7x7 depthwise, đặt ở đầu block (giống window attention của Swin).
  7. Micro: GELU thay ReLU, ít activation/norm hơn, LayerNorm thay BN,
     downsample bằng layer riêng giữa các stage.

ConvNeXt block (dim = C):
  x → dwconv 7x7 (p3, groups=C) → LayerNorm → Linear C→4C → GELU → Linear 4C→C → × gamma → + x
  - Linear trên channel = conv 1x1. Làm ở NHWC: x.permute(0, 2, 3, 1), dùng nn.LayerNorm(C)
    và nn.Linear, xong permute(0, 3, 1, 2) lại.
  - gamma = layer scale: nn.Parameter(1e-6 * torch.ones(C)).
  - Stochastic depth (drop path): optional, bỏ qua được.
Downsample giữa các stage: LayerNorm (theo channel) → conv 2x2 s2.
  LayerNorm trên tensor NCHW: tự viết LayerNorm2d (permute → nn.LayerNorm → permute).

ConvNeXt-T gốc: stem 4x4 s4, depths (3,3,9,3), dims (96,192,384,768).
Bản CIFAR (khuyên): stem conv 2x2 s2 → LayerNorm2d (32 → 16), depths (2,2,6,2), dims (64,128,256,512)
  stage 16x16 → 8x8 → 4x4 → 2x2 → global avg pool → LayerNorm → fc num_classes.

Train: `python train.py --model convnext --opt adamw --epochs 100`.
Đừng ngạc nhiên nếu thua ResNet18 khi train ngắn / augmentation yếu: như ViT, ConvNeXt cần
recipe dài + augmentation mạnh. Đó chính là bài học số 1 của paper.
"""
import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm theo channel cho tensor (N, C, H, W)."""

    def __init__(self, dim):
        super().__init__()
        raise NotImplementedError("TODO")

    def forward(self, x):
        raise NotImplementedError("TODO")


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        raise NotImplementedError("TODO")

    def forward(self, x):
        raise NotImplementedError("TODO")


class ConvNeXt(nn.Module):
    def __init__(self, depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), num_classes=10):
        super().__init__()
        raise NotImplementedError("TODO")

    def forward(self, x):
        raise NotImplementedError("TODO")
