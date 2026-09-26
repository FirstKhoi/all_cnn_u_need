"""
BÀI 1 — LeNet-5. LeCun et al., 1998, "Gradient-Based Learning Applied to Document Recognition".
Đọc số viết tay (MNIST, check ngân hàng). CNN đầu tiên được dùng thực tế.

Ý tưởng gốc (so với MLP thuần):
  1. Conv + weight sharing thay fully-connected → ít params, khai thác cấu trúc không gian.
  2. Xen kẽ conv (trích đặc trưng) và subsampling (giảm kích thước) → hierarchy đặc trưng.
  3. Train end-to-end bằng backprop.

Kiến trúc (input 32x32, khớp luôn CIFAR, chỉ đổi 1 → 3 channel):
  conv 6,  5x5 → tanh → avgpool 2     32 → 28 → 14
  conv 16, 5x5 → tanh → avgpool 2     14 → 10 → 5
  flatten 16*5*5 = 400
  fc 120 → tanh → fc 84 → tanh → fc num_classes

Thí nghiệm:
  - Thay tanh → ReLU, avgpool → maxpool (--tag lenet_relu). Nhanh hơn? Tốt hơn? Đó là bước AlexNet.
  - `python visualize.py summary lenet`: đếm xem params nằm ở conv hay fc nhiều hơn.
Kỳ vọng: ~60-65%. CIFAR khó hơn MNIST nhiều → model nhỏ underfit (train acc cũng thấp).
"""
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),    # (3, 32, 32) -> (6, 28, 28)
            nn.Tanh(),
            nn.MaxPool2d(2, 2),                # -> (6, 14, 14)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),   # (6, 14, 14) -> (16, 10, 10)
            nn.Tanh(),
            nn.MaxPool2d(2, 2),                # -> (16, 5, 5)
        )
        self.outputs = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)     # (16, 5, 5) -> 400
        x = self.outputs(x)
        return x
