"""
BÀI 6 — ResNet. He et al., 2015, "Deep Residual Learning for Image Recognition".
Thắng ILSVRC 2015 (top-5 3.57%) với 152 layer.

Vấn đề: DEGRADATION. Chồng thêm layer lên mạng plain → train error lại CAO hơn.
Đây không phải overfit (train error cũng tệ), mà là khó tối ưu.
Paper: plain 56-layer thua plain 20-layer trên CIFAR-10.

Cải tiến: học phần dư F(x) = H(x) - x thay vì học thẳng H(x):
    out = ReLU(F(x) + x)
  - Nếu identity là tối ưu, chỉ cần đẩy F về 0, dễ hơn nhiều so với học identity bằng chồng conv.
  - Gradient chảy thẳng qua shortcut → train được mạng rất sâu.
  - BatchNorm sau mỗi conv. Không dropout, GAP thay FC lớn (như GoogLeNet).

BasicBlock (ResNet-18/34):
  x → conv3x3(stride) → BN → ReLU → conv3x3 → BN → (+ shortcut(x)) → ReLU
  shortcut = identity; hoặc conv1x1(stride) + BN khi stride != 1 hay số channel đổi (projection).
  shortcut=False → bỏ phép cộng → thành mạng "plain" cùng độ sâu để so sánh.
Bottleneck (ResNet-50+): 1x1 giảm → 3x3 → 1x1 tăng x4. Không bắt buộc ở đây.

ResNet-18 cho CIFAR (num_blocks=(2,2,2,2)):
  stem: conv3x3 64 s1 p1 → BN → ReLU     (bản gốc: 7x7 s2 + maxpool, quá mạnh cho 32x32)
  layer1: 2 block, 64 ch,  stride 1      32x32
  layer2: 2 block, 128 ch, stride 2      16x16
  layer3: 2 block, 256 ch, stride 2      8x8
  layer4: 2 block, 512 ch, stride 2      4x4
  AdaptiveAvgPool2d(1) → fc 512 → num_classes
  (chỉ block ĐẦU mỗi layer có stride 2; các block sau stride 1)
ResNet-34: num_blocks=(3,4,6,3).

Thí nghiệm (tái hiện Fig. 4 của paper): train plain18, plain34, resnet18, resnet34.
Kỳ vọng: plain34 ≤ plain18 (degradation), resnet34 ≥ resnet18. Có BN nên hiệu ứng nhẹ hơn paper;
nhìn train loss trong `visualize.py curves` rõ hơn test acc.
"""
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, shortcut=True):
        super().__init__()
        raise NotImplementedError("TODO")

    def forward(self, x):
        raise NotImplementedError("TODO")


class ResNet(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, shortcut=True):
        super().__init__()
        raise NotImplementedError("TODO")

    def forward(self, x):
        raise NotImplementedError("TODO")
