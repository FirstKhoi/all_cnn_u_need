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
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, shortcut=True):
        super().__init__()
        self.residual = nn.Sequential(                                       # F(x)
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_shortcut = shortcut
        self.shortcut = nn.Identity()
        if shortcut and (stride != 1 or in_ch != out_ch):
            # projection: x has a different shape than F(x), so match it with a 1x1 conv
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.residual(x)
        if self.use_shortcut:
            out = out + self.shortcut(x)                                     # F(x) + x
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, shortcut=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),                      # (3, 32, 32) -> (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, num_blocks[0], 1, shortcut)     # -> (64, 32, 32)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], 2, shortcut)    # -> (128, 16, 16)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], 2, shortcut)   # -> (256, 8, 8)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], 2, shortcut)   # -> (512, 4, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)                                   # -> (512, 1, 1)
        self.outputs = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, n, stride, shortcut):
        # only the first block of a layer downsamples and changes channels
        blocks = [BasicBlock(in_ch, out_ch, stride, shortcut)]
        blocks += [BasicBlock(out_ch, out_ch, 1, shortcut) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)                                    # (512, 1, 1) -> 512
        x = self.outputs(x)
        return x
