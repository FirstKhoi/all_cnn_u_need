"""
BÀI 4 — VGG. Simonyan & Zisserman, 2014, "Very Deep Convolutional Networks for Large-Scale Image Recognition".
Á quân ILSVRC 2014 (top-5 7.3%).

Cải tiến: bỏ kernel lớn, CHỈ dùng conv 3x3 (s1 p1) + maxpool 2x2, xếp chồng thật sâu (11-19 layer).
  - 2 conv 3x3 = receptive field 5x5; 3 conv 3x3 = 7x7.
  - Nhưng ít params hơn: 3 * (3*3*C*C) = 27C²  vs  7*7*C*C = 49C².
  - Thêm ReLU giữa các lớp → phi tuyến hơn.
  - Thiết kế đồng nhất: mỗi lần pool thì nhân đôi channel. Dễ nhân bản, dễ hiểu.
Nhược: rất nặng (VGG16 ~138M params, phần lớn nằm ở FC 4096) và chậm.
Chuyện bên lề: không có BatchNorm, paper phải train VGG11 trước rồi dùng nó init VGG16.

Bản CIFAR: 32 → 5 lần pool → 512 x 1 x 1 → bỏ FC khổng lồ, chỉ cần nn.Linear(512, num_classes).
Mỗi số trong CFG = conv 3x3 với số channel đó (→ BN → ReLU), "M" = maxpool 2x2.

Gợi ý: viết hàm dựng self.features bằng vòng for qua CFG[cfg].
Thêm nn.BatchNorm2d sau mỗi conv (VGG-BN; BN ra đời 2015, sau VGG).

Thí nghiệm:
  - vgg11 vs vgg16: sâu hơn có tốt hơn không, đổi lại chậm bao nhiêu?
  - Bỏ BN khỏi vgg16 (--tag vgg16_nobn): có hội tụ nổi không với lr 0.05? → lý do BN tồn tại.
"""
import torch.nn as nn
import torch

CFG = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(self, cfg="vgg16", num_classes=10):
        super().__init__()
        # Dựng chuỗi feature extraction từ cấu hình CFG
        self.features = self._make_layers(CFG[cfg])
        
        # Sau 5 lần pool: 32 -> 16 -> 8 -> 4 -> 2 -> 1 (512 * 1 * 1)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg_list):
        layers = []
        in_channels = 3
        
        for v in cfg_list:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # (N, 512, 1, 1) -> (N, 512)
        x = self.classifier(x)
        return x
