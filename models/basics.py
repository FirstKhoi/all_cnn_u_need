"""
BÀI 0 — CNN cơ bản. Tự viết convolution & pooling bằng vòng lặp để hiểu chúng làm gì.
Kiểm tra: `python check.py` (so với F.conv2d / F.max_pool2d của PyTorch).

Khái niệm cần nắm:
- Convolution: trượt kernel (C_in x k x k) trên ảnh; mỗi vị trí = tích vô hướng + bias.
  Có C_out kernel → C_out feature map.
- Kích thước output: out = floor((n + 2p - k) / s) + 1
- Local connectivity + weight sharing → cực ít tham số so với fully-connected.
  VD 32x32x3 → 32x32x16:  FC cần 3072*16384 ≈ 50M params,  conv 3x3 cần 3*3*3*16 + 16 = 448.
- Pooling: giảm H,W → tăng receptive field, bất biến với dịch chuyển nhỏ.
- Receptive field: vùng ảnh gốc mà 1 neuron "nhìn thấy". Càng sâu càng lớn.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def output_size(n, k, s=1, p=0):
    """Kích thước không gian sau conv/pool."""
    return (n + 2 * p - k) // s + 1
    raise NotImplementedError("TODO: 1 dòng")


def conv2d_naive(x, w, b=None, stride=1, padding=0):
    """x: (N, C_in, H, W), w: (C_out, C_in, k, k), b: (C_out,) → (N, C_out, H_out, W_out)

    Gợi ý:
      - F.pad(x, (padding,) * 4) để pad.
      - 2 vòng for qua (i, j) của output. Tại mỗi vị trí:
          patch = x[:, :, i*s : i*s+k, j*s : j*s+k]            # (N, C_in, k, k)
          out[:, :, i, j] = (patch[:, None] * w[None]).sum((2, 3, 4))   # broadcast qua N và C_out
    """
    N, C_in, H, W = x.shape
    C_out, _, k, k = w.shape
    
    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1
    
    x_pad = F.pad(x, (padding, padding, padding, padding))
    out = torch.zeros((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            patch = x_pad[:, :, h_start : h_start + k, w_start : w_start + k]
            out[:, :, i, j] = (patch[:, None] * w[None]).sum((2, 3, 4))
            
    if b is not None:
        out += b.view(1, -1, 1, 1)
        
    return out


def maxpool2d_naive(x, k=2, stride=2):
    """x: (N, C, H, W) → (N, C, H_out, W_out). Cùng khung vòng lặp như conv, thay tích bằng max."""
    N, C, H, W = x.shape
    
    H_out = (H - k) // stride + 1
    W_out = (W - k) // stride + 1
    
    out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            patch = x[:, :, h_start : h_start + k, w_start : w_start + k]
            
            out[:, :, i, j] = patch.amax(dim=(2, 3))
    return out

class SimpleCNN(nn.Module):
    """Baseline nhỏ nhất, dùng nn.Conv2d thật.

    conv 3→32, 3x3 p1 → ReLU → maxpool 2      (32 → 16)
    conv 32→64, 3x3 p1 → ReLU → maxpool 2     (16 → 8)
    flatten (64*8*8) → Linear → num_classes

    Kỳ vọng CIFAR-10: ~70%. Đây là mốc để so mọi model sau.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.outputs = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.outputs(x)
        return x
