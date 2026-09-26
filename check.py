"""Run after each implementation: python check.py   (add a name to check one model: python check.py resnet18)"""
import sys

import torch
import torch.nn.functional as F

from models import MODELS, basics


def check_basics():
    try:
        assert basics.output_size(32, 5) == 28
        assert basics.output_size(32, 3, 1, 1) == 32
        assert basics.output_size(32, 3, 2, 1) == 16
        x, w, b = torch.randn(2, 3, 9, 9), torch.randn(4, 3, 3, 3), torch.randn(4)
        for s, p in [(1, 0), (1, 1), (2, 1)]:
            assert torch.allclose(basics.conv2d_naive(x, w, b, s, p), F.conv2d(x, w, b, s, p), atol=1e-4), (s, p)
        assert torch.allclose(basics.maxpool2d_naive(x), F.max_pool2d(x, 2, 2))
        assert torch.allclose(basics.maxpool2d_naive(x, 3, 2), F.max_pool2d(x, 3, 2))
        print(f"{'basics':12} OK")
    except NotImplementedError:
        print(f"{'basics':12} TODO")


def check_model(name):
    try:
        model = MODELS[name]()
        out = model(torch.randn(2, 3, 32, 32))
    except NotImplementedError:
        print(f"{name:12} TODO")
        return
    assert out.shape == (2, 10), f"{name}: output {tuple(out.shape)}, expected (2, 10)"
    out.sum().backward()  # gradients must reach every parameter
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, f"{name}: no grad for {dead}"
    print(f"{name:12} OK  {sum(p.numel() for p in model.parameters()):>12,} params")


if __name__ == "__main__":
    names = sys.argv[1:] or ["basics", *MODELS]
    for n in names:
        check_basics() if n == "basics" else check_model(n)
