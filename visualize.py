"""Visualize runs & models. Infra — đã viết sẵn. Ảnh lưu vào runs/.

    python visualize.py summary  resnet18       # shape + params từng layer (không cần train)
    python visualize.py curves                  # test acc / train loss của mọi run
    python visualize.py compare                 # accuracy vs #params, bảng tổng kết
    python visualize.py filters  alexnet        # filter conv đầu tiên (đẹp nhất với kernel lớn: AlexNet/ZFNet)
    python visualize.py features resnet18 --idx 7   # feature map qua các tầng conv (8 channel kích hoạt mạnh nhất)

`name` = tag của run (runs/<tag>.json). Chưa train → dùng weights random (so sánh random vs trained rất đáng xem).
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid

from data import CLASSES, MEAN, STD, test_set
from models import MODELS

RUNS = Path(os.environ.get("RUNS_DIR", "runs"))


def save(fig, name):
    RUNS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(RUNS / name, dpi=120)
    plt.close(fig)
    print("saved", RUNS / name)


def load_runs():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]


def load_model(name, trained=True):
    """Return (model, img_size). Loads trained weights if runs/<name>.pt exists (and trained=True)."""
    meta = RUNS / f"{name}.json"
    run = json.loads(meta.read_text()) if meta.exists() else {"model": name, "args": {}}
    model = MODELS[run["model"]]()
    weights = RUNS / f"{name}.pt"
    if trained and weights.exists():
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    elif trained:
        print(f"no {weights}, using random init")
    return model.eval(), run["args"].get("img_size", 32)


def summary(a):
    model, size = load_model(a.name)
    rows = []
    for n, m in model.named_modules():
        if not list(m.children()):  # leaf layers only
            m.register_forward_hook(lambda m, i, o, n=n: rows.append(
                (n, type(m).__name__, tuple(o.shape[1:]), sum(p.numel() for p in m.parameters()))))
    with torch.no_grad():
        model(torch.randn(1, 3, size, size))
    print(f"{'layer':32} {'type':18} {'output (C,H,W)':20} {'params':>12}")
    for n, t, shape, params in rows:
        print(f"{n:32} {t:18} {str(shape):20} {params:>12,}")
    print(f"total params: {sum(p.numel() for p in model.parameters()):,}")


def curves(_):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in load_runs():
        epochs = range(1, len(r["test_acc"]) + 1)
        a1.plot(epochs, r["test_acc"], label=r["tag"])
        a2.plot(epochs, r["train_loss"], label=r["tag"])
    a1.set(title="Test accuracy", xlabel="epoch")
    a2.set(title="Train loss", xlabel="epoch", yscale="log")
    a1.grid(alpha=.3), a2.grid(alpha=.3), a1.legend()
    save(fig, "curves.png")


def compare(_):
    runs = sorted(load_runs(), key=lambda r: max(r["test_acc"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"{'run':20} {'params':>12} {'best acc':>9} {'s/epoch':>8}")
    for r in runs:
        acc = max(r["test_acc"])
        ax.scatter(r["params"], acc)
        ax.annotate(r["tag"], (r["params"], acc), textcoords="offset points", xytext=(5, 5))
        print(f"{r['tag']:20} {r['params']:>12,} {acc:>9.4f} {sum(r['epoch_time']) / len(r['epoch_time']):>8.1f}")
    ax.set(xscale="log", xlabel="#params", ylabel="best test acc", title="Accuracy vs model size")
    ax.grid(alpha=.3)
    save(fig, "compare.png")


def filters(a):
    model, _ = load_model(a.name)
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.detach()[:64]
    lo, hi = w.amin((1, 2, 3), keepdim=True), w.amax((1, 2, 3), keepdim=True)
    grid = make_grid((w - lo) / (hi - lo + 1e-8), nrow=8, padding=1, pad_value=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid.permute(1, 2, 0))
    ax.set_title(f"{a.name}: first conv {tuple(conv.weight.shape)}")
    ax.axis("off")
    save(fig, f"{a.name}_filters.png")


def features(a):
    model, size = load_model(a.name)
    img, label = test_set(size)[a.idx]
    convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    picks = torch.linspace(0, len(convs) - 1, min(6, len(convs))).long().tolist()  # 6 layers, shallow -> deep
    convs = [convs[i] for i in picks]
    acts = {}
    for n, m in convs:
        m.register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o[0]))
    with torch.no_grad():
        pred = model(img[None]).argmax(1).item()

    fig, axes = plt.subplots(len(convs), 9, figsize=(14, 1.7 * len(convs)), squeeze=False)
    raw = img * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    axes[0][0].imshow(raw.permute(1, 2, 0).clamp(0, 1))
    for row, (n, _) in zip(axes, convs):
        fmap = acts[n]
        top = fmap.mean((1, 2)).topk(min(8, len(fmap))).indices  # channels firing hardest on this image
        row[1].set_title(f"{n} {tuple(fmap.shape)}", fontsize=7, loc="left")
        for ax, c in zip(row[1:], top.tolist()):
            ax.imshow(fmap[c], cmap="viridis")
        for ax in row:
            ax.axis("off")
    fig.suptitle(f"{a.name}: true={CLASSES[label]} pred={CLASSES[pred]}")
    save(fig, f"{a.name}_features.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    for fn in (summary, filters, features):
        s = sub.add_parser(fn.__name__)
        s.add_argument("name")
        s.set_defaults(fn=fn)
    sub.choices["features"].add_argument("--idx", type=int, default=0, help="test image index")
    for fn in (curves, compare):
        sub.add_parser(fn.__name__).set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)
