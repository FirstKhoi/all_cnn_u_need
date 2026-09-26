"""Train + evaluate one model on CIFAR-10. Infra — đã viết sẵn.

    python train.py --model lenet
    python train.py --model lenet --limit 256 --epochs 50 --bs 32     # sanity check: phải overfit ~100% train acc
    python train.py --model convnext --opt adamw --epochs 100
    python train.py --model alexnet --tag alexnet_no_dropout  # đặt tên riêng cho từng thí nghiệm

Kết quả: runs/<tag>.json (history) + runs/<tag>.pt (weights).
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import get_loaders
from models import MODELS

RUNS = Path(os.environ.get("RUNS_DIR", "runs"))  # Colab points this at Google Drive


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(model, loader, device, criterion, scaler, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # float16 mixed precision when --amp on CUDA; plain float32 otherwise
            with torch.autocast(device, dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(x)
                loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--tag", help="run name, default = model name")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--opt", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, help="default: 0.05 sgd / 1e-3 adamw")
    p.add_argument("--wd", type=float, help="default: 5e-4 sgd / 0.05 adamw")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--limit", type=int, help="train on first N samples only")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="float16 mixed precision, CUDA only (faster on Colab GPUs)")
    args = p.parse_args(argv)
    tag = args.tag or args.model
    sgd = args.opt == "sgd"
    lr = args.lr or (0.05 if sgd else 1e-3)
    wd = args.wd if args.wd is not None else (5e-4 if sgd else 0.05)

    device = get_device()
    torch.backends.cudnn.benchmark = True  # fixed input size -> let cuDNN pick the fastest kernels
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")
    model = MODELS[args.model]().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{tag}: {params:,} params on {device}")

    train_loader, test_loader = get_loaders(args.bs, args.img_size, args.limit, args.workers)
    criterion = nn.CrossEntropyLoss()
    if sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    RUNS.mkdir(exist_ok=True)
    hist = dict(tag=tag, model=args.model, params=params, args=vars(args),
                train_loss=[], train_acc=[], test_loss=[], test_acc=[], epoch_time=[])
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, scaler, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader, device, criterion, scaler)
        scheduler.step()
        dt = time.time() - t0
        for k, v in zip(("train_loss", "train_acc", "test_loss", "test_acc", "epoch_time"),
                        (tr_loss, tr_acc, te_loss, te_acc, dt)):
            hist[k].append(v)
        print(f"[{epoch:3d}/{args.epochs}] loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"test loss {te_loss:.3f} acc {te_acc:.3f} | {dt:.0f}s")
        # save every epoch so Ctrl+C keeps what was trained
        (RUNS / f"{tag}.json").write_text(json.dumps(hist, indent=1))
        torch.save(model.state_dict(), RUNS / f"{tag}.pt")
    print(f"best test acc: {max(hist['test_acc']):.4f}")


if __name__ == "__main__":
    main()
