"""Look inside a CNN: what each layer outputs, Grad-CAM, the branches of one block, interventions.
Infra — đã viết sẵn, không phải phần cần học. Pure PyTorch; app.py is the UI on top of it.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

from data import MEAN, STD
from visualize import load_model

# block type -> [(child attribute, label)] in data-flow order; branches() opens these up
PARTS = {
    "BasicBlock": [("residual", "F(x): conv3×3 → BN → ReLU → conv3×3 → BN"), ("shortcut", "shortcut(x)")],
    "Inception": [("branch1", "nhánh 1: conv 1×1"), ("branch2", "nhánh 2: 1×1 → 3×3"),
                  ("branch3", "nhánh 3: 1×1 → 5×5"), ("branch4", "nhánh 4: maxpool 3×3 → 1×1")],
    "DepthwiseSeparable": [("depthwise", "depthwise 3×3: mỗi channel lọc riêng, không trộn"),
                           ("pointwise", "pointwise 1×1: trộn các channel")],
    "ConvNeXtBlock": [("dwconv", "depthwise 7×7"), ("norm", "LayerNorm"), ("pwconv1", "Linear C → 4C (mở rộng)"),
                      ("act", "GELU"), ("pwconv2", "Linear 4C → C (thu hẹp)")],
}
OUTPUT = {
    "BasicBlock": "output = ReLU(F(x) + shortcut(x))",
    "Inception": "output = nối 4 nhánh theo channel",
    "DepthwiseSeparable": "output",
    "ConvNeXtBlock": "output = x + γ·F(x)",
}
BLOCKS = tuple(PARTS) + ("LayerNorm2d",)                            # shown as a single step in the overview
POINTS = (nn.ReLU, nn.Tanh, nn.GELU, nn.MaxPool2d, nn.AvgPool2d)    # overview: after each nonlinearity / pool

_MEAN = torch.tensor(MEAN)[:, None, None]
_STD = torch.tensor(STD)[:, None, None]


def prepare(model):
    """eval mode + no in-place ReLU: an in-place ReLU overwrites the conv output a hook just saw
    and breaks the autograd graph Grad-CAM needs."""
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model.eval()


def load(name, trained=True):
    torch.manual_seed(0)  # the same "random" weights every time
    return prepare(load_model(name, trained)[0])


def to_input(img):
    """(3,H,W) in [0,1] -> normalized (1,3,H,W) batch."""
    return ((img - _MEAN) / _STD)[None]


def denorm(x):
    """Normalized (3,H,W) -> (3,H,W) in [0,1]."""
    return (x * _STD + _MEAN).clamp(0, 1)


def _chw(m, out):
    # Linear / LayerNorm / GELU only meet 4D tensors inside ConvNeXt, where they run channels-last
    if isinstance(m, (nn.Linear, nn.LayerNorm, nn.GELU)) and out.dim() == 4:
        return out.permute(0, 3, 1, 2)
    return out


def trace(model, x, detail="overview"):
    """Run x through the model and record what the layers output, in execution order.
    detail="overview": one step per block / nonlinearity / pool; "all": every leaf layer.
    Steps whose map is 1x1 are dropped. Returns (logits, steps); step = dict(name, type, act (C,H,W), params)."""
    mods = dict(model.named_modules())
    in_block = {f"{n}.{c}" for n, m in mods.items() if type(m).__name__ in BLOCKS for c, _ in m.named_modules() if c}

    def keep(n, m):
        if detail == "all":
            return not list(m.children()) and not isinstance(m, (nn.Identity, nn.Dropout))
        return n not in in_block and (type(m).__name__ in BLOCKS or isinstance(m, POINTS))

    steps = []

    def hook(m, out, n):
        a = _chw(m, out)
        if a.dim() == 4 and a.shape[2] * a.shape[3] > 1:
            steps.append(dict(name=n, type=type(m).__name__, act=a[0].detach().clone(),
                              params=sum(p.numel() for p in m.parameters())))

    handles = [m.register_forward_hook(lambda m, i, o, n=n: hook(m, o, n)) for n, m in mods.items() if n and keep(n, m)]
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        for h in handles:
            h.remove()
    return logits, steps


def gradcam(model, x, layer):
    """Grad-CAM (Selvaraju et al., 2017) at `layer`: weight each channel by its mean gradient for a class
    score, sum over channels, keep the positive part. One map per class.
    Returns (cams (num_classes, H, W) in [0,1] at x's resolution, logits)."""
    m = dict(model.named_modules())[layer]
    got = {}
    h = m.register_forward_hook(lambda m, i, o: got.__setitem__("a", o))
    try:
        with torch.enable_grad():
            logits = model(x)
            a = got["a"]
            cams = []
            for c in range(logits.shape[1]):
                g = torch.autograd.grad(logits[0, c], a, retain_graph=True, allow_unused=True)[0] if a.requires_grad else None
                if g is None:  # layer switched off by an intervention: nothing flows through it
                    cams.append(torch.zeros(_chw(m, a).shape[2:]))
                    continue
                act, g = _chw(m, a), _chw(m, g)
                cams.append(F.relu((g.mean((2, 3), keepdim=True) * act).sum(1))[0])
    finally:
        h.remove()
    cams = F.interpolate(torch.stack(cams)[:, None].detach(), size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cams / cams.flatten(1).amax(1).clamp_min(1e-8)[:, None, None], logits.detach()


def blocks(model, size=32):
    """Every multi-branch block in execution order: dict(name, type, skippable, shortcut).
    skippable = input and output have the same shape; shortcut = a ResNet block whose shortcut is on."""
    info = []

    def hook(m, i, o, n):
        info.append(dict(name=n, type=type(m).__name__, skippable=i[0].shape == o.shape,
                         shortcut=getattr(m, "use_shortcut", False)))

    handles = [m.register_forward_hook(lambda m, i, o, n=n: hook(m, i, o, n))
               for n, m in model.named_modules() if type(m).__name__ in PARTS]
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, size, size))
    finally:
        for h in handles:
            h.remove()
    return info


def branches(model, x, block):
    """Open one block: its input, every intermediate tensor, its output. Returns [(label, act (C,H,W))]."""
    b = dict(model.named_modules())[block]
    kind = type(b).__name__
    got = {}
    handles = [getattr(b, a).register_forward_hook(lambda m, i, o, a=a: got.__setitem__(a, _chw(m, o)[0].detach().clone()))
               for a, _ in PARTS[kind]]
    handles.append(b.register_forward_hook(lambda m, i, o: got.update(input=i[0][0].clone(), output=o[0].clone())))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    out = [("input x", got["input"])] + [(label, got[a]) for a, label in PARTS[kind] if a in got]
    final = OUTPUT[kind]
    if kind == "BasicBlock" and "shortcut" in got:
        out.append(("F(x) + shortcut(x)", got["residual"] + got["shortcut"]))
    elif kind == "BasicBlock":  # plain nets (or a switched-off shortcut) never call the shortcut
        final = "output = ReLU(F(x)), không có shortcut"
    if kind == "ConvNeXtBlock":
        out.append(("× γ (layer scale)", b.gamma.detach()[:, None, None] * got["pwconv2"]))
    return out + [(final, got["output"])]


@contextlib.contextmanager
def intervene(model, skip=(), drop=(), no_shortcut=(), zero=None):
    """Temporarily edit a model; everything is restored on exit.
    skip: blocks whose output is replaced by their input.  drop: modules whose output becomes 0.
    no_shortcut: ResNet blocks that compute ReLU(F(x)) without + x.  zero: (layer, [channels]) set to 0."""
    mods = dict(model.named_modules())
    handles = [mods[n].register_forward_hook(lambda m, i, o: i[0]) for n in skip]
    handles += [mods[n].register_forward_hook(lambda m, i, o: torch.zeros_like(o)) for n in drop]
    if zero:
        layer, chans = zero

        def zero_hook(m, i, o):
            o = o.clone()
            _chw(m, o)[:, chans] = 0  # permute returns a view, so this writes into o
            return o

        handles.append(mods[layer].register_forward_hook(zero_hook))
    saved = {n: mods[n].use_shortcut for n in no_shortcut}
    for n in saved:
        mods[n].use_shortcut = False
    try:
        yield model
    finally:
        for h in handles:
            h.remove()
        for n, v in saved.items():
            mods[n].use_shortcut = v


def adjust_image(img, brightness=1.0, contrast=1.0, noise=0.0, flip=False, angle=0, occlude=None):
    """Edit a (3,H,W) image in [0,1]. occlude = (row, col, size) gray square.
    Noise is seeded, so the same settings always give the same image."""
    x = img
    if flip:
        x = x.flip(-1)
    if angle:
        x = rotate(x, angle)
    if contrast != 1:
        x = (x - x.mean()) * contrast + x.mean()
    if brightness != 1:
        x = x * brightness
    if noise:
        x = x + noise * torch.randn(x.shape, generator=torch.Generator().manual_seed(0))
    if occlude:
        r, c, s = occlude
        x = x.clone()
        x[:, r:r + s, c:c + s] = 0.5
    return x.clamp(0, 1)
