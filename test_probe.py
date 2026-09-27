"""Self-check for probe.py on every model (random weights) + a headless smoke run of app.py.

    python test_probe.py
"""
import torch

import probe
from models import MODELS
from visualize import RUNS

x = torch.randn(1, 3, 32, 32)
for key in MODELS:
    torch.manual_seed(0)
    model = probe.prepare(MODELS[key]())
    base, steps = probe.trace(model, x)
    assert base.shape == (1, 10) and steps, key
    _, all_steps = probe.trace(model, x, "all")
    assert len(all_steps) >= len(steps), key
    for s in steps + all_steps:
        C, H, W = s["act"].shape
        assert H * W > 1, (key, s["name"])

    layer = [s for s in steps if s["act"].shape[1] >= 4][-1]["name"]
    cams, logits = probe.gradcam(model, x, layer)
    assert cams.shape == (10, 32, 32) and not cams.isnan().any() and 0 <= cams.min() and cams.max() <= 1, key
    assert torch.allclose(logits, base, atol=1e-5), key

    info = probe.blocks(model)
    for b in info:
        parts = probe.branches(model, x, b["name"])
        assert parts[0][0] == "input x" and all(a.dim() == 3 for _, a in parts), (key, b["name"])

    # every intervention changes the output and is fully undone on exit.
    # Random plain34 is the exception that proves ResNet's point: its signal fades to ~1e-13 over
    # 16 blocks, so the logits ignore the input and no edit can change them.
    sensitive = not torch.equal(probe.trace(model, torch.randn(1, 3, 32, 32))[0], base)
    edits = [dict(zero=(all_steps[0]["name"], list(range(len(all_steps[0]["act"])))))]
    edits += [dict(skip=[b["name"]]) for b in info if b["skippable"]][:1]
    edits += [dict(no_shortcut=[b["name"] for b in info if b["shortcut"]])] if any(b["shortcut"] for b in info) else []
    edits += [dict(drop=[f"{b['name']}.branch1"]) for b in info if b["type"] == "Inception"][:1]
    for edit in edits:
        with probe.intervene(model, **edit):
            changed = not torch.equal(probe.trace(model, x)[0], base)  # tiny for random ConvNeXt (γ = 1e-6)
            assert changed or not sensitive, (key, edit)
            probe.gradcam(model, x, layer)
        assert torch.equal(probe.trace(model, x)[0], base), (key, edit)
    print(f"{key:14} ok  {len(steps):3} overview / {len(all_steps):3} layers, {len(info):2} blocks, {len(edits)} edits")

img = torch.rand(3, 32, 32)
assert torch.equal(probe.adjust_image(img), img)
assert (probe.adjust_image(img, occlude=(4, 4, 8))[:, 4:12, 4:12] == 0.5).all()
assert torch.allclose(probe.denorm(probe.to_input(img)[0]), img, atol=1e-6)

if list(RUNS.glob("*.pt")):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=300).run()
    assert not at.exception, at.exception
    for mode in ("2 model khác nhau", "Trained vs random", "Gốc vs đã chỉnh"):
        at.radio(key="mode").set_value(mode).run()
        assert not at.exception, (mode, at.exception)
    at.radio(key="detail").set_value("Mọi layer").run()
    assert not at.exception, at.exception
    for name in ("googlenet", "mobilenet", "convnext", "lenet", "vgg16"):
        at.selectbox(key="model_a").set_value(name).run()
        assert not at.exception, (name, at.exception)
    print("app.py ok")
print("all checks passed")
