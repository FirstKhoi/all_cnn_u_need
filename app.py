"""CNN Explorer: đưa 1 ảnh qua model đã train, xem ảnh biến đổi ra sao qua từng layer.

    python app.py            (KHÔNG dùng `streamlit run app.py`, xem ghi chú ở dưới)

Infra — đã viết sẵn. Logic (hook, Grad-CAM, can thiệp) nằm ở probe.py; file này chỉ là giao diện.
Dùng các run có đủ runs/<tag>.json + runs/<tag>.pt.
"""
import sys
import threading

import streamlit as st

# torch must first be imported on the main thread. Imported inside Streamlit's script thread, it leaves
# thread-local Python objects that segfault the whole server when that thread exits (seen with torch 2.12
# on macOS). `python app.py` imports torch here on the main thread, then starts Streamlit in this process.
if "torch" not in sys.modules and threading.current_thread() is not threading.main_thread():
    st.error("Chạy app bằng `python app.py` (không dùng `streamlit run app.py`).")
    st.stop()

import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

import probe
from data import CLASSES, test_set
from models import MODELS
from visualize import RUNS

if __name__ == "__main__" and not st.runtime.exists():  # `python app.py`: start Streamlit on this script
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(cli.main())

st.set_page_config(page_title="CNN Explorer", page_icon="🔬", layout="wide")

EXPLAIN = {
    "BasicBlock": "Block ResNet: nhánh chính F(x) chỉ cần học **phần chênh lệch** so với x, còn shortcut đưa thẳng x "
                  "qua. Nhờ đường tắt này gradient chạy thẳng về các layer đầu, nên mạng 34 layer vẫn train tốt "
                  "(so với plain34). Model plain không có shortcut nên chỉ thấy F(x).",
    "Inception": "4 nhánh song song nhìn ảnh ở nhiều cỡ (1×1, 3×3, 5×5, pool) rồi nối lại theo channel. "
                 "Conv 1×1 đứng trước 3×3/5×5 để giảm số channel, đỡ tốn tính toán.",
    "DepthwiseSeparable": "Tách conv thường thành 2 bước: **depthwise** lọc từng channel riêng (không trộn), "
                          "**pointwise** 1×1 trộn các channel. Rẻ hơn khoảng 8–9 lần so với conv 3×3 thường.",
    "ConvNeXtBlock": "Giống block Transformer: depthwise 7×7 (trộn theo không gian) → LayerNorm → MLP mở rộng 4× "
                     "(trộn channel) → nhân γ (khởi tạo 1e-6) → cộng x.",
}
INCEPTION_BRANCH = {"1": "1×1", "2": "1×1→3×3", "3": "1×1→5×5", "4": "pool→1×1"}


# ---------- cached data & models ----------
@st.cache_resource
def get_model(name, trained=True):
    return probe.load(name, trained)


@st.cache_resource
def cifar():
    ds = test_set()
    return ds, [[i for i, t in enumerate(ds.targets) if t == c] for c in range(10)]


@st.cache_data
def runs():
    """Runs that have trained weights, in the order the models were invented."""
    order = list(MODELS)
    found = {p.stem: json.loads(p.read_text()) for p in RUNS.glob("*.json") if p.with_suffix(".pt").exists()}
    return dict(sorted(found.items(), key=lambda kv: (order.index(kv[1]["model"]), kv[0])))


@st.cache_data
def model_info(name):
    """(blocks, [(layer name, #channels)]) of a model."""
    model = get_model(name)
    _, steps = probe.trace(model, torch.zeros(1, 3, 32, 32), "all")
    return probe.blocks(model), [(s["name"], s["act"].shape[0]) for s in steps]


# ---------- drawing ----------
def colorize(a, cmap, size, lo=None, hi=None):
    """(H,W) map -> RGB tile. Nearest upscaling keeps every activation a crisp square."""
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    z = ((a - lo) / (hi - lo + 1e-8)).clamp(0, 1).numpy()
    return Image.fromarray((colormaps[cmap](z)[..., :3] * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


def rgb(img, size):
    return Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


def overlay(img, heat, size, alpha, cmap="jet"):
    """Heat map in [0,1] (at image resolution) blended over the image."""
    return Image.blend(rgb(img, size), colorize(heat, cmap, size, 0, 1), alpha)


def strip(tiles, gap=3):
    out = Image.new("RGB", (len(tiles) * (tiles[0].width + gap) - gap, tiles[0].height), "white")
    for k, t in enumerate(tiles):
        out.paste(t, (k * (t.width + gap), 0))
    return out


def order(act, opt):
    return act.mean((1, 2)).argsort(descending=True) if opt["sort"] == "Mạnh nhất" else torch.arange(len(act))


def show_act(act, opt):
    """Mean over channels, then opt['n'] channels, as one image strip."""
    chans = order(act, opt)[:opt["n"]].tolist()
    lo, hi = (act.min(), act.max()) if opt["scale"] == "Chung cả layer" else (None, None)
    tiles = [colorize(act.mean(0), opt["cmap"], opt["size"])]
    tiles += [colorize(act[c], opt["cmap"], opt["size"], lo, hi) for c in chans]
    st.image(strip(tiles))
    st.caption("TB (trung bình mọi channel) · " + " ".join(f"#{c}" for c in chans))


def step_label(s, i):
    C, H, W = s["act"].shape
    return f"{i + 1}. {s['name']} ({s['type']}, {C}×{H}×{W})"


# ---------- tabs ----------
def journey(v, steps, logits, opt):
    p = logits.softmax(1)[0]
    top = p.argmax().item()
    st.image(rgb(v["img"], 128), caption="Đầu vào 3×32×32")
    st.markdown(f"Dự đoán: **{CLASSES[top]}** ({p[top]:.0%})")
    st.bar_chart(pd.DataFrame({"xác suất": p.numpy()}, index=list(CLASSES)), horizontal=True, sort=False, height=260)
    for i, s in enumerate(steps):
        st.markdown(f"**{step_label(s, i)}** · {s['params']:,} params")
        show_act(s["act"], opt)


def inspect_layer(v, steps, opt, key):
    i = st.selectbox("Layer", range(len(steps)), index=len(steps) // 2,
                     format_func=lambda i: step_label(steps[i], i), key=f"layer{key}")
    act = steps[i]["act"]
    chans = order(act, opt).tolist()
    n = st.slider("Số channel hiện", 1, len(act), min(64, len(act)), key=f"n{key}_{i}")
    small = [colorize(act[c], opt["cmap"], 40) for c in chans[:n]]
    for r in range(0, n, 16):
        st.image(strip(small[r:r + 16]))
    st.caption(("Mạnh nhất trước" if opt["sort"] == "Mạnh nhất" else "Theo thứ tự") + ", trái → phải, trên → dưới: "
               + " ".join(f"#{c}" for c in chans[:n]))
    c = st.number_input("Phóng to channel #", 0, len(act) - 1, chans[0], key=f"c{key}_{i}")
    heat = F.interpolate(act[c][None, None], size=v["img"].shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    st.image([colorize(act[c], opt["cmap"], 224), overlay(v["img"], heat, 224, 0.5, opt["cmap"])],
             caption=[f"channel #{c}: {act[c].shape[0]}×{act[c].shape[1]}", "phủ lên ảnh gốc"])


def cam_tab(v, steps, opt, key):
    default = max((i for i, s in enumerate(steps) if s["act"].shape[1] >= 4), default=len(steps) - 1)
    i = st.selectbox("Layer", range(len(steps)), index=default,
                     format_func=lambda i: step_label(steps[i], i), key=f"camlayer{key}")
    cams, logits = probe.gradcam(v["model"], probe.to_input(v["img"]), steps[i]["name"])
    p = logits.softmax(1)[0]
    c = st.selectbox("Giải thích lớp", [-1] + list(range(10)), key=f"camcls{key}",
                     format_func=lambda c: "lớp model đoán" if c < 0 else f"{CLASSES[c]} ({p[c]:.0%})")
    c = p.argmax().item() if c < 0 else c
    alpha = st.slider("Độ đậm heatmap", 0.0, 1.0, 0.5, key=f"alpha{key}")
    st.image(overlay(v["img"], cams[c], 256, alpha), caption=f"Vùng làm tăng điểm của lớp '{CLASSES[c]}'")
    st.image([overlay(v["img"], cams[k], 48, alpha) for k in range(10)],
             caption=[f"{CLASSES[k]} {p[k]:.0%}" for k in range(10)])
    st.caption("Layer càng sâu thì càng hiểu nghĩa nhưng càng thô (ảnh 32×32 chỉ còn 4×4 hoặc 2×2 ở cuối).")


def branch_tab(v, opt, key):
    blocks = [b for b in model_info(v["name"])[0] if b["type"] in probe.PARTS]
    if not blocks:
        st.info("Model này chỉ xếp các layer nối tiếp, không có block nhiều nhánh. "
                "Thử resnet18, plain18, googlenet, mobilenet hoặc convnext.")
        return
    kind = {b["name"]: b["type"] for b in blocks}
    name = st.selectbox("Block", list(kind), format_func=lambda n: f"{n} ({kind[n]})", key=f"block{key}")
    st.markdown(EXPLAIN[kind[name]])
    for label, act in probe.branches(v["model"], probe.to_input(v["img"]), name):
        C, H, W = act.shape
        st.markdown(f"**{label}** · {C}×{H}×{W}")
        show_act(act, opt)


# ---------- sidebar ----------
R = runs()
if not R:
    st.error(f"Không có run nào đủ weights trong `{RUNS}/` (cần `<tag>.json` + `<tag>.pt`).")
    st.stop()
names = list(R)
fmt = lambda t: f"{t} · {max(R[t]['test_acc']):.1%}"
sb = st.sidebar

sb.header("🧠 Model")
name_a = sb.selectbox("Model", names, index=names.index("resnet18") if "resnet18" in names else 0,
                      format_func=fmt, key="model_a")
mode = sb.radio("So sánh", ["Không", "2 model khác nhau", "Trained vs random", "Gốc vs đã chỉnh"], key="mode")
if mode == "2 model khác nhau":
    name_b = sb.selectbox("Model B", names, index=names.index("plain18") if "plain18" in names else 0,
                          format_func=fmt, key="model_b")

sb.header("🖼️ Ảnh")
ds, by_class = cifar()
label = None
if sb.radio("Nguồn", ["CIFAR-10 test", "Upload"], horizontal=True, key="src") == "Upload":
    f = sb.file_uploader("Ảnh bất kỳ (tự cắt vuông, thu về 32×32)", type=["png", "jpg", "jpeg", "webp"])
    if f is None:
        st.info("Chọn 1 ảnh để upload ở thanh bên trái.")
        st.stop()
    img = to_tensor(ImageOps.fit(Image.open(f).convert("RGB"), (32, 32), Image.BICUBIC))
else:
    st.session_state.setdefault("cls", 0)
    st.session_state.setdefault("k", 0)

    def roll():
        st.session_state.cls = random.randrange(10)
        st.session_state.k = random.randrange(len(by_class[st.session_state.cls]))

    sb.button("🎲 Ảnh ngẫu nhiên", on_click=roll)
    cls = sb.selectbox("Lớp", range(10), format_func=CLASSES.__getitem__, key="cls")
    k = sb.slider("Ảnh thứ", 0, len(by_class[cls]) - 1, key="k")
    x0, label = ds[by_class[cls][k]]
    img = probe.denorm(x0)

with sb.expander("🎛️ Chỉnh ảnh đầu vào"):
    adj = dict(
        brightness=st.slider("Độ sáng", 0.2, 2.0, 1.0, 0.05),
        contrast=st.slider("Tương phản", 0.0, 2.0, 1.0, 0.05),
        noise=st.slider("Nhiễu (σ)", 0.0, 0.5, 0.0, 0.01),
        flip=st.checkbox("Lật ngang"),
        angle=st.slider("Xoay (độ)", -180, 180, 0, 5),
    )
    if st.checkbox("Che 1 ô vuông màu xám"):
        size = st.slider("Cỡ ô", 2, 16, 8)
        adj["occlude"] = (st.slider("Hàng", 0, 32 - size, 12), st.slider("Cột", 0, 32 - size, 12), size)

blocks, layers = model_info(name_a)
with sb.expander(f"🔧 Tắt/bật thành phần ({name_a})"):
    iv = {}
    shortcuts = [b["name"] for b in blocks if b["shortcut"]]
    if shortcuts:
        iv["no_shortcut"] = shortcuts if st.checkbox("Tắt MỌI shortcut") else st.multiselect("Tắt shortcut ở block", shortcuts)
    inception = [b["name"] for b in blocks if b["type"] == "Inception"]
    if inception:
        iv["drop"] = st.multiselect("Tắt nhánh Inception", [f"{n}.branch{k}" for n in inception for k in range(1, 5)],
                                    format_func=lambda s: f"{s[:-8]} · nhánh {s[-1]} ({INCEPTION_BRANCH[s[-1]]})")
    skippable = [b["name"] for b in blocks if b["skippable"]]
    if skippable:
        iv["skip"] = st.multiselect("Bỏ qua block (output = input)", skippable)
    layer = st.selectbox("Tắt channel ở layer", [None] + [n for n, _ in layers],
                         format_func=lambda n: "(không)" if n is None else n)
    if layer:
        C = dict(layers)[layer]
        iv["zero"] = (layer, st.multiselect(f"Channel cần tắt (0–{C - 1})", range(C)))
    iv = {k: v for k, v in iv.items() if v and (k != "zero" or v[1])}

with sb.expander("👁️ Hiển thị", expanded=True):
    opt = dict(
        detail=st.radio("Mức chi tiết", ["Tổng quan", "Mọi layer"], horizontal=True, key="detail"),
        n=st.slider("Số channel mỗi layer", 4, 32, 8),
        sort=st.radio("Chọn channel", ["Mạnh nhất", "Theo thứ tự"], horizontal=True),
        scale=st.radio("Thang màu", ["Riêng từng channel", "Chung cả layer"], horizontal=True),
        cmap=st.selectbox("Colormap", ["viridis", "magma", "gray", "RdBu_r"]),
        size=st.slider("Cỡ ô (px)", 32, 96, 56, 8),
    )

# ---------- views: one column per model being compared ----------
img_adj = probe.adjust_image(img, **adj)
edited = " · đã chỉnh" if iv or not torch.equal(img_adj, img) else ""
A = get_model(name_a)
if mode == "Không":
    views = [dict(title=name_a + edited, name=name_a, model=A, img=img_adj, iv=iv)]
elif mode == "2 model khác nhau":
    views = [dict(title=name_a + edited, name=name_a, model=A, img=img_adj, iv=iv),
             dict(title=name_b, name=name_b, model=get_model(name_b), img=img_adj, iv={})]
elif mode == "Trained vs random":
    views = [dict(title=f"{name_a} · đã train{edited}", name=name_a, model=A, img=img_adj, iv=iv),
             dict(title=f"{name_a} · weights ngẫu nhiên{edited}", name=name_a, model=get_model(name_a, False),
                  img=img_adj, iv=iv)]
else:
    views = [dict(title=f"{name_a} · gốc", name=name_a, model=A, img=img, iv={}),
             dict(title=f"{name_a} · đã chỉnh", name=name_a, model=A, img=img_adj, iv=iv)]

st.title("🔬 CNN Explorer")
st.caption("Đưa 1 ảnh qua model, xem nó biến đổi ra sao qua từng layer. Mọi nút chỉnh nằm ở thanh bên trái.")
if label is not None:
    st.markdown(f"Nhãn thật: **{CLASSES[label]}**")

tabs = st.tabs(["🛤️ Hành trình qua layer", "🔍 Soi 1 layer", "🔥 Grad-CAM", "🌿 Nhánh trong block"])
columns = [t.columns(len(views)) for t in tabs]
for key, (v, *cols) in enumerate(zip(views, *columns)):
    with probe.intervene(v["model"], **v["iv"]):
        logits, steps = probe.trace(v["model"], probe.to_input(v["img"]),
                                    "overview" if opt["detail"] == "Tổng quan" else "all")
        for col in cols:
            col.subheader(v["title"])
        with cols[0]:
            journey(v, steps, logits, opt)
        with cols[1]:
            inspect_layer(v, steps, opt, key)
        with cols[2]:
            cam_tab(v, steps, opt, key)
        with cols[3]:
            branch_tab(v, opt, key)
