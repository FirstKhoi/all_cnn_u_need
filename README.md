# All CNN U Need

Tự implement lại các CNN kinh điển theo thứ tự thời gian, train trên CIFAR-10, visualize để thấy mỗi model cải tiến gì so với model trước.

## Cấu trúc

```
data.py        CIFAR-10 loaders                      (viết sẵn)
train.py       train + eval, lưu runs/<tag>.json|pt  (viết sẵn)
visualize.py   summary / curves / compare / filters / features  (viết sẵn)
check.py       kiểm tra shape + gradient mỗi model   (viết sẵn)
app.py         CNN Explorer: xem ảnh đi qua từng layer (viết sẵn, logic ở probe.py)
models/        ← PHẦN BẠN VIẾT. Mỗi file: docstring = bài học + kiến trúc + thí nghiệm
```

## Lộ trình

| # | File | Model | Cải tiến chính | Thí nghiệm để "thấy" cải tiến |
|---|------|-------|----------------|-------------------------------|
| 0 | `basics.py` | Conv/Pool tự viết, SimpleCNN | weight sharing, local connectivity | so với `F.conv2d` |
| 1 | `lenet.py` | LeNet-5 (1998) | conv + subsampling, end-to-end | tanh vs ReLU |
| 2 | `alexnet.py` | AlexNet (2012) | ReLU, dropout, sâu hơn, GPU | bỏ dropout; xem filters |
| 3 | `zfnet.py` | ZFNet (2013) | visualize → sửa conv1 (kernel/stride nhỏ hơn) | filters/features vs AlexNet |
| 4 | `vgg.py` | VGG (2014) | chỉ conv 3x3, rất sâu | vgg11 vs vgg16; có/không BN |
| 5 | `googlenet.py` | GoogLeNet (2014) | Inception, 1x1 bottleneck, GAP | params vs VGG16 |
| 6 | `resnet.py` | ResNet (2015) | residual shortcut | plain18/34 vs resnet18/34 |
| 7 | `mobilenet.py` | MobileNet (2017) | depthwise separable conv | params & tốc độ vs ResNet |
| 8 | `convnext.py` | ConvNeXt (2022) | hiện đại hoá ResNet theo ViT | AdamW, epoch dài |

## Quy trình mỗi bài

```bash
# 1. Đọc docstring trong models/<file>.py, implement
# 2. Kiểm tra shape + gradient
python check.py lenet
python visualize.py summary lenet            # shape & params từng layer
# 3. Sanity check: model phải overfit được 256 ảnh (train acc → ~100%)
python train.py --model lenet --limit 256 --epochs 50 --bs 32 --tag lenet_overfit
# 4. Train thật
python train.py --model lenet
# 5. So sánh
python visualize.py curves
python visualize.py compare
python visualize.py filters lenet
python visualize.py features lenet --idx 3
```

Thí nghiệm biến thể (bỏ dropout, đổi activation...): sửa code tạm thời, rồi train với `--tag <tên>` để không đè lên run cũ.

Mốc tham khảo trên CIFAR-10 (train từ đầu, khoảng 30 epoch, sai số vài %): LeNet ~65%, AlexNet-CIFAR ~80%, VGG16-BN ~92%, GoogLeNet ~93%, ResNet18 ~94%, MobileNet ~90%.

## CNN Explorer: xem ảnh đi qua từng layer

```bash
python app.py        # mở http://localhost:8501. KHÔNG dùng `streamlit run app.py` (torch crash server, xem đầu app.py)
python test_probe.py # tự kiểm tra probe.py trên mọi model + chạy thử app
```

Chọn model và ảnh (CIFAR-10 test hoặc upload), rồi xem 4 tab: hành trình qua từng layer, soi 1 layer, Grad-CAM, nhánh trong block (ResNet/Inception/MobileNet/ConvNeXt). Thanh bên trái: so sánh 2 model / trained vs random / gốc vs đã chỉnh, chỉnh ảnh đầu vào (sáng, nhiễu, xoay, che 1 vùng), tắt/bật thành phần (shortcut, nhánh Inception, bỏ qua block, tắt channel). Cần `runs/<tag>.pt`.

## Train trên Google Colab

Mở [`colab.ipynb` trên Colab](https://colab.research.google.com/github/FirstKhoi/all_cnn_u_need/blob/main/colab.ipynb) (repo private thì Colab hỏi quyền GitHub → tick *Include private repos*). Notebook clone repo, train toàn bộ model với `--amp`, lưu kết quả vào Google Drive `MyDrive/all_cnn_u_need/runs/`. Bị ngắt kết nối thì chạy lại: model đã train xong tự bỏ qua.

Xem kết quả trên Mac: tải thư mục `runs/` trên Drive về, hoặc chạy thẳng `RUNS_DIR=<đường dẫn Drive> python visualize.py compare`.
