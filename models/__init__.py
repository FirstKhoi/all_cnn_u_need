from functools import partial

from .alexnet import AlexNet
from .basics import SimpleCNN
from .convnext import ConvNeXt
from .googlenet import GoogLeNet
from .lenet import LeNet5
from .mobilenet import MobileNet
from .resnet import ResNet
from .vgg import VGG
from .zfnet import ZFNet

# Chronological order = learning order. Every entry: callable() -> nn.Module, (N,3,32,32) -> (N,10).
MODELS = {
    "simple": SimpleCNN,
    "lenet": LeNet5,
    "alexnet": AlexNet,
    "zfnet": ZFNet,
    "vgg11": partial(VGG, "vgg11"),
    "vgg16": partial(VGG, "vgg16"),
    "googlenet": GoogLeNet,
    "plain18": partial(ResNet, (2, 2, 2, 2), shortcut=False),
    "plain34": partial(ResNet, (3, 4, 6, 3), shortcut=False),
    "resnet18": partial(ResNet, (2, 2, 2, 2)),
    "resnet34": partial(ResNet, (3, 4, 6, 3)),
    "mobilenet": MobileNet,
    "convnext": ConvNeXt,
}
