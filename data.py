"""CIFAR-10 loaders. Infra — đã viết sẵn, không phải phần cần học."""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = "data"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def _transform(augment, img_size):
    t = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    if img_size != 32:
        t.append(transforms.Resize(img_size))  # for faithful 224x224 AlexNet/ZFNet/VGG
    return transforms.Compose(t + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def test_set(img_size=32):
    return datasets.CIFAR10(ROOT, train=False, download=True, transform=_transform(False, img_size))


def get_loaders(batch_size=128, img_size=32, limit=None, workers=2):
    # limit = tiny subset for the "overfit a few samples" sanity check, so no augmentation
    train = datasets.CIFAR10(ROOT, train=True, download=True, transform=_transform(not limit, img_size))
    if limit:
        train = Subset(train, range(limit))
    kw = dict(batch_size=batch_size, num_workers=workers, persistent_workers=workers > 0)
    return DataLoader(train, shuffle=True, **kw), DataLoader(test_set(img_size), **kw)
