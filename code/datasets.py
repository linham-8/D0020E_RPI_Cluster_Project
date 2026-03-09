import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, TensorDataset
from config import Config

"""
    TODO Göra Download optional?
    TODO Lägga till fler dataset eller ett custom?
    TODO Docstring för funktionerna
"""

def get_FashionMNIST_training_data() -> Dataset:
    """
    Docstring for get_FashionMNIST_training_data
    """
    raw_dataset = datasets.FashionMNIST(
        root=Config.DATA_ROOT, train=True, download=True
    )

    data = raw_dataset.data.float().reshape(-1, 784) / 255.0
    targets = raw_dataset.targets.long()

    return TensorDataset(data, targets)

def get_FashionMNIST_test_data() -> Dataset:
    """
    Docstring for get_FashionMNIST_test_data
    """
    raw_dataset = datasets.FashionMNIST(
        root=Config.DATA_ROOT, train=False, download=True
    )

    data = raw_dataset.data.float().reshape(-1, 784) / 255.0
    targets = raw_dataset.targets.long()

    return TensorDataset(data, targets)

def get_EMNIST_training_data() -> Dataset:
    """
    Docstring for get_EMNIST_training_data
    """
    raw_dataset = datasets.EMNIST(
        root=Config.DATA_ROOT, split='digits', train=True, download=True
    )

    data = raw_dataset.data.float().reshape(-1, 784) / 255.0
    targets = raw_dataset.targets.long()

    return TensorDataset(data, targets)

def get_EMNIST_test_data() -> Dataset:
    """
    Docstring for get_EMNIST_test_data
    """
    raw_dataset = datasets.EMNIST(
        root=Config.DATA_ROOT, split='digits', train=False, download=True
    )
    data = raw_dataset.data.float().reshape(-1, 784) / 255.0
    targets = raw_dataset.targets.long()

    return TensorDataset(data, targets)