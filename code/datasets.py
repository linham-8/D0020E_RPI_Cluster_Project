import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
    TODO Göra Download optional?
    TODO Lägga till fler dataset eller ett custom?
    TODO Docstring för funktionerna
"""

def get_FashionMNIST_training_data() -> datasets.FashionMNIST:
    """
    Docstring for get_FashionMNIST_training_data
    """
    return datasets.FashionMNIST(
        root ="data",
        train = True,
        download = True,
        transform = ToTensor()
    )
def get_FashionMNIST_test_data() -> datasets.FashionMNIST:
    """
    Docstring for get_FashionMNIST_test_data
    """
    return datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor()
    )
def get_EMNIST_training_data() -> datasets.EMNIST:
    """
    Docstring for get_EMNIST_training_data
    """
    return datasets.EMNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor()
    )

def get_EMNIST_test_data() -> datasets.EMNIST:
    """
    Docstring for get_EMNIST_test_data
    """
    return datasets.EMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor()
    )
  