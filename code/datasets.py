import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor




def get_FasionMnist_training_data():
    """_summary_

    Returns:
        FashionMNIST:Dataset
    """
    return datasets.FashionMNIST(
        root="/TrainingData/data",
        train = True,
        downlaod = True,
        tranform = ToTensor()
    )

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)