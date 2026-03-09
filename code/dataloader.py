import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import datasets

"""
    TODO Add docstrings
    TODO Kanske göra "en" dataset selector funktion med argument training
"""

def training_dataset_selector(dataset:str):
    """
    Docstring for training_dataset_selector

    :param dataset: Dataset to be used(EMNIST or FashionMNIST)
    :type dataset: str
    """
    match dataset:
        case "EMNIST":
            return datasets.get_EMNIST_training_data()
        case "FashionMNIST":
            return datasets.get_FashionMNIST_training_data()
        case _:
            print("Dataset not available :(")


def test_dataset_selector(dataset:str):
    match dataset:
        case "EMNIST":
            return datasets.get_EMNIST_test_data()
        case "FashionMNIST":
            return datasets.get_FashionMNIST_test_data()
        case _:
            print("Dataset not available :(")


def get_data(training:bool, dataset:str, batch_size:int, shuffle:bool=True, world_size:int=1, rank:int=0) -> DataLoader:
    """
    Docstring for get_data

    :param training: if used for training set True
    :type training: bool
    :param dataset: Dataset to use(EMNIST or FashionMNIST)
    :type dataset: str
    :param batch_size: Batch size
    :type batch_size: int
    :param shuffle: Shuffle on or off
    :type shuffle: bool
    :param world_size: Number of nodes in cluster
    :type world_size: int
    :param rank: Node ID
    :type rank: int
    """

    if training:
        if dataset == "EMNIST":
            data_obj = datasets.get_EMNIST_training_data()
        else:
            return None

        if world_size > 1:
            sampler = DistributedSampler(data_obj, num_replicas=world_size, rank=rank, shuffle=shuffle)
            return DataLoader(data_obj, batch_size=batch_size, sampler=sampler, num_workers=0)
        else:
            return DataLoader(data_obj, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    else:
        data_obj = datasets.get_EMNIST_test_data()
        return DataLoader(data_obj, batch_size=batch_size, shuffle=False, num_workers=0)