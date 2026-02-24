from torch.utils.data import DataLoader
from dataloader.datasets import * 

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
            return get_EMNIST_training_data()    
        case "FashionMNIST":
            return get_FashionMNIST_training_data()
        case _:
            print("Dataset not avaiable:(")
            

def test_dataset_selector(dataset:str):
    match dataset:
        case "EMNIST":
            return get_EMNIST_test_data()
        case "FasionMNIST":
            return get_FashionMNIST_test_data()
        case _:
            print("Dataset not avaiable:(")


def get_data(training:bool, dataset:str, batch_size:int, shuffle:bool) -> DataLoader:
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
    """
    if training:
        training_data = training_dataset_selector(dataset)
        return DataLoader(training_data, batch_size, shuffle) 
    else:
        test_data = test_dataset_selector(dataset)
        return DataLoader(test_data, batch_size, shuffle)
             
    



            
        
    
    