from torch.utils.data import DataLoader
from datasets import * 

"""
    TODO Add docstrings
    TODO Kanske göra en dataset selector funktion med argument training
"""

def training_dataset_selector(dataset:str):
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


def get_data(training:bool, dataset:str, batch_size:int, shuffle:bool):
    if training:
        training_data = training_dataset_selector(dataset)
        return DataLoader(training_data, batch_size, shuffle) 
    else:
        test_data = test_dataset_selector(dataset)
        return DataLoader(test_data, batch_size, shuffle)
             
    



            
        
    
    