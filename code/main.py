from configs.distconf import getDistConfig
from dataloader.dataloader import get_data
from models_alternative.data_parallel_model import DataParallelModel


def main():    
    dist_conf = getDistConfig()
    
    data_loader = get_data("EMNIST")
    
    #CPU_logger = CPULogger()
    #memory_logger = MemoryLogger()
    #network_logger = NetworkLogger()
    
    model = DataParallelModel(data_loader=data_loader, dist_conf = dist_conf)
    
    
    #loop eller kör träning och loggin i två olika trådar, loggern körs på interupts varje sekund
    
    while (model.epoch < 5): 
        model.train_step()
        #CPU_logger.collect()
        #memory_logger.collect()
        #network_logger.collect()
    
        
    
    
    