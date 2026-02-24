from models import BaseModel
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DataParallelModel(BaseModel):
    
    def __init__(self, data_loader, dist_config):
        super().__init__(data_loader, dist_config)
        


    def build_model(self):
        #Kanske kan ligga i models.py
        sample_x, sample_y = next(iter(self.data_loader))
        
        #dim av sample_X
        x_dim = sample_x.view(sample_x(0), -1).size(1)
        #number of unique labels/classes
        num_classes = len(torch.unique(sample_y))
        return DDP(torch.nn.Linear(x_dim, num_classes))

    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.01)

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()
