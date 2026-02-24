from models import BaseModel
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DataParallelModel(BaseModel):
    
    def __init__(self, data_loader, dist_config):
        super().__init__(self, data_loader, dist_config)
        


    def build_model(self):
        return DDP(torch.nn.Linear())

    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.01)

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()
