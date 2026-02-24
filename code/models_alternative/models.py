from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class BaseModel(ABC):
    def __init__(self, data_loader, dist_config):
        self.device = torch.device("cpu")
        
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.epoch = 0

        self.init_distributed(dist_config)
        
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.criterion = self.build_criterion()

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def build_optimizer(self):
        pass

    @abstractmethod
    def build_criterion(self): #Loss Function
        pass

    def init_distributed(self, dist_config):
        dist.init_process_group(
            backend= dist_config.backend,
            rank=dist_config.rank,
            world_size=dist_config.world_size,
            store=dist.FileStore("/scratch/temp/data_parallel_sync", dist_config.world_size)
        )

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def train_step(self):
        try:
            batch_x, batch_y = next(self.data_iter)
        except StopIteration:
            self.epoch += 1

            if hasattr(self.data_loader.sampler, "set_epoch"):
                self.data_loader.sampler.set_epoch(self.epoch)
            
            self.data_iter = iter(self.data_loader)
            batch_x, batch_y = next(self.data_iter)

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        loss.backward()
        self.optimizer.step()
