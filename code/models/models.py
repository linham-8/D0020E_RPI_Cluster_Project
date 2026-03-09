import os
import time
from abc import ABC, abstractmethod
import torch
import torch.distributed as dist

from loggers.traininglogg import TrainingLogger, TestLogger
from loggers.cpulogg import CPUlogger
from loggers.memorylogg import MemoryLogger
from loggers.networklogg import NetworkLogger
from config import Config

class BaseModel(ABC):
    def __init__(self, data_loader, test_loader, dist_config, archive_dir, use_saved, parallelism_type):
        self.device = torch.device("cpu")
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.dist_config = dist_config
        self.archive_dir = archive_dir
        self.use_saved = use_saved
        self.parallelism_type = parallelism_type

        self.rank = dist_config['rank']
        self.world_size = dist_config['world_size']
        self.num_compute_nodes = self.world_size - 1

        self.init_distributed()

        self.init_loggers()



        first_image, _ = self.data_loader.dataset[0]

        self.input_dim = first_image.view(-1).size(0)

        if hasattr(self.data_loader.dataset, 'classes'):
            self.num_classes = len(self.data_loader.dataset.classes)
        elif hasattr(self.data_loader.dataset, 'tensors'):
            self.num_classes = len(torch.unique(self.data_loader.dataset.tensors[1]))
        elif hasattr(self.data_loader.dataset, 'targets'):
            self.num_classes = len(torch.unique(self.data_loader.dataset.targets))
        else:
            self.num_classes = 10

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

    def init_distributed(self):
        sync_file = os.path.join(Config.TEMP_DIR, f"{self.parallelism_type}_sync")
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(sync_file, self.world_size)
        )
        print(f"Rank {self.rank}: Connected to cluster.")

    def init_loggers(self):
        temp_base = str(Config.TEMP_DIR)
        live_dir = os.path.join(temp_base, "live")
        latest_dir = os.path.join(temp_base, "latest")

        os.makedirs(live_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)

        if self.rank == 0:
            self.train_logger = TrainingLogger(
                os.path.join(latest_dir, "latest.log"),
                os.path.join(live_dir, "live.log"),
                self.archive_dir, self.parallelism_type, self.world_size
            )
            self.test_logger = TestLogger(
                os.path.join(latest_dir, "latest.log"),
                self.archive_dir, self.parallelism_type, self.world_size, self.use_saved
            )

        self.system_loggers = [
            CPUlogger(
                os.path.join(latest_dir, "cpu.log"),
                os.path.join(live_dir, "cpu_live.log"),
                self.archive_dir, rank=self.rank
            ),
            MemoryLogger(
                os.path.join(latest_dir, "mem.log"),
                os.path.join(live_dir, "mem_live.log"),
                self.archive_dir, rank=self.rank
            ),
            NetworkLogger(
                os.path.join(latest_dir, "net.log"),
                os.path.join(live_dir, "net_live.log"),
                self.archive_dir, rank=self.rank
            )
        ]

    def collect_system_logs(self):
        """Kör collect() för alla systemloggrar"""
        for logger in self.system_loggers:
            logger.collect()

    def cleanup(self):
        if hasattr(self, 'system_loggers'):
            for logger in self.system_loggers:
                if hasattr(logger, 'finalize_run'):
                    logger.finalize_run()

        if dist.is_initialized():
            dist.barrier()
            print(f"[Rank {self.rank}] Destroying process group.")
            dist.destroy_process_group()

    @abstractmethod
    def train(self, epochs=Config.EPOCHS, time_limit=None):
        pass

    @abstractmethod
    def test(self):
        pass
