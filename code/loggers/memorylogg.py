import psutil
from loggers.logger import Logger
import time
import torch
import torch.distributed as dist


class MemoryLogger(Logger):

    def __init__(self, filepath: str, filepathLiveLog: str, archive_dir: str = None, rank: int = 0):
        super().__init__(filepath, filepathLiveLog, archive_dir, rank=rank, node_specific=True)
        self.total_memory = psutil.virtual_memory().total
        self.total_mem_percent = 0.0
        self.total_mem_used = 0
        self.measure_count = 0

    def collect(self) -> None:
        memory_stats = psutil.virtual_memory()

        self.total_mem_percent += memory_stats.percent
        self.total_mem_used += memory_stats.used
        self.measure_count += 1

        avg_percent = self.total_mem_percent / self.measure_count
        avg_used = self.total_mem_used / self.measure_count

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")

        live_log = {
            "timestamp": formatted_time,
            "memory_used_percent": memory_stats.percent,
            "memory_used": memory_stats.used
        }
        self.updateLiveLogFile(live_log)

    def finalize_run(self) -> None:
        avg_percent = self.total_mem_percent / self.measure_count if self.measure_count > 0 else 0.0
        avg_used = self.total_mem_used / self.measure_count if self.measure_count > 0 else 0.0
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
        overall_log = {
            "timestamp": formatted_time,
            "overall_avg_memory_percent": round(avg_percent, 2),
            "overall_avg_memory_used": int(avg_used)
        }
        self.updateLogFile(overall_log, mode="w")

        if dist.is_initialized():
            tensor_avgs = torch.tensor([avg_percent, avg_used], dtype=torch.float32)
            dist.reduce(tensor_avgs, dst=0, op=dist.ReduceOp.SUM)
            global_percent = tensor_avgs[0].item() / dist.get_world_size()
            global_used = tensor_avgs[1].item() / dist.get_world_size()
        else:
            global_percent = avg_percent
            global_used = avg_used

        if self.rank == 0:
            global_log = {
                "timestamp": formatted_time,
                "overall_avg_memory_percent": round(global_percent, 2),
                "overall_avg_memory_used": int(global_used)
            }
            self.updateGeneralLogFile(global_log, mode="w")