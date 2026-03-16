import psutil
from loggers.logger import Logger
import time
import torch
import torch.distributed as dist

class CPUlogger(Logger):
    def __init__(self, filepath: str = None, filepathLiveLog: str = None, archive_dir: str = None, interval=0, rank: int = 0):
        super().__init__(filepath=filepath, filepathLiveLog=filepathLiveLog, archive_dir=archive_dir, rank=rank, node_specific=True)
        self.interval = interval
        self.total_cpu = 0.0
        self.measure_count = 0
        self.process = psutil.Process()
        self.process.cpu_percent(interval=None)
        "Remove files if already exists"

    def collect(self) -> None:
        current_cpu = self.process.cpu_percent(interval=None) / psutil.cpu_count()

        self.total_cpu += current_cpu
        self.measure_count += 1

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")

        live_log = {
            "timestamp": formatted_time,
            "cpu-usage": current_cpu,
            #"count": psutil.cpu_count()
            #"cpu_stats": psutil.cpu_stats()._asdict()
            #ADD relevant
        }
        self.updateLiveLogFile(live_log)

    def finalize_run(self) -> None:
        avg_cpu = self.total_cpu / self.measure_count if self.measure_count > 0 else 0.0
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
        overall_log = {
            "timestamp": formatted_time,
            "overall_avg_cpu_usage": round(avg_cpu, 2)
        }
        self.updateLogFile(overall_log, mode="w")

        if dist.is_initialized():
            tensor_avg = torch.tensor([avg_cpu], dtype=torch.float32)
            dist.reduce(tensor_avg, dst=0, op=dist.ReduceOp.SUM)
            global_avg = tensor_avg.item() / dist.get_world_size()
        else:
            global_avg = avg_cpu

        if self.rank == 0:
            global_log = {
                "timestamp": formatted_time,
                "overall_avg_cpu_usage": round(global_avg, 2)
            }
            self.updateGeneralLogFile(global_log, mode="w")


