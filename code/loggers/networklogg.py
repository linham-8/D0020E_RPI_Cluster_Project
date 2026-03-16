import psutil
from loggers.logger import Logger
import time
import torch
import torch.distributed as dist
import subprocess
from config import Config


class NetworkLogger(Logger):
    def __init__(self, filepath: str, filepathLiveLog: str, archive_dir: str = None, rank: int = 0):
        super().__init__(filepath, filepathLiveLog, archive_dir, rank=rank, node_specific=True)
        net_io = psutil.net_io_counters()
        self.prev_bytes_recv = max(0, net_io.bytes_recv)
        self.start_bytes_sent = net_io.bytes_sent
        self.start_bytes_recv = net_io.bytes_recv
        self.prev_time = time.time()
        self.total_bytes_per_sec = 0.0
        self.total_latency = 0.0
        self.measure_count = 0

    def get_ping_latency(self) -> float:
        try:
            target = Config.COMPUTE_NODES[0] if self.rank != 0 else "localhost"
            output = subprocess.check_output(["ping", "-c", "1", "-W", "1", target], stderr=subprocess.STDOUT, text=True)
            for line in output.split('\n'):
                if 'time=' in line:
                    return float(line.split('time=')[1].split(' ')[0])
        except Exception:
            pass
        return 0.0

    def collect(self) -> None:
        net_stats = psutil.net_io_counters()
        bytes_per_second = self.bytesRecvPerSecond(net_stats.bytes_recv)
        if bytes_per_second is None:
            bytes_per_second = 0

        current_latency = self.get_ping_latency()

        self.total_bytes_per_sec += bytes_per_second
        self.total_latency += current_latency
        self.measure_count += 1

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")

        live_log = {
            "timestamp": formatted_time,
            "bytes_per_second":round(bytes_per_second / (1024**2), 2),
            "bytes_sent":round(net_stats.bytes_sent / (1024**2), 2),     #number of bytes sent #wraparound Risk??
            "bytes_recv":round(net_stats.bytes_recv / (1024**2), 2),     #number of bytes received
            "packets_sent": net_stats.packets_sent, #number of packets sent
            "packets_recv": net_stats.packets_recv, #number of packets received
            "errin": net_stats.errin,               #total number of errors while receiving
            "errout": net_stats.errout,             #total number of errors while sending
            "dropin": net_stats.dropin,             #total number of incoming packets which were dropped
            "dropout": net_stats.dropout,            #total number of outgoing packets which were dropped(always 0 on macOS and BSD)
            "latency_ms": current_latency
        }
        self.updateLiveLogFile(live_log)

    def finalize_run(self) -> None:
        avg_bytes_per_sec = self.total_bytes_per_sec / self.measure_count if self.measure_count > 0 else 0.0
        avg_latency = self.total_latency / self.measure_count if self.measure_count > 0 else 0.0
        net_stats = psutil.net_io_counters()
        total_sent = net_stats.bytes_sent - self.start_bytes_sent
        total_recv = net_stats.bytes_recv - self.start_bytes_recv

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
        overall_log = {
            "timestamp": formatted_time,
            "overall_avg_bytes_per_second": round(avg_bytes_per_sec / (1024**2), 2),
            "total_bytes_sent_during_run": round(total_sent / (1024**2), 2),
            "total_bytes_recv_during_run": round(total_recv / (1024**2), 2),
            "overall_avg_latency_ms": round(avg_latency, 2)
        }
        self.updateLogFile(overall_log, mode="w")

        if dist.is_initialized():
            tensor_avgs = torch.tensor([avg_bytes_per_sec, total_sent, total_recv, avg_latency], dtype=torch.float32)
            dist.reduce(tensor_avgs, dst=0, op=dist.ReduceOp.SUM)
            global_bps = tensor_avgs[0].item() / dist.get_world_size()
            global_sent = tensor_avgs[1].item() / dist.get_world_size()
            global_recv = tensor_avgs[2].item() / dist.get_world_size()
            global_lat = tensor_avgs[3].item() / dist.get_world_size()
        else:
            global_bps = avg_bytes_per_sec
            global_sent = total_sent
            global_recv = total_recv
            global_lat = avg_latency

        if self.rank == 0:
            global_log = {
                "timestamp": formatted_time,
                "overall_avg_bytes_per_second": round(global_bps / (1024**2), 2),
                "total_bytes_sent_during_run": round(global_sent / (1024**2), 2),
                "total_bytes_recv_during_run": round(global_recv / (1024**2), 2),
                "overall_avg_latency_ms": round(global_lat, 2)
            }
            self.updateGeneralLogFile(global_log, mode="w")

    def bytesRecvPerSecond(self, bytes_recv: int) -> int:
        if self.prev_bytes_recv is not None:
            current_time = time.time()
            elapsed = current_time - self.prev_time
            if elapsed > 0:
                bytes_per_second = int((bytes_recv - self.prev_bytes_recv) / elapsed)
            else:
                bytes_per_second = 0
            self.prev_time = current_time
            self.prev_bytes_recv = bytes_recv
            return bytes_per_second


    def packetsPerSecond(self):
        pass
