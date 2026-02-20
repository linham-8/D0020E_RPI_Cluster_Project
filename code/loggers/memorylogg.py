import psutil
from logger import Logger


class Networklogger(Logger):
    
    def collect():
        net_stats = psutil.net_io_counters()
        return {
            "bytes_sent": net_stats.bytes_sent,
            "bytes_recv": net_stats.bytes_recv,
            "dropin": net_stats.dropin
        }

    