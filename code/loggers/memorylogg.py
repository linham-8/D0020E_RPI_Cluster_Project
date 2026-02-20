import psutil
from logger import Logger


class MemoryLogger(Logger):

    def __init__(self, filepath, filepathLiveLog):
        super().__init__(filepath, filepathLiveLog)
        self.total_memory = psutil.virtual_memory().total 

    def collect(self):
        memory_stats = psutil.virtual_memory()
        log = {
            "memory_used_percent": memory_stats.percent,
            "memory_used": memory_stats.used
        }
        self.updateLiveLogFile(log)
        self.updateLogFile(log)