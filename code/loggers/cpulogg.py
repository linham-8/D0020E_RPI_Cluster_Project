import psutil
from logger import Logger

class CPUlogger(Logger):
    
    def __init__(self, filepath = "code/loggers/cpu_metrics.json",
                 filepathLiveLog = "code/loggers/live_cpu_metrics.json",
                 interval = 1.0):

            super().__init__(filepath, filepathLiveLog)
            self.interval = interval
            "Remove files if already exists"
    
    def collect(self):
        log = {
            "cpu-usage": psutil.cpu_percent(self.interval)
            #"count": psutil.cpu_count()
            #"cpu_stats": psutil.cpu_stats()
            #ADD relevant
        }
        self.updateLiveLogFile(log)
        self.updateLogFile(log)
        
    

