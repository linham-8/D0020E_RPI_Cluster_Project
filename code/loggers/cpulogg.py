import psutil
from logger import Logger
import json 

class CPUlogger(Logger):
    
    def __init__(self, filepath = "code/loggers/cpu_metrics.json",
                 filepathLiveLog = "code/loggers/live_cpu_metrics.json",
                 interval = 1.0):

            self.filepath = filepath
            self.filepathLiveLog = filepathLiveLog
            self.interval = interval
            "Remove files if already exists"
    
    def collect(self):
        latestlog = {
            "cpu-usage": psutil.cpu_percent(self.interval)
            #"count": psutil.cpu_count()
            #"cpu_stats": psutil.cpu_stats()
        }
        self.updateLogFile(latestlog)
        self.updateLiveLogFile(latestlog)
        
        
    def updateLogFile(self, log):      
        "Add saftey if file not exist"
        if log:
            with open(self.filepath, "a") as f:
                json.dump(log, f)
                f.write("\n")
                
    def updateLiveLogFile(self, log):      
        "Add saftey if file not exist"
        if log:
            with open(self.filepathLiveLog, "w") as f:
                json.dump(log, f)
                f.write("\n")

