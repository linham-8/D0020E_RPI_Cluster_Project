from abc import abstractmethod
import json


class Logger():
    
    def __init__(self, filepath, filepathLiveLog):
        self.filepath = filepath
        self.filepathLiveLog = filepathLiveLog

    
    @abstractmethod
    def collect(self)->None:
        """Collects data from logger"""
        log = {}
        
        self.updateLiveLogFile(log)
        self.updateLogFile(log)


    def updateLogFile(self, log) -> None:
        if log:
            with open(self.filepath, "a") as f:
                json.dump(log, f)
                f.write("\n")
    

    def updateLiveLogFile(self, log) -> None:
        """Writes collected data to self.filepathLiveLog"""
        if log:
            with open(self.filepathLiveLog, "w") as f:
                json.dump(log, f)
                f.write("\n")        
        