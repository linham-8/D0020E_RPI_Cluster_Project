from abc import abstractmethod, ABC
import json


class Logger(ABC):
    
    def __init__(self, filepath: str, filepathLiveLog: str):
        self.filepath = filepath
        self.filepathLiveLog = filepathLiveLog

    
    @abstractmethod
    def collect(self) -> None:
        """Collects data from logger"""
        log = {}
        
        self.updateLiveLogFile(log)
        self.updateLogFile(log)


    def updateLogFile(self, log: dict) -> None:
        with open(self.filepath, "a") as f:
            json.dump(log, f)
            f.write("\n")
    

    def updateLiveLogFile(self, log: dict) -> None:
        """Writes collected data to self.filepathLiveLog"""
        with open(self.filepathLiveLog, "w") as f:
            json.dump(log, f)
            f.write("\n")        
        