from abc import ABC, abstractmethod
import json


class Logger(ABC):
    
    @abstractmethod
    def __init__(self, filepath, filepathLiveLog):
        self.filepath = filepath
        self.filepathLiveLog = filepathLiveLog

    
    @abstractmethod
    def collect(self)->None:
        """Collects data from logger"""
        log = {}
        
        self.updateLiveLogFile(log)
        self.updateLogFile(log)
    
    @abstractmethod
    def updateLogFile(self,log) -> None:
        """Writes collected data to self.filepath"""
    
    @abstractmethod
    def updateLiveLogFile(self, log) -> None:
        """Writes collected data to self.filepathLiveLog"""