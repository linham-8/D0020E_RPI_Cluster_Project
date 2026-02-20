import psutil
from logger import Logger
import time

class NetworkLogger(Logger):
    
    def __init__(self, filepath, filepathLiveLog):
        super().__init__(filepath,filepathLiveLog)
        self.prev_bytes_recv = max(0, psutil.net_io_counters().bytes_recv) #Maybe not needed
        self.prev_time = time.time()   
    
    def collect(self):
        net_stats = psutil.net_io_counters()
        #latency = latency(net_stats.bytes_sent, net_stats.bytes_recv)
        bytes_per_second = self.bytesRecvPerSecond(net_stats.bytes_recv)
        log = {
            "bytes_per_second": bytes_per_second,
            "bytes_sent": net_stats.bytes_sent,     #number of bytes sent #wraparound Risk??
            "bytes_recv": net_stats.bytes_recv,     #number of bytes received
            "packets_sent": net_stats.packets_sent, #number of packets sent
            "packets_recv": net_stats.packets_recv, #number of packets received
            "errin": net_stats.errin,               #total number of errors while receiving
            "errout": net_stats.errout,             #total number of errors while sending
            "dropin": net_stats.dropin,             #total number of incoming packets which were dropped
            "dropout": net_stats.dropout            #total number of outgoing packets which were dropped(always 0 on macOS and BSD)
            #"latency": latency
        }
        self.updateLiveLogFile(log)
        self.updateLogFile(log)
 
    def bytesRecvPerSecond(self, bytes_recv) -> int:
        if self.prev_bytes_recv is not None: #Maybe not needed since the raspberry pi will recv other bytes on startup it should not be 0 or None
            current_time = time.time()
            bytes_per_second = int((bytes_recv - self.prev_bytes_recv) / (current_time - self.prev_time)) #Protection against division by 0 may be needed
            self.prev_time = current_time
            self.prev_bytes_recv = bytes_recv
            return bytes_per_second
            
 
    def packetsPerSecond(self):
        pass
        