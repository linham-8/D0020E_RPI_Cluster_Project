from abc import abstractmethod, ABC
import json
import os
from config import Config

class Logger(ABC):

    def __init__(self, filepath: str = None, filepathLiveLog: str = None, archive_dir: str = None, rank: int = 0, node_specific: bool = False):
        self.rank = rank
        self.archive_dir = archive_dir
        self.node_specific = node_specific

        self.general_filepath = filepath
        self.general_filepathLiveLog = filepathLiveLog

        self.filepath = filepath
        self.filepathLiveLog = filepathLiveLog

        if self.node_specific:
            if filepath:
                self.filepath = self._get_node_specific_path(filepath)
            if filepathLiveLog:
                self.filepathLiveLog = self._get_node_specific_path(filepathLiveLog)

    @abstractmethod
    def collect(self) -> None:
        """Collects data from logger"""
        pass

    def _get_node_specific_path(self, path: str) -> str:
        base_dir, file_name = os.path.split(path)
        base_name, ext = os.path.splitext(file_name)
        nodes_dir = os.path.join(base_dir, "nodes")
        os.makedirs(nodes_dir, exist_ok=True)
        return os.path.join(nodes_dir, f"{base_name}_rank{self.rank}{ext}")




    def _get_archive_filepath(self, target_filepath: str, default_subfolder: str) -> str:
        if not self.archive_dir or not target_filepath:
            return None
        temp_prefix = str(Config.TEMP_DIR) + "/"
        if target_filepath.startswith(temp_prefix):
            rel_path = os.path.relpath(target_filepath, str(Config.TEMP_DIR))
            return os.path.join(self.archive_dir, rel_path)
        return os.path.join(self.archive_dir, default_subfolder, os.path.basename(target_filepath))

    def _write_log(self, path: str, log: dict, mode: str, archive_folder: str) -> None:
        if not path:
            return
        with open(path, mode) as f:
            f.write(json.dumps(log) + "\n")
        if self.archive_dir:
            archive_filepath = self._get_archive_filepath(path, archive_folder)
            if archive_filepath:
                os.makedirs(os.path.dirname(archive_filepath), exist_ok=True)
                with open(archive_filepath, mode) as f:
                    f.write(json.dumps(log) + "\n")

    def updateLogFile(self, log: dict, mode: str = "a") -> None:
        self._write_log(self.filepath, log, mode, "latest")

    def updateGeneralLogFile(self, log: dict, mode: str = "a") -> None:
        self._write_log(self.general_filepath, log, mode, "latest")

    def updateLiveLogFile(self, log: dict) -> None:
        self._write_log(self.filepathLiveLog, log, "a", "live")
        if self.node_specific and self.rank == 0:
            self._write_log(self.general_filepathLiveLog, log, "a", "live")
