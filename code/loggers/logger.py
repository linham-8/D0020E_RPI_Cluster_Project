from abc import abstractmethod, ABC
import json
import os
from config import Config

class Logger(ABC):

    def __init__(self, filepath: str, filepathLiveLog: str, archive_dir: str = None, rank: int = 0, node_specific: bool = False):
        self.rank = rank
        self.archive_dir = archive_dir
        self.node_specific = node_specific

        self.general_filepath = filepath
        self.general_filepathLiveLog = filepathLiveLog

        if self.node_specific:
            if filepath:
                base_dir, file_name = os.path.split(filepath)
                base_name, ext = os.path.splitext(file_name)
                nodes_dir = os.path.join(base_dir, "nodes")
                os.makedirs(nodes_dir, exist_ok=True)
                self.filepath = os.path.join(nodes_dir, f"{base_name}_rank{self.rank}{ext}")
            else:
                self.filepath = None

            if filepathLiveLog:
                base_dir, file_name = os.path.split(filepathLiveLog)
                base_name, ext = os.path.splitext(file_name)
                nodes_dir = os.path.join(base_dir, "nodes")
                os.makedirs(nodes_dir, exist_ok=True)
                self.filepathLiveLog = os.path.join(nodes_dir, f"{base_name}_rank{self.rank}{ext}")
            else:
                self.filepathLiveLog = None
        else:
            self.filepath = filepath
            self.filepathLiveLog = filepathLiveLog

    @abstractmethod
    def collect(self) -> None:
        """Collects data from logger"""
        pass


    def _get_archive_filepath(self, target_filepath: str, default_subfolder: str) -> str:
        temp_prefix = str(Config.TEMP_DIR) + "/"
        if target_filepath and target_filepath.startswith(temp_prefix):
            rel_path = os.path.relpath(target_filepath, str(Config.TEMP_DIR))
            return os.path.join(self.archive_dir, rel_path)
        return os.path.join(self.archive_dir, default_subfolder, os.path.basename(target_filepath))

    def updateLogFile(self, log: dict, mode: str = "a") -> None:
        if self.filepath:
            with open(self.filepath, mode) as f:
                f.write(json.dumps(log) + "\n")
        if self.archive_dir and os.path.isdir(self.archive_dir) and self.filepath:
            archive_filepath = self._get_archive_filepath(self.filepath, "latest")
            os.makedirs(os.path.dirname(archive_filepath), exist_ok=True)
            with open(archive_filepath, mode) as f:
                f.write(json.dumps(log) + "\n")

    def updateGeneralLogFile(self, log: dict, mode: str = "a") -> None:
        if self.general_filepath:
            with open(self.general_filepath, mode) as f:
                f.write(json.dumps(log) + "\n")
        if self.archive_dir and os.path.isdir(self.archive_dir) and self.general_filepath:
            archive_filepath = self._get_archive_filepath(self.general_filepath, "latest")
            os.makedirs(os.path.dirname(archive_filepath), exist_ok=True)
            with open(archive_filepath, mode) as f:
                f.write(json.dumps(log) + "\n")

    def updateLiveLogFile(self, log: dict) -> None:
        """Writes collected data to self.filepathLiveLog"""
        if self.filepathLiveLog:
            with open(self.filepathLiveLog, "a") as f:
                f.write(json.dumps(log) + "\n")

            if self.archive_dir and os.path.isdir(self.archive_dir):
                archive_filepath = self._get_archive_filepath(self.filepathLiveLog, "live")
                os.makedirs(os.path.dirname(archive_filepath), exist_ok=True)
                with open(archive_filepath, "a") as f:
                    f.write(json.dumps(log) + "\n")

            if self.node_specific and self.rank == 0 and getattr(self, 'general_filepathLiveLog', None):
                with open(self.general_filepathLiveLog, "a") as f:
                    f.write(json.dumps(log) + "\n")

                if self.archive_dir and os.path.isdir(self.archive_dir):
                    archive_filepath = self._get_archive_filepath(self.general_filepathLiveLog, "live")
                    os.makedirs(os.path.dirname(archive_filepath), exist_ok=True)
                    with open(archive_filepath, "a") as f:
                        f.write(json.dumps(log) + "\n")
