import os
import json
import time
from loggers.logger import Logger

class TrainingLogger(Logger):
    def __init__(self, filepath: str = None, filepathLiveLog: str = None, archive_dir: str = None, parallelism_type: str = "unknown", world_size: int = 1):
        super().__init__(filepath=filepath, filepathLiveLog=filepathLiveLog, archive_dir=archive_dir)
        self.parallelism_type = parallelism_type
        self.world_size = world_size

    def collect(self):
        pass

    def log_live_progress(self, current_image: int, total_images: int, elapsed_seconds: float = 0.0, time_limit: int = 0, is_finished: bool = False):
        if time_limit > 0:
            progress = min(round((elapsed_seconds / time_limit) * 100, 2), 100.0)
            limit_type = "time"
            total_images = current_image
        else:
            progress = min(round((current_image / total_images) * 100, 2) if total_images > 0 else 0, 100.0)
            limit_type = "epochs"

        if is_finished:
            progress = 100.0
            if time_limit == 0:
                current_image = total_images

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")

        live_log = {
            "progress": progress,
            "current_image": current_image,
            "total_images": total_images,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "time_limit": time_limit,
            "limit_type": limit_type,
            "timestamp": formatted_time,
        }
        self.updateLiveLogFile(live_log)

    def log_training_result(self, training_time: float, total_images: int, epochs: int, batch_size: int):
        run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total_batches = total_images / batch_size
        throughput = total_images / training_time
        avg_iteration_time = (training_time / total_batches) * 1000

        train_log = {
            "type": "training_result",
            "parallelism_type": self.parallelism_type,
            "training_time": round(training_time, 2),
            "throughput": round(throughput, 2),
            "iteration_time_ms": round(avg_iteration_time, 2),
            "world_size": self.world_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "timestamp": run_timestamp,
        }

        self.updateLogFile(train_log, mode="w")

        if self.archive_dir:
            logs_dir = os.path.join(self.archive_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            try:
                with open(os.path.join(logs_dir, "training.log"), "w") as f:
                    json.dump(train_log, f)
            except OSError as e:
                print(f"Rank 0: Training archive logging failed: {e}")


class TestLogger(Logger):
    def __init__(self, filepath: str = None, archive_dir: str = None, parallelism_type: str = "unknown", world_size: int = 1, use_saved: str = "no"):
        super().__init__(filepath=filepath, archive_dir=archive_dir)
        self.parallelism_type = parallelism_type
        self.world_size = world_size
        self.use_saved = use_saved

    def collect(self):
        pass

    def log_test_result(self, test_time: float, accuracy: float, total_test_images: int, batch_size: int):
        run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        inference_throughput = total_test_images / test_time
        inference_iteration_time = (test_time / (total_test_images / batch_size)) * 1000

        test_log = {
            "type": "test_result",
            "parallelism_type": self.parallelism_type,
            "accuracy": float(accuracy),
            "test_time": round(test_time, 2),
            "inference_throughput": round(inference_throughput, 2),
            "inference_iteration_time_ms": round(inference_iteration_time, 2),
            "world_size": self.world_size,
            "timestamp": run_timestamp,
        }

        merged_log = {}
        if self.filepath:
            try:
                with open(self.filepath, "r") as f:
                    merged_log = json.load(f)
            except Exception:
                pass

        merged_log.update(test_log)

        self.updateLogFile(merged_log, mode="w")

        if self.archive_dir and self.use_saved != "yes":
            logs_dir = os.path.join(self.archive_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            try:
                with open(os.path.join(logs_dir, "test.log"), "w") as f:
                    json.dump(test_log, f)
            except OSError as e:
                print(f"Rank 0: Test archive logging failed: {e}")