import json
from pathlib import Path

def read_latest_log():
    """Reads the newest log file from /scratch/temp and returns its data."""
    default_data = {
        "parallelism_type": "N/A",
        "accuracy": "N/A",
        "training_time": "N/A",
        "test_time": "N/A",
        "throughput": "N/A",
        "latency_per_batch": "N/A",
        "world_size": "N/A",
        "epochs": "N/A",
        "timestamp": "N/A",
    }
    try:
        latest_log_path = Path("/scratch/temp/latest.log")

        with open(latest_log_path, "r") as f:
            data = json.load(f)
            return {
                "parallelism_type": data.get("parallelism_type", "N/A"),
                "accuracy": data.get("accuracy", "N/A"),
                "training_time": data.get("training_time", "N/A"),
                "test_time": data.get("test_time", "N/A"),
                "throughput": data.get("throughput", "N/A"),
                "latency_per_batch": data.get("latency_per_batch", "N/A"),
                "world_size": data.get("world_size", "N/A"),
                "epochs": data.get("epochs", "N/A"),
                "timestamp": data.get("timestamp", "N/A"),
                "has_data": True,
                "log_file": str(latest_log_path.name),
            }
    except FileNotFoundError:
        print("Log file not found.")
        return default_data
    except json.JSONDecodeError:
        print("Error decoding JSON from log file.")
        return default_data

def read_history_log():
    history = []
    try:
        history_log_path = Path("/scratch/temp/history.log")
        if not history_log_path.exists():
            return history

        with open(history_log_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                history.append(entry)
        return history
    except Exception as e:
        print(f"Error reading history log: {e}")
        return history