import json
import os
from pathlib import Path
from config import Config

def read_latest_log():
    """Reads the newest log file from /scratch/temp and returns its data."""
    default_data = {
        "model_type": "N/A",
        "accuracy": "N/A",
        "training_time": "N/A",
        "test_time": "N/A",
        "throughput": "N/A",
        "latency_per_batch_ms": "N/A",
        "inference_throughput": "N/A",
        "inference_latency_ms": "N/A",
        "world_size": "N/A",
        "epochs": "N/A",
        "timestamp": "N/A",
        "type": "N/A",
    }
    try:
        latest_log_path = Path("/scratch/temp/latest.log")

        if not latest_log_path.exists() or latest_log_path.stat().st_size == 0:
            return default_data
        
        with open(latest_log_path, "r") as f:
            data = json.load(f)
            return {
                "type": data.get("type", "N/A"),
                "parallelism_type": data.get(
                    "model_type", data.get("parallelism_type", "N/A")
                ),
                "accuracy": data.get("accuracy", "N/A"),
                "training_time": data.get("training_time", "N/A"),
                "test_time": data.get("test_time", "N/A"),
                "throughput": data.get("throughput", "N/A"),
                "latency_per_batch_ms": data.get(
                    "latency_per_batch_ms", data.get("latency_per_batch", "N/A")
                ),
                "inference_throughput": data.get("inference_throughput", "N/A"),
                "inference_latency_ms": data.get("inference_latency_ms", "N/A"),
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


def get_archived_runs(filter_type=None):
    """Läser igenom archive-mappen och returnerar lista med träningsrundor
    och deras tillhörande test-resultat."""
    runs = []
    if not Config.ARCHIVE_DIR.exists():
        return []

    for run_folder in Config.ARCHIVE_DIR.iterdir():
        if not run_folder.is_dir():
            continue

        train_log = run_folder / "train" / "training.log"

        if not train_log.exists():
            pass

        run_data = None
        if train_log.exists():
            try:
                with open(train_log, "r") as f:
                    run_data = json.load(f)
                    run_data["path"] = str(run_folder)
            except:
                continue

        if run_data:
            if filter_type and run_data.get("parallelism_type") != filter_type:
                continue

            tests = []

            std_test = run_folder / "test" / "test.log"
            if std_test.exists():
                try:
                    with open(std_test, "r") as f:
                        tests.append(json.load(f))
                except:
                    pass

            for f_path in run_folder.glob("test_log_*.json"):
                try:
                    with open(f_path, "r") as f:
                        tests.append(json.load(f))
                except:
                    pass

            tests.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            run_data["tests"] = tests

            runs.append(run_data)

    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return runs


def read_history_log():
    merged_data = {}
    try:
        history_log_path = Path("/scratch/temp/history.log")
        if not history_log_path.exists():
            return []

        with open(history_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = entry.get("timestamp")

                    if ts:
                        if ts not in merged_data:
                            merged_data[ts] = entry
                        else:
                            merged_data[ts].update(entry)
                except json.JSONDecodeError:
                    continue

        return list(merged_data.values())[::-1]

    except Exception as e:
        print(f"Error reading history log: {e}")
        return []
