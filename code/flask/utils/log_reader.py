import json
import os
from pathlib import Path
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..', '..')
sys.path.append(os.path.abspath(parent_dir))
from config import Config

def read_json(filepath):
    try:
        if filepath.exists() and filepath.stat().st_size > 0:
            with open(filepath, "r") as f:
                return json.load(f)
    except:
        pass
    return {}

def read_latest_log():
    """Reads the newest log file and returns its data."""
    default_data = {
        "parallelism_type": "N/A",
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
        latest_dir = Config.TEMP_DIR / "latest"
        latest_log_path = latest_dir / "latest.log"

        if not latest_log_path.exists() or latest_log_path.stat().st_size == 0:
            return default_data

        data = read_json(latest_log_path)

        result = {
            "type": data.get("type", "N/A"),
            "parallelism_type": data.get("parallelism_type", "N/A"),
            "accuracy": data.get("accuracy", "N/A"),
            "training_time": data.get("training_time", "N/A"),
            "test_time": data.get("test_time", "N/A"),
            "throughput": data.get("throughput", "N/A"),
            "latency_per_batch_ms": data.get("latency_per_batch_ms", "N/A"),
            "inference_throughput": data.get("inference_throughput", "N/A"),
            "inference_latency_ms": data.get("inference_latency_ms", "N/A"),
            "world_size": data.get("world_size", "N/A"),
            "epochs": data.get("epochs", "N/A"),
            "timestamp": data.get("timestamp", "N/A"),
            "has_data": True,
            "log_file": str(latest_log_path.name),
        }

        result.update(read_json(latest_dir / "cpu.log"))
        result.update(read_json(latest_dir / "mem.log"))
        result.update(read_json(latest_dir / "net.log"))

        return result
    except Exception as e:
        print(f"Error reading latest log: {e}")
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

        latest_dir = run_folder / "latest"
        log_path = latest_dir / "latest.log"

        if log_path.exists():
            try:
                run_data = read_json(log_path)
                if not run_data:
                    continue

                run_data["path"] = str(run_folder)

                if filter_type and run_data.get("parallelism_type") != filter_type:
                    continue

                run_data.update(read_json(latest_dir / "cpu.log"))
                run_data.update(read_json(latest_dir / "mem.log"))
                run_data.update(read_json(latest_dir / "net.log"))

                if "tests" not in run_data:
                    run_data["tests"] = [run_data.copy()]

                runs.append(run_data)
            except:
                continue

    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return runs


def read_history_log():
    merged_data = {}
    try:
        history_log_path = Config.HISTORY_LOG
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

                        if "tests" not in merged_data[ts]:
                            merged_data[ts]["tests"] = [merged_data[ts].copy()]

                except json.JSONDecodeError:
                    continue

        return list(merged_data.values())[::-1]

    except Exception as e:
        print(f"Error reading history log: {e}")
        return []