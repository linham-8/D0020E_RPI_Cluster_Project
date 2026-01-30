from flask import Flask, request, redirect, url_for, render_template, jsonify
import json
import subprocess
import os
import signal
from pathlib import Path


app = Flask(__name__)
active_tasks = {}
has_data = False

base_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(base_dir)
launch_script = os.path.join(code_dir, "launch.py")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form["action"]
        if action == "start":
            model = request.form["model"]
            parallelism = request.form["parallelism"]
            saved = request.form["saved"]

            print(
                f"Starting training: Model={model}, Parallelism={parallelism}, Saved={saved}"
            )
            proc = subprocess.Popen(["python", launch_script, parallelism, saved])

            active_tasks["training"] = proc.pid
            print(f"Started training with PID {proc.pid}")
            return redirect(url_for("index"))

        elif action == "stop":
            pid = active_tasks.get("training")
            if pid:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    active_tasks.pop("training", None)
                except ProcessLookupError:
                    pass
            return redirect(url_for("index"))

        elif action == "clear-history":
            try:
                os.remove("/scratch/temp/history.log")
            except FileNotFoundError:
                pass
            return redirect(url_for("index"))

    log_data = read_latest_log()
    history_data = read_history_log()
    print(f"Log Data: {log_data}")
    print(f"History Data: {history_data}")
    return render_template(
        "index.html",
        **log_data,
        history=history_data,
        has_history=len(history_data) > 0,
    )


@app.route("/api/start", methods=["POST"])
def api_start():
    # Start logic here
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ## Stop logic here
    return jsonify({"status": "stopped"})


@app.route("/api/status", methods=["GET"])
def api_status():
    pid = active_tasks.get("training")
    status = "running" if pid else "stopped"
    return jsonify({"status": status})


# Helper functions

# TODO function to clear history log


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
