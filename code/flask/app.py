from flask import Flask, request, redirect, url_for, render_template, jsonify
from utils.log_reader import read_latest_log, read_history_log, get_archived_runs
import subprocess
import os
import signal
import logging
import shutil
import sys
from config import Config
import stop

app = Flask(__name__)

#Slår av loggin, kommentera ut bara
#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)

active_tasks = {}
has_data = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(CODE_DIR)
launch_script = os.path.join(CODE_DIR, "launch.py")


@app.route("/", methods=["GET", "POST"])
def index():
    """Main page route used to start and stop training, and view results and graphs"""
    return render_template("index.html")

@app.route("/logs")
def logs():
    """Browseable logs"""
    return """
    <h2>
        <a href="/">Back to main page</a>
    </h2>
    <ul>
        <li><a href="/api/latest">Latest Log</a></li>
        <li><a href="/api/history">History log</a></li>
    </ul>
    """

@app.route("/api/training/start", methods=["POST"])
def api_start():
    data = request.get_json() or {}
    
    if is_process_running():
        return jsonify({"status": "error", "message": "Training already in progress"}), 400

    model = data.get("model")
    parallelism = data.get("parallelism")
    saved = data.get("saved")
    time_limit_min = str(data.get("time_limit", "0"))
    time_limit_sec = int(time_limit_min) * 60 if time_limit_min.isdigit() else 0
    custom_name = str(data.get("custom_name", "")).strip()

    try:
        cmd = [
            "python", launch_script,
            "--model", parallelism, 
            "--saved", saved,
            "--time_limit", str(time_limit_sec)
        ]

        if custom_name:
            cmd.extend(["--name", custom_name])

        proc = subprocess.Popen(cmd, cwd=Config.ROOT_DIR)
        
        active_tasks["training"] = {
            "proc": proc,
            "pid": proc.pid,
            "parallelism": parallelism,
            "model": model
        }
        
        return jsonify({
            "status": "success",
            "message": f"Started {parallelism} training (PID: {proc.pid})"
        }), 202

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/training/stop", methods=["POST"])
def api_stop():
    """API route used to stop training"""
    task_info = active_tasks.get("training")
    
    if not task_info:
        return jsonify({"status": "error", "message": "No active task found"}), 404

    try:
        stop.stop_session()
            
        active_tasks.pop("training", None)
        
        return jsonify({"status": "success", "message": "Training stopped"}), 200
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/training/status", methods=["GET"])
def api_status():
    if is_process_running():
        task_data = active_tasks["training"]
        return jsonify({
            "active": True,
            "details": {
                "pid": task_data.get("pid"),
                "parallelism": task_data.get("parallelism"),
                "model": task_data.get("model")
            }
        })
    return jsonify({"active": False})

@app.route("/api/logs/clear", methods=["POST"])
def api_clear_logs():
    """API route that triggers the deletion logic in stop.py"""
    data = request.get_json() or {}
    target = data.get("target")
    
    try:
        if target == "latest":
            stop.clean_temp_folders()
        elif target == "history":
            stop.clean_history()
        else:
            return jsonify({"status": "error", "message": "Invalid target"}), 400
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/latest")
def api_latest():
    """API route returning the latest log"""
    return jsonify(read_latest_log())

#@app.route("/api/live")
#def api_latest():
#    """API route returning the live log"""
#    return jsonify(read_latest_log())

@app.route("/api/history")
def api_history():
    """API route returning the history log"""
    return jsonify(get_archived_runs(filter_type=None))

@app.route("/api/archives/<model_type>")
def api_archives(model_type):
    """Api route returing the archived log"""
    runs = get_archived_runs(filter_type=model_type)
    return jsonify(runs)

# Status helper
def is_process_running():
    """Checks the stored 'proc' object to see if training is still alive."""
    task = active_tasks.get("training")
    
    if not task or "proc" not in task:
        return False

    proc = task["proc"]
    
    if proc.poll() is None:
        return True

    active_tasks.pop("training", None)
    return False

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
