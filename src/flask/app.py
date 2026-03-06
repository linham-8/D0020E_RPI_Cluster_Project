from flask import Flask, request, redirect, url_for, render_template, jsonify
from .utils.log_reader import read_latest_log, read_history_log, get_archived_runs
import subprocess
import os
import signal
from config import Config

app = Flask(__name__)
active_tasks = {}
has_data = False

@app.route("/")
def index():
    """Main page route used to start and stop training, and view results and graphs"""
    return render_template("index.html")

@app.route("/api/docs")
def api_docs():
    """Browseaable API GET rotues"""
    return """
    <h2>
        <a href="/">Back to main page</a>
    </h2>
    <ul>
        <li><a href="/api/training/status">Status</a></li>
        <li><a href="/api/latest">Latest Log</a></li>
        <li><a href="/api/live">Live Log</a></li>
        <li><a href="/api/history">History log</a></li>
    </ul>
    """

@app.route("/api/training/start", methods=["POST"])
def api_start():
    """API route used to start training"""
    data = request.get_json() or {}
    if "training" in active_tasks:
        pid = active_tasks["training"]["pid"]
        if is_process_running(pid):
            return jsonify({"status": "error", "message": "Training already in progress"}), 400
        else:
            active_tasks.pop("training")

    model = data.get("model")
    parallelism = data.get("parallelism")
    saved = data.get("saved")

    log_path = Config.LATEST_LOG
    try:
        os.remove(log_path)
    except FileNotFoundError:
        pass

    try:
        proc = subprocess.Popen(
                ["python", "-m", "src.launch", parallelism, saved], cwd=Config.ROOT_DIR)
        
        active_tasks["training"] = {
            "proc": proc,
            "pid": proc.pid,
            "parallelism": parallelism,
            "model": model
        }
        
        return jsonify({
            "status": "success",
            "pid": proc.pid,
            "message": f"Started {parallelism} training."
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
        subprocess.run(
            ["python", "-m", "src.stop", task_info["parallelism"]], cwd=Config.ROOT_DIR)
            
        active_tasks.pop("training")
        return jsonify({"status": "success", "message": "Training stopped"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/training/status", methods=["GET"])
def api_status():
    """API route returning training status"""
    
    if is_process_running():
        task_data = active_tasks["training"]
        details = {
            "pid": task_data.get("pid"),
            "parallelism": task_data.get("parallelism"),
            "model": task_data.get("model")
        }
        
        return jsonify({
            "active": True,
            "details": details
        })
    
    active_tasks.pop("training", None)
    return jsonify({"active": False})

@app.route("/api/logs/clear", methods=["POST"]) # TODO delete correct history file/dir
def api_clear_logs():
    """API route used to delete log files"""
    data = request.get_json() or {}
    target = data.get("target")
    
    paths = {
        "latest": Config.LATEST_LOG,
        "history": Config.HISTORY_LOG
    }
    
    if target not in paths:
        return jsonify({"status": "error", "message": "Invalid target"}), 400
    
    try:
        os.remove(paths[target])
    except FileNotFoundError:
        pass
    
    return jsonify({"status": "success"})

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
    
    return False