from flask import Flask, request, redirect, url_for, render_template, jsonify
from .utils.log_reader import read_latest_log, read_history_log, get_archived_runs
import subprocess
import os
import signal
import psutil
from config import Config

app = Flask(__name__)
active_tasks = {}
has_data = False

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page route used to start and stop training, and view results and graphs"""
    if request.method == "POST":
        action = request.form["action"]
        if action == "start":
            model = request.form["model"]
            parallelism = request.form["parallelism"]
            saved = request.form["saved"]

            print(
                f"Starting training: Model={model}, Parallelism={parallelism}, Saved={saved}"
            )
            proc = subprocess.Popen(
                ["python", "-m", "src.launch", parallelism, saved], cwd=Config.ROOT_DIR)

            active_tasks["training"] = {"pid": proc.pid, "parallelism": parallelism}
            print(f"Started training with PID {proc.pid}")
            return redirect(url_for("index"))

        elif action == "stop":
            task_info = active_tasks.get("training")
            if task_info:
                parallelism = task_info["parallelism"]

                print(f"Running stop script for: {parallelism}")
                subprocess.run(["python", "-m", "src.stop", parallelism], cwd=Config.ROOT_DIR)

                active_tasks.pop("training", None)
            return redirect(url_for("index"))

        elif action == "clear-latest":
            try:
                os.remove("/scratch/temp/latest.log")
            except FileNotFoundError:
                pass
            return redirect(url_for("index"))
        
        elif action == "clear-history":
            try:
                os.remove("/scratch/temp/history.log")
            except FileNotFoundError:
                pass
            return redirect(url_for("index"))

    return render_template("index.html")

@app.route("/api/docs")
def api_docs():
    """Browseaable API GET rotues"""
    return """
    <h2>
        <a href="/">Back to main page</a>
    </h2>
    <ul>
        <li><a href="/api/status">Status</a></li>
        <li><a href="/api/latest">Latest Log</a></li>
        <li><a href="/api/live">Live Log</a></li>
        <li><a href="/api/history">History log</a></li>
    </ul>
    """

@app.route("/api/start", methods=["POST"])
def api_start():
    """API route used to start training"""
    # Start logic here
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    """API route used to stop training"""
    ## Stop logic here
    return jsonify({"status": "stopped"})

@app.route("/api/status")
def api_status():
    """API route returning training status"""
    task_info = active_tasks.get("training")
    if task_info and is_process_running(task_info["pid"]):
        status = "running"
    else:
        active_tasks.pop("training", None)
        status = "stopped"
        
    return jsonify({"status": status})

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
def is_process_running(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False