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

launch_script = os.path.join(CODE_DIR, "launch.py")

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page route used to start and stop training, and view results and graphs"""
    if request.method == "POST":
        action = request.form["action"]
        if action == "start":
            model = request.form["model"]
            parallelism = request.form["parallelism"]
            saved = request.form["saved"]

            time_limit_min = request.form.get("time_limit", "0")
            time_limit_sec = int(time_limit_min) * 60 if time_limit_min.isdigit() else 0

            custom_name = request.form.get("custom_name", "").strip()

            print(
                f"Starting training: Model={model}, Parallelism={parallelism}, Saved={saved}, TimeLimit={time_limit_sec}s, Name={custom_name}"
            )

            cmd = [
                "python", launch_script,
                "--model", parallelism,
                "--saved", saved,
                "--time_limit", str(time_limit_sec)
            ]

            if custom_name:
                cmd.extend(["--name", custom_name])

            proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

            active_tasks["training"] = proc.pid
            print(f"Started training with PID {proc.pid}")
            return redirect(url_for("index"))

        elif action == "stop":
            stop.stop_session()
            active_tasks.pop("training", None)
            return redirect(url_for("index"))

        elif action == "clear-latest":
            stop.clean_temp_folders()
            return redirect(url_for("index"))

        elif action == "clear-history":
            stop.clean_history()
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
    pid = active_tasks.get("training")
    status = "running" if pid else "stopped"
    return jsonify({"status": status})

@app.route("/api/latest")
def api_latest():
    """API route returning the latest log"""
    return jsonify(read_latest_log())

@app.route("/api/history")
def api_history():
    """API route returning the history log"""
    return jsonify(get_archived_runs(filter_type=None))

@app.route("/api/archives/<model_type>")
def api_archives(model_type):
    """Api route returing the archived log"""
    runs = get_archived_runs(filter_type=model_type)
    return jsonify(runs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
