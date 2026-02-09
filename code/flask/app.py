from flask import Flask, request, redirect, url_for, render_template, jsonify
from utils import read_latest_log, read_history_log
import subprocess
import os
import signal

app = Flask(__name__)
active_tasks = {}
has_data = False

base_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(base_dir)
launch_script = os.path.join(code_dir, "launch.py")

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
    return jsonify(read_history_log())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
