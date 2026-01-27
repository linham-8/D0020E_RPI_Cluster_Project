from flask import Flask, request, redirect, url_for, render_template
import json
import subprocess
import os
import signal

app = Flask(__name__)
active_tasks = {}

base_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(base_dir)
launch_script = os.path.join(code_dir, "launch.py")
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form['action']
        if action == 'start': 
            model = request.form['model']
            parallelism = request.form['parallelism']
            saved = request.form['saved']
            
            proc = subprocess.Popen(["python", launch_script, parallelism, saved])

            active_tasks['training'] = proc.pid
            return redirect(url_for('index', model=model, parallelism=parallelism))
            
        elif action == 'stop':
            pid = active_tasks.get('training')
            if pid:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    active_tasks.pop('training', None)
                except ProcessLookupError:
                    pass
            return redirect(url_for('index', model=None, parallelism=None))

    model = None
    parallelism = None
    accuracy = None
    execution_time = None
    throughput = None
    latency_per_batch = None
    world_size = None
    epochs = None
    timestamp = "N/A"
    has_data = False

    try:
        with open("/scratch/temp/data_parallel.log", "r") as f:
            data = json.load(f)
            parallelism = data.get("model_type", "N/A")
            accuracy = data.get("accuracy", "N/A")
            execution_time = data.get("execution_time", "N/A")
            throughput = data.get("throughput", "N/A")
            latency_per_batch = data.get("latency_per_batch", "N/A")
            world_size = data.get("world_size", "N/A")
            epochs = data.get("epochs", "N/A")
            timestamp = data.get("timestamp", "N/A")
            has_data=True
    except FileNotFoundError:
        has_data=False
    
    return render_template("index.html", 
    model=model,
    parallelism=parallelism,
    accuracy=accuracy,
    execution_time=execution_time,
    throughput=throughput,
    latency_per_batch=latency_per_batch,
    world_size=world_size,
    epochs=epochs,
    timestamp=timestamp,
    has_data=has_data)

@app.route('/api/status', methods=['GET'])
def api_status():
    pid = active_tasks.get('training')
    status = 'running' if pid else 'stopped'
    return json.dumps({'status': status})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000) # Använd (, Debug=True) om hotloading.