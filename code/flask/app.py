from flask import Flask
import json

app = Flask(__name__)

@app.route('/')
def index():
    try:
        with open("/scratch/temp/accuracy.log", "r") as f:
            data = json.load(f)
            accuracy = data.get("accuracy", "N/A")
            return f"<h1>Result</h1><p>Accuracy: {accuracy}%</p>"
    except FileNotFoundError:
        return f"<h1>No Data Yet</h1><p>Waiting for log file</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000) # Använd (, Debug=True) om hotloading.