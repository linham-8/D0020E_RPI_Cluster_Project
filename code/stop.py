import subprocess
import sys
import os

COMPUTE_NODES = ["pi1", "pi2", "pi3", "pi4"]
TEMP_DIR = "/scratch/temp"

MODELS = {
    "data_parallel": "data_parallel.py",
    "expert_parallel": "expert_parallel.py",
    "pipeline_parallel": "pipeline_parallel.py",
    "model_parallel": "model_parallel.py",
}

def safe_remove(path):
    """Tar bort fil utan att krascha om den saknas."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(f"Could not remove {path}: {e}.")

def kill_process(pattern, node=None):
    """Dödar processer lokalt eller remote."""
    try:
        if node:
            cmd = ["ssh", node, f"pkill -f {pattern}"]
        else:
            cmd = ["pkill", "-f", pattern]
        
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error at pkill ({node if node else 'local'}): {e}")

def stop_cluster(selected_model):
    """Stoppar en specifik modell."""
    model_filename = MODELS[selected_model]
    print(f"Stopping {selected_model}.")
    
    kill_process(model_filename)
    kill_process("launch.py")

    for node in COMPUTE_NODES:
        kill_process(model_filename, node)
            
    print(f"Cleaning temporary files.")
    safe_remove(os.path.join(TEMP_DIR, f"{selected_model}_sync"))
    safe_remove(os.path.join(TEMP_DIR, "live.log"))
    safe_remove(os.path.join(TEMP_DIR, "latest.log"))
    
    print(f"Done.")

def stop_session():
    """Stoppar allt och rensar rubbet."""
    print(f"Stopping the session and all models.")
    
    for model_file in MODELS.values():
        kill_process(model_file)
        for node in COMPUTE_NODES:
            kill_process(model_file, node)

    kill_process("launch.py")
    
    print(f"Cleaning all logs and sync files.")
    
    safe_remove(os.path.join(TEMP_DIR, "live.log"))
    safe_remove(os.path.join(TEMP_DIR, "latest.log"))
    safe_remove(os.path.join(TEMP_DIR, "history.log"))
    
    for model in MODELS:
        safe_remove(os.path.join(TEMP_DIR, f"{model}_sync"))
        
    print(f"Session cleared.")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] in MODELS:
            stop_cluster(sys.argv[1])
        else:
            stop_session()
    except KeyboardInterrupt:
        print(f"Stopped by user.")
    except Exception as e:
        print(f"Something went wrong in stop.py: {e}")
