import subprocess
import sys
import os
import shutil
import argparse
from config import Config

COMPUTE_NODES = Config.COMPUTE_NODES
TEMP_DIR = str(Config.TEMP_DIR)

SUPPORTED_MODELS = Config.SUPPORTED_MODELS

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
            cmd = ["ssh", node, f"pkill -f '{pattern}'"]
        else:
            cmd = ["pkill", "-f", pattern]

        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error at pkill ({node if node else 'local'}): {e}")

def kill_run_nodes(selected_model=None):
    """Dödar run_node.py processer, antingen för en specifik modell eller alla."""
    pattern = f"run_node.py --model {selected_model}" if selected_model else "run_node.py"
    kill_process(pattern)
    for node in COMPUTE_NODES:
        kill_process(pattern, node)

def clean_live_folders():
    """Rensar enbart live-mappen (används när en ny körning startas)."""
    shutil.rmtree(os.path.join(TEMP_DIR, "live"), ignore_errors=True)

def clean_temp_folders():
    """Rensar mapparna live och latest."""
    shutil.rmtree(os.path.join(TEMP_DIR, "live"), ignore_errors=True)
    shutil.rmtree(os.path.join(TEMP_DIR, "latest"), ignore_errors=True)

def clean_history():
    """Rensar historikloggen."""
    safe_remove(str(Config.HISTORY_LOG))

def stop_cluster(selected_model):
    """Stoppar en specifik modell."""
    print(f"Stopping {selected_model}.")

    kill_run_nodes(selected_model)
    kill_process("launch.py")

    print(f"Cleaning sync files.")
    safe_remove(os.path.join(TEMP_DIR, f"{selected_model}_sync"))

    print(f"Done.")

def stop_session():
    """Stoppar processer men behåller loggar för hemsidan."""
    print(f"Stopping the session and all models.")

    kill_run_nodes()
    kill_process("launch.py")

    print(f"Cleaning sync files.")
    for model in SUPPORTED_MODELS:
        safe_remove(os.path.join(TEMP_DIR, f"{model}_sync"))

    print(f"Session stopped.")

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Stop the cluster.")
        parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, help="Stop a specific model", default=None)
        args = parser.parse_args()

        if args.model:
            stop_cluster(args.model)
        else:
            stop_session()
    except KeyboardInterrupt:
        print(f"Stopped by user.")
    except Exception as e:
        print(f"Something went wrong in stop.py: {e}")