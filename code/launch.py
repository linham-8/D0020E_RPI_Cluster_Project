import subprocess
import sys
import os
import datetime
import time
import shutil

COMPUTE_NODES = ["pi1", "pi2", "pi3", "pi4"]
TEMP_DIR = "/scratch/temp"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("/mnt/usb/scratch", "/scratch", 1)
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODELS = {
    "data_parallel": "data_parallel.py",
    "expert_parallel": "expert_parallel.py",
    "pipeline_parallel": "pipeline_parallel.py",
    "model_parallel": "model_parallel.py",
}

def clean_file(path):
    """Tar bort en fil säkert om den finns."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(f"Could not remove {path}: {e}")

def zero_file(path):
    """Tömmer en fil utan att ta bort den. Skapar den om den inte finns."""
    try:
        with open(path, "w") as f:
            pass
    except OSError as e:
        print(f"Could not reset {path}: {e}")

def kill_remote_process(node, pattern):
    """Dödar processer på en nod via SSH."""
    try:
        subprocess.run(
            ["ssh", node, f"pkill -f {pattern}"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Could not kill processes at {node}: {e}")

def start_cluster(selected_model, model_path, saved_choice):
    """Startar Klustret"""
    model_filename = MODELS[selected_model]
    processes = []

    print(f"Initiating cluster for: {selected_model}")

    sync_file = os.path.join(TEMP_DIR, f"{selected_model}_sync")
    clean_file(sync_file)

    live_log_path = os.path.join(TEMP_DIR, "live.log")
    latest_log_path = os.path.join(TEMP_DIR, "latest.log")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "train" if saved_choice == "no" else "test"
    archive_dir = os.path.join(TEMP_DIR, "archive", f"{run_id}_{selected_model}_{mode}")

    try:
        os.makedirs(archive_dir, exist_ok=True)
        print(f"Archive created: {archive_dir}")
    except OSError as e:
        print(f"Could not create archive folder: {e}")
        sys.exit(1)

    if saved_choice == "no":
        zero_file(live_log_path)
        zero_file(latest_log_path)
    else:
        print(f'Running in test mode (saved="yes"). Result saved in test archive.')

    print(f"Cleaning old processes")
    subprocess.run(["pkill", "-f", model_filename], check=False)
    for node in COMPUTE_NODES:
        kill_remote_process(node, model_filename)

    print(f"Starting head node")
    try:
        cmd_0 = ["python", "-u", model_path, "0", saved_choice, archive_dir]
        process_0 = subprocess.Popen(cmd_0)
        processes.append(process_0)
    except OSError as e:
        print(f"Could not start head node: {e}")
        sys.exit(1)

    for rank, node in enumerate(COMPUTE_NODES, start=1):
        print(f"Starting {rank} on {node}.")
        try:
            remote_cmd = f"python -u {model_path} {rank} {saved_choice} {archive_dir}"
            ssh_cmd = ["ssh", node, remote_cmd]
            
            p = subprocess.Popen(ssh_cmd)
            processes.append(p)
        except OSError as e:
            print(f"Could not start Rank: {rank} on {node}: {e}")

    print(f"All nodes started. Waiting for head node to finish.")
    
    try:
        exit_code = process_0.wait()
        if exit_code == 0:
            print(f"Finished successfully.")
        else:
            print(f"Finished with errors (Exit code: {exit_code})")
    except KeyboardInterrupt:
        print(f"Stopping the cluster.")
        subprocess.run(["python", os.path.join(os.path.dirname(__file__), "stop.py"), selected_model])

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(f"Usage: python launch.py <model_name> [saved=no]")
            print(f"Available models: {', '.join(MODELS.keys())}")
            sys.exit(1)

        selected_model = sys.argv[1]
        saved_choice = sys.argv[2] if len(sys.argv) > 2 else "no"

        if selected_model in MODELS:
            final_path = os.path.join(MODELS_DIR, MODELS[selected_model])
            if not os.path.exists(final_path):
                print(f"File: {final_path} not found.")
                sys.exit(1)
                
            start_cluster(selected_model, final_path, saved_choice)
        else:
            print(f"Model '{selected_model}' doesn't exist.")
            print(f"Choose one of: {', '.join(MODELS.keys())}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Something went wrong in launch.py: {e}")
        sys.exit(1)