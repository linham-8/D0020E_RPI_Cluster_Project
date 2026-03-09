import subprocess
import sys
import os
import datetime
import json
import argparse
import shutil
from config import Config
import stop

COMPUTE_NODES = Config.COMPUTE_NODES
TEMP_DIR = str(Config.TEMP_DIR)
BASE_DIR = str(Config.BASE_DIR)
MODELS_DIR = str(Config.MODELS_DIR)
ARCHIVE_DIR = str(Config.ARCHIVE_DIR)

PYTHON_EXEC = "python3"

SUPPORTED_MODELS = Config.SUPPORTED_MODELS

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


def start_cluster(selected_model, saved_choice, custom_name=None, time_limit=0):
    """Startar Klustret"""
    processes = []
    print(f"Initiating cluster for: {selected_model}")

    if saved_choice == "no":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_dir_name = f"{custom_name}_{timestamp}" if custom_name else f"{selected_model}_{timestamp}"
        archive_dir = os.path.join(ARCHIVE_DIR, archive_dir_name)
        use_saved_flag = "no"

        try:
            os.makedirs(archive_dir, exist_ok=True)
            print(f"Archive created: {archive_dir}")
        except OSError as e:
            print(f"Could not create archive folder: {e}")
            sys.exit(1)
    else:
        archive_dir = saved_choice
        use_saved_flag = "yes"
        if not os.path.isdir(archive_dir):
            print(f"Archive directory not found: {archive_dir}")
            sys.exit(1)
        print(f"Using existing archive: {archive_dir}")

    sync_file = os.path.join(TEMP_DIR, f"{selected_model}_sync")
    clean_file(sync_file)

    latest_log_path = os.path.join(TEMP_DIR, "latest", "latest.log")

    if saved_choice == "no":
        stop.clean_live_folders()
        os.makedirs(os.path.join(TEMP_DIR, "live", "nodes"), exist_ok=True)
        os.makedirs(os.path.join(TEMP_DIR, "latest", "nodes"), exist_ok=True)
    else:
        print(f'Running in test mode (saved="yes"). Result saved in test archive.')

    print(f"Cleaning old processes")
    stop.kill_run_nodes()

    print(f"Starting head node")
    runner_script = os.path.join(BASE_DIR, "run_node.py")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = BASE_DIR

        cmd_0 = [
            PYTHON_EXEC, "-u", runner_script,
            "--model", selected_model,
            "--rank", "0",
            "--use_saved", use_saved_flag,
            "--archive_dir", str(archive_dir),
            "--time_limit", str(time_limit)
        ]
        process_0 = subprocess.Popen(cmd_0, env=env)
        processes.append(process_0)
    except OSError as e:
        print(f"Could not start head node: {e}")
        sys.exit(1)

    for rank, node in enumerate(COMPUTE_NODES, start=1):
        print(f"Starting {rank} on {node}.")
        try:
            remote_cmd = f"PYTHONPATH={BASE_DIR} {PYTHON_EXEC} -u {runner_script} --model {selected_model} --rank {rank} --use_saved {use_saved_flag} --archive_dir {archive_dir} --time_limit {time_limit}"
            ssh_cmd = ["ssh", node, remote_cmd]

            p = subprocess.Popen(ssh_cmd)
            processes.append(p)
        except OSError as e:
            print(f"Could not start Rank: {rank} on {node}: {e}")

    print(f"All nodes started. Waiting for all nodes to finish.")

    try:
        exit_code = process_0.wait()

        for p in processes:
            if p != process_0:
                p.wait()

        clean_file(sync_file)

        if exit_code == 0:
            print(f"Finished successfully.")
            if use_saved_flag == "yes" and os.path.exists(latest_log_path):
                try:
                    with open(latest_log_path, 'r') as f:
                        data = json.load(f)

                    if data.get("type") == "test_result":
                        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        logs_dir = os.path.join(archive_dir, "latest")
                        os.makedirs(logs_dir, exist_ok=True)
                        dest = os.path.join(logs_dir, f"test_log_{ts}.log")
                        shutil.copy(latest_log_path, dest)
                        print(f"Saved unique test log to: {dest}")
                except Exception as e:
                    print(f"Warning: Could not save unique test log: {e}")
        else:
            print(f"Finished with errors (Exit code: {exit_code})")

    except KeyboardInterrupt:
        print(f"Stopping the cluster.")
        subprocess.run([PYTHON_EXEC, os.path.join(os.path.dirname(__file__), "stop.py"), "--model", selected_model])

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Launch the PyTorch cluster.")
        parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS, help=f"Model to run. Available: {', '.join(SUPPORTED_MODELS)}")
        parser.add_argument("--saved", type=str, default="no", help="Use saved data (provide path or 'no')")
        parser.add_argument("--name", type=str, default=None, help="Custom name for the benchmark run")
        parser.add_argument("--time_limit", type=int, default=0, help="Time limit for training in seconds (0 = use epochs)")

        args = parser.parse_args()

        start_cluster(args.model, args.saved, args.name, args.time_limit)

    except Exception as e:
        print(f"Something went wrong in launch.py: {e}")
        sys.exit(1)