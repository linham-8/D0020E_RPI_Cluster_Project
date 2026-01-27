import subprocess
import sys
import os
import time

compute_nodes = ["pi1", "pi2", "pi3", "pi4"]

base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = base_dir.replace("/mnt/usb/scratch", "/scratch", 1) # Nån anledning letar den efter /mnt/usb/scratch. Weird.

models_dir = os.path.join(base_dir, "models")
temp_dir = "/scratch/temp"

models = {
    "data_parallel": "data_parallel.py",
    "expert_parallel": "expert_parallel.py",
    "pipeline_parallel": "pipeline_parallel.py",
    "model_parallel": "model_parallel.py"
}

def start_cluster(model_selected, model_path, saved_choice):
    model_filename = models[model_selected]
    processes = []
    
    clean_cmd = f"rm -f {temp_dir}/{model_selected}_sync {temp_dir}/{model_selected}.log"
    subprocess.run(clean_cmd, shell=True)

    subprocess.run(f"pkill -f {model_filename}", shell=True)
    for node in compute_nodes:
        subprocess.run(f"ssh {node} 'pkill -f {model_filename}'", shell=True)
    
    time.sleep(1)

    print(f"Starting Head Node for {model_selected} (Saved: {saved_choice})")
    process_0 = subprocess.Popen(["python", "-u", model_path, "0", saved_choice])
    processes.append(process_0)

    for rank, node in enumerate(compute_nodes, start=1):
        print(f"Starting Rank {rank} on {node}")
        cmd = f"ssh {node} 'python -u {model_path} {rank} {saved_choice}'"
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    print(f"All nodes started. Waiting for head node")
    process_0.wait()
    print(f"Finished.")

if __name__ == "__main__":
      
    model_selected = sys.argv[1]
    saved_choice = sys.argv[2]
    
    if model_selected in models:
        final_path = os.path.join(models_dir, models[model_selected])
        start_cluster(model_selected, final_path, saved_choice)