import subprocess
import sys
import os

compute_nodes = ["pi1", "pi2", "pi3", "pi4"]
temp_dir = "/scratch/temp"

models = {
    "data_parallel": "data_parallel.py",
    "expert_parallel": "expert_parallel.py",
    "pipeline_parallel": "pipeline_parallel.py",
    "model_parallel": "model_parallel.py"
}

def stop_cluster(selected_model):
    model_filename = models[selected_model]
    print(f"Stopping {selected_model}")
    
    subprocess.run(f"pkill -f {model_filename}", shell=True)
    subprocess.run(f"pkill -f launch.py", shell=True)

    for node in compute_nodes:
        subprocess.run(f"ssh {node} 'pkill -f {model_filename}'", shell=True)
            
    subprocess.run(f"rm -f {temp_dir}/{selected_model}_sync", shell=True)
    subprocess.run(f"rm -f {temp_dir}/{selected_model}.log", shell=True)

if __name__ == "__main__":
    selected_model = sys.argv[1]
    if selected_model in models:
        stop_cluster(selected_model)