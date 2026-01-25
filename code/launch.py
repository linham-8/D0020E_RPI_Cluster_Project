import subprocess
import sys
import os
import time

compute_nodes = ["pi1", "pi2", "pi3", "pi4"]
path = "/scratch/D0020E_RPI_Cluster_Project/code/models"

models = {
    "data_parallel": "data_parallel.py",
    "expert_parallel": "expert_parallel.py",
    "pipeline_parallel": "pipeline_parallel.py",
    "model_parallel": "model_parallel.py"
}

def start_cluster(model_selected, model_path):
    model_filename = models[model_selected]
    processes = []
    print(f"Cleaning up for {model_selected}")
    clean_cmd = f"rm -f /scratch/temp/{model_selected}_sync /scratch/temp/{model_selected}.log /scratch/temp/{model_selected}*.pt"
    subprocess.run(clean_cmd, shell=True)

    subprocess.run(f"pkill -f {model_filename}", shell=True)
    for node in compute_nodes:
        subprocess.run(f"ssh {node} 'pkill -f {model_filename}'", shell=True)
    
    time.sleep(1)

    print(f"Starting Head Node for {model_selected}")
    process_0 = subprocess.Popen(["python", "-u", model_path, "0"])
    processes.append(process_0)

    for rank, node in enumerate(compute_nodes, start=1):
        print(f"Starting Rank {rank} on {node}")
        cmd = f"ssh {node} 'python -u {model_path} {rank}'"
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    print(f"All nodes started. Waiting for head node")
    process_0.wait()
    print(f"Finished.")

if __name__ == "__main__":
    model_selected = sys.argv[1]
    final_path = os.path.join(path, models[model_selected])
    start_cluster(model_selected, final_path)