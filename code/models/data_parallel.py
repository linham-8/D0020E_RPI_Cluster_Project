import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import time

os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
rank = int(sys.argv[1])
use_saved = sys.argv[2] if len(sys.argv) > 2 else "no"
world_size = 5

print(f"Rank {rank}: Trying to connect")

dist.init_process_group(
    backend="gloo",
    rank=rank,
    world_size=world_size,
    store=dist.FileStore("/scratch/temp/data_parallel_sync", world_size=world_size),
)

print(f"Rank {rank}: Connected")


def load(path, offset):
    with open(path, "rb") as f:
        return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)


X = (
    load("/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte", 16)
    .float()
    .reshape(-1, 784)
    / 255.0
)
Y = load("/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte", 8).long()

X = X[rank::world_size]
Y = Y[rank::world_size]

model = DDP(torch.nn.Linear(784, 10))
opt = torch.optim.SGD(model.parameters(), lr=0.01)
crit = torch.nn.CrossEntropyLoss()

if use_saved == "yes":
    model.load_state_dict(torch.load("/scratch/temp/data_parallel_model.pt"))

dist.barrier()
if rank == 0:
    start_time = time.time()

if use_saved != "yes":
    last_log_time = 0
    for epoch in range(5):
        for i in range(0, len(X), 64):
            opt.zero_grad()
            loss = crit(model(X[i : i + 64]), Y[i : i + 64])
            loss.backward()
            opt.step()

            if rank == 0:
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    current_image = (epoch * len(X) + i) * world_size
                    total_images = len(X) * 5 * world_size
                    live_log = {
                        "progress": round((current_image / total_images) * 100, 2),
                        "current": current_image,
                        "total": total_images,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    with open("/scratch/temp/live.log", "a") as f:
                        json.dump(live_log, f)
                        f.write("\n")
                    last_log_time = current_time

if rank == 0:
    end_time = time.time()
    training_time = end_time - start_time

    if use_saved != "yes":
        torch.save(model.state_dict(), "/scratch/temp/data_parallel_model.pt")

    Xt = (
        load("/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte", 16)
        .float()
        .reshape(-1, 784)
        / 255.0
    )
    Yt = load("/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte", 8).long()

    test_start = time.time()
    with torch.no_grad():
        acc = (model(Xt).argmax(dim=1) == Yt).float().mean() * 100
    test_time = time.time() - test_start

    print(f"Accuracy: {acc:.2f}%")
    print(f"Total Training Time: {training_time:.2f}s")

    if use_saved != "yes":
        total_images_processed = len(X) * world_size * 5
        throughput = total_images_processed / training_time
        total_batches = (len(X) / 64) * 5
        avg_batch_latency = training_time / total_batches
    else:
        throughput = len(Xt) / test_time
        avg_batch_latency = test_time / (len(Xt) / 64)

    log = {
        "parallelism_type": "data_parallel",
        "accuracy": float(acc),
        "training_time": round(training_time, 2),
        "test_time": round(test_time, 2),
        "throughput": round(throughput, 2),
        "latency_per_batch": round(avg_batch_latency, 2),
        "world_size": world_size,
        "epochs": 5 if use_saved != "yes" else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("/scratch/temp/latest.log", "w") as f:
        f.write(json.dumps(log))
    with open("/scratch/temp/history.log", "a") as f:
        f.write(json.dumps(log) + "\n")

dist.destroy_process_group()
