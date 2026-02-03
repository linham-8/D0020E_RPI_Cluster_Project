import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
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
    store=dist.FileStore("/scratch/temp/expert_parallel_sync", world_size=world_size),
)

print(f"Rank {rank}: Connected")


def load(path, offset):
    with open(path, "rb") as f:
        return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)


if rank == 0:
    X = (
        load("/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte", 16)
        .float()
        .reshape(-1, 784)
        / 255.0
    )
    Y = load("/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte", 8).long()

    Xt = (
        load("/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte", 16)
        .float()
        .reshape(-1, 784)
        / 255.0
    )
    Yt = load("/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte", 8).long()

    gate_model = nn.Linear(784, 4)
    gate_opt = torch.optim.SGD(gate_model.parameters(), lr=0.01)

    if use_saved == "yes":
        gate_model.load_state_dict(
            torch.load("/scratch/temp/expert_parallel_gate_model.pt")
        )

    dist.barrier()
    start_time = time.time()

    if use_saved != "yes":
        dist.broadcast(torch.tensor([1]), src=0)
        last_log_time = 0

        for epoch in range(5):
            perm = torch.randperm(len(X))
            X_shuff = X[perm]
            Y_shuff = Y[perm]

            for i in range(0, len(X), 64):
                batch_x = X_shuff[i : i + 64]
                batch_y = Y_shuff[i : i + 64]
                if len(batch_x) < 64:
                    continue

                dist.broadcast(torch.tensor([1]), src=0)

                with torch.no_grad():
                    gate_scores = gate_model(batch_x)
                    expert_assignments = torch.argmax(gate_scores, dim=1) + 1

                expert_losses = []
                for r in range(1, 5):
                    mask = expert_assignments == r
                    sub_x = batch_x[mask]
                    sub_y = batch_y[mask]

                    count = torch.tensor([len(sub_x)])
                    dist.send(count, dst=r)

                    if len(sub_x) > 0:
                        dist.send(sub_x, dst=r)
                        pred_buffer = torch.zeros(len(sub_x), 10)
                        dist.recv(pred_buffer, src=r)

                        pred_buffer.requires_grad = True
                        loss = nn.CrossEntropyLoss()(pred_buffer, sub_y)
                        loss.backward()
                        expert_losses.append(loss.item())
                        dist.send(pred_buffer.grad, dst=r)

                gate_opt.zero_grad()
                gate_loss = torch.tensor(expert_losses).sum()
                gate_loss.requires_grad = True
                gate_opt.step()

                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    current_image = epoch * len(X) + i
                    total_images = len(X) * 5
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

        dist.broadcast(torch.tensor([0]), src=0)
    else:
        dist.broadcast(torch.tensor([0]), src=0)

    end_time = time.time()

    dist.broadcast(torch.tensor([2]), src=0)

    correct = 0
    total = 0
    test_start = time.time()

    with torch.no_grad():
        for i in range(0, len(Xt), 64):
            batch_xt = Xt[i : i + 64]
            batch_yt = Yt[i : i + 64]
            if len(batch_xt) < 64:
                continue

            dist.broadcast(torch.tensor([1]), src=0)
            gate_scores = gate_model(batch_xt)
            expert_assignments = torch.argmax(gate_scores, dim=1) + 1
            batch_preds = torch.zeros(64, 10)

            for r in range(1, 5):
                mask = expert_assignments == r
                sub_x = batch_xt[mask]

                count = torch.tensor([len(sub_x)])
                dist.send(count, dst=r)

                if len(sub_x) > 0:
                    dist.send(sub_x, dst=r)
                    pred_buffer = torch.zeros(len(sub_x), 10)
                    dist.recv(pred_buffer, src=r)
                    batch_preds[mask] = pred_buffer

            acc = (batch_preds.argmax(dim=1) == batch_yt).sum().item()
            correct += acc
            total += 64

    test_time = time.time() - test_start
    dist.broadcast(torch.tensor([0]), src=0)

    final_acc = (correct / total) * 100
    print(f"Accuracy: {final_acc:.2f}%")

    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f}s")

    if use_saved != "yes":
        total_images_processed = len(X) * 5
        throughput = total_images_processed / training_time
        total_batches = (len(X) / 64) * 5
        avg_batch_latency = training_time / total_batches
    else:
        throughput = len(Xt) / test_time
        avg_batch_latency = test_time / (len(Xt) / 64)

    log = {
        "parallelism_type": "expert_parallel",
        "accuracy": float(final_acc),
        "training_time": round(training_time, 2),
        "test_time": round(test_time, 2),
        "throughput": round(throughput, 2),
        "latency_per_batch": round(avg_batch_latency, 2),
        "world_size": world_size,
        "epochs": 5 if use_saved != "yes" else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if use_saved != "yes":
        torch.save(
            gate_model.state_dict(), "/scratch/temp/expert_parallel_gate_model.pt"
        )

    with open("/scratch/temp/latest.log", "w") as f:
        f.write(json.dumps(log))
    with open("/scratch/temp/history.log", "a") as f:
        f.write(json.dumps(log) + "\n")

else:
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    if use_saved == "yes":
        model.load_state_dict(
            torch.load(f"/scratch/temp/expert_parallel_rank{rank}_model.pt")
        )

    dist.barrier()

    mode_signal = torch.tensor([0])
    dist.broadcast(mode_signal, src=0)

    if mode_signal.item() == 1:
        step_signal = torch.tensor([0])
        while True:
            dist.broadcast(step_signal, src=0)
            if step_signal.item() == 0:
                break

            count_tensor = torch.tensor([0])
            dist.recv(count_tensor, src=0)
            curr_count = count_tensor.item()

            if curr_count > 0:
                input_data = torch.zeros(curr_count, 784)
                dist.recv(input_data, src=0)

                input_data.requires_grad = True
                opt.zero_grad()
                output = model(input_data)

                dist.send(output, dst=0)

                grad_in = torch.zeros(curr_count, 10)
                dist.recv(grad_in, src=0)

                output.backward(grad_in)
                opt.step()

        torch.save(
            model.state_dict(), f"/scratch/temp/expert_parallel_rank{rank}_model.pt"
        )

    dist.broadcast(mode_signal, src=0)
    if mode_signal.item() == 2:
        step_signal = torch.tensor([0])
        with torch.no_grad():
            while True:
                dist.broadcast(step_signal, src=0)
                if step_signal.item() == 0:
                    break

                count_tensor = torch.tensor([0])
                dist.recv(count_tensor, src=0)
                curr_count = count_tensor.item()

                if curr_count > 0:
                    input_data = torch.zeros(curr_count, 784)
                    dist.recv(input_data, src=0)
                    output = model(input_data)
                    dist.send(output, dst=0)

dist.destroy_process_group()
