import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import json
import time

def get_unique_filename(directory, base_name, extension):
    """Hittar ett unikt filnamn genom att lägga till _1, _2 osv."""
    counter = 1
    file_path = os.path.join(directory, f"{base_name}.{extension}")
    while os.path.exists(file_path):
        file_path = os.path.join(directory, f"{base_name}_{counter}.{extension}")
        counter += 1
    return file_path

def main():
    # Argumenthantering
    try:
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
        rank = int(sys.argv[1])
        use_saved = sys.argv[2] if len(sys.argv) > 2 else "no"

        archive_arg = sys.argv[3] if len(sys.argv) > 3 else "None"
        archive_dir = archive_arg if archive_arg != "None" else None

        world_size = 5
    except (IndexError, ValueError) as e:
        print(
            f"Invalid arguments. Usage: python pipeline_parallel.py <rank> <saved> <archive>. Error: {e}"
        )
        sys.exit(1)

    print(f"Rank {rank}: Starting initialization.")

    try:
        # Initiera processgrupp
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            store=dist.FileStore(
                "/scratch/temp/pipeline_parallel_sync", world_size=world_size
            ),
        )
        print(f"Rank {rank}: Connected to cluster.")

        def load(path, offset):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset not found at {path}")
            with open(path, "rb") as f:
                return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

        # Rank 0: Orchestrator
        if rank == 0:
            # Inläsning av data
            X = (
                load("/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte", 16)
                .float()
                .reshape(-1, 784)
                / 255.0
            )
            Y = load(
                "/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte", 8
            ).long()

            Xt = (
                load("/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte", 16)
                .float()
                .reshape(-1, 784)
                / 255.0
            )
            Yt = load(
                "/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte", 8
            ).long()

            dist.barrier()
            start_time = time.time()

            # Träningsloopen
            if use_saved != "yes":
                dist.broadcast(torch.tensor([1]), src=0)
                last_log_time = 0

                for epoch in range(5):
                    for i in range(0, len(X), 64):
                        batch_x = X[i : i + 64]
                        batch_y = Y[i : i + 64]
                        if len(batch_x) < 64:
                            continue

                        dist.broadcast(torch.tensor([1]), src=0)
                        dist.send(batch_x, dst=1)

                        preds = torch.zeros(64, 10)
                        dist.recv(preds, src=4)
                        preds.requires_grad = True
                        loss = nn.CrossEntropyLoss()(preds, batch_y)

                        grad_initial = torch.autograd.grad(loss, preds)[0]
                        dist.send(grad_initial, dst=4)

                        current_time = time.time()
                        if current_time - last_log_time >= 1.0:
                            current_img = epoch * len(X) + i
                            total_img = len(X) * 5
                            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")

                            live_log = {
                                "progress": round(
                                    (current_img / total_img) * 100, 2
                                ),
                                "current": current_img,
                                "total": total_img,
                                "timestamp": formatted_time,
                            }
                            try:
                                with open("/scratch/temp/live.log", "a") as f:
                                    json.dump(live_log, f)
                                    f.write("\n")
                                if archive_dir and os.path.isdir(archive_dir):
                                    with open(
                                        os.path.join(archive_dir, "live.log"), "a"
                                    ) as f:
                                        json.dump(live_log, f)
                                        f.write("\n")
                            except OSError as e:
                                print(f"Rank 0: Logging failed: {e}")

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
                    dist.send(batch_xt, dst=1)
                    preds = torch.zeros(64, 10)
                    dist.recv(preds, src=4)

                    acc = (preds.argmax(dim=1) == batch_yt).sum().item()
                    correct += acc
                    total += 64

            test_time = time.time() - test_start
            dist.broadcast(torch.tensor([0]), src=0)

            final_acc = (correct / total) * 100
            print(f"Accuracy: {final_acc:.2f}%")
            training_time = end_time - start_time
            print(f"Total Training Time: {training_time:.2f}s")

            if use_saved != "yes":
                total_batches = (len(X) / 64) * 5
                throughput = (len(X) * 5) / training_time
                avg_latency = (training_time / total_batches) * 1000

                log = {
                    "type": "training_result",
                    "model_type": "pipeline_parallel",
                    "accuracy": float(final_acc),
                    "training_time": round(training_time, 2),
                    "test_time": round(test_time, 2),
                    "throughput": round(throughput, 2),
                    "latency_per_batch_ms": round(avg_latency, 2),
                    "world_size": world_size,
                    "epochs": 5,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                log_filename = "training_result"
            else:
                throughput = len(Xt) / test_time
                avg_latency = (test_time / (len(Xt) / 64)) * 1000

                log = {
                    "type": "test_result",
                    "model_type": "pipeline_parallel",
                    "accuracy": float(final_acc),
                    "test_time": round(test_time, 2),
                    "inference_throughput": round(throughput, 2),
                    "inference_latency_ms": round(avg_latency, 2),
                    "world_size": world_size,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                log_filename = "test_result"

            try:
                with open("/scratch/temp/latest.log", "w") as f:
                    json.dump(log, f)

                with open("/scratch/temp/history.log", "a") as f:
                    json.dump(log, f)
                    f.write("\n")

                if archive_dir and os.path.isdir(archive_dir):
                    unique_path = get_unique_filename(archive_dir, log_filename, "log")
                    with open(unique_path, "w") as f:
                        json.dump(log, f)
                    with open(os.path.join(archive_dir, "history.log"), "a") as f:
                        json.dump(log, f)
                        f.write("\n")
            except OSError as e:
                print(f"Rank 0: Final logging failed: {e}")

        # Rank 1-4: Stages
        else:
            if rank == 1:
                model = nn.Sequential(nn.Linear(784, 128), nn.ReLU())
            elif rank == 2:
                model = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            elif rank == 3:
                model = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            elif rank == 4:
                model = nn.Linear(128, 10)

            opt = torch.optim.SGD(model.parameters(), lr=0.01)

            if use_saved == "yes":
                model_path = f"/scratch/temp/pipeline_parallel_rank{rank}_model.pt"
                if archive_dir and os.path.exists(os.path.join(archive_dir, f"pipeline_parallel_rank{rank}_model.pt")):
                    model_path = os.path.join(archive_dir, f"pipeline_parallel_rank{rank}_model.pt")
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))

            dist.barrier()

            mode_signal = torch.tensor([0])
            dist.broadcast(mode_signal, src=0)

            if mode_signal.item() == 1:
                step_signal = torch.tensor([0])
                while True:
                    dist.broadcast(step_signal, src=0)
                    if step_signal.item() == 0:
                        break

                    if rank == 1:
                        input_data = torch.zeros(64, 784)
                        dist.recv(input_data, src=0)
                    else:
                        input_data = torch.zeros(64, 128)
                        dist.recv(input_data, src=rank - 1)

                    input_data.requires_grad = True
                    output = model(input_data)

                    if rank == 4:
                        dist.send(output, dst=0)
                    else:
                        dist.send(output, dst=rank + 1)

                    if rank == 4:
                        grad_in = torch.zeros(64, 10)
                        dist.recv(grad_in, src=0)
                    else:
                        grad_in = torch.zeros(64, 128)
                        dist.recv(grad_in, src=rank + 1)

                    opt.zero_grad()
                    output.backward(grad_in)
                    opt.step()

                    if rank > 1:
                        dist.send(input_data.grad, dst=rank - 1)

                torch.save(
                    model.state_dict(), f"/scratch/temp/pipeline_parallel_rank{rank}_model.pt"
                )
                if archive_dir and os.path.isdir(archive_dir):
                    torch.save(
                        model.state_dict(),
                        os.path.join(archive_dir, f"pipeline_parallel_rank{rank}_model.pt"),
                    )

            dist.broadcast(mode_signal, src=0)
            if mode_signal.item() == 2:
                step_signal = torch.tensor([0])
                with torch.no_grad():
                    while True:
                        dist.broadcast(step_signal, src=0)
                        if step_signal.item() == 0:
                            break
                        if rank == 1:
                            input_data = torch.zeros(64, 784)
                            dist.recv(input_data, src=0)
                        else:
                            input_data = torch.zeros(64, 128)
                            dist.recv(input_data, src=rank - 1)
                        output = model(input_data)
                        if rank == 4:
                            dist.send(output, dst=0)
                        else:
                            dist.send(output, dst=rank + 1)

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed.")


if __name__ == "__main__":
    main()