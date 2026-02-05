import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import json
import time
import traceback


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
            f"Invalid arguments. Usage: python model_parallel.py <rank> <saved> <archive>. Error: {e}"
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
                "/scratch/temp/model_parallel_sync", world_size=world_size
            ),
        )
        print(f"Rank {rank}: Connected to cluster.")

        compute_group = dist.new_group([1, 2, 3, 4])

        def load(path, offset):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset not found at {path}")
            with open(path, "rb") as f:
                return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

        # Rank 0: Master
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
            X, Y = X[Y < 8], Y[Y < 8]

            Xt = (
                load("/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte", 16)
                .float()
                .reshape(-1, 784)
                / 255.0
            )
            Yt = load(
                "/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte", 8
            ).long()
            Xt, Yt = Xt[Yt < 8], Yt[Yt < 8]

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
                        dist.broadcast(batch_x, src=0)
                        dist.broadcast(batch_y, src=0)

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
                    if len(batch_xt) < 64:
                        continue

                    dist.broadcast(torch.tensor([1]), src=0)
                    dist.broadcast(batch_xt, src=0)

                    preds = []
                    for r in range(1, 5):
                        buffer = torch.zeros(64, 2)
                        dist.recv(buffer, src=r)
                        preds.append(buffer)

                    full_logits = torch.cat(preds, dim=1)
                    acc = (full_logits.argmax(dim=1) == Yt[i : i + 64]).sum().item()
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
                "model_type": "model_parallel",
                "accuracy": float(final_acc),
                "training_time": round(training_time, 2),
                "test_time": round(test_time, 2),
                "throughput": round(throughput, 2),
                "latency_per_batch": round(avg_batch_latency, 2),
                "world_size": world_size,
                "epochs": 5 if use_saved != "yes" else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            try:
                with open("/scratch/temp/latest.log", "w") as f:
                    json.dump(log, f)

                with open("/scratch/temp/history.log", "a") as f:
                    json.dump(log, f)
                    f.write("\n")

                if archive_dir and os.path.isdir(archive_dir):
                    with open(os.path.join(archive_dir, "history.log"), "a") as f:
                        json.dump(log, f)
                        f.write("\n")
            except OSError as e:
                print(f"Rank 0: Final logging failed: {e}")

        # Rank 1-4: Compute
        else:
            model = nn.Linear(784, 2)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)

            if use_saved == "yes":
                model_path = f"/scratch/temp/model_parallel_rank{rank}_model.pt"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path))

            dist.barrier()

            mode_signal = torch.tensor([0])

            dist.broadcast(mode_signal, src=0)
            if mode_signal.item() == 1:
                batch_x = torch.zeros(64, 784)
                batch_y = torch.zeros(64, dtype=torch.long)
                step_signal = torch.tensor([0])

                while True:
                    dist.broadcast(step_signal, src=0)
                    if step_signal.item() == 0:
                        break

                    dist.broadcast(batch_x, src=0)
                    dist.broadcast(batch_y, src=0)

                    opt.zero_grad()
                    local_out = model(batch_x)

                    gathered = [torch.zeros(64, 2) for _ in range(4)]
                    dist.all_gather(gathered, local_out.data, group=compute_group)

                    full_parts = []
                    for r, tensor in enumerate(gathered):
                        if r == (rank - 1):
                            full_parts.append(local_out)
                        else:
                            full_parts.append(tensor)

                    full_logits = torch.cat(full_parts, dim=1)
                    loss = nn.CrossEntropyLoss()(full_logits, batch_y)
                    loss.backward()
                    opt.step()

                torch.save(
                    model.state_dict(), f"/scratch/temp/model_parallel_rank{rank}_model.pt"
                )
                if archive_dir and os.path.isdir(archive_dir):
                    torch.save(
                        model.state_dict(),
                        os.path.join(archive_dir, f"model_parallel_rank{rank}_model.pt"),
                    )

            dist.broadcast(mode_signal, src=0)
            if mode_signal.item() == 2:
                batch_x = torch.zeros(64, 784)
                step_signal = torch.tensor([0])

                with torch.no_grad():
                    while True:
                        dist.broadcast(step_signal, src=0)
                        if step_signal.item() == 0:
                            break

                        dist.broadcast(batch_x, src=0)
                        local_out = model(batch_x)
                        dist.send(local_out, dst=0)

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed.")


if __name__ == "__main__":
    main()