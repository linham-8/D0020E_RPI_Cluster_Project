import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import time

def get_unique_filename(directory, base_name, extension):
    """Hittar ett unikt filnamn, börjar alltid på _0."""
    counter = 0
    while True:
        file_path = os.path.join(directory, f"{base_name}_{counter}.{extension}")
        if not os.path.exists(file_path):
            return file_path
        counter += 1

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
            f"Invalid arguments. Usage: python data_parallel.py <rank> <saved> <archive>. Error: {e}"
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
                "/scratch/temp/data_parallel_sync", world_size=world_size
            ),
        )
        print(f"Rank {rank}: Connected to cluster.")

        # Inläsning av data
        def load(path, offset):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset not found at {path}")
            with open(path, "rb") as f:
                return torch.frombuffer(bytearray(f.read()[offset:]), dtype=torch.uint8)

        X = (
            load("/scratch/mnist_dataset/emnist-digits-train-images-idx3-ubyte", 16)
            .float()
            .reshape(-1, 784)
            / 255.0
        )
        Y = load(
            "/scratch/mnist_dataset/emnist-digits-train-labels-idx1-ubyte", 8
        ).long()

        # Sharding
        X = X[rank::world_size]
        Y = Y[rank::world_size]

        model = DDP(torch.nn.Linear(784, 10))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = torch.nn.CrossEntropyLoss()

        if use_saved == "yes":
            model_path = os.path.join(archive_dir, "train", "model.pt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                print(f"Rank {rank}: Loaded saved model from {model_path}")
            else:
                print(f"Rank {rank}: Saved model not found.")

        dist.barrier()
        if rank == 0:
            start_time = time.time()

        # Träningsloopen
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
                            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")

                            live_log = {
                                "progress": round(
                                    (current_image / total_images) * 100, 2
                                ),
                                "current": current_image,
                                "total": total_images,
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

        # Resultat
        if rank == 0:
            end_time = time.time()
            training_time = end_time - start_time

            run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            train_dir = None
            test_dir = None
            
            if archive_dir and os.path.isdir(archive_dir):
                train_dir = os.path.join(archive_dir, "train")
                test_dir = os.path.join(archive_dir, "test")
                
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)

            if use_saved != "yes":
                if train_dir:
                    torch.save(model.state_dict(), os.path.join(train_dir, "model.pt"))

                total_batches = (len(X) / 64) * 5
                throughput = (len(X) * world_size * 5) / training_time
                avg_latency = (training_time / total_batches) * 1000

                train_log = {
                    "type": "training_result",
                    "parallelism_type": "data_parallel",
                    "training_time": round(training_time, 2),
                    "throughput": round(throughput, 2),
                    "latency_per_batch_ms": round(avg_latency, 2),
                    "world_size": world_size,
                    "epochs": 5,
                    "timestamp": run_timestamp,
                }
                
                print(f"Total Training Time: {training_time:.2f}s")

                try:
                    with open("/scratch/temp/latest.log", "w") as f:
                        json.dump(train_log, f)

                    if train_dir:
                        with open(os.path.join(train_dir, "training.log"), "w") as f:
                            json.dump(train_log, f)


                except OSError as e:
                    print(f"Rank 0: Training logging failed: {e}")

            Xt = (
                load("/scratch/mnist_dataset/emnist-digits-test-images-idx3-ubyte", 16)
                .float()
                .reshape(-1, 784)
                / 255.0
            )
            Yt = load(
                "/scratch/mnist_dataset/emnist-digits-test-labels-idx1-ubyte", 8
            ).long()

            test_start = time.time()
            with torch.no_grad():
                acc = (model(Xt).argmax(dim=1) == Yt).float().mean() * 100
            test_time = time.time() - test_start

            print(f"Accuracy: {acc:.2f}%")

            inference_throughput = len(Xt) / test_time
            inference_latency = (test_time / (len(Xt) / 64)) * 1000

            test_log = {
                "type": "test_result",
                "parallelism_type": "data_parallel",
                "accuracy": float(acc),
                "test_time": round(test_time, 2),
                "inference_throughput": round(inference_throughput, 2),
                "inference_latency_ms": round(inference_latency, 2),
                "world_size": world_size,
                "timestamp": run_timestamp,
            }

            try:
                if os.path.exists("/scratch/temp/latest.log"):
                    with open("/scratch/temp/latest.log", "r") as f:
                        try:
                            merged_log = json.load(f)
                        except json.JSONDecodeError:
                            merged_log = {}
                    merged_log.update(test_log)
                else:
                    merged_log = test_log

                with open("/scratch/temp/latest.log", "w") as f:
                    json.dump(merged_log, f)

                if test_dir and use_saved != "yes":
                    with open(os.path.join(test_dir, "test.log"), "w") as f:
                        json.dump(test_log, f)
                    
            except OSError as e:
                print(f"Rank 0: Test logging failed: {e}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()