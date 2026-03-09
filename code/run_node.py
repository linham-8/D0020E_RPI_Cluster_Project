import sys
import os
import argparse

from dataloader import get_data

from models.data_parallel import DataParallelModel
from models.model_parallel import ModelParallelModel
from models.expert_parallel import ExpertParallelModel
from models.pipeline_parallel import PipelineParallelModel
from config import Config


def main():
    os.environ["GLOO_SOCKET_IFNAME"] = "eth0"

    parser = argparse.ArgumentParser(description="Run a compute node.")
    parser.add_argument("--model", type=str, required=True, help="Type of parallelism model")
    parser.add_argument("--rank", type=int, required=True, help="Node rank")
    parser.add_argument("--use_saved", type=str, required=True, help="Use saved ('yes' or 'no')")
    parser.add_argument("--archive_dir", type=str, required=True, help="Directory for archive")
    parser.add_argument("--time_limit", type=int, default=0, help="Time limit in seconds")

    args = parser.parse_args()

    model_type = args.model
    rank = args.rank
    use_saved = args.use_saved
    archive_dir = args.archive_dir if args.archive_dir != "None" else None
    time_limit = args.time_limit if args.time_limit > 0 else None

    world_size = Config.WORLD_SIZE
    batch_size = Config.BATCH_SIZE

    dist_config = {
        "rank": rank,
        "world_size": world_size
    }

    print(f"Rank {rank}: Initializing model '{model_type}'.")

    if model_type == "data_parallel":
        train_loader = get_data(training=True, dataset="EMNIST", batch_size=batch_size, world_size=world_size, rank=rank)
    else:
        train_loader = get_data(training=True, dataset="EMNIST", batch_size=batch_size, world_size=1, rank=0)

    test_loader = get_data(training=False, dataset="EMNIST", batch_size=batch_size)

    if model_type == "data_parallel":
        model = DataParallelModel(train_loader, test_loader, dist_config, archive_dir, use_saved)
    elif model_type == "model_parallel":
        model = ModelParallelModel(train_loader, test_loader, dist_config, archive_dir, use_saved)
    elif model_type == "expert_parallel":
        model = ExpertParallelModel(train_loader, test_loader, dist_config, archive_dir, use_saved)
    elif model_type == "pipeline_parallel":
        model = PipelineParallelModel(train_loader, test_loader, dist_config, archive_dir, use_saved)
    else:
        print(f"Rank {rank}: Unknown model type '{model_type}'")
        sys.exit(1)

    try:
        model.train(epochs=Config.EPOCHS, time_limit=time_limit)
        model.test()
    except KeyboardInterrupt:
        print(f"\n[Rank {rank}] Ctrl+C detected, cleanup starting.")
    finally:
        model.cleanup()

if __name__ == "__main__":
    main()