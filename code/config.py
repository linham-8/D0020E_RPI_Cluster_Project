import os
from pathlib import Path

class Config():
    # Project root
    ROOT_DIR = Path(__file__).resolve().parent

    # Temp and archive
    TEMP_DIR = Path("/scratch/temp")
    ARCHIVE_DIR = TEMP_DIR / "archive"

    # BASE_DIR used by launch.py
    RAW_BASE = ROOT_DIR
    BASE_DIR = Path(str(RAW_BASE).replace("/mnt/usb/scratch", "/scratch", 1))

    # Subdirectories
    SRC_DIR = BASE_DIR / "code"
    MODELS_DIR = SRC_DIR / "models"

    # Files
    LAUNCH_SCRIPT = SRC_DIR / "launch.py"
    LATEST_LOG = TEMP_DIR / "latest.log"
    HISTORY_LOG = TEMP_DIR / "history.log"

    # Nodes
    COMPUTE_NODES = ["pi1", "pi2", "pi3", "pi4"]

    # World Size
    WORLD_SIZE = len(COMPUTE_NODES) + 1

    # Models & Training
    SUPPORTED_MODELS = [
        "data_parallel",
        "expert_parallel",
        "pipeline_parallel",
        "model_parallel",
    ]
    EPOCHS = 5
    BATCH_SIZE = 512
    LEARNING_RATE = 0.01
    HIDDEN_DIM = 128
    DATA_ROOT = Path("/scratch/datasets")