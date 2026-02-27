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
    SRC_DIR = BASE_DIR / "src"
    MODELS_DIR = SRC_DIR / "models"
    LAUNCH_SCRIPT = SRC_DIR / "launch.py"

    # Nodes
    COMPUTE_NODES = ["pi1", "pi2", "pi3", "pi4"]
