import os
import sys

def getDistConfig():
    try:
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"

        rank = int(sys.argv[1])
        world_size = int(sys.argv[2])
        training_mode = sys.argv[3] if len(sys.argv) > 3 else "no"
        archive_dir = sys.argv[4] if len(sys.argv) > 4 else None

        return {
            "backend": "gloo",
            "rank": rank,
            "world_size": world_size,
            "training_mode": training_mode,
            "archive_dir": archive_dir,
        }

    except (IndexError, ValueError) as e:
        print(f"Invalid arguments: {e}")
        sys.exit(1)