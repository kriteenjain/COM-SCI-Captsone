import json
import os
import socket
import time
from pathlib import Path

from . import training
from .heartbeat import send_join, start_heartbeat_sender


TF_CONFIG_FILENAME = "tf_config.json"


def _config_dir() -> Path:
    path = os.getenv("CONFIG_DIR", "/shared/config")
    return Path(path)


def _load_tf_config_for_worker(worker_id: str) -> tuple[str | None, int]:
    cfg_path = _config_dir() / TF_CONFIG_FILENAME
    if not cfg_path.exists():
        print(f"[worker {worker_id}] No TF_CONFIG file at {cfg_path}, running single-worker.")
        return None, 0

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    configs = data.get("configs", {})
    cfg = configs.get(worker_id)
    if not cfg:
        print(f"[worker {worker_id}] No TF_CONFIG entry for this worker in {cfg_path}, running single-worker.")
        return None, 0

    generation = int(data.get("generation", 0))
    num_workers = len(data.get("workers", []))
    worker_list = data.get("workers", [])

    print("")
    print("=" * 60)
    print(f"[worker {worker_id}] CLUSTER CONFIG LOADED")
    print(f"[worker {worker_id}]   Generation:  {generation}")
    print(f"[worker {worker_id}]   Workers:     {worker_list}")
    print(f"[worker {worker_id}]   My role:     worker {worker_id} of {num_workers}")
    print("=" * 60)
    print("")

    return json.dumps(cfg), generation


def run_worker() -> None:
    worker_id = os.getenv("WORKER_ID", "0")
    controller_host = os.getenv("CONTROLLER_HOST", "controller")
    heartbeat_port = int(os.getenv("HEARTBEAT_PORT", "5000"))
    startup_sleep_secs = float(os.getenv("STARTUP_SLEEP_SECS", "20"))
    tf_port = int(os.getenv("TF_PORT", "12345"))

    host = os.getenv("WORKER_HOST", "localhost")

    # Heartbeat is now handled by a separate process (heartbeat_sender.py) started
    # by the bash wrapper, so it survives TF crashes. No in-process heartbeat here.
    print(f"[worker {worker_id}] Heartbeat handled by external process (survives TF crashes).")

    if startup_sleep_secs > 0:
        print(f"[worker {worker_id}] Sleeping {startup_sleep_secs:.0f}s to allow all workers to register...")
        time.sleep(startup_sleep_secs)

    tf_config, generation = _load_tf_config_for_worker(worker_id)
    if tf_config:
        os.environ["TF_CONFIG"] = tf_config
        os.environ["TF_GENERATION"] = str(generation)
    else:
        os.environ.pop("TF_CONFIG", None)
        os.environ.pop("TF_GENERATION", None)

    print(f"[worker {worker_id}] Launching distributed training...")
    training.main()

    print("")
    print("*" * 60)
    print(f"[worker {worker_id}] Training finished successfully!")
    print("*" * 60)
    print("")


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()
