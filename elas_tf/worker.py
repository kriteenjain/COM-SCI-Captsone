import json
import os
import socket
from pathlib import Path

from . import training
from .heartbeat import send_join, start_heartbeat_sender


TF_CONFIG_FILENAME = "tf_config.json"


def _config_dir() -> Path:
    path = os.getenv("CONFIG_DIR", "/shared/config")
    return Path(path)


def _load_tf_config_for_worker(worker_id: str) -> str | None:
    cfg_path = _config_dir() / TF_CONFIG_FILENAME
    if not cfg_path.exists():
        print(f"[worker {worker_id}] No TF_CONFIG file at {cfg_path}, running single-worker.")
        return None

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    configs = data.get("configs", {})
    cfg = configs.get(worker_id)
    if not cfg:
        print(f"[worker {worker_id}] No TF_CONFIG entry for this worker in {cfg_path}, running single-worker.")
        return None

    generation = data.get("generation", 0)
    print(f"[worker {worker_id}] Loaded TF_CONFIG (generation={generation}) from {cfg_path}")
    return json.dumps(cfg)


def run_worker() -> None:
    worker_id = os.getenv("WORKER_ID", "0")
    controller_host = os.getenv("CONTROLLER_HOST", "controller")
    heartbeat_port = int(os.getenv("HEARTBEAT_PORT", "5000"))

    # MultiWorker ports are arbitrary here; all workers share the same port.
    tf_port = int(os.getenv("TF_PORT", "12345"))

    # Announce ourselves to the controller and start periodic heartbeats.
    host = socket.gethostname()
    send_join(controller_host, heartbeat_port, worker_id=worker_id, host=host, port=tf_port)
    stop_hb = start_heartbeat_sender(
        controller_host=controller_host,
        controller_port=heartbeat_port,
        worker_id=worker_id,
        host=host,
        port=tf_port,
    )

    try:
        tf_config = _load_tf_config_for_worker(worker_id)
        if tf_config:
            os.environ["TF_CONFIG"] = tf_config
        else:
            os.environ.pop("TF_CONFIG", None)

        print(f"[worker {worker_id}] Starting training process.")
        training.main()
        print(f"[worker {worker_id}] Training finished.")
    finally:
        stop_hb.set()


def main() -> None:
    run_worker()


if __name__ == "__main__":
    main()

