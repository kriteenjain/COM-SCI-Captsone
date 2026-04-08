import json
import os
import time
from pathlib import Path

import requests

from . import training


TF_CONFIG_FILENAME = "tf_config.json"


def _config_dir() -> Path:
    path = os.getenv("CONFIG_DIR", "/shared/config")
    return Path(path)


def _load_tf_config_via_http(worker_id: str) -> tuple:
    """Fetch TF_CONFIG from the controller HTTP API."""
    controller_url = os.getenv("CONTROLLER_URL")
    if not controller_url:
        return None, 0

    try:
        resp = requests.get(f"{controller_url}/config/{worker_id}", timeout=5)
        if resp.status_code == 404:
            print(f"[worker {worker_id}] Controller returned 404 — not registered yet.")
            return None, 0
        resp.raise_for_status()
        data = resp.json()
        tf_config = data["tf_config"]
        generation = int(data.get("generation", 0))
        workers = data.get("workers", [])
        num_workers = len(workers)

        print("")
        print("=" * 60)
        print(f"[worker {worker_id}] CLUSTER CONFIG LOADED (via HTTP)")
        print(f"[worker {worker_id}]   Generation:  {generation}")
        print(f"[worker {worker_id}]   Workers:     {workers}")
        print(f"[worker {worker_id}]   My role:     worker {worker_id} of {num_workers}")
        print("=" * 60)
        print("")

        return json.dumps(tf_config), generation
    except Exception as e:
        print(f"[worker {worker_id}] HTTP config fetch failed: {e}")
        return None, 0


def _load_tf_config_from_file(worker_id: str) -> tuple:
    """Fallback: load TF_CONFIG from local tf_config.json (for local mode)."""
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
    print(f"[worker {worker_id}] CLUSTER CONFIG LOADED (from file)")
    print(f"[worker {worker_id}]   Generation:  {generation}")
    print(f"[worker {worker_id}]   Workers:     {worker_list}")
    print(f"[worker {worker_id}]   My role:     worker {worker_id} of {num_workers}")
    print("=" * 60)
    print("")

    return json.dumps(cfg), generation


def _load_tf_config_for_worker(worker_id: str) -> tuple:
    """Load TF_CONFIG, preferring HTTP API when CONTROLLER_URL is set."""
    if os.getenv("CONTROLLER_URL"):
        return _load_tf_config_via_http(worker_id)
    return _load_tf_config_from_file(worker_id)


def run_worker() -> None:
    worker_id = os.getenv("WORKER_ID", "0")
    startup_sleep_secs = float(os.getenv("STARTUP_SLEEP_SECS", "20"))

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
