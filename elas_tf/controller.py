import json
import os
import time
from pathlib import Path
from typing import Dict, List

from .heartbeat import HeartbeatEvent, HeartbeatMonitor


TF_CONFIG_FILENAME = "tf_config.json"


def _config_dir() -> Path:
    path = os.getenv("CONFIG_DIR", "/shared/config")
    return Path(path)


def _write_tf_config(worker_states: Dict[str, Dict], generation: int) -> None:
    """Write a TF_CONFIG mapping for all workers to the shared config dir.

    The file format is:
    {
      "generation": int,
      "workers": ["0", "1", ...],
      "configs": {
        "0": {<TF_CONFIG for worker 0>},
        "1": {<TF_CONFIG for worker 1>}
      }
    }
    """
    cfg_dir = _config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / TF_CONFIG_FILENAME

    worker_ids: List[str] = sorted(worker_states.keys(), key=lambda x: int(x))
    cluster_workers = []
    for idx, wid in enumerate(worker_ids):
        state = worker_states[wid]
        host = state["host"]
        port = state["port"]
        cluster_workers.append(f"{host}:{port}")

    configs: Dict[str, Dict] = {}
    for idx, wid in enumerate(worker_ids):
        tf_config = {
            "cluster": {"worker": cluster_workers},
            "task": {"type": "worker", "index": idx},
        }
        configs[wid] = tf_config

    payload = {"generation": generation, "workers": worker_ids, "configs": configs}
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[controller] Wrote TF_CONFIG for workers={worker_ids} to {path}")


def run_controller() -> None:
    port = int(os.getenv("HEARTBEAT_PORT", "5000"))
    monitor = HeartbeatMonitor(port=port)
    monitor.start()

    print(f"[controller] Heartbeat monitor listening on 0.0.0.0:{port}")

    generation = 0
    known_workers: Dict[str, Dict] = {}

    try:
        while True:
            # Handle incoming heartbeat/join events.
            events = monitor.poll_events()
            membership_changed = False
            for ev in events:
                if ev.event_type == "join":
                    if ev.worker_id not in known_workers:
                        print(f"[controller] Worker join detected: {ev.worker_id} ({ev.host}:{ev.port})")
                        membership_changed = True
                    known_workers[ev.worker_id] = {"host": ev.host or "", "port": ev.port or 12345}
                elif ev.event_type == "heartbeat":
                    # Heartbeats should not trigger a cluster reconfiguration; they only update liveness.
                    if ev.worker_id in known_workers:
                        known_workers[ev.worker_id] = {"host": ev.host or known_workers[ev.worker_id]["host"], "port": ev.port or known_workers[ev.worker_id]["port"]}
                elif ev.event_type == "failure":
                    if ev.worker_id in known_workers:
                        print(f"[controller] Worker failure detected: {ev.worker_id}")
                        known_workers.pop(ev.worker_id, None)
                        membership_changed = True

            # Detect failures based on timeout and generate corresponding events.
            failures: List[HeartbeatEvent] = monitor.detect_failures()
            for ev in failures:
                if ev.worker_id in known_workers:
                    print(f"[controller] Worker timed out: {ev.worker_id}")
                    known_workers.pop(ev.worker_id, None)
                    membership_changed = True

            # If membership changed, bump generation and rewrite TF_CONFIG.
            if membership_changed:
                active_ids = sorted(known_workers.keys(), key=lambda x: int(x)) if known_workers else []
                print(f"[controller] Membership change detected. Active workers now: {active_ids}")

                if known_workers:
                    generation += 1
                    _write_tf_config(known_workers, generation)
                    print(f"[controller] Cluster generation updated to {generation}")

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[controller] Shutting down (KeyboardInterrupt).")
    finally:
        monitor.stop()


def main() -> None:
    run_controller()


if __name__ == "__main__":
    main()

