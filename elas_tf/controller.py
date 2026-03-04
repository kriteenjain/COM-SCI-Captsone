import json
import os
import time
from pathlib import Path
from typing import Dict, List

from .heartbeat import HeartbeatEvent, HeartbeatMonitor


TF_CONFIG_FILENAME = "tf_config.json"
REMAINING_WORKERS_FILENAME = "remaining_workers"


def _config_dir() -> Path:
    path = os.getenv("CONFIG_DIR", "/shared/config")
    return Path(path)


def _write_tf_config(worker_states: Dict[str, Dict], generation: int) -> None:
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


def run_controller() -> None:
    port = int(os.getenv("HEARTBEAT_PORT", "5000"))
    monitor = HeartbeatMonitor(port=port)
    monitor.start()

    print("")
    print("=" * 60)
    print(f"[controller] ElasTF Controller started")
    print(f"[controller] Heartbeat monitor listening on 0.0.0.0:{port}")
    print(f"[controller] Heartbeat timeout: 15s")
    print(f"[controller] Waiting for workers to register...")
    print("=" * 60)
    print("")

    generation = 0
    known_workers: Dict[str, Dict] = {}
    ever_had_workers = False

    prev_worker_count = 0
    stabilization_deadline: float = 0.0
    STABILIZATION_WINDOW = 10.0

    try:
        while True:
            membership_changed = False

            config_changed = False

            events = monitor.poll_events()
            for ev in events:
                if ev.event_type == "join":
                    old = known_workers.get(ev.worker_id)
                    if old is None:
                        print(f"[controller] >>> WORKER JOIN: worker {ev.worker_id} at {ev.host}:{ev.port}")
                        membership_changed = True
                    elif old.get("port") != ev.port:
                        print(f"[controller] >>> WORKER REJOIN: worker {ev.worker_id} new port {ev.port} (was {old.get('port')})")
                        config_changed = True
                    known_workers[ev.worker_id] = {"host": ev.host or "", "port": ev.port or 12345}
                elif ev.event_type == "heartbeat":
                    if ev.worker_id in known_workers:
                        known_workers[ev.worker_id]["host"] = ev.host or known_workers[ev.worker_id]["host"]
                elif ev.event_type == "failure":
                    if ev.worker_id in known_workers:
                        print(f"[controller] !!! WORKER FAILURE: worker {ev.worker_id}")
                        known_workers.pop(ev.worker_id, None)
                        membership_changed = True

            failures: List[HeartbeatEvent] = monitor.detect_failures()
            for ev in failures:
                if ev.worker_id in known_workers:
                    print(f"[controller] !!! WORKER TIMEOUT: worker {ev.worker_id} (no heartbeat received)")
                    known_workers.pop(ev.worker_id, None)
                    membership_changed = True

            if membership_changed or config_changed:
                active_ids = sorted(known_workers.keys(), key=lambda x: int(x)) if known_workers else []
                num_active = len(active_ids)

                reason = "CLUSTER MEMBERSHIP CHANGED" if membership_changed else "CLUSTER CONFIG UPDATED (port change)"
                print("")
                print("-" * 60)
                print(f"[controller] {reason}")
                print(f"[controller]   Active workers: {active_ids}")
                print(f"[controller]   Worker count:   {num_active}")

                if known_workers:
                    ever_had_workers = True
                    generation += 1
                    _write_tf_config(known_workers, generation)
                    print(f"[controller]   New generation: {generation}")
                    print(f"[controller]   TF_CONFIG written with {num_active} workers")
                    print(f"[controller]   Data will be sharded: 60000 / {num_active} = ~{60000 // num_active} samples/worker")
                elif ever_had_workers:
                    print(f"[controller]   All workers have left.")
                    print(f"[controller]   Waiting for new workers to join...")

                if membership_changed and ever_had_workers and num_active < prev_worker_count:
                    stabilization_deadline = time.time() + STABILIZATION_WINDOW
                    print(f"[controller]   Stabilization window: waiting {STABILIZATION_WINDOW:.0f}s for all survivor heartbeats...")

                prev_worker_count = num_active

                print("-" * 60)
                print("")

            # After the stabilization window expires, write the final survivor list.
            if stabilization_deadline > 0 and time.time() >= stabilization_deadline:
                active_ids = sorted(known_workers.keys(), key=lambda x: int(x)) if known_workers else []
                remaining_path = _config_dir() / REMAINING_WORKERS_FILENAME
                remaining_path.write_text("\n".join(active_ids) + "\n", encoding="utf-8")
                print(f"[controller] Stabilization complete. Remaining workers: {active_ids}")
                stabilization_deadline = 0.0

            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[controller] Shutting down (KeyboardInterrupt).")
    finally:
        monitor.stop()


def main() -> None:
    run_controller()


if __name__ == "__main__":
    main()
