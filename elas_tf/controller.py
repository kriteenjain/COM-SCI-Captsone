import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request as flask_request

from .heartbeat import HeartbeatEvent, HeartbeatMonitor


TF_CONFIG_FILENAME = "tf_config.json"
REMAINING_WORKERS_FILENAME = "remaining_workers"


def _config_dir() -> Path:
    path = os.getenv("CONFIG_DIR", "/shared/config")
    return Path(path)


class ControllerState:
    """Thread-safe shared state between the heartbeat loop and HTTP API."""

    def __init__(self):
        self._lock = threading.Lock()
        self.generation: int = 0
        self.known_workers: Dict[str, Dict] = {}
        self.tf_configs: Dict[str, Dict] = {}
        self.worker_ids: List[str] = []
        self.restart_signals: Dict[str, bool] = {}

    def update_cluster(self, known_workers: Dict[str, Dict], generation: int) -> None:
        with self._lock:
            self.generation = generation
            self.known_workers = dict(known_workers)
            worker_ids = sorted(known_workers.keys(), key=lambda x: int(x))
            self.worker_ids = worker_ids

            cluster_workers = []
            for wid in worker_ids:
                state = known_workers[wid]
                cluster_workers.append(f"{state['host']}:{state['port']}")

            configs: Dict[str, Dict] = {}
            for idx, wid in enumerate(worker_ids):
                configs[wid] = {
                    "cluster": {"worker": cluster_workers},
                    "task": {"type": "worker", "index": idx},
                }
            self.tf_configs = configs

    def get_config_for_worker(self, worker_id: str) -> tuple:
        with self._lock:
            cfg = self.tf_configs.get(worker_id)
            return cfg, self.generation, list(self.worker_ids)

    def set_restart_signal(self, worker_id: str) -> None:
        with self._lock:
            self.restart_signals[worker_id] = True

    def check_restart_signal(self, worker_id: str) -> bool:
        with self._lock:
            return self.restart_signals.pop(worker_id, False)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "generation": self.generation,
                "workers": list(self.worker_ids),
                "num_workers": len(self.worker_ids),
                "worker_details": {
                    wid: {"host": s["host"], "port": s["port"]}
                    for wid, s in self.known_workers.items()
                },
            }


def _write_tf_config(worker_states: Dict[str, Dict], generation: int) -> None:
    """Write tf_config.json to disk (kept for backward compatibility with local mode)."""
    cfg_dir = _config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / TF_CONFIG_FILENAME

    worker_ids: List[str] = sorted(worker_states.keys(), key=lambda x: int(x))
    cluster_workers = []
    for wid in worker_ids:
        state = worker_states[wid]
        cluster_workers.append(f"{state['host']}:{state['port']}")

    configs: Dict[str, Dict] = {}
    for idx, wid in enumerate(worker_ids):
        tf_config = {
            "cluster": {"worker": cluster_workers},
            "task": {"type": "worker", "index": idx},
        }
        configs[wid] = tf_config

    payload = {"generation": generation, "workers": worker_ids, "configs": configs}
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_http_app(state: ControllerState) -> Flask:
    app = Flask("elastf-controller")
    app.logger.disabled = True

    @app.route("/config/<worker_id>", methods=["GET"])
    def get_worker_config(worker_id: str):
        cfg, generation, workers = state.get_config_for_worker(worker_id)
        if cfg is None:
            return jsonify({"error": "worker not registered"}), 404
        return jsonify({
            "tf_config": cfg,
            "generation": generation,
            "workers": workers,
        })

    @app.route("/config/generation", methods=["GET"])
    def get_generation():
        status = state.get_status()
        return jsonify({"generation": status["generation"], "num_workers": status["num_workers"]})

    @app.route("/signal/restart/<worker_id>", methods=["GET"])
    def check_restart(worker_id: str):
        triggered = state.check_restart_signal(worker_id)
        return jsonify({"restart": triggered})

    @app.route("/signal/restart/<worker_id>", methods=["POST"])
    def post_restart(worker_id: str):
        state.set_restart_signal(worker_id)
        return jsonify({"ok": True})

    @app.route("/status", methods=["GET"])
    def cluster_status():
        return jsonify(state.get_status())

    return app


def run_controller() -> None:
    heartbeat_port = int(os.getenv("HEARTBEAT_PORT", "5000"))
    http_port = int(os.getenv("HTTP_PORT", "8080"))

    monitor = HeartbeatMonitor(port=heartbeat_port)
    monitor.start()

    state = ControllerState()

    app = create_http_app(state)
    http_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=http_port, threaded=True),
        name="http-api",
        daemon=True,
    )
    http_thread.start()

    print("")
    print("=" * 60)
    print(f"[controller] ElasTF Controller started")
    print(f"[controller] Heartbeat monitor listening on 0.0.0.0:{heartbeat_port}")
    print(f"[controller] HTTP API listening on 0.0.0.0:{http_port}")
    print(f"[controller] Heartbeat timeout: 8s")
    print(f"[controller] Waiting for workers to register...")
    print("=" * 60)
    print("")

    generation = 0
    known_workers: Dict[str, Dict] = {}
    ever_had_workers = False

    prev_worker_count = 0
    stabilization_deadline: float = 0.0
    STABILIZATION_WINDOW = 5.0

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
                    state.update_cluster(known_workers, generation)
                    print(f"[controller]   New generation: {generation}")
                    print(f"[controller]   TF_CONFIG written with {num_active} workers")
                    total_samples = 50000  # CIFAR-10
                    print(f"[controller]   Data will be sharded: {total_samples} / {num_active} = ~{total_samples // num_active} samples/worker")

                    if membership_changed:
                        for wid in active_ids:
                            state.set_restart_signal(wid)
                        print(f"[controller]   Restart signals set for: {active_ids}")

                elif ever_had_workers:
                    print(f"[controller]   All workers have left.")
                    print(f"[controller]   Waiting for new workers to join...")

                if membership_changed and ever_had_workers and num_active < prev_worker_count:
                    stabilization_deadline = time.time() + STABILIZATION_WINDOW
                    print(f"[controller]   Stabilization window: waiting {STABILIZATION_WINDOW:.0f}s for all survivor heartbeats...")

                prev_worker_count = num_active

                print("-" * 60)
                print("")

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
