"""Cloud worker entrypoint — replaces the bash worker loop from launch.sh.

This script manages the full lifecycle of a worker on a GCP VM:
  1. Discover own internal IP
  2. Start heartbeat sender as a subprocess
  3. Wait for the controller to register enough workers
  4. Fetch TF_CONFIG from the controller HTTP API
  5. Run training
  6. On crash/reconfiguration: poll for restart signal, re-fetch config, restart
  7. On success: exit cleanly
"""

import os
import signal
import socket
import subprocess
import sys
import time

import requests


def _get_internal_ip() -> str:
    """Get this VM's internal IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _wait_for_controller(controller_url: str, timeout: int = 120) -> bool:
    """Wait until the controller HTTP API is reachable."""
    print(f"[entrypoint] Waiting for controller at {controller_url}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{controller_url}/status", timeout=3)
            if resp.status_code == 200:
                print(f"[entrypoint] Controller is up: {resp.json()}")
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"[entrypoint] Controller not reachable after {timeout}s")
    return False


def _wait_for_stable_cluster(
    controller_url: str,
    expected_workers: int = 0,
    stability_secs: int = 15,
    timeout: int = 300,
) -> int:
    """Wait until expected workers have registered, then wait for stability."""
    if expected_workers > 0:
        print(f"[entrypoint] Waiting for {expected_workers} worker(s) to register...")
    else:
        print(f"[entrypoint] Waiting for cluster to stabilize ({stability_secs}s of no changes)...")

    deadline = time.time() + timeout
    last_gen = -1
    last_change = time.time()

    while time.time() < deadline:
        try:
            resp = requests.get(f"{controller_url}/config/generation", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                gen = data.get("generation", 0)
                num = data.get("num_workers", 0)
                if gen != last_gen:
                    last_gen = gen
                    last_change = time.time()
                    print(f"[entrypoint]   Generation {gen}, {num} worker(s)...")

                if expected_workers > 0 and num >= expected_workers:
                    print(f"[entrypoint]   All {num} expected worker(s) registered!")
                    time.sleep(5)
                    return gen
                elif expected_workers == 0 and num > 0 and (time.time() - last_change) >= stability_secs:
                    print(f"[entrypoint]   Cluster stable: generation {gen}, {num} worker(s)")
                    return gen
        except Exception:
            pass
        time.sleep(3)

    print(f"[entrypoint]   Timed out, proceeding with current config")
    return last_gen


def _poll_restart_signal(controller_url: str, worker_id: str, timeout: int = 120) -> bool:
    """Poll the controller for a restart signal."""
    print(f"[entrypoint] Polling for restart signal (worker {worker_id})...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{controller_url}/signal/restart/{worker_id}", timeout=3
            )
            if resp.status_code == 200 and resp.json().get("restart"):
                print(f"[entrypoint] Restart signal received!")
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"[entrypoint] No restart signal after {timeout}s")
    return False


_active_worker_proc: subprocess.Popen | None = None


def _handle_reconfigure(signum, frame):
    """SIGUSR1 handler: kill the running training subprocess to trigger restart."""
    global _active_worker_proc
    if _active_worker_proc and _active_worker_proc.poll() is None:
        print(f"[entrypoint] Received SIGUSR1 — killing training process (pid={_active_worker_proc.pid})")
        _active_worker_proc.kill()


def main() -> None:
    signal.signal(signal.SIGUSR1, _handle_reconfigure)

    worker_id = os.getenv("WORKER_ID", "0")
    controller_host = os.getenv("CONTROLLER_HOST", "elastf-controller")
    heartbeat_port = int(os.getenv("HEARTBEAT_PORT", "5000"))
    http_port = int(os.getenv("HTTP_PORT", "8080"))
    tf_port = int(os.getenv("TF_PORT", "35000"))
    epochs = os.getenv("EPOCHS", "10")
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/tmp/elastf_checkpoints")
    gcs_bucket = os.getenv("GCS_BUCKET", "")
    expected_workers = int(os.getenv("EXPECTED_WORKERS", "0"))

    controller_url = os.getenv(
        "CONTROLLER_URL", f"http://{controller_host}:{http_port}"
    )
    my_ip = _get_internal_ip()

    print("")
    print("=" * 60)
    print(f"[entrypoint] ElasTF Worker Entrypoint")
    print(f"[entrypoint]   Worker ID:       {worker_id}")
    print(f"[entrypoint]   My IP:           {my_ip}")
    print(f"[entrypoint]   TF port:         {tf_port}")
    print(f"[entrypoint]   Controller:      {controller_url}")
    print(f"[entrypoint]   Heartbeat:       {controller_host}:{heartbeat_port}")
    print(f"[entrypoint]   Checkpoint dir:  {checkpoint_dir}")
    if gcs_bucket:
        print(f"[entrypoint]   GCS bucket:      {gcs_bucket}")
    print("=" * 60)
    print("")

    if not _wait_for_controller(controller_url):
        print("[entrypoint] FATAL: Controller unreachable. Exiting.")
        sys.exit(1)

    os.makedirs(checkpoint_dir, exist_ok=True)

    real_start_file = os.path.join(checkpoint_dir, "real_start_time")
    if not os.path.exists(real_start_file):
        with open(real_start_file, "w") as f:
            f.write(f"{time.time():.4f}\n")
        print(f"[entrypoint] Recorded real start time")

    hb_proc = subprocess.Popen(
        [
            sys.executable, "-m", "elas_tf.heartbeat_sender",
            controller_host, str(heartbeat_port),
            worker_id, my_ip, str(tf_port), "2",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[entrypoint] Heartbeat sender started (pid={hb_proc.pid})")

    generation = 0
    while True:
        generation += 1
        print(f"\n[entrypoint] === Generation {generation} ===")

        if generation == 1:
            _wait_for_stable_cluster(
                controller_url,
                expected_workers=expected_workers,
                stability_secs=15,
                timeout=300,
            )
        else:
            print(f"[entrypoint] Sleeping 10s for restart stabilization...")
            time.sleep(10)

        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        env["CONTROLLER_URL"] = controller_url
        env["CONTROLLER_HOST"] = controller_host
        env["HEARTBEAT_PORT"] = str(heartbeat_port)
        env["CHECKPOINT_DIR"] = checkpoint_dir
        env["EPOCHS"] = epochs
        env["TF_PORT"] = str(tf_port)
        env["STARTUP_SLEEP_SECS"] = "0"
        if gcs_bucket:
            env["GCS_BUCKET"] = gcs_bucket

        print(f"[entrypoint] Launching worker process...")
        global _active_worker_proc
        worker_proc = subprocess.Popen(
            [sys.executable, "-m", "elas_tf.worker"],
            env=env,
        )
        _active_worker_proc = worker_proc
        exit_code = worker_proc.wait()
        _active_worker_proc = None

        if exit_code == 0:
            print("")
            print("*" * 60)
            print(f"[entrypoint] Training completed successfully! (generation {generation})")
            print("*" * 60)
            break

        print(f"[entrypoint] Worker process exited with code {exit_code}")
        print(f"[entrypoint] Heartbeat sender still running (pid={hb_proc.pid})")

        if not _poll_restart_signal(controller_url, worker_id, timeout=90):
            print(f"[entrypoint] No restart signal — this worker was removed. Exiting.")
            break

        print(f"[entrypoint] Restarting for generation {generation + 1}...")

    if hb_proc.poll() is None:
        hb_proc.terminate()
        try:
            hb_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            hb_proc.kill()
    print("[entrypoint] Heartbeat sender stopped. Goodbye.")


if __name__ == "__main__":
    main()
