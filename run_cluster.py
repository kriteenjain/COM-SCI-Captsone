import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """
    Convenience launcher for ElasTF on bare metal.

    Runs:
      - 1 controller process
      - 3 worker processes (WORKER_ID 0, 1, 2)

    All logs are printed into this same terminal.
    """
    base_dir = Path(__file__).resolve().parent
    shared_dir = base_dir / "shared"
    checkpoint_dir = shared_dir / "checkpoints"
    config_dir = shared_dir / "config"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Always start from a clean slate for checkpoints/config.
    for path in checkpoint_dir.glob("*"):
        try:
            path.unlink()
        except IsADirectoryError:
            # There shouldn't be subdirs, but be defensive.
            pass
    for path in config_dir.glob("*"):
        try:
            path.unlink()
        except IsADirectoryError:
            pass

    heartbeat_port = int(os.environ.get("HEARTBEAT_PORT", "6000"))

    # Base environment for all subprocesses.
    base_env = os.environ.copy()
    base_env["CHECKPOINT_DIR"] = str(checkpoint_dir)
    base_env["CONFIG_DIR"] = str(config_dir)
    base_env["HEARTBEAT_PORT"] = str(heartbeat_port)

    procs: list[subprocess.Popen] = []

    # Start controller.
    controller_env = base_env.copy()
    controller_cmd = [sys.executable, "-m", "elas_tf.controller"]
    print(f"[launcher] Starting controller on port {heartbeat_port}...")
    controller_proc = subprocess.Popen(
        controller_cmd,
        cwd=str(base_dir),
        env=controller_env,
    )
    procs.append(controller_proc)

    # Give the controller a moment to bind the port.
    time.sleep(2.0)

    # Start three workers.
    for worker_id in range(3):
        env = base_env.copy()
        env["WORKER_ID"] = str(worker_id)
        env["CONTROLLER_HOST"] = "localhost"
        cmd = [sys.executable, "-m", "elas_tf.worker"]
        print(f"[launcher] Starting worker {worker_id}...")
        proc = subprocess.Popen(
            cmd,
            cwd=str(base_dir),
            env=env,
        )
        procs.append(proc)

    print("[launcher] Controller and 3 workers started.")
    print("[launcher] Press Ctrl-C here to terminate all processes.")

    try:
        # Wait for workers to finish (controller typically runs indefinitely).
        for p in procs[1:]:
            p.wait()
    except KeyboardInterrupt:
        print("\n[launcher] Caught KeyboardInterrupt, terminating child processes...")
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass

        # Give processes a moment to exit gracefully, then force kill if needed.
        deadline = time.time() + 5.0
        for p in procs:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass

        print("[launcher] All child processes terminated.")


if __name__ == "__main__":
    main()

