"""Standalone heartbeat sender — runs as a separate process so it survives TF crashes.

Usage:
    python3 -m elas_tf.heartbeat_sender <controller_host> <controller_port> \
        <worker_id> <worker_host> <worker_port> [interval_secs]
"""
import json
import signal
import socket
import sys
import time

                                                                               
                                                                             
                                                                          
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)


def _send(host: str, port: int, payload: dict) -> None:
    try:
        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall(json.dumps(payload).encode("utf-8"))
    except OSError:
        pass


def main() -> None:
    if len(sys.argv) < 6:
        print("Usage: python3 -m elas_tf.heartbeat_sender "
              "<ctrl_host> <ctrl_port> <worker_id> <worker_host> <worker_port> [interval]")
        sys.exit(1)

    ctrl_host = sys.argv[1]
    ctrl_port = int(sys.argv[2])
    worker_id = sys.argv[3]
    worker_host = sys.argv[4]
    worker_port = int(sys.argv[5])
    interval = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0

    _send(ctrl_host, ctrl_port, {
        "type": "join", "worker_id": worker_id,
        "host": worker_host, "port": worker_port,
    })

    while True:
        time.sleep(interval)
        _send(ctrl_host, ctrl_port, {
            "type": "heartbeat", "worker_id": worker_id,
            "host": worker_host, "port": worker_port,
        })


if __name__ == "__main__":
    main()
