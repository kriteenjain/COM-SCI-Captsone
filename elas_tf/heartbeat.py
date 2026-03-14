import json
import socket
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Dict, List, Optional


DEFAULT_HEARTBEAT_INTERVAL_SECS = 2.0
DEFAULT_HEARTBEAT_TIMEOUT_SECS = 8.0


@dataclass
class HeartbeatEvent:

    event_type: str                                    
    worker_id: str
    host: Optional[str] = None
    port: Optional[int] = None


@dataclass
class WorkerHeartbeatState:
    worker_id: str
    host: str
    port: int
    last_seen: float = field(default_factory=time.time)
    alive: bool = True


class HeartbeatMonitor:

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, timeout_secs: float = DEFAULT_HEARTBEAT_TIMEOUT_SECS) -> None:
        self._host = host
        self._port = port
        self._timeout_secs = timeout_secs

        self._sock: Optional[socket.socket] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._workers: Dict[str, WorkerHeartbeatState] = {}
        self._events: "Queue[HeartbeatEvent]" = Queue()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._listener_thread is not None:
            return

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.listen()

        self._listener_thread = threading.Thread(target=self._serve_forever, name="heartbeat-listener", daemon=True)
        self._listener_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def _serve_forever(self) -> None:
        assert self._sock is not None
        self._sock.settimeout(1.0)
        while not self._stop_event.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                try:
                    data = conn.recv(4096)
                    if not data:
                        continue
                    msg = json.loads(data.decode("utf-8"))
                    self._handle_message(msg)
                except Exception:
                    continue

    def _handle_message(self, msg: Dict) -> None:
        worker_id = str(msg.get("worker_id", ""))
        host = str(msg.get("host", ""))
        port = int(msg.get("port", 0))
        msg_type = msg.get("type")

        if not worker_id or not msg_type:
            return

        now = time.time()
        with self._lock:
            state = self._workers.get(worker_id)
            if state is None:
                state = WorkerHeartbeatState(worker_id=worker_id, host=host, port=port, last_seen=now, alive=True)
                self._workers[worker_id] = state
            else:
                state.host = host or state.host
                state.port = port or state.port
                state.last_seen = now
                state.alive = True

            event_type = "heartbeat" if msg_type == "heartbeat" else "join"
            self._events.put(HeartbeatEvent(event_type=event_type, worker_id=worker_id, host=state.host, port=state.port))

    def poll_events(self, max_events: int = 100) -> List[HeartbeatEvent]:
        items: List[HeartbeatEvent] = []
        for _ in range(max_events):
            try:
                item = self._events.get_nowait()
            except Empty:
                break
            else:
                items.append(item)
        return items

    def detect_failures(self) -> List[HeartbeatEvent]:
        now = time.time()
        failures: List[HeartbeatEvent] = []
        with self._lock:
            for worker_id, state in list(self._workers.items()):
                if state.alive and (now - state.last_seen) > self._timeout_secs:
                    state.alive = False
                    failures.append(HeartbeatEvent(event_type="failure", worker_id=worker_id, host=state.host, port=state.port))
                    self._events.put(failures[-1])
        return failures

    def get_worker_states(self) -> Dict[str, WorkerHeartbeatState]:
        with self._lock:
            return dict(self._workers)

