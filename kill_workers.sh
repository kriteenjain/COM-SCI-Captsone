if [ $# -lt 1 ]; then
    echo "Usage: ./kill_workers.sh <worker_id> [worker_id2] ..."
    echo "Example: ./kill_workers.sh 0 1"
    exit 1
fi

TMPDIR_LAUNCH="/tmp/elastf_launch"

for id in "$@"; do
    PIDFILE="$TMPDIR_LAUNCH/worker_${id}.pid"
    if [ ! -f "$PIDFILE" ]; then
        echo "Worker $id: PID file not found ($PIDFILE). Skipping."
        continue
    fi

    PID=$(cat "$PIDFILE")
    if [ -z "$PID" ]; then
        echo "Worker $id: PID file is empty. Skipping."
        continue
    fi

    CHILDREN=$(pgrep -P "$PID" 2>/dev/null)
    kill -9 "$PID" 2>/dev/null
    for cpid in $CHILDREN; do
        kill -9 "$cpid" 2>/dev/null
    done
    echo "Killed worker $id (pid=$PID, children: ${CHILDREN:-none})"
done

echo ""
echo "Controller will detect failure via heartbeat timeout (~8s)."
echo "Supervisor will restart surviving workers automatically."
