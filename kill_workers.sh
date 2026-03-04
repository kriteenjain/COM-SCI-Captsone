#!/usr/bin/env bash
# Usage:  ./kill_workers.sh <id1> [id2] ...
#   Kills specific workers by index. Survivors get restart signals.
#
# Examples (with 3 workers: 0, 1, 2):
#   ./kill_workers.sh 0        → kill worker 0, restart workers 1 and 2
#   ./kill_workers.sh 0 1      → kill workers 0 and 1, restart worker 2
#   ./kill_workers.sh 1 2      → kill workers 1 and 2, restart worker 0

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

    # Find all child PIDs (heartbeat_sender, python) before killing the parent.
    CHILDREN=$(pgrep -P "$PID" 2>/dev/null)
    kill -9 "$PID" 2>/dev/null
    for cpid in $CHILDREN; do
        kill -9 "$cpid" 2>/dev/null
    done
    echo "Killed worker $id (pid=$PID, children: ${CHILDREN:-none})"
done

echo ""
echo "Controller will detect failure via heartbeat timeout (~15s)."
echo "Supervisor will restart surviving workers automatically."
