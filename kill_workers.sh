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

PIDS_FILE="shared/config/worker_pids"
KILLED_IDS_FILE="shared/config/killed_ids"

if [ ! -f "$PIDS_FILE" ]; then
    echo "No worker_pids file found. Is launch.sh running?"
    exit 1
fi

# Read PIDs into an array (macOS-compatible, no mapfile).
ALL_PIDS=()
while IFS= read -r line; do
    [ -n "$line" ] && ALL_PIDS+=("$line")
done < "$PIDS_FILE"

TOTAL=${#ALL_PIDS[@]}
if [ "$TOTAL" -lt 1 ]; then
    echo "No workers found in PID file."
    exit 1
fi

# Kill each requested worker AND all its child processes (heartbeat sender, python).
> "$KILLED_IDS_FILE"
for id in "$@"; do
    echo "$id" >> "$KILLED_IDS_FILE"
    if [ "$id" -lt "$TOTAL" ] 2>/dev/null; then
        PID="${ALL_PIDS[$id]}"
        # Find all child PIDs (heartbeat_sender, python) before killing the parent.
        CHILDREN=$(pgrep -P "$PID" 2>/dev/null)
        kill -9 "$PID" 2>/dev/null
        for cpid in $CHILDREN; do
            kill -9 "$cpid" 2>/dev/null
        done
        echo "Killed worker $id (pid=$PID, children: ${CHILDREN:-none})"
    else
        echo "Worker $id not found (only $TOTAL workers running: 0-$((TOTAL-1)))"
    fi
done

REMAINING=$((TOTAL - $#))
[ "$REMAINING" -lt 0 ] && REMAINING=0
echo "Controller will detect failure via heartbeat timeout (~15s)."
echo "Supervisor will restart $REMAINING surviving worker(s)."
