NUM_TO_ADD=${1:-1}

if [ "$NUM_TO_ADD" -lt 1 ] 2>/dev/null; then
    echo "Usage: ./add_worker.sh [number_of_workers]"
    echo "Example: ./add_worker.sh 3"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HEARTBEAT_PORT=6000
EPOCHS=5
TMPDIR_LAUNCH="/tmp/elastf_launch"
SIGNAL_DIR="$PROJECT_DIR/shared/config/signals"
TF_CONFIG_FILE="$PROJECT_DIR/shared/config/tf_config.json"
SCALE_UP_FILE="$PROJECT_DIR/shared/config/scale_up"

if [ ! -f "$TF_CONFIG_FILE" ]; then
    echo "[add_worker] No tf_config.json found. Is the cluster running?"
    exit 1
fi

CURRENT_IDS=$(python3 -c "
import json
d = json.load(open('$TF_CONFIG_FILE'))
print(' '.join(d.get('workers', [])))
")

if [ -z "$CURRENT_IDS" ]; then
    echo "[add_worker] No workers found in tf_config.json."
    exit 1
fi

MAX_ID=0
for id in $CURRENT_IDS; do
    [ "$id" -gt "$MAX_ID" ] && MAX_ID="$id"
done

CURRENT_COUNT=$(echo $CURRENT_IDS | wc -w | tr -d ' ')
NEW_TOTAL=$((CURRENT_COUNT + NUM_TO_ADD))

NEW_IDS=()
for n in $(seq 1 "$NUM_TO_ADD"); do
    NEW_IDS+=($((MAX_ID + n)))
done

echo ""
echo "============================================================"
echo "[add_worker] SCALING UP: $CURRENT_COUNT -> $NEW_TOTAL workers"
echo "[add_worker] Current workers: $CURRENT_IDS"
echo "[add_worker] Adding $NUM_TO_ADD worker(s): ${NEW_IDS[*]}"
echo "============================================================"
echo ""

mkdir -p "$SIGNAL_DIR"

NEW_BASE_PORT=$((30000 + RANDOM % 10000))

for idx in "${!NEW_IDS[@]}"; do
    WID=${NEW_IDS[$idx]}
    WTF_PORT=$((NEW_BASE_PORT + idx))
    SCRIPT="$TMPDIR_LAUNCH/worker_${WID}.sh"
    PIDFILE="$TMPDIR_LAUNCH/worker_${WID}.pid"

    cat > "$SCRIPT" <<OUTER
#!/usr/bin/env bash
cd "$PROJECT_DIR"
source .venv/bin/activate
export WORKER_ID=$WID
export CONTROLLER_HOST=localhost
export WORKER_HOST=localhost
export HEARTBEAT_PORT=$HEARTBEAT_PORT
export CONFIG_DIR=shared/config
export CHECKPOINT_DIR=shared/checkpoints

HB_PID=""
CHILD=""
KILLED_BY_USER=0

cleanup_and_exit() {
    [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
    [ -n "\$CHILD" ] && kill -9 \$CHILD 2>/dev/null
    KILLED_BY_USER=1
    echo ""
    echo "[worker $WID] Ctrl+C detected. Stopping heartbeat and exiting."
}
trap cleanup_and_exit INT TERM

GENERATION=0
while true; do
    GENERATION=\$((GENERATION + 1))

    SIGNAL_FILE="$SIGNAL_DIR/restart_${WID}"
    if [ -f "\$SIGNAL_FILE" ]; then
        source "\$SIGNAL_FILE"
        rm -f "\$SIGNAL_FILE"
    fi

    export TF_PORT=\${TF_PORT:-$WTF_PORT}
    export STARTUP_SLEEP_SECS=\${STARTUP_SLEEP_SECS:-10}
    export EPOCHS=$EPOCHS

    echo ""
    echo "=== WORKER $WID | Generation \$GENERATION | port \$TF_PORT ==="
    echo \$\$ > "$PIDFILE"

    [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
    python3 -m elas_tf.heartbeat_sender localhost $HEARTBEAT_PORT "$WID" localhost "\$TF_PORT" 2 &
    HB_PID=\$!
    echo "[worker $WID] Heartbeat sender started (pid=\$HB_PID)"

    python3 -m elas_tf.worker &
    CHILD=\$!
    wait \$CHILD 2>/dev/null
    EXIT_CODE=\$?
    CHILD=""

    if [ \$KILLED_BY_USER -eq 1 ]; then
        echo "[worker $WID] Killed by user. Exiting."
        break
    fi

    if [ \$EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[worker $WID] Training completed successfully. Done."
        touch "$PROJECT_DIR/shared/config/training_done"
        [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
        break
    fi

    echo ""
    echo "[worker $WID] Python crashed (TF cascade, code=\$EXIT_CODE). Heartbeat still running."
    echo "[worker $WID] Waiting for restart signal from supervisor..."

    WAITED=0
    while [ ! -f "\$SIGNAL_FILE" ] && [ \$WAITED -lt 45 ]; do
        sleep 2
        WAITED=\$((WAITED + 2))
    done

    if [ ! -f "\$SIGNAL_FILE" ]; then
        echo "[worker $WID] No restart signal received. This worker was removed. Exiting."
        [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
        break
    fi

    echo "[worker $WID] Restart signal received! Restarting..."
done
OUTER
    chmod +x "$SCRIPT"

    osascript -e "tell application \"Terminal\" to do script \"echo '=== WORKER $WID ==='; $SCRIPT ; exit;\""
    echo "[add_worker] Worker $WID terminal launched (port $WTF_PORT)"
done

echo ""
echo "[add_worker] Waiting for controller to register ${#NEW_IDS[@]} new worker(s)..."
REGISTERED_ALL=0
for attempt in $(seq 1 30); do
    sleep 1
    if [ -f "$TF_CONFIG_FILE" ]; then
        ALL_FOUND=1
        for wid in "${NEW_IDS[@]}"; do
            HAS=$(python3 -c "
import json
d = json.load(open('$TF_CONFIG_FILE'))
print('yes' if '$wid' in d.get('workers', []) else 'no')
")
            if [ "$HAS" != "yes" ]; then
                ALL_FOUND=0
                break
            fi
        done
        if [ "$ALL_FOUND" -eq 1 ]; then
            echo "[add_worker] All ${#NEW_IDS[@]} new worker(s) registered!"
            REGISTERED_ALL=1
            break
        fi
    fi
    echo "[add_worker]   ... waiting ($attempt/30s)"
done

if [ "$REGISTERED_ALL" -eq 0 ]; then
    echo "[add_worker] WARNING: Not all workers registered within 30s. Proceeding anyway."
fi

echo ""
echo "[add_worker] Stopping existing workers' training for reconfiguration..."
for id in $CURRENT_IDS; do
    WRAPPER_PIDFILE="$TMPDIR_LAUNCH/worker_${id}.pid"
    if [ -f "$WRAPPER_PIDFILE" ]; then
        WRAPPER_PID=$(cat "$WRAPPER_PIDFILE")
        TRAINING_PID=$(pgrep -P "$WRAPPER_PID" -f "elas_tf.worker" 2>/dev/null)
        if [ -n "$TRAINING_PID" ]; then
            kill -9 "$TRAINING_PID" 2>/dev/null
            echo "[add_worker]   Killed training for worker $id (training pid=$TRAINING_PID)"
        else
            echo "[add_worker]   Worker $id training not found (may have already exited)"
        fi
    else
        echo "[add_worker]   Worker $id PID file not found"
    fi
done

for id in $CURRENT_IDS; do
    cat > "$SIGNAL_DIR/restart_${id}" <<SIGEOF
export STARTUP_SLEEP_SECS=10
SIGEOF
    echo "[add_worker]   Restart signal for worker $id"
done

printf '%s\n' "${NEW_IDS[@]}" > "$SCALE_UP_FILE"

echo ""
echo "============================================================"
echo "[add_worker] Scale-up complete!"
echo "[add_worker] Cluster now has $NEW_TOTAL workers: $CURRENT_IDS ${NEW_IDS[*]}"
echo "[add_worker] Existing workers will restart in their terminals."
echo "[add_worker] ${#NEW_IDS[@]} new worker(s) starting in new terminals."
echo "============================================================"
