#!/usr/bin/env bash
# add_worker.sh — Add a new worker to a running ElasTF cluster.
#
# Automatically determines the next worker ID, creates a terminal,
# and reconfigures the cluster so all workers (old + new) train together.
#
# Usage:
#   ./add_worker.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HEARTBEAT_PORT=6000
EPOCHS=5
TMPDIR_LAUNCH="/tmp/elastf_launch"
SIGNAL_DIR="$PROJECT_DIR/shared/config/signals"
TF_CONFIG_FILE="$PROJECT_DIR/shared/config/tf_config.json"
SCALE_UP_FILE="$PROJECT_DIR/shared/config/scale_up"

# --- Determine current cluster state ---
if [ ! -f "$TF_CONFIG_FILE" ]; then
    echo "[add_worker] No tf_config.json found. Is the cluster running?"
    exit 1
fi

CURRENT_IDS=$(python3 -c "
import json, sys
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
NEW_ID=$((MAX_ID + 1))
CURRENT_COUNT=$(echo $CURRENT_IDS | wc -w | tr -d ' ')
NEW_TOTAL=$((CURRENT_COUNT + 1))

echo ""
echo "============================================================"
echo "[add_worker] SCALING UP: $CURRENT_COUNT -> $NEW_TOTAL workers"
echo "[add_worker] Current workers: $CURRENT_IDS"
echo "[add_worker] New worker ID:   $NEW_ID"
echo "============================================================"
echo ""

# --- Assign port for new worker ---
NEW_TF_PORT=$((30000 + RANDOM % 10000))

# --- Create the worker script (same structure as launch.sh workers) ---
SCRIPT="$TMPDIR_LAUNCH/worker_${NEW_ID}.sh"
PIDFILE="$TMPDIR_LAUNCH/worker_${NEW_ID}.pid"
mkdir -p "$SIGNAL_DIR"

cat > "$SCRIPT" <<OUTER
#!/usr/bin/env bash
cd "$PROJECT_DIR"
source .venv/bin/activate
export WORKER_ID=$NEW_ID
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
    echo "[worker $NEW_ID] Ctrl+C detected. Stopping heartbeat and exiting."
}
trap cleanup_and_exit INT TERM

GENERATION=0
while true; do
    GENERATION=\$((GENERATION + 1))

    SIGNAL_FILE="$SIGNAL_DIR/restart_${NEW_ID}"
    if [ -f "\$SIGNAL_FILE" ]; then
        source "\$SIGNAL_FILE"
        rm -f "\$SIGNAL_FILE"
    fi

    export TF_PORT=\${TF_PORT:-$NEW_TF_PORT}
    export STARTUP_SLEEP_SECS=\${STARTUP_SLEEP_SECS:-20}
    export EPOCHS=$EPOCHS

    echo ""
    echo "=== WORKER $NEW_ID | Generation \$GENERATION | port \$TF_PORT ==="
    echo \$\$ > "$PIDFILE"

    [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
    python3 -m elas_tf.heartbeat_sender localhost $HEARTBEAT_PORT "$NEW_ID" localhost "\$TF_PORT" 2 &
    HB_PID=\$!
    echo "[worker $NEW_ID] Heartbeat sender started (pid=\$HB_PID)"

    python3 -m elas_tf.worker &
    CHILD=\$!
    wait \$CHILD 2>/dev/null
    EXIT_CODE=\$?
    CHILD=""

    if [ \$KILLED_BY_USER -eq 1 ]; then
        echo "[worker $NEW_ID] Killed by user. Exiting."
        break
    fi

    if [ \$EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[worker $NEW_ID] Training completed successfully. Done."
        touch "$PROJECT_DIR/shared/config/training_done"
        [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
        break
    fi

    echo ""
    echo "[worker $NEW_ID] Python crashed (TF cascade, code=\$EXIT_CODE). Heartbeat still running."
    echo "[worker $NEW_ID] Waiting for restart signal from supervisor..."

    WAITED=0
    while [ ! -f "\$SIGNAL_FILE" ] && [ \$WAITED -lt 90 ]; do
        sleep 2
        WAITED=\$((WAITED + 2))
    done

    if [ ! -f "\$SIGNAL_FILE" ]; then
        echo "[worker $NEW_ID] No restart signal received. This worker was removed. Exiting."
        [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
        break
    fi

    echo "[worker $NEW_ID] Restart signal received! Restarting..."
done
OUTER
chmod +x "$SCRIPT"

# --- Launch the new worker terminal ---
osascript -e "tell application \"Terminal\" to do script \"echo '=== WORKER $NEW_ID ==='; $SCRIPT ; exit;\""
echo "[add_worker] Worker $NEW_ID terminal launched."

# --- Wait for controller to register the new worker ---
echo "[add_worker] Waiting for controller to register worker $NEW_ID..."
REGISTERED=0
for attempt in $(seq 1 30); do
    sleep 1
    if [ -f "$TF_CONFIG_FILE" ]; then
        HAS_WORKER=$(python3 -c "
import json
d = json.load(open('$TF_CONFIG_FILE'))
print('yes' if '$NEW_ID' in d.get('workers', []) else 'no')
")
        if [ "$HAS_WORKER" = "yes" ]; then
            echo "[add_worker] Controller registered worker $NEW_ID!"
            REGISTERED=1
            break
        fi
    fi
    echo "[add_worker]   ... waiting ($attempt/30s)"
done

if [ "$REGISTERED" -eq 0 ]; then
    echo "[add_worker] WARNING: Controller did not register worker $NEW_ID within 30s."
    echo "[add_worker] The worker terminal is running. It may join later."
    exit 1
fi

# --- Reconfigure existing workers ---
# Kill only the python training children (not heartbeat senders, not bash wrappers).
# Heartbeat senders stay alive so the controller still sees these workers.
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

# Write restart signals for existing workers (keep their current ports — they're
# freed when the python process dies, so no need for new ones).
for id in $CURRENT_IDS; do
    cat > "$SIGNAL_DIR/restart_${id}" <<SIGEOF
export STARTUP_SLEEP_SECS=20
SIGEOF
    echo "[add_worker]   Restart signal for worker $id"
done

# --- Notify supervisor about the new worker ---
echo "$NEW_ID" > "$SCALE_UP_FILE"

echo ""
echo "============================================================"
echo "[add_worker] Scale-up complete!"
echo "[add_worker] Cluster now has $NEW_TOTAL workers: $CURRENT_IDS $NEW_ID"
echo "[add_worker] Existing workers will restart in their terminals."
echo "[add_worker] New worker $NEW_ID is starting in a new terminal."
echo "============================================================"
