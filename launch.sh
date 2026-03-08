#!/usr/bin/env bash
# launch.sh — Elastic supervisor for ElasTF.
#
# Heartbeat-based failure detection. When a worker is killed, TF crashes all
# workers (collective failure). The controller detects the failure via heartbeat
# timeout. The supervisor reads killed_count (written by kill_workers.sh or
# Ctrl+C trap) and restarts (ACTIVE - killed) workers in the same terminals.
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HEARTBEAT_PORT=6000
NUM_WORKERS=6
STARTUP_SLEEP=10
EPOCHS=5
TMPDIR_LAUNCH="/tmp/elastf_launch"
REMAINING_FILE="$PROJECT_DIR/shared/config/remaining_workers"
SIGNAL_DIR="$PROJECT_DIR/shared/config/signals"
SCALE_UP_FILE="$PROJECT_DIR/shared/config/scale_up"

echo "[supervisor] Killing any old ElasTF processes..."
pkill -9 -f "elas_tf.controller" 2>/dev/null
pkill -9 -f "elas_tf.worker" 2>/dev/null
pkill -9 -f "elas_tf.heartbeat_sender" 2>/dev/null
lsof -ti tcp:${HEARTBEAT_PORT} | xargs kill -9 2>/dev/null
sleep 3

echo "[supervisor] Clearing old state..."
rm -rf "$PROJECT_DIR/shared/checkpoints/"* "$PROJECT_DIR/shared/config/"*
mkdir -p "$PROJECT_DIR/shared/checkpoints" "$PROJECT_DIR/shared/config" "$SIGNAL_DIR"
rm -rf "$TMPDIR_LAUNCH"
mkdir -p "$TMPDIR_LAUNCH"

# --- Write the controller script ---
cat > "$TMPDIR_LAUNCH/controller.sh" <<EOF
#!/usr/bin/env bash
cd "$PROJECT_DIR"
source .venv/bin/activate
export CONFIG_DIR=shared/config
export CHECKPOINT_DIR=shared/checkpoints
export HEARTBEAT_PORT=$HEARTBEAT_PORT
echo "=== CONTROLLER ==="
echo \$\$ > "$TMPDIR_LAUNCH/controller.pid"
exec python3 -m elas_tf.controller
EOF
chmod +x "$TMPDIR_LAUNCH/controller.sh"

# --- Open controller Terminal ---
echo "[supervisor] Starting controller..."
open -a Terminal "$TMPDIR_LAUNCH/controller.sh"
sleep 3

if [ -f "$TMPDIR_LAUNCH/controller.pid" ]; then
    CTRL_PID=$(cat "$TMPDIR_LAUNCH/controller.pid")
    echo "[supervisor] Controller started (pid=$CTRL_PID)."
else
    echo "[supervisor] Controller started."
fi

# --- Trap Ctrl-C ---
cleanup() {
    echo ""
    echo "[supervisor] Shutting down..."
    pkill -9 -f "elas_tf.controller" 2>/dev/null
    pkill -9 -f "elas_tf.worker" 2>/dev/null
    pkill -9 -f "elas_tf.heartbeat_sender" 2>/dev/null
    rm -rf "$TMPDIR_LAUNCH"
    echo "[supervisor] Done."
    exit 0
}
trap cleanup INT TERM

# --- Launch worker terminals (once — they loop internally) ---
ACTIVE_WORKERS=$NUM_WORKERS
BASE_PORT=$((30000 + RANDOM % 10000))

echo ""
echo "============================================================"
echo "[supervisor] Starting $ACTIVE_WORKERS workers"
echo "[supervisor] TF ports: $BASE_PORT - $((BASE_PORT + ACTIVE_WORKERS - 1))"
echo "============================================================"

for i in $(seq 0 $((ACTIVE_WORKERS - 1))); do
    TF_PORT=$((BASE_PORT + i))
    PIDFILE="$TMPDIR_LAUNCH/worker_${i}.pid"
    SCRIPT="$TMPDIR_LAUNCH/worker_${i}.sh"

    # Worker script: runs heartbeat as a SEPARATE process (survives TF crashes).
    # When python crashes (TF cascade), heartbeat keeps running → controller still
    # sees this worker alive. Only when bash exits (Ctrl+C/removed) does heartbeat stop.
    cat > "$SCRIPT" <<OUTER
#!/usr/bin/env bash
cd "$PROJECT_DIR"
source .venv/bin/activate
export WORKER_ID=$i
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
    echo "[worker $i] Ctrl+C detected. Stopping heartbeat and exiting."
}
trap cleanup_and_exit INT TERM

GENERATION=0
while true; do
    GENERATION=\$((GENERATION + 1))

    SIGNAL_FILE="$SIGNAL_DIR/restart_${i}"
    if [ -f "\$SIGNAL_FILE" ]; then
        source "\$SIGNAL_FILE"
        rm -f "\$SIGNAL_FILE"
    fi

    export TF_PORT=\${TF_PORT:-$TF_PORT}
    export STARTUP_SLEEP_SECS=\${STARTUP_SLEEP_SECS:-$STARTUP_SLEEP}
    export EPOCHS=$EPOCHS

    echo ""
    echo "=== WORKER $i | Generation \$GENERATION | port \$TF_PORT ==="
    echo \$\$ > "$PIDFILE"

    # Start heartbeat sender as a separate process (survives python/TF crashes).
    [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
    python3 -m elas_tf.heartbeat_sender localhost $HEARTBEAT_PORT "$i" localhost "\$TF_PORT" 2 &
    HB_PID=\$!
    echo "[worker $i] Heartbeat sender started (pid=\$HB_PID)"

    # Run training.
    python3 -m elas_tf.worker &
    CHILD=\$!
    wait \$CHILD 2>/dev/null
    EXIT_CODE=\$?
    CHILD=""

    if [ \$KILLED_BY_USER -eq 1 ]; then
        echo "[worker $i] Killed by user. Exiting."
        break
    fi

    if [ \$EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[worker $i] Training completed successfully. Done."
        touch "$PROJECT_DIR/shared/config/training_done"
        [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
        break
    fi

    # TF cascade crash — heartbeat is still running, controller still sees us alive.
    echo ""
    echo "[worker $i] Python crashed (TF cascade, code=\$EXIT_CODE). Heartbeat still running."
    echo "[worker $i] Waiting for restart signal from supervisor..."

    WAITED=0
    while [ ! -f "\$SIGNAL_FILE" ] && [ \$WAITED -lt 45 ]; do
        sleep 2
        WAITED=\$((WAITED + 2))
    done

    if [ ! -f "\$SIGNAL_FILE" ]; then
        echo "[worker $i] No restart signal received. This worker was removed. Exiting."
        [ -n "\$HB_PID" ] && kill -9 \$HB_PID 2>/dev/null
        break
    fi

    echo "[worker $i] Restart signal received! Restarting..."
done
OUTER
    chmod +x "$SCRIPT"

    open -a Terminal "$SCRIPT"
    echo "[supervisor] Opened Terminal for worker $i (port $TF_PORT)"
done

# --- Main supervisor loop ---
GENERATION=0
# Track which worker IDs are alive (by original index, e.g. 0 1 2).
ALIVE_IDS=()
for i in $(seq 0 $((ACTIVE_WORKERS - 1))); do
    ALIVE_IDS+=("$i")
done

while true; do
    GENERATION=$((GENERATION + 1))

    # Wait for PID files — use ALIVE_IDS (not 0..N-1) to read correct PID files.
    echo "[supervisor] Waiting for worker PIDs..."
    sleep 4

    WORKER_PIDS=()
    WORKER_PID_IDS=()
    for i in "${ALIVE_IDS[@]}"; do
        PIDFILE="$TMPDIR_LAUNCH/worker_${i}.pid"
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            WORKER_PIDS+=("$PID")
            WORKER_PID_IDS+=("$i")
            echo "[supervisor] Worker $i pid=$PID"
        else
            echo "[supervisor] WARNING: Could not get PID for worker $i"
        fi
    done

    # Write current PIDs so kill_workers.sh can use them.
    WORKER_PIDS_FILE="$PROJECT_DIR/shared/config/worker_pids"
    printf '%s\n' "${WORKER_PIDS[@]}" > "$WORKER_PIDS_FILE"

    # Clear remaining_workers from previous generation.
    rm -f "$REMAINING_FILE"

    echo ""
    echo "[supervisor] Generation $GENERATION: ${#ALIVE_IDS[@]} workers running (IDs: ${ALIVE_IDS[*]})"
    echo "[supervisor] To simulate failure: Ctrl+C in a worker terminal, or from another terminal:"
    echo ""
    echo "    cd ~/Downloads/comsci214/ElasTF && ./kill_workers.sh ${ALIVE_IDS[0]}"
    if [ ${#ALIVE_IDS[@]} -ge 2 ]; then
    echo "    cd ~/Downloads/comsci214/ElasTF && ./kill_workers.sh ${ALIVE_IDS[0]} ${ALIVE_IDS[1]}"
    fi
    echo ""
    echo "[supervisor] (Current PIDs: ${WORKER_PIDS[*]})"
    echo "[supervisor] Heartbeat timeout is 8s — controller will detect failures after that."
    echo ""

    # Wait for a worker PID to exit OR a scale-up signal.
    EXITED_PID=""
    SCALED_UP=0
    while [ -z "$EXITED_PID" ] && [ "$SCALED_UP" -eq 0 ]; do
        if [ -f "$SCALE_UP_FILE" ]; then
            while IFS= read -r line || [ -n "$line" ]; do
                line=$(echo "$line" | tr -d ' \n')
                [ -n "$line" ] && ALIVE_IDS+=("$line")
            done < "$SCALE_UP_FILE"
            rm -f "$SCALE_UP_FILE"
            ACTIVE_WORKERS=${#ALIVE_IDS[@]}
            SCALED_UP=1
            break
        fi
        for pid in "${WORKER_PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                EXITED_PID=$pid
                break
            fi
        done
        sleep 1
    done

    if [ "$SCALED_UP" -eq 1 ]; then
        echo ""
        echo "[supervisor] Scale-up detected! New ALIVE_IDS: ${ALIVE_IDS[*]} ($ACTIVE_WORKERS workers)"
        echo "[supervisor] Looping back to re-read PIDs for the larger cluster."
        echo ""
        continue
    fi

    echo ""
    echo "[supervisor] Worker exit detected (pid=$EXITED_PID)."

    # Check if training completed (workers write this marker on exit code 0).
    TRAINING_DONE_FILE="$PROJECT_DIR/shared/config/training_done"
    sleep 2
    if [ -f "$TRAINING_DONE_FILE" ]; then
        # Wait for ALL workers to finish before declaring done.
        echo "[supervisor] Training done marker found. Waiting for all workers to finish..."
        sleep 5
        echo ""
        echo "============================================================"
        echo "[supervisor] Training completed successfully!"
        echo "============================================================"
        break
    fi

    echo "[supervisor] Waiting for controller to detect failure via heartbeat timeout..."

    # Poll for the controller's remaining_workers file.
    WAIT_LIMIT=25
    WAITED=0
    while [ ! -f "$REMAINING_FILE" ] && [ $WAITED -lt $WAIT_LIMIT ]; do
        sleep 2
        WAITED=$((WAITED + 2))
        echo "[supervisor]   ... waiting for controller ($WAITED/${WAIT_LIMIT}s)"
    done

    # Read surviving worker IDs from controller.
    SURVIVOR_IDS=()
    if [ -f "$REMAINING_FILE" ]; then
        while IFS= read -r line || [ -n "$line" ]; do
            line=$(echo "$line" | tr -d ' \n')
            [ -n "$line" ] && SURVIVOR_IDS+=("$line")
        done < "$REMAINING_FILE"
        rm -f "$REMAINING_FILE"
        echo "[supervisor] Controller reports surviving workers: ${SURVIVOR_IDS[*]}"
    else
        echo "[supervisor] Timed out waiting for controller."
        SURVIVOR_IDS=("${ALIVE_IDS[@]}")
    fi

    [ ${#SURVIVOR_IDS[@]} -lt 1 ] && SURVIVOR_IDS=("${ALIVE_IDS[0]}")

    echo ""
    echo "============================================================"
    echo "[supervisor] ELASTIC RECOVERY: restarting ${#SURVIVOR_IDS[@]} worker(s): ${SURVIVOR_IDS[*]}"
    echo "============================================================"

    # Write restart signals for surviving workers (keep their current ports —
    # the old python processes are dead so ports are freed).
    ALIVE_IDS=()
    for i in "${SURVIVOR_IDS[@]}"; do
        cat > "$SIGNAL_DIR/restart_${i}" <<SIGEOF
export STARTUP_SLEEP_SECS=10
SIGEOF
        echo "[supervisor] Restart signal for worker $i"
        ALIVE_IDS+=("$i")
    done

    ACTIVE_WORKERS=${#ALIVE_IDS[@]}
    echo "[supervisor] Survivors will restart in their existing terminals. Killed terminals will exit."
    echo ""
done

# Clean up.
pkill -9 -f "elas_tf.controller" 2>/dev/null
pkill -9 -f "elas_tf.heartbeat_sender" 2>/dev/null
rm -rf "$TMPDIR_LAUNCH"
echo "[supervisor] Controller stopped. All done."
