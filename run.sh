#!/usr/bin/env bash
# run.sh — Crash-safe wrapper for the ML-in-C pipeline with log rotation
# and phase-level progress indicators.
#
# Usage:
#   ./run.sh                       # full pipeline (build+tune+benchmark+plot) for all models
#   ./run.sh --model cnn           # full pipeline for CNN only
#   ./run.sh --skip-tune           # skip tuning phase
#   ./run.sh --keep 5              # keep last 5 log directories (default: 10)
#   ./run.sh --model mlp --skip-tune

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPELINE="${PROJECT_ROOT}/src/scripts/pipeline.py"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Defaults
MODEL="all"
KEEP=10
SKIP_TUNE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --keep)    KEEP="$2"; shift 2 ;;
        --skip-tune) SKIP_TUNE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--model mlp|cnn|all] [--keep N] [--skip-tune]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create timestamped log directory
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${LOGS_DIR}/${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# Log rotation: keep only the last N run directories
rotate_logs() {
    local dirs
    dirs=$(find "${LOGS_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
    local count
    count=$(echo "$dirs" | grep -c . || true)
    if [[ $count -gt $KEEP ]]; then
        local to_remove=$((count - KEEP))
        echo "$dirs" | head -n "$to_remove" | while read -r d; do
            rm -rf "$d"
        done
    fi
}

# Phase runner with spinner — stdout+stderr to log, spinner on terminal
run_phase() {
    local phase="$1"
    local label="$2"
    shift 2
    local model_suffix=""
    # Extract --model arg for per-model log filenames
    local i
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--model" ]]; then
            local j=$((i+1))
            model_suffix="_${!j}"
            break
        fi
    done
    local logfile="${RUN_DIR}/${phase}${model_suffix}.log"
    local pid

    printf "  %-20s " "${label}..."

    # Run pipeline phase, capture ALL output to log
    if "${PYTHON}" "${PIPELINE}" "$phase" "$@" > "${logfile}" 2>&1 & pid=$!; then
        # Spinner while process runs
        local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
        local i=0
        while kill -0 "$pid" 2>/dev/null; do
            printf "\r  %-20s %s" "${label}..." "${spin:i%${#spin}:1}"
            i=$((i + 1))
            sleep 0.15
        done
        wait "$pid"
        local rc=$?
        if [[ $rc -eq 0 ]]; then
            printf "\r  %-20s done\n" "${label}"
        else
            printf "\r  %-20s FAILED (rc=%d)\n" "${label}" "$rc"
            echo "  Log: ${logfile}"
            tail -20 "${logfile}" | sed 's/^/    /'
            return "$rc"
        fi
    fi
}

# Live phase runner — stdout to log, stderr (tqdm progress) to terminal
run_phase_live() {
    local phase="$1"
    local label="$2"
    shift 2
    local model_suffix=""
    local i
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--model" ]]; then
            local j=$((i+1))
            model_suffix="_${!j}"
            break
        fi
    done
    local logfile="${RUN_DIR}/${phase}${model_suffix}.log"

    echo "  ${label}"

    # stdout to log for detailed results, stderr (tqdm) to terminal
    "${PYTHON}" "${PIPELINE}" "$phase" "$@" > "${logfile}"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        printf "  %-20s FAILED (rc=%d)\n" "${label}" "$rc"
        echo "  Log: ${logfile}"
        tail -20 "${logfile}" | sed 's/^/    /'
        return "$rc"
    fi
}

echo "============================================"
echo "  Full Pipeline"
echo "  Model: ${MODEL}  |  Logs: ${RUN_DIR}"
echo "============================================"
echo ""

START_TIME=$SECONDS

# Build phase (always runs) — use spinner (no tqdm)
MODEL_ARGS=()
if [[ "$MODEL" != "all" ]]; then
    MODEL_ARGS=(--model "$MODEL")
fi
run_phase build "Build" "${MODEL_ARGS[@]}"

# Tune phase — live output (tqdm progress visible)
if [[ "$SKIP_TUNE" == "false" ]]; then
    if [[ "$MODEL" == "all" ]]; then
        run_phase_live tune "Tune MLP" --model mlp
        run_phase_live tune "Tune CNN" --model cnn
    else
        run_phase_live tune "Tune ${MODEL^^}" --model "$MODEL"
    fi
fi

# Benchmark + scaling + plot phases
if [[ "$MODEL" == "all" ]]; then
    run_phase_live benchmark "Benchmark MLP" --model mlp
    run_phase_live scaling "Scaling MLP" --model mlp
    run_phase plot "Plot MLP" --model mlp
    run_phase_live benchmark "Benchmark CNN" --model cnn
    run_phase_live scaling "Scaling CNN" --model cnn
    run_phase plot "Plot CNN" --model cnn
else
    run_phase_live benchmark "Benchmark ${MODEL^^}" --model "$MODEL"
    run_phase_live scaling "Scaling ${MODEL^^}" --model "$MODEL"
    run_phase plot "Plot ${MODEL^^}" --model "$MODEL"
fi

ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================"
echo "  Complete in ${MINS}m ${SECS}s"
echo "  Logs: ${RUN_DIR}"
echo "============================================"

# Rotate old logs
rotate_logs
