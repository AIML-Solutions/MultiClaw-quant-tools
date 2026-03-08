#!/usr/bin/env bash
set -euo pipefail

WS="/home/aimls-dtd/.openclaw/workspace"
LOG="$WS/logs/qc-nightly-mlflow-suite.log"

export PATH="$HOME/.local/bin:/home/aimls-dtd/.npm-global/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-qc-nightly-backtests}"
# Low-burn defaults: run fewer projects unless explicitly overridden.
export QC_SUITE_MAX_PROJECTS="${QC_SUITE_MAX_PROJECTS:-2}"
export QC_BACKTEST_TIMEOUT_SEC="${QC_BACKTEST_TIMEOUT_SEC:-1800}"

cd "$WS"

echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] suite_start" >> "$LOG"
# Hard stop protects against runaway jobs.
/usr/bin/timeout 70m /usr/bin/python3 "$WS/automation/qc_nightly_mlflow_suite.py" >> "$LOG" 2>&1 || true
/usr/bin/timeout 8m /usr/bin/python3 "$WS/automation/qc_mlflow_visual_report.py" >> "$LOG" 2>&1 || true
echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] suite_done" >> "$LOG"
