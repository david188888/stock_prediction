#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/configs/default.yaml}"
PREPARED_DATA="${PREPARED_DATA:-$ROOT_DIR/data/processed/prepared_stock_data.csv}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-smoke_$(date +"%Y%m%d_%H%M%S")}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-$ROOT_DIR/outputs/runs/$SMOKE_RUN_NAME}"
SMOKE_SYMBOL_COUNT="${SMOKE_SYMBOL_COUNT:-2}"
SMOKE_SYMBOLS="${SMOKE_SYMBOLS:-}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$1"
}

fail() {
  log "ERROR: $1"
  exit 1
}

check_paths() {
  [[ -f "$BASE_CONFIG" ]] || fail "Config not found: $BASE_CONFIG"
  [[ -f "$PREPARED_DATA" ]] || fail "Prepared dataset not found: $PREPARED_DATA"
}

check_python_dependencies() {
  log "Checking runtime dependencies."
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required_modules = [
    "yaml",
    "numpy",
    "pandas",
    "torch",
    "matplotlib",
]
missing = []
for module_name in required_modules:
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(module_name)

if missing:
    print("Missing modules:", ", ".join(missing))
    sys.exit(1)
PY
}

resolve_symbols() {
  if [[ -n "$SMOKE_SYMBOLS" ]]; then
    echo "$SMOKE_SYMBOLS"
    return
  fi

  PREPARED_DATA="$PREPARED_DATA" SMOKE_SYMBOL_COUNT="$SMOKE_SYMBOL_COUNT" "$PYTHON_BIN" - <<'PY'
import csv
import os
from collections import Counter
from pathlib import Path

path = Path(os.environ["PREPARED_DATA"])
target_count = int(os.environ["SMOKE_SYMBOL_COUNT"])
counts = Counter()
with path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        symbol = row.get("symbol")
        if symbol:
            counts[symbol] += 1

selected = [symbol for symbol, _ in counts.most_common(target_count)]
print(" ".join(selected))
PY
}

run_smoke_training() {
  local symbols="$1"
  log "Running smoke training with symbols: $symbols"

  RUN_NAME="$SMOKE_RUN_NAME" \
  OUTPUT_ROOT="$SMOKE_OUTPUT_ROOT" \
  STOCK_SYMBOLS="$symbols" \
  MODELS="linear_regression linear_regression_scaled arima garch_return lstm gru transformer arima_residual_lstm" \
  PREDICTION_HORIZONS="1" \
  WINDOW_SIZES="20" \
  EVALUATION_MODES="holdout walk_forward" \
  WALK_FORWARD_MODELS="linear_regression linear_regression_scaled arima garch_return lstm gru transformer arima_residual_lstm" \
  WALK_FORWARD_STEPS="3" \
  LSTM_EPOCHS="3" \
  GRU_EPOCHS="3" \
  TRANSFORMER_EPOCHS="3" \
  HYBRID_EPOCHS="3" \
  LSTM_BATCH_SIZE="16" \
  GRU_BATCH_SIZE="16" \
  TRANSFORMER_BATCH_SIZE="16" \
  HYBRID_BATCH_SIZE="16" \
  TUNE_ARIMA_ORDER="false" \
  CONTINUE_ON_ERROR="false" \
  RESUME_IF_EXISTS="false" \
  RUN_PREPARE_DATA="auto" \
  bash "$ROOT_DIR/scripts/run_overnight_full_training.sh"
}

verify_outputs() {
  log "Verifying smoke-run artifacts under $SMOKE_OUTPUT_ROOT"
  SMOKE_OUTPUT_ROOT="$SMOKE_OUTPUT_ROOT" "$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

output_root = Path(os.environ["SMOKE_OUTPUT_ROOT"])
status_path = output_root / "run_status.csv"
if not status_path.exists():
    raise SystemExit(f"Missing run status file: {status_path}")

with status_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    raise SystemExit("run_status.csv is empty")

failed = [row for row in rows if row["status"] != "success"]
if failed:
    raise SystemExit(f"Smoke run contains non-success rows: {failed}")

metrics_files = sorted((output_root / "metrics").glob("*_metrics.csv"))
prediction_files = sorted((output_root / "predictions").glob("*_predictions.csv"))
research_figures = sorted((output_root / "figures" / "research").glob("*.png"))

if len(metrics_files) < 8:
    raise SystemExit(f"Expected at least 8 metrics files, found {len(metrics_files)}")
if len(prediction_files) < 8:
    raise SystemExit(f"Expected at least 8 prediction files, found {len(prediction_files)}")
if not (output_root / "metrics" / "leaderboard.csv").exists():
    raise SystemExit("Missing aggregated leaderboard.csv")
if not (output_root / "metrics" / "leaderboard.md").exists():
    raise SystemExit("Missing aggregated leaderboard.md")
if not (output_root / "metrics" / "conclusions.md").exists():
    raise SystemExit("Missing aggregated conclusions.md")
if len(research_figures) < 3:
    raise SystemExit(f"Expected research figures, found {len(research_figures)}")

training_logs = sorted((output_root / "metrics").glob("*_training_log.csv"))
if not training_logs:
    raise SystemExit("No training_log.csv files found")

print(f"Smoke verification passed. metrics={len(metrics_files)} predictions={len(prediction_files)} figures={len(research_figures)}")
PY
}

main() {
  log "Starting overnight smoke test."
  check_paths
  check_python_dependencies

  local symbols
  symbols="$(resolve_symbols)"
  [[ -n "$symbols" ]] || fail "Could not resolve smoke-test symbols."

  run_smoke_training "$symbols"
  verify_outputs

  log "Smoke test passed. It is reasonable to start the overnight full run."
  log "Smoke outputs: $SMOKE_OUTPUT_ROOT"
}

main "$@"
