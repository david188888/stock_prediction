#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/configs/default.yaml}"
RUN_NAME="${RUN_NAME:-overnight_$(date +"%Y%m%d_%H%M%S")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/runs/$RUN_NAME}"
LOG_DIR="$OUTPUT_ROOT/logs"
CONFIG_DIR="$OUTPUT_ROOT/configs"
STATUS_CSV="$OUTPUT_ROOT/run_status.csv"
CONSOLE_LOG="$LOG_DIR/console.log"
SYSTEM_INFO="$OUTPUT_ROOT/system_info.txt"

# User-tunable experiment controls.
MODELS="${MODELS:-linear_regression linear_regression_scaled arima garch_return lstm gru transformer arima_residual_lstm}"
STOCK_SYMBOLS="${STOCK_SYMBOLS:-}"
PREDICTION_HORIZONS="${PREDICTION_HORIZONS:-1 5}"
WINDOW_SIZES="${WINDOW_SIZES:-20 60}"
EVALUATION_MODES="${EVALUATION_MODES:-holdout walk_forward}"
WALK_FORWARD_MODELS="${WALK_FORWARD_MODELS:-$MODELS}"
WALK_FORWARD_STEPS="${WALK_FORWARD_STEPS:-0}"
GENERATE_FIGURES="${GENERATE_FIGURES:-true}"
SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.70}"
VAL_RATIO="${VAL_RATIO:-0.15}"
TEST_RATIO="${TEST_RATIO:-0.15}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
TUNE_ARIMA_ORDER="${TUNE_ARIMA_ORDER:-true}"
EXTREME_VOLATILITY_QUANTILE="${EXTREME_VOLATILITY_QUANTILE:-0.95}"
WINSORIZE_LOWER="${WINSORIZE_LOWER:-0.01}"
WINSORIZE_UPPER="${WINSORIZE_UPPER:-0.99}"
PRICE_COLUMN="${PRICE_COLUMN:-adjclose}"
TARGET_COLUMN="${TARGET_COLUMN:-target_next_adjclose}"
FEATURE_SET="${FEATURE_SET:-auto}"

# Sequence / hybrid training controls.
LSTM_HIDDEN_SIZE="${LSTM_HIDDEN_SIZE:-64}"
LSTM_NUM_LAYERS="${LSTM_NUM_LAYERS:-1}"
LSTM_DROPOUT="${LSTM_DROPOUT:-0.0}"
LSTM_EPOCHS="${LSTM_EPOCHS:-30}"
LSTM_BATCH_SIZE="${LSTM_BATCH_SIZE:-64}"
LSTM_LEARNING_RATE="${LSTM_LEARNING_RATE:-0.001}"

GRU_HIDDEN_SIZE="${GRU_HIDDEN_SIZE:-64}"
GRU_NUM_LAYERS="${GRU_NUM_LAYERS:-1}"
GRU_DROPOUT="${GRU_DROPOUT:-0.0}"
GRU_EPOCHS="${GRU_EPOCHS:-30}"
GRU_BATCH_SIZE="${GRU_BATCH_SIZE:-64}"
GRU_LEARNING_RATE="${GRU_LEARNING_RATE:-0.001}"

TRANSFORMER_HIDDEN_SIZE="${TRANSFORMER_HIDDEN_SIZE:-64}"
TRANSFORMER_NUM_LAYERS="${TRANSFORMER_NUM_LAYERS:-2}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"
TRANSFORMER_EPOCHS="${TRANSFORMER_EPOCHS:-30}"
TRANSFORMER_BATCH_SIZE="${TRANSFORMER_BATCH_SIZE:-64}"
TRANSFORMER_LEARNING_RATE="${TRANSFORMER_LEARNING_RATE:-0.001}"

HYBRID_ARIMA_ORDER="${HYBRID_ARIMA_ORDER:-5 1 0}"
HYBRID_HIDDEN_SIZE="${HYBRID_HIDDEN_SIZE:-32}"
HYBRID_NUM_LAYERS="${HYBRID_NUM_LAYERS:-1}"
HYBRID_DROPOUT="${HYBRID_DROPOUT:-0.0}"
HYBRID_EPOCHS="${HYBRID_EPOCHS:-30}"
HYBRID_BATCH_SIZE="${HYBRID_BATCH_SIZE:-64}"
HYBRID_LEARNING_RATE="${HYBRID_LEARNING_RATE:-0.001}"

# Runner behavior.
RUN_PREPARE_DATA="${RUN_PREPARE_DATA:-auto}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-true}"

mkdir -p "$LOG_DIR" "$CONFIG_DIR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  local message="$1"
  printf '[%s] %s\n' "$(timestamp)" "$message" | tee -a "$CONSOLE_LOG"
}

to_csv_list() {
  local value="$1"
  echo "$value" | tr ' ' ',' | sed 's/,,*/,/g; s/^,//; s/,$//'
}

run_python_module() {
  PYTHONPATH="$ROOT_DIR/src" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m "$@"
}

write_runtime_config() {
  local model="$1"
  local horizon="$2"
  local window_size="$3"
  local config_path="$4"
  export ROOT_DIR BASE_CONFIG OUTPUT_ROOT model horizon window_size config_path
  export MODELS STOCK_SYMBOLS PREDICTION_HORIZONS WINDOW_SIZES EVALUATION_MODES WALK_FORWARD_MODELS WALK_FORWARD_STEPS
  export GENERATE_FIGURES SEED TRAIN_RATIO VAL_RATIO TEST_RATIO EARLY_STOPPING_PATIENCE MIN_DELTA TUNE_ARIMA_ORDER
  export EXTREME_VOLATILITY_QUANTILE WINSORIZE_LOWER WINSORIZE_UPPER PRICE_COLUMN TARGET_COLUMN FEATURE_SET
  export LSTM_HIDDEN_SIZE LSTM_NUM_LAYERS LSTM_DROPOUT LSTM_EPOCHS LSTM_BATCH_SIZE LSTM_LEARNING_RATE
  export GRU_HIDDEN_SIZE GRU_NUM_LAYERS GRU_DROPOUT GRU_EPOCHS GRU_BATCH_SIZE GRU_LEARNING_RATE
  export TRANSFORMER_HIDDEN_SIZE TRANSFORMER_NUM_LAYERS TRANSFORMER_DROPOUT TRANSFORMER_EPOCHS TRANSFORMER_BATCH_SIZE TRANSFORMER_LEARNING_RATE
  export HYBRID_ARIMA_ORDER HYBRID_HIDDEN_SIZE HYBRID_NUM_LAYERS HYBRID_DROPOUT HYBRID_EPOCHS HYBRID_BATCH_SIZE HYBRID_LEARNING_RATE
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import yaml


def parse_list(name: str) -> list[str]:
    value = os.environ.get(name, "").strip()
    return value.split() if value else []


def parse_bool(name: str) -> bool:
    return os.environ.get(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def parse_float(name: str) -> float:
    return float(os.environ[name])


def parse_int(name: str) -> int:
    return int(os.environ[name])


with Path(os.environ["BASE_CONFIG"]).open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)

root_dir = Path(os.environ["ROOT_DIR"]).resolve()
model = os.environ["model"]
horizon = parse_int("horizon")
window_size = parse_int("window_size")
output_root = str(Path(os.environ["OUTPUT_ROOT"]).resolve())

dataset = payload["dataset"]
for key in ["raw_dir", "extracted_dir", "processed_dir", "interim_dir", "schema_path"]:
    path = Path(dataset[key])
    if not path.is_absolute():
        path = root_dir / path
    dataset[key] = str(path.resolve())

experiment = payload["experiment"]
experiment["stock_symbols"] = parse_list("STOCK_SYMBOLS")
experiment["feature_set"] = os.environ["FEATURE_SET"]
experiment["target_column"] = os.environ["TARGET_COLUMN"]
experiment["price_column"] = os.environ["PRICE_COLUMN"]
experiment["prediction_horizon"] = horizon
experiment["prediction_horizons"] = [horizon]
experiment["window_size"] = window_size
experiment["window_sizes"] = [window_size]
experiment["train_ratio"] = parse_float("TRAIN_RATIO")
experiment["val_ratio"] = parse_float("VAL_RATIO")
experiment["test_ratio"] = parse_float("TEST_RATIO")
experiment["seed"] = parse_int("SEED")
experiment["output_dir"] = output_root
experiment["generate_figures"] = parse_bool("GENERATE_FIGURES")
experiment["evaluation_modes"] = parse_list("EVALUATION_MODES")
experiment["walk_forward_steps"] = parse_int("WALK_FORWARD_STEPS")
experiment["walk_forward_models"] = parse_list("WALK_FORWARD_MODELS")
experiment["selected_models"] = [model]
experiment["early_stopping_patience"] = parse_int("EARLY_STOPPING_PATIENCE")
experiment["min_delta"] = parse_float("MIN_DELTA")
experiment["tune_arima_order"] = parse_bool("TUNE_ARIMA_ORDER")
experiment["extreme_volatility_quantile"] = parse_float("EXTREME_VOLATILITY_QUANTILE")
experiment["winsorize_limits"] = [parse_float("WINSORIZE_LOWER"), parse_float("WINSORIZE_UPPER")]

payload["models"]["lstm"].update(
    {
        "hidden_size": parse_int("LSTM_HIDDEN_SIZE"),
        "num_layers": parse_int("LSTM_NUM_LAYERS"),
        "dropout": parse_float("LSTM_DROPOUT"),
        "epochs": parse_int("LSTM_EPOCHS"),
        "batch_size": parse_int("LSTM_BATCH_SIZE"),
        "learning_rate": parse_float("LSTM_LEARNING_RATE"),
    }
)
payload["models"]["gru"].update(
    {
        "hidden_size": parse_int("GRU_HIDDEN_SIZE"),
        "num_layers": parse_int("GRU_NUM_LAYERS"),
        "dropout": parse_float("GRU_DROPOUT"),
        "epochs": parse_int("GRU_EPOCHS"),
        "batch_size": parse_int("GRU_BATCH_SIZE"),
        "learning_rate": parse_float("GRU_LEARNING_RATE"),
    }
)
payload["models"]["transformer"].update(
    {
        "hidden_size": parse_int("TRANSFORMER_HIDDEN_SIZE"),
        "num_layers": parse_int("TRANSFORMER_NUM_LAYERS"),
        "dropout": parse_float("TRANSFORMER_DROPOUT"),
        "epochs": parse_int("TRANSFORMER_EPOCHS"),
        "batch_size": parse_int("TRANSFORMER_BATCH_SIZE"),
        "learning_rate": parse_float("TRANSFORMER_LEARNING_RATE"),
    }
)
payload["models"]["hybrid"].update(
    {
        "arima_order": [int(value) for value in parse_list("HYBRID_ARIMA_ORDER")],
        "hidden_size": parse_int("HYBRID_HIDDEN_SIZE"),
        "num_layers": parse_int("HYBRID_NUM_LAYERS"),
        "dropout": parse_float("HYBRID_DROPOUT"),
        "epochs": parse_int("HYBRID_EPOCHS"),
        "batch_size": parse_int("HYBRID_BATCH_SIZE"),
        "learning_rate": parse_float("HYBRID_LEARNING_RATE"),
    }
)

config_path = Path(os.environ["config_path"])
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
PY
}

prepare_data_if_needed() {
  local prepared_rel_path
  prepared_rel_path="$(BASE_CONFIG="$BASE_CONFIG" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml
import os

config_path = Path(os.environ["BASE_CONFIG"])
with config_path.open("r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
dataset = payload["dataset"]
root = config_path.parent.parent
processed_dir = Path(dataset["processed_dir"])
if not processed_dir.is_absolute():
    processed_dir = root / processed_dir
print(processed_dir / dataset["prepared_filename"])
PY
)"

  if [[ "$RUN_PREPARE_DATA" == "false" ]]; then
    log "Skipping prepare-data because RUN_PREPARE_DATA=false."
    return
  fi
  if [[ "$RUN_PREPARE_DATA" == "auto" && -f "$prepared_rel_path" ]]; then
    log "Prepared dataset already exists at $prepared_rel_path. Skipping prepare-data."
    return
  fi

  log "Running prepare-data using $BASE_CONFIG."
  if ! run_python_module stock_prediction.cli.main prepare-data --config "$BASE_CONFIG" 2>&1 | tee -a "$CONSOLE_LOG"; then
    log "prepare-data failed."
    exit 1
  fi
}

write_system_info() {
  {
    echo "run_name=$RUN_NAME"
    echo "timestamp=$(timestamp)"
    echo "root_dir=$ROOT_DIR"
    echo "python_bin=$PYTHON_BIN"
    echo "base_config=$BASE_CONFIG"
    echo "output_root=$OUTPUT_ROOT"
    echo "models=$(to_csv_list "$MODELS")"
    echo "prediction_horizons=$(to_csv_list "$PREDICTION_HORIZONS")"
    echo "window_sizes=$(to_csv_list "$WINDOW_SIZES")"
    echo "evaluation_modes=$(to_csv_list "$EVALUATION_MODES")"
    echo "walk_forward_models=$(to_csv_list "$WALK_FORWARD_MODELS")"
    echo "walk_forward_steps=$WALK_FORWARD_STEPS"
    echo "seed=$SEED"
    echo "python_version=$("$PYTHON_BIN" --version 2>&1)"
    echo "uname=$(uname -a)"
    if command -v sw_vers >/dev/null 2>&1; then
      sw_vers
    fi
    if command -v sysctl >/dev/null 2>&1; then
      echo "hw.memsize=$(sysctl -n hw.memsize 2>/dev/null || true)"
      echo "machdep.cpu.brand_string=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    fi
    echo "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  } > "$SYSTEM_INFO"
}

append_status_header() {
  if [[ ! -f "$STATUS_CSV" ]]; then
    echo "model,prediction_horizon,window_size,status,started_at,finished_at,metrics_path,log_path" > "$STATUS_CSV"
  fi
}

main() {
  log "Starting overnight training run: $RUN_NAME"
  write_system_info
  append_status_header
  prepare_data_if_needed

  local failures=0
  local total_runs=0
  local start_at
  start_at="$(timestamp)"

  local horizon
  local window_size
  local model
  for horizon in $PREDICTION_HORIZONS; do
    for window_size in $WINDOW_SIZES; do
      for model in $MODELS; do
        total_runs=$((total_runs + 1))
        local combo_tag="${model}_h${horizon}_w${window_size}"
        local combo_config="$CONFIG_DIR/${combo_tag}.yaml"
        local combo_log="$LOG_DIR/${combo_tag}.log"
        local metrics_path="$OUTPUT_ROOT/metrics/${model}_${PRICE_COLUMN}_h${horizon}_w${window_size}_metrics.csv"
        local started_at
        local finished_at
        started_at="$(timestamp)"

        write_runtime_config "$model" "$horizon" "$window_size" "$combo_config"

        if [[ "$RESUME_IF_EXISTS" == "true" && -f "$metrics_path" ]]; then
          log "Skipping $combo_tag because metrics already exist at $metrics_path."
          finished_at="$(timestamp)"
          echo "$model,$horizon,$window_size,skipped,$started_at,$finished_at,$metrics_path,$combo_log" >> "$STATUS_CSV"
          continue
        fi

        log "Running $combo_tag"
        if /usr/bin/time -lp env PYTHONPATH="$ROOT_DIR/src" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m stock_prediction.cli.main train --model "$model" --config "$combo_config" \
          2>&1 | tee -a "$combo_log" | tee -a "$CONSOLE_LOG"; then
          finished_at="$(timestamp)"
          log "Completed $combo_tag"
          echo "$model,$horizon,$window_size,success,$started_at,$finished_at,$metrics_path,$combo_log" >> "$STATUS_CSV"
        else
          finished_at="$(timestamp)"
          failures=$((failures + 1))
          log "Failed $combo_tag"
          echo "$model,$horizon,$window_size,failed,$started_at,$finished_at,$metrics_path,$combo_log" >> "$STATUS_CSV"
          if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
            log "Stopping because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR."
            exit 1
          fi
        fi
      done
    done
  done

  log "Post-processing aggregated outputs and research figures."
  if ! PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/render_research_figures.py" --output-root "$OUTPUT_ROOT" \
    2>&1 | tee -a "$CONSOLE_LOG"; then
    log "Post-processing failed."
    exit 1
  fi

  log "Run finished. total_runs=$total_runs failures=$failures started_at=$start_at ended_at=$(timestamp)"
  log "Artifacts written under $OUTPUT_ROOT"
}

main "$@"
