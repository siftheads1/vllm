#!/usr/bin/env bash
# Run the Milestone 5 Step 5.1 runtime latency matrix.
#
# This script intentionally keeps VLLM_MPR_DEBUG_DIR unset for latency runs so
# JSONL debug writing does not enter the production-like overhead baseline.

set -euo pipefail

MODEL="Qwen/Qwen3-8B"
DTYPE="half"
MAX_MODEL_LEN="2048"
MAX_TOKENS="512"
BLOCK_SIZE="32"
GPU_MEMORY_UTILIZATION="0.75"
TENSOR_PARALLEL_SIZE="1"
SEED="0"
PYTHON_BIN="python"
WORK_DIR=""
WARMUP_RUNS="1"
MEASURED_RUNS="5"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/mpr_run_m5_step51_latency.sh [options]

Options:
  --work-dir PATH                 Output directory. Defaults to /tmp/mpr_m5_step51_<timestamp>.
  --python PATH                   Python executable. Default: python.
  --warmup-runs N                 Warmup generations per mode. Default: 1.
  --measured-runs N               Measured generations per mode. Default: 5.
  --model NAME                    Model name/path. Default: Qwen/Qwen3-8B.
  --dtype DTYPE                   vLLM dtype. Default: half.
  --max-model-len N               Max model length. Default: 2048.
  --max-tokens N                  Generated token count. Default: 512.
  --block-size N                  KV block size. Default: 32.
  --gpu-memory-utilization FLOAT  vLLM GPU memory utilization. Default: 0.75.
  --tensor-parallel-size N        Tensor parallel size. Default: 1.
  --seed N                        Deterministic seed. Default: 0.
  --extra-arg VALUE               Append one argument to mpr_benchmark_step_latency.py.
                                  Repeat for flags and values, e.g.
                                  --extra-arg --trust-remote-code.
  -h, --help                      Show this help.

The script runs seven modes:
  baseline, mpr_enable_only, backup_only, scoring_only, fp16_recovery,
  mixed_int8, mixed_int4.

Each mode writes:
  <mode>_warmup_<idx>.log/.csv
  <mode>_measured_<idx>.log/.csv
  runtime_summary.csv
EOF
}

require_value() {
  local name="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "error: ${name} requires a value" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      require_value "$1" "${2:-}"
      WORK_DIR="$2"
      shift 2
      ;;
    --python)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --warmup-runs)
      require_value "$1" "${2:-}"
      WARMUP_RUNS="$2"
      shift 2
      ;;
    --measured-runs)
      require_value "$1" "${2:-}"
      MEASURED_RUNS="$2"
      shift 2
      ;;
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --dtype)
      require_value "$1" "${2:-}"
      DTYPE="$2"
      shift 2
      ;;
    --max-model-len)
      require_value "$1" "${2:-}"
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-tokens)
      require_value "$1" "${2:-}"
      MAX_TOKENS="$2"
      shift 2
      ;;
    --block-size)
      require_value "$1" "${2:-}"
      BLOCK_SIZE="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      require_value "$1" "${2:-}"
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      require_value "$1" "${2:-}"
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --seed)
      require_value "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --extra-arg)
      require_value "$1" "${2:-}"
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_nonnegative_int "$WARMUP_RUNS"; then
  echo "error: --warmup-runs must be a non-negative integer" >&2
  exit 2
fi
if ! is_nonnegative_int "$MEASURED_RUNS" || [[ "$MEASURED_RUNS" -eq 0 ]]; then
  echo "error: --measured-runs must be a positive integer" >&2
  exit 2
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="/tmp/mpr_m5_step51_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$WORK_DIR"

BASE_ENV=(
  env
  -u VLLM_MPR_ENABLE
  -u VLLM_MPR_CPU_BACKUP
  -u VLLM_MPR_SCORING_ENABLE
  -u VLLM_MPR_RECOVERY_ENABLE
  -u VLLM_MPR_BACKUP_STORAGE_MODE
  -u VLLM_MPR_PRECISION_TIERING_ENABLE
  -u VLLM_MPR_PRECISION_POLICY
  -u VLLM_MPR_TIER_FP16_RATIO
  -u VLLM_MPR_TIER_INT8_RATIO
  -u VLLM_MPR_TIER_INT4_RATIO
  -u VLLM_MPR_TIER_HIGH_THRESHOLD
  -u VLLM_MPR_TIER_MID_THRESHOLD
  -u VLLM_MPR_TIER_LOW_THRESHOLD
  -u VLLM_MPR_RECOVERY_POLICY
  -u VLLM_MPR_RECOVERY_THRESHOLD
  -u VLLM_MPR_RECOVERY_TOPK
  -u VLLM_MPR_RECOVERY_TEST_MUTATE
  -u VLLM_MPR_RECOVERY_TEST_MODE
  -u VLLM_MPR_DEBUG_DIR
  VLLM_USE_V1=1
)

COMMON_ARGS=(
  --model "$MODEL"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-tokens "$MAX_TOKENS"
  --block-size "$BLOCK_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --seed "$SEED"
  --ignore-eos
)

write_manifest() {
  local manifest="$WORK_DIR/manifest.txt"
  {
    echo "benchmark: mpr_m5_step51_latency"
    echo "work_dir: $WORK_DIR"
    echo "python: $PYTHON_BIN"
    echo "model: $MODEL"
    echo "dtype: $DTYPE"
    echo "max_model_len: $MAX_MODEL_LEN"
    echo "max_tokens: $MAX_TOKENS"
    echo "block_size: $BLOCK_SIZE"
    echo "gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
    echo "tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
    echo "seed: $SEED"
    echo "warmup_runs: $WARMUP_RUNS"
    echo "measured_runs: $MEASURED_RUNS"
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      printf 'extra_args:'
      printf ' %q' "${EXTRA_ARGS[@]}"
      printf '\n'
    else
      echo "extra_args:"
    fi
  } > "$manifest"
}

write_runtime_summary() {
  "$PYTHON_BIN" - "$WORK_DIR" <<'PY'
import ast
import csv
import re
import statistics
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
modes = [
    "baseline",
    "mpr_enable_only",
    "backup_only",
    "scoring_only",
    "fp16_recovery",
    "mixed_int8",
    "mixed_int4",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def parse_float_line(text: str, name: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}:\s+([0-9.]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_boundary_steps(text: str) -> set[int]:
    match = re.search(r"^predicted_boundary_steps:\s+(\[.*\])", text,
                      re.MULTILINE)
    if not match:
        return set()
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list):
        return set()
    return {int(item) for item in value}


rows: list[dict[str, str | int | float | bool]] = []
for mode in modes:
    logs = sorted(work_dir.glob(f"{mode}_measured_*.log"))
    generate_elapsed: list[float] = []
    tokens_per_sec: list[float] = []
    step_latencies: list[float] = []
    boundary_latencies: list[float] = []
    non_boundary_latencies: list[float] = []
    csv_paths: list[str] = []

    for log_path in logs:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        generate_elapsed_value = parse_float_line(text, "generate_elapsed_sec")
        if generate_elapsed_value is not None:
            generate_elapsed.append(generate_elapsed_value)
        tokens_per_sec_value = parse_float_line(text, "generated_tokens_per_sec")
        if tokens_per_sec_value is not None:
            tokens_per_sec.append(tokens_per_sec_value)

        boundary_steps = parse_boundary_steps(text)
        csv_path = log_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        csv_paths.append(str(csv_path))
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                step_idx = int(row["step_idx"])
                latency = float(row["latency_ms"])
                step_latencies.append(latency)
                if step_idx in boundary_steps:
                    boundary_latencies.append(latency)
                else:
                    non_boundary_latencies.append(latency)

    step_median = median(step_latencies)
    step_p95 = percentile(step_latencies, 95)
    step_max = max(step_latencies) if step_latencies else 0.0
    outlier = bool(
        step_latencies
        and step_median > 0.0
        and (step_p95 > 1.5 * step_median or step_max > 2.0 * step_median)
    )
    rows.append({
        "mode": mode,
        "measured_runs": len(logs),
        "generate_elapsed_sec_mean": mean(generate_elapsed),
        "generate_elapsed_sec_median": median(generate_elapsed),
        "generated_tokens_per_sec_mean": mean(tokens_per_sec),
        "generated_tokens_per_sec_median": median(tokens_per_sec),
        "engine_step_count": len(step_latencies),
        "step_latency_ms_mean": mean(step_latencies),
        "step_latency_ms_median": step_median,
        "step_latency_ms_p95": step_p95,
        "step_latency_ms_p99": percentile(step_latencies, 99),
        "step_latency_ms_max": step_max,
        "boundary_step_latency_ms_mean": mean(boundary_latencies),
        "boundary_step_latency_ms_max": (
            max(boundary_latencies) if boundary_latencies else 0.0
        ),
        "non_boundary_step_latency_ms_mean": mean(non_boundary_latencies),
        "non_boundary_step_latency_ms_max": (
            max(non_boundary_latencies) if non_boundary_latencies else 0.0
        ),
        "outlier_rerun_recommended": outlier,
        "csv_paths": ";".join(csv_paths),
    })

summary_path = work_dir / "runtime_summary.csv"
fieldnames = list(rows[0].keys()) if rows else []
with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
    writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

outlier_path = work_dir / "outlier_check.txt"
with outlier_path.open("w", encoding="utf-8") as outlier_file:
    outlier_modes = [
        str(row["mode"]) for row in rows if row["outlier_rerun_recommended"]
    ]
    if outlier_modes:
        outlier_file.write(
            "Rerun these modes with --warmup-runs 3 --measured-runs 5:\n"
        )
        for mode in outlier_modes:
            outlier_file.write(f"{mode}\n")
    else:
        outlier_file.write("No p95/max outlier rerun recommended.\n")

print(f"runtime_summary_csv: {summary_path}")
print(f"outlier_check: {outlier_path}")
PY
}

run_one() {
  local mode="$1"
  local phase="$2"
  local idx="$3"
  shift 3
  local prefix="$WORK_DIR/${mode}_${phase}_${idx}"
  echo "running ${mode} ${phase} ${idx}"
  "${BASE_ENV[@]}" "$@" \
    "$PYTHON_BIN" scripts/mpr_benchmark_step_latency.py \
      "${COMMON_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      --csv "${prefix}.csv" \
    > "${prefix}.log" 2>&1
}

run_mode() {
  local mode="$1"
  shift
  local idx
  for ((idx = 1; idx <= WARMUP_RUNS; idx += 1)); do
    run_one "$mode" warmup "$idx" "$@"
  done
  for ((idx = 1; idx <= MEASURED_RUNS; idx += 1)); do
    run_one "$mode" measured "$idx" "$@"
  done
}

write_manifest

run_mode baseline

run_mode mpr_enable_only \
  VLLM_MPR_ENABLE=1 \
  VLLM_MPR_CPU_BACKUP=0 \
  VLLM_MPR_SCORING_ENABLE=0 \
  VLLM_MPR_RECOVERY_ENABLE=0

run_mode backup_only \
  VLLM_MPR_ENABLE=1 \
  VLLM_MPR_CPU_BACKUP=1 \
  VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only \
  VLLM_MPR_SCORING_ENABLE=0 \
  VLLM_MPR_RECOVERY_ENABLE=0

run_mode scoring_only \
  VLLM_MPR_ENABLE=1 \
  VLLM_MPR_CPU_BACKUP=0 \
  VLLM_MPR_SCORING_ENABLE=1 \
  VLLM_MPR_RECOVERY_ENABLE=0

run_mode fp16_recovery \
  VLLM_MPR_ENABLE=1 \
  VLLM_MPR_CPU_BACKUP=1 \
  VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only \
  VLLM_MPR_SCORING_ENABLE=1 \
  VLLM_MPR_RECOVERY_ENABLE=1 \
  VLLM_MPR_PRECISION_TIERING_ENABLE=0 \
  VLLM_MPR_RECOVERY_POLICY=threshold_block \
  VLLM_MPR_RECOVERY_THRESHOLD=-1e30 \
  VLLM_MPR_RECOVERY_TOPK=1 \
  VLLM_MPR_RECOVERY_TEST_MUTATE=off

run_mode mixed_int8 \
  VLLM_MPR_ENABLE=1 \
  VLLM_MPR_CPU_BACKUP=1 \
  VLLM_MPR_BACKUP_STORAGE_MODE=eager_fp16_int8 \
  VLLM_MPR_SCORING_ENABLE=1 \
  VLLM_MPR_RECOVERY_ENABLE=1 \
  VLLM_MPR_PRECISION_TIERING_ENABLE=1 \
  VLLM_MPR_PRECISION_POLICY=top_ratio \
  VLLM_MPR_TIER_FP16_RATIO=0.25 \
  VLLM_MPR_TIER_INT8_RATIO=0.75 \
  VLLM_MPR_TIER_INT4_RATIO=0.0 \
  VLLM_MPR_RECOVERY_POLICY=threshold_block \
  VLLM_MPR_RECOVERY_THRESHOLD=-1e30 \
  VLLM_MPR_RECOVERY_TOPK=1 \
  VLLM_MPR_RECOVERY_TEST_MUTATE=off

run_mode mixed_int4 \
  VLLM_MPR_ENABLE=1 \
  VLLM_MPR_CPU_BACKUP=1 \
  VLLM_MPR_BACKUP_STORAGE_MODE=eager_fp16_int8_int4 \
  VLLM_MPR_SCORING_ENABLE=1 \
  VLLM_MPR_RECOVERY_ENABLE=1 \
  VLLM_MPR_PRECISION_TIERING_ENABLE=1 \
  VLLM_MPR_PRECISION_POLICY=top_ratio \
  VLLM_MPR_TIER_FP16_RATIO=0.25 \
  VLLM_MPR_TIER_INT8_RATIO=0.25 \
  VLLM_MPR_TIER_INT4_RATIO=0.50 \
  VLLM_MPR_RECOVERY_POLICY=threshold_block \
  VLLM_MPR_RECOVERY_THRESHOLD=-1e30 \
  VLLM_MPR_RECOVERY_TOPK=1 \
  VLLM_MPR_RECOVERY_TEST_MUTATE=off

write_runtime_summary

echo "runtime logs, CSVs, and manifest: $WORK_DIR"
