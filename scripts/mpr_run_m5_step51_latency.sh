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
MODES="baseline,mpr_enable_only,backup_only,scoring_only,fp16_recovery,mixed_int8,mixed_int4"
OBSERVE_BACKEND="slot"
BOUNDARY_PROFILE="0"
SCORING_PROFILE="0"

usage() {
  cat <<'EOF'
Usage: bash scripts/mpr_run_m5_step51_latency.sh [options]

Options:
  --work-dir PATH                 Output directory. Defaults to ../results/mpr_m5_step51_<timestamp>.
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
  --modes CSV                     Comma-separated modes to run. Default: all modes.
                                  Example: baseline,mpr_enable_only
  --observe-backend NAME          MPR KV observe backend: slot or counter. Default: slot.
  --boundary-profile              Enable targeted MPR boundary-step profiling counters.
  --scoring-profile               Enable targeted MPR scoring-path profiling counters.
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
    --modes)
      require_value "$1" "${2:-}"
      MODES="$2"
      shift 2
      ;;
    --observe-backend)
      require_value "$1" "${2:-}"
      OBSERVE_BACKEND="$2"
      shift 2
      ;;
    --boundary-profile)
      BOUNDARY_PROFILE="1"
      shift
      ;;
    --scoring-profile)
      SCORING_PROFILE="1"
      shift
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
if [[ "$OBSERVE_BACKEND" != "slot" && "$OBSERVE_BACKEND" != "counter" ]]; then
  echo "error: --observe-backend must be slot or counter" >&2
  exit 2
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="/home/han/KV_cache_quant/proposed_method_develop/results/mpr_m5_step51_$(date +%Y%m%d_%H%M%S)"
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
  -u VLLM_MPR_ENABLE_LOGGING
  -u VLLM_MPR_OBSERVE_BACKEND
  -u VLLM_MPR_BOUNDARY_PROFILE
  -u VLLM_MPR_SCORING_PROFILE
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
    echo "modes: $MODES"
    echo "observe_backend: $OBSERVE_BACKEND"
    echo "boundary_profile: $BOUNDARY_PROFILE"
    echo "scoring_profile: $SCORING_PROFILE"
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
  "$PYTHON_BIN" - "$WORK_DIR" "$MODES" <<'PY'
import ast
import csv
import json
import re
import statistics
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
modes = [mode for mode in sys.argv[2].split(",") if mode]


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


def read_mpr_observe_timing(log_path: Path) -> dict[str, float]:
    base = log_path.with_suffix(".mpr_observe_timing.json")
    timing_paths = sorted(log_path.parent.glob(base.name + ".*.json"))
    count = 0.0
    total_ms = 0.0
    max_ms = 0.0
    for timing_path in timing_paths:
        try:
            data = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        count += float(data.get("count", 0.0))
        total_ms += float(data.get("total_ms", 0.0))
        max_ms = max(max_ms, float(data.get("max_ms", 0.0)))
    mean_ms = total_ms / count if count > 0.0 else 0.0
    return {
        "count": count,
        "total_ms": total_ms,
        "mean_ms": mean_ms,
        "max_ms": max_ms,
    }


def parse_mpr_observe_timing(text: str) -> dict[str, float]:
    count = parse_float_line(text, "mpr_observe_hook_count") or 0.0
    total_ms = parse_float_line(text, "mpr_observe_hook_total_ms") or 0.0
    mean_ms = parse_float_line(text, "mpr_observe_hook_mean_ms") or 0.0
    max_ms = parse_float_line(text, "mpr_observe_hook_max_ms") or 0.0
    return {
        "count": count,
        "total_ms": total_ms,
        "mean_ms": mean_ms,
        "max_ms": max_ms,
    }


COUNTER_OBSERVE_KEYS = [
    "counter_observe_candidate_count",
    "counter_observe_accepted",
    "counter_observe_used",
    "counter_observe_missing_attn_metadata",
    "counter_observe_missing_block_table",
    "counter_observe_bad_query_len",
    "counter_observe_unsupported_num_reqs",
    "counter_observe_leader_selected",
    "counter_observe_prefill_initialized",
    "counter_observe_decode_advanced",
    "counter_observe_uninitialized",
    "counter_observe_not_pure_decode",
    "counter_observe_fallback_required",
    "counter_observe_no_boundary",
    "counter_observe_boundary_requests",
    "counter_observe_digest_created",
]

BOUNDARY_PROFILE_TIMING_KEYS = [
    "counter_prepare",
    "counter_observe",
    "counter_key_cache",
    "counter_block_lookup",
    "counter_create_digest",
    "counter_summarize_key_block",
    "counter_to_block_digest",
    "counter_append_quest_metadata",
    "counter_backup",
]

SCORING_PROFILE_TIMING_KEYS = [
    "scoring_estimate_query_scores",
    "scoring_record_estimated",
    "scoring_should_record",
    "scoring_query_clone",
    "scoring_window_stack_mean",
    "scoring_block_size",
    "scoring_request_block_context",
    "scoring_request_ctx_seq_lens",
    "scoring_request_ctx_block_table_lookup",
    "scoring_request_ctx_block_table_row",
    "scoring_request_ctx_candidates",
    "scoring_select_digest_blocks",
    "scoring_quest_packed_prefix",
    "scoring_quest_packed_estimate",
    "quest_packed_estimate_output_alloc",
    "quest_packed_estimate_query_prepare",
    "quest_packed_estimate_custom_op",
    "quest_packed_estimate_transpose",
    "quest_packed_estimate_aggregate",
    "quest_packed_estimate_result_build",
    "scoring_pack_layer_digests",
    "scoring_backend_estimate",
    "scoring_score_packing_debug",
    "scoring_block_topk",
    "scoring_head_topk_debug",
    "scoring_block_debug_fields",
    "scoring_context_build",
]


rows: list[dict[str, str | int | float | bool]] = []
for mode in modes:
    logs = sorted(work_dir.glob(f"{mode}_measured_*.log"))
    generate_elapsed: list[float] = []
    tokens_per_sec: list[float] = []
    step_latencies: list[float] = []
    prefill_latencies: list[float] = []
    decode_latencies: list[float] = []
    boundary_decode_latencies: list[float] = []
    non_boundary_decode_latencies: list[float] = []
    mpr_observe_hook_counts: list[float] = []
    mpr_observe_hook_total_ms: list[float] = []
    mpr_observe_hook_mean_ms: list[float] = []
    mpr_observe_hook_max_ms: list[float] = []
    mpr_observe_hook_per_decode_step_ms: list[float] = []
    mpr_observe_pre_block_offsets_ms: list[float] = []
    mpr_observe_pre_block_offsets_per_decode_step_ms: list[float] = []
    mpr_observe_block_offsets_ms: list[float] = []
    mpr_observe_block_offsets_per_decode_step_ms: list[float] = []
    mpr_counter_observe_values = {
        key: [] for key in COUNTER_OBSERVE_KEYS
    }
    mpr_boundary_profile_values = {
        f"{key}_{field}": []
        for key in BOUNDARY_PROFILE_TIMING_KEYS
        for field in ("count", "total_ms", "mean_ms", "max_ms")
    }
    mpr_scoring_profile_values = {
        f"{key}_{field}": []
        for key in SCORING_PROFILE_TIMING_KEYS
        for field in ("count", "total_ms", "mean_ms", "max_ms")
    }
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
        run_decode_count = 0
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                step_idx = int(row["step_idx"])
                latency = float(row["latency_ms"])
                phase = row.get("phase")
                if phase not in {"prefill", "decode"}:
                    phase = "prefill" if step_idx == 0 else "decode"
                step_latencies.append(latency)
                if phase == "prefill":
                    prefill_latencies.append(latency)
                    continue
                decode_latencies.append(latency)
                run_decode_count += 1
                if step_idx in boundary_steps:
                    boundary_decode_latencies.append(latency)
                else:
                    non_boundary_decode_latencies.append(latency)

        timing = read_mpr_observe_timing(log_path)
        if timing["count"] <= 0.0:
            timing = parse_mpr_observe_timing(text)
        if timing["count"] > 0.0:
            mpr_observe_hook_counts.append(timing["count"])
            mpr_observe_hook_total_ms.append(timing["total_ms"])
            mpr_observe_hook_mean_ms.append(timing["mean_ms"])
            mpr_observe_hook_max_ms.append(timing["max_ms"])
            mpr_observe_hook_per_decode_step_ms.append(
                timing["total_ms"] / run_decode_count
                if run_decode_count > 0
                else 0.0
            )
        pre_offsets_ms = parse_float_line(
            text,
            "mpr_observe_pre_block_offsets_total_ms",
        )
        if pre_offsets_ms is not None:
            mpr_observe_pre_block_offsets_ms.append(pre_offsets_ms)
        pre_offsets_per_decode_ms = parse_float_line(
            text,
            "mpr_observe_pre_block_offsets_total_per_decode_step_ms",
        )
        if pre_offsets_per_decode_ms is not None:
            mpr_observe_pre_block_offsets_per_decode_step_ms.append(
                pre_offsets_per_decode_ms
            )
        block_offsets_ms = parse_float_line(
            text,
            "mpr_observe_block_offsets_total_ms",
        )
        if block_offsets_ms is not None:
            mpr_observe_block_offsets_ms.append(block_offsets_ms)
        block_offsets_per_decode_ms = parse_float_line(
            text,
            "mpr_observe_block_offsets_total_per_decode_step_ms",
        )
        if block_offsets_per_decode_ms is not None:
            mpr_observe_block_offsets_per_decode_step_ms.append(
                block_offsets_per_decode_ms
            )
        for key in COUNTER_OBSERVE_KEYS:
            value = parse_float_line(text, f"mpr_{key}")
            if value is not None:
                mpr_counter_observe_values[key].append(value)
        for key in BOUNDARY_PROFILE_TIMING_KEYS:
            for field in ("count", "total_ms", "mean_ms", "max_ms"):
                value = parse_float_line(
                    text,
                    f"mpr_boundary_profile_{key}_{field}",
                )
                if value is not None:
                    mpr_boundary_profile_values[f"{key}_{field}"].append(value)
        for key in SCORING_PROFILE_TIMING_KEYS:
            for field in ("count", "total_ms", "mean_ms", "max_ms"):
                value = parse_float_line(
                    text,
                    f"mpr_scoring_profile_{key}_{field}",
                )
                if value is not None:
                    mpr_scoring_profile_values[f"{key}_{field}"].append(value)

    step_median = median(step_latencies)
    step_p95 = percentile(step_latencies, 95)
    step_max = max(step_latencies) if step_latencies else 0.0
    decode_median = median(decode_latencies)
    decode_p95 = percentile(decode_latencies, 95)
    decode_max = max(decode_latencies) if decode_latencies else 0.0
    boundary_decode_median = median(boundary_decode_latencies)
    boundary_decode_p90 = percentile(boundary_decode_latencies, 90)
    boundary_decode_p95 = percentile(boundary_decode_latencies, 95)
    boundary_decode_p99 = percentile(boundary_decode_latencies, 99)
    boundary_decode_max = (
        max(boundary_decode_latencies) if boundary_decode_latencies else 0.0
    )
    non_boundary_decode_median = median(non_boundary_decode_latencies)
    non_boundary_decode_p90 = percentile(non_boundary_decode_latencies, 90)
    non_boundary_decode_p95 = percentile(non_boundary_decode_latencies, 95)
    non_boundary_decode_p99 = percentile(non_boundary_decode_latencies, 99)
    non_boundary_decode_max = (
        max(non_boundary_decode_latencies)
        if non_boundary_decode_latencies
        else 0.0
    )
    outlier = bool(
        decode_latencies
        and decode_median > 0.0
        and (
            decode_p95 > 1.5 * decode_median
            or decode_max > 2.0 * decode_median
        )
    )
    row = {
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
        "prefill_step_latency_ms_mean": mean(prefill_latencies),
        "prefill_step_latency_ms_max": (
            max(prefill_latencies) if prefill_latencies else 0.0
        ),
        "decode_step_count": len(decode_latencies),
        "decode_step_latency_ms_mean": mean(decode_latencies),
        "decode_step_latency_ms_median": decode_median,
        "decode_step_latency_ms_p95": decode_p95,
        "decode_step_latency_ms_p99": percentile(decode_latencies, 99),
        "decode_step_latency_ms_max": decode_max,
        "boundary_decode_step_latency_ms_mean": mean(boundary_decode_latencies),
        "boundary_decode_step_latency_ms_median": boundary_decode_median,
        "boundary_decode_step_latency_ms_p90": boundary_decode_p90,
        "boundary_decode_step_latency_ms_p95": boundary_decode_p95,
        "boundary_decode_step_latency_ms_p99": boundary_decode_p99,
        "boundary_decode_step_latency_ms_max": boundary_decode_max,
        "non_boundary_decode_step_latency_ms_mean": (
            mean(non_boundary_decode_latencies)
        ),
        "non_boundary_decode_step_latency_ms_median": non_boundary_decode_median,
        "non_boundary_decode_step_latency_ms_p90": non_boundary_decode_p90,
        "non_boundary_decode_step_latency_ms_p95": non_boundary_decode_p95,
        "non_boundary_decode_step_latency_ms_p99": non_boundary_decode_p99,
        "non_boundary_decode_step_latency_ms_max": non_boundary_decode_max,
        "mpr_observe_hook_count_mean": mean(mpr_observe_hook_counts),
        "mpr_observe_hook_count_total": sum(mpr_observe_hook_counts),
        "mpr_observe_hook_total_ms_mean": mean(mpr_observe_hook_total_ms),
        "mpr_observe_hook_mean_ms_mean": mean(mpr_observe_hook_mean_ms),
        "mpr_observe_hook_max_ms_max": (
            max(mpr_observe_hook_max_ms) if mpr_observe_hook_max_ms else 0.0
        ),
        "mpr_observe_hook_total_per_decode_step_ms_mean": (
            mean(mpr_observe_hook_per_decode_step_ms)
        ),
        "mpr_observe_pre_block_offsets_total_ms_mean": (
            mean(mpr_observe_pre_block_offsets_ms)
        ),
        "mpr_observe_pre_block_offsets_total_per_decode_step_ms_mean": (
            mean(mpr_observe_pre_block_offsets_per_decode_step_ms)
        ),
        "mpr_observe_block_offsets_total_ms_mean": (
            mean(mpr_observe_block_offsets_ms)
        ),
        "mpr_observe_block_offsets_total_per_decode_step_ms_mean": (
            mean(mpr_observe_block_offsets_per_decode_step_ms)
        ),
        "outlier_rerun_recommended": outlier,
        "csv_paths": ";".join(csv_paths),
    }
    for key in COUNTER_OBSERVE_KEYS:
        row[f"mpr_{key}_mean"] = mean(mpr_counter_observe_values[key])
    for key in BOUNDARY_PROFILE_TIMING_KEYS:
        for field in ("count", "total_ms", "mean_ms", "max_ms"):
            row[f"mpr_boundary_profile_{key}_{field}_mean"] = mean(
                mpr_boundary_profile_values[f"{key}_{field}"]
            )
    for key in SCORING_PROFILE_TIMING_KEYS:
        for field in ("count", "total_ms", "mean_ms", "max_ms"):
            row[f"mpr_scoring_profile_{key}_{field}_mean"] = mean(
                mpr_scoring_profile_values[f"{key}_{field}"]
            )
    rows.append(row)

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
  local profile_env=()
  if [[ "$BOUNDARY_PROFILE" == "1" ]]; then
    profile_env=(VLLM_MPR_BOUNDARY_PROFILE=1)
  fi
  if [[ "$SCORING_PROFILE" == "1" ]]; then
    profile_env+=("VLLM_MPR_SCORING_PROFILE=1")
  fi
  echo "running ${mode} ${phase} ${idx}"
  "${BASE_ENV[@]}" "${profile_env[@]}" "$@" \
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

mode_enabled() {
  local mode="$1"
  [[ ",$MODES," == *",$mode,"* ]]
}

write_manifest

if mode_enabled baseline; then
  run_mode baseline
fi

if mode_enabled mpr_enable_only; then
  run_mode mpr_enable_only \
    VLLM_MPR_ENABLE=1 \
    VLLM_MPR_OBSERVE_BACKEND="$OBSERVE_BACKEND" \
    VLLM_MPR_CPU_BACKUP=0 \
    VLLM_MPR_SCORING_ENABLE=0 \
    VLLM_MPR_RECOVERY_ENABLE=0
fi

if mode_enabled backup_only; then
  run_mode backup_only \
    VLLM_MPR_ENABLE=1 \
    VLLM_MPR_OBSERVE_BACKEND="$OBSERVE_BACKEND" \
    VLLM_MPR_CPU_BACKUP=1 \
    VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only \
    VLLM_MPR_SCORING_ENABLE=0 \
    VLLM_MPR_RECOVERY_ENABLE=0
fi

if mode_enabled scoring_only; then
  run_mode scoring_only \
    VLLM_MPR_ENABLE=1 \
    VLLM_MPR_OBSERVE_BACKEND="$OBSERVE_BACKEND" \
    VLLM_MPR_CPU_BACKUP=0 \
    VLLM_MPR_SCORING_ENABLE=1 \
    VLLM_MPR_RECOVERY_ENABLE=0
fi

if mode_enabled fp16_recovery; then
  run_mode fp16_recovery \
    VLLM_MPR_ENABLE=1 \
    VLLM_MPR_OBSERVE_BACKEND="$OBSERVE_BACKEND" \
    VLLM_MPR_CPU_BACKUP=1 \
    VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only \
    VLLM_MPR_SCORING_ENABLE=1 \
    VLLM_MPR_RECOVERY_ENABLE=1 \
    VLLM_MPR_PRECISION_TIERING_ENABLE=0 \
    VLLM_MPR_RECOVERY_POLICY=threshold_block \
    VLLM_MPR_RECOVERY_THRESHOLD=-1e30 \
    VLLM_MPR_RECOVERY_TOPK=1 \
    VLLM_MPR_RECOVERY_TEST_MUTATE=off
fi

if mode_enabled mixed_int8; then
  run_mode mixed_int8 \
    VLLM_MPR_ENABLE=1 \
    VLLM_MPR_OBSERVE_BACKEND="$OBSERVE_BACKEND" \
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
fi

if mode_enabled mixed_int4; then
  run_mode mixed_int4 \
    VLLM_MPR_ENABLE=1 \
    VLLM_MPR_OBSERVE_BACKEND="$OBSERVE_BACKEND" \
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
fi

write_runtime_summary

echo "runtime logs, CSVs, and manifest: $WORK_DIR"
