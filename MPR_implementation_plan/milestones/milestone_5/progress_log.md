# Milestone 5 Progress Log

## 2026-06-09: Step 5.1 Measurement Baseline Command Set

Step 5.1 starts with measurement only. Do not change recovery semantics or
cleanup code before collecting this baseline.

Decision:

```text
Run Step 5.1 first, then proceed step-by-step through Milestone 5.
Use the M4.5 workload shape.
Measure seven runtime modes, including MPR enable-only observe overhead.
Use one warmup run and five measured runs per mode.
Increase warmup only for modes whose p95/max values are large outliers.
```

Common runtime benchmark arguments:

```bash
COMMON_ARGS=(
  --model Qwen/Qwen3-8B
  --dtype half
  --max-model-len 2048
  --max-tokens 512
  --block-size 32
  --gpu-memory-utilization 0.75
  --tensor-parallel-size 1
  --seed 0
  --ignore-eos
)
```

Runtime mode matrix:

```text
baseline:
  unset all VLLM_MPR_*

mpr_enable_only:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_CPU_BACKUP=0
  VLLM_MPR_SCORING_ENABLE=0
  VLLM_MPR_RECOVERY_ENABLE=0
  VLLM_MPR_DEBUG_DIR unset

backup_only:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_CPU_BACKUP=1
  VLLM_MPR_SCORING_ENABLE=0
  VLLM_MPR_RECOVERY_ENABLE=0
  VLLM_MPR_DEBUG_DIR unset

scoring_only:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_CPU_BACKUP=0
  VLLM_MPR_SCORING_ENABLE=1
  VLLM_MPR_RECOVERY_ENABLE=0
  VLLM_MPR_DEBUG_DIR unset

fp16_recovery:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_CPU_BACKUP=1
  VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only
  VLLM_MPR_SCORING_ENABLE=1
  VLLM_MPR_RECOVERY_ENABLE=1
  VLLM_MPR_PRECISION_TIERING_ENABLE=0
  VLLM_MPR_RECOVERY_POLICY=threshold_block
  VLLM_MPR_RECOVERY_THRESHOLD=-1e30
  VLLM_MPR_RECOVERY_TOPK=1
  VLLM_MPR_RECOVERY_TEST_MUTATE=off

mixed_int8:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_CPU_BACKUP=1
  VLLM_MPR_BACKUP_STORAGE_MODE=eager_fp16_int8
  VLLM_MPR_SCORING_ENABLE=1
  VLLM_MPR_RECOVERY_ENABLE=1
  VLLM_MPR_PRECISION_TIERING_ENABLE=1
  VLLM_MPR_PRECISION_POLICY=top_ratio
  VLLM_MPR_TIER_FP16_RATIO=0.25
  VLLM_MPR_TIER_INT8_RATIO=0.75
  VLLM_MPR_TIER_INT4_RATIO=0.0
  VLLM_MPR_RECOVERY_POLICY=threshold_block
  VLLM_MPR_RECOVERY_THRESHOLD=-1e30
  VLLM_MPR_RECOVERY_TOPK=1
  VLLM_MPR_RECOVERY_TEST_MUTATE=off

mixed_int4:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_CPU_BACKUP=1
  VLLM_MPR_BACKUP_STORAGE_MODE=eager_fp16_int8_int4
  VLLM_MPR_SCORING_ENABLE=1
  VLLM_MPR_RECOVERY_ENABLE=1
  VLLM_MPR_PRECISION_TIERING_ENABLE=1
  VLLM_MPR_PRECISION_POLICY=top_ratio
  VLLM_MPR_TIER_FP16_RATIO=0.25
  VLLM_MPR_TIER_INT8_RATIO=0.25
  VLLM_MPR_TIER_INT4_RATIO=0.50
  VLLM_MPR_RECOVERY_POLICY=threshold_block
  VLLM_MPR_RECOVERY_THRESHOLD=-1e30
  VLLM_MPR_RECOVERY_TOPK=1
  VLLM_MPR_RECOVERY_TEST_MUTATE=off
```

Repeat policy:

```text
For each runtime mode:
  run 1 warmup generation
  run 5 measured generations
  exclude the warmup run from summary tables

If p95 > 1.5x median or max > 2.0x median:
  rerun only that mode with 3 warmup generations and 5 measured generations
  if the spread remains, keep the result and mark it as runtime variance
```

Record per mode:

```text
generate_elapsed_sec
generated_tokens_per_sec
engine step latency mean, median, p95, p99, max
predicted boundary step latency mean, max
non-boundary step latency mean, max
CSV path
warmup count and measured run count
outlier rerun status
```

Runtime command:

```bash
bash scripts/mpr_run_m5_step51_latency.sh \
  --work-dir /tmp/mpr_m5_step51_$(date +%Y%m%d_%H%M%S)
```

The script records `manifest.txt`, per-mode warmup/measured `.log` and `.csv`
files, `runtime_summary.csv`, and `outlier_check.txt`. It clears inherited
`VLLM_MPR_*` environment for every run, keeps `VLLM_MPR_DEBUG_DIR` unset for
latency measurements, and injects only the mode environment being measured.

Semantic smoke commands after latency collection:

```bash
WORK_DIR=/tmp/mpr_m5_step51_tiered_$(date +%Y%m%d_%H%M%S)
python scripts/mpr_smoke_tiered_recovery.py \
  --work-dir "$WORK_DIR" \
  --tier-ratios \
  1.00:0.00:0.00,0.50:0.25:0.25,0.25:0.25:0.50,0.00:0.50:0.50,0.00:0.00:1.00 \
  --summary-json "$WORK_DIR/summary.json" \
  --text-preview-chars 500

WORK_DIR=/tmp/mpr_m5_step51_degraded_$(date +%Y%m%d_%H%M%S)
python scripts/mpr_smoke_tiered_degraded_residency.py \
  --work-dir "$WORK_DIR" \
  --tier-ratios \
  0.25:0.25:0.25,0.25:0.25:0.10,0.25:0.10:0.25,0.10:0.25:0.25 \
  --summary-json "$WORK_DIR/summary.json" \
  --text-preview-chars 500
```

Synthetic microbenchmark commands:

```bash
python scripts/mpr_benchmark_cpu_copy.py
python scripts/mpr_benchmark_cpu_backup.py
python scripts/mpr_benchmark_scoring.py
```

Status:

```text
Step 5.1 command set recorded.
Runtime command script added at scripts/mpr_run_m5_step51_latency.sh.
Runtime measurements not yet recorded.
Semantic smoke measurements not yet recorded.
Synthetic microbenchmark measurements not yet recorded.
```
