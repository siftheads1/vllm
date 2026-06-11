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
  VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only
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
all engine step latency mean, median, p95, p99, max
prefill step latency mean, max
decode step latency mean, median, p95, p99, max
predicted boundary decode step latency mean, max
non-boundary decode step latency mean, max
CSV path
warmup count and measured run count
outlier rerun status based on decode p95/max
```

Runtime command:

```bash
bash scripts/mpr_run_m5_step51_latency.sh \
  --work-dir /tmp/mpr_m5_step51_$(date +%Y%m%d_%H%M%S)
```

The script records `manifest.txt`, per-mode warmup/measured `.log` and `.csv`
files, `runtime_summary.csv`, and `outlier_check.txt`. The benchmark CSV marks
`step_idx=0` as `prefill` and later steps as `decode`; the primary M5 latency
comparison uses decode-only summary fields. The runner clears inherited
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
First runtime summary recorded below.
Semantic smoke measurements not yet recorded.
Synthetic microbenchmark measurements not yet recorded.
```

## 2026-06-10: Step 5.1 First Runtime Summary and Bottleneck Targets

Received first runtime summary from the GPU runtime environment.

Detailed report:

```text
MPR_implementation_plan/milestones/milestone_5/reports/step_5_1_runtime_bottleneck_report.md
```

Runtime summary:

```text
mode             decode mean   median    p95      max       tok/s   mean vs baseline
baseline          25.00 ms    22.02    37.14     52.29    39.46       1.00x
mpr_enable_only   43.06 ms    40.86    54.66     66.24    23.03       1.72x
backup_only       42.77 ms    40.46    54.85     78.82    23.18       1.71x
scoring_only      90.53 ms    95.10   103.29    181.33    11.01       3.62x
fp16_recovery    126.45 ms   132.28   168.42    206.03     7.89       5.06x
mixed_int8       204.79 ms   153.45   865.31   1757.53     4.88       8.19x
mixed_int4       296.15 ms   202.45  1398.78   3348.17     3.38      11.85x
```

Primary bottleneck targets recorded in the report:

```text
1. MPR observe/digest base overhead
2. Scoring path
3. FP16 recovery materialization
4. Mixed INT8/INT4 tail latency
```

Target 1 code map:

```text
MPR_implementation_plan/milestones/milestone_5/reports/step_5_1_target_1_observe_digest_code_map.md
```

## 2026-06-10: Step 5.1 Target 1 Observe Hook Attribution Probe

Before changing observe/digest behavior, added temporary timing attribution for
the `unified_kv_cache_update -> _maybe_observe_mpr_kv_write` hook.

Path check:

```text
Decode-only exception probes confirmed both baseline and MPR-enable runs enter
vllm/model_executor/layers/attention/attention.py::unified_kv_cache_update.
The alternate-attention-backend/fused-path explanation for the missing timing
counter was rejected for the Qwen3-8B benchmark path used here.
```

Timing collection note:

```text
Default V1 multiprocessing does not expose the worker-local module counter to
the benchmark driver process. In that mode, the runtime summary can still show
zero mpr_observe_hook_count even though the hook path is executed.

For this attribution probe, rerun baseline and mpr_enable_only with:
  VLLM_ENABLE_V1_MULTIPROCESSING=0
so the benchmark process can read the hook counter directly.
```

Command shape:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
bash scripts/mpr_run_m5_step51_latency.sh \
  --python /home/han/anaconda3/envs/20260528_vllm/bin/python \
  --work-dir /tmp/mpr_observe_inproc_$(date +%Y%m%d_%H%M%S) \
  --modes baseline,mpr_enable_only \
  --warmup-runs 1 \
  --measured-runs 3 \
  --max-tokens 128
```

Observed conclusion:

```text
The in-process attribution run confirmed that the MPR enable-only overhead is
dominated by the observe KV write hook. This supports making
RecoverySidecar.observe_kv_write the first optimization target before changing
scoring, recovery materialization, or precision policy.
```

Non-in-process reference artifact:

```text
/tmp/mpr_observe_enable_only_20260610_151730/runtime_summary.csv

baseline decode mean:        17.28 ms
mpr_enable_only decode mean: 31.10 ms
increment:                  +13.82 ms

The same artifact has mpr_observe_hook_count=0 because the run used default V1
multiprocessing and did not recover worker-local timing counters.
```

## 2026-06-10: Step 5.2 Observe Logging Gate

Implemented the first observe hot-path cleanup by separating debug logging from
required KV-write observation.

Change:

```text
Added VLLM_MPR_ENABLE_LOGGING.
JSONL debug writing is active only when:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_ENABLE_LOGGING=1
  VLLM_MPR_DEBUG_DIR is set

RecoverySidecar.observe_kv_write no longer computes debug-only fields when
logging is disabled:
  per-layer dump gating
  unique block id list for JSONL
  min/max block offset for JSONL
  shape fields and digest-count fields for observe_kv_write JSONL
```

Preserved behavior:

```text
block offset observation still runs
full-block digest creation still runs
CPU backup creation still runs when enabled
event counters are still updated by _record for existing unit-test/stat users
debug/smoke scripts that require JSONL now set VLLM_MPR_ENABLE_LOGGING=1
```

Validation:

```bash
python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/debug.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/envs.py \
  scripts/mpr_smoke_tiered_recovery.py \
  scripts/mpr_smoke_tiered_degraded_residency.py \
  scripts/mpr_smoke_recovery_quality.py

bash -n scripts/mpr_run_m5_step51_latency.sh

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q
```

Result:

```text
test_digest.py + test_scoring.py: 52 passed, 2 skipped
test_recovery.py + test_debug_jsonl_validator.py: 60 passed
```

## 2026-06-10: Step 5.2 Observe KV Write Split Timing

Added temporary, lightweight timing attribution inside
`RecoverySidecar.observe_kv_write` to split the observe hot path into:

```text
pre_observe_block_offsets:
  slot_mapping flatten
  invalid negative slot check
  PAD filtering
  block id / block offset derivation

observe_block_offsets:
  _observe_block_offsets(...) call itself
```

The benchmark script now prints both totals and per-decode-step summaries, and
the step-51 runner carries those fields into `runtime_summary.csv`.

Validation:

```bash
python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/v1/mixed_precision_recovery/__init__.py \
  scripts/mpr_benchmark_step_latency.py

bash -n scripts/mpr_run_m5_step51_latency.sh

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q
```

Result:

```text
py_compile: passed
bash -n: passed
test_digest.py + test_scoring.py: 52 passed, 2 skipped
test_recovery.py + test_debug_jsonl_validator.py: 60 passed
```

## 2026-06-10: Step 5.2 Counter-Based KV Observe Backend

Added an experimental `VLLM_MPR_OBSERVE_BACKEND=counter` path for
`mpr_enable_only`-style overhead isolation. The default remains
`VLLM_MPR_OBSERVE_BACKEND=slot`.

Initial implementation notes:

```text
counter backend:
  enabled only for pure decode batches
  uses query_start_loc_cpu to reject prefill/mixed continuous batching
  uses CPU seq lengths plus block table to detect block boundaries
  creates a digest when post-update seq_len % block_size == 1
  falls back to slot backend at the attention hook when unsupported

shared digest helper:
  slot and counter paths now both use _create_digest_for_full_block
  digest cache, Quest metadata, CPU backup, and digest_created logging stay shared
```

Benchmark runner update:

```bash
bash scripts/mpr_run_m5_step51_latency.sh \
  --modes baseline,mpr_enable_only \
  --observe-backend counter
```

The runner now defaults output directories under:

```text
/home/han/KV_cache_quant/proposed_method_develop/results
```

Validation:

```bash
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/model_executor/layers/attention/attention.py \
  scripts/mpr_benchmark_step_latency.py

bash -n scripts/mpr_run_m5_step51_latency.sh

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

git diff --check
```

Result:

```text
py_compile: passed
bash -n: passed
test_digest.py: 7 passed
test_scoring.py: 48 passed, 2 skipped
test_recovery.py + test_debug_jsonl_validator.py: 60 passed
git diff --check: passed
```

## 2026-06-10: Step 5.2 Counter Backend Runtime Result and Boundary Distribution

Latest artifact:

```text
/home/han/KV_cache_quant/proposed_method_develop/results/mpr_m5_step51_20260610_191946/runtime_summary.csv
```

The sidecar-owned counter backend successfully took the decode path:

```text
mpr_counter_observe_candidate_count_mean: 4680
mpr_counter_observe_accepted_mean: 4572
mpr_counter_observe_used_mean: 4572
mpr_counter_observe_prefill_initialized_mean: 1
mpr_counter_observe_decode_advanced_mean: 127
mpr_counter_observe_boundary_requests_mean: 288
mpr_counter_observe_digest_created_mean: 288
```

Observed effect:

```text
baseline decode mean:                  17.07 ms
counter mpr_enable_only decode mean:   17.97 ms
remaining decode overhead:             ~0.90 ms/step

counter hook total per decode step:     ~1.49 ms
slot pre-block-offset path per step:    ~0.06 ms
slot _observe_block_offsets per step:   ~0.01 ms
```

Compared with the slot fallback run, the hot slot scan overhead was removed:

```text
slot fallback hook per decode step:     ~9.51 ms
counter hook per decode step:           ~1.49 ms
```

Boundary decode steps still show a latency spike, consistent with digest
creation being concentrated on block boundaries:

```text
boundary decode mean:       ~31.07 ms
non-boundary decode mean:   ~17.55 ms
```

Before profiling/optimizing boundary digest creation, the benchmark output was
extended to report distribution statistics for boundary and non-boundary decode
steps, matching the regular decode latency reporting style:

```text
predicted_boundary_decode_step_latency_ms:
  mean
  median
  p90
  p95
  p99
  max

non_boundary_decode_step_latency_ms:
  mean
  median
  p90
  p95
  p99
  max
```

The step-51 runner now carries these fields into `runtime_summary.csv`:

```text
boundary_decode_step_latency_ms_median
boundary_decode_step_latency_ms_p90
boundary_decode_step_latency_ms_p95
boundary_decode_step_latency_ms_p99
non_boundary_decode_step_latency_ms_median
non_boundary_decode_step_latency_ms_p90
non_boundary_decode_step_latency_ms_p95
non_boundary_decode_step_latency_ms_p99
```

Validation:

```bash
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  scripts/mpr_benchmark_step_latency.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/model_executor/layers/attention/attention.py

bash -n scripts/mpr_run_m5_step51_latency.sh

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

git diff --check
```

Result:

```text
py_compile: passed
bash -n: passed
test_digest.py: 7 passed
test_scoring.py: 48 passed, 2 skipped
test_recovery.py + test_debug_jsonl_validator.py: 60 passed
git diff --check: passed
```

## 2026-06-10: Step 5.2 Remove Query-Start CPU Dependency

A follow-up `--observe-backend counter` run showed the shared counter path was
still falling back because per-layer FlashAttention metadata does not expose
`query_start_loc_cpu`:

```text
mpr_counter_observe_candidate_count_mean: 4680
mpr_counter_observe_accepted_mean: 0
mpr_counter_observe_missing_query_start_loc_cpu_mean: 4680
```

The counter backend no longer reads `query_start_loc_cpu`. It now derives the
single-request query length from per-layer metadata fields that FlashAttention
does expose:

```text
single request if num_actual_tokens == max_query_len
query_len = num_actual_tokens
multi-request / mixed batches fall back to slot backend
```

The benchmark diagnostics were updated accordingly:

```text
removed:
  mpr_counter_observe_missing_query_start_loc_cpu
  mpr_counter_observe_bad_query_start_loc_cpu

added:
  mpr_counter_observe_bad_query_len
```

Validation:

```bash
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  scripts/mpr_benchmark_step_latency.py

bash -n scripts/mpr_run_m5_step51_latency.sh

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

git diff --check
```

Result:

```text
py_compile: passed
bash -n: passed
test_digest.py: 7 passed
test_scoring.py: 48 passed, 2 skipped
test_recovery.py + test_debug_jsonl_validator.py: 60 passed
git diff --check: passed
```

## 2026-06-10: Step 5.2 Counter Fallback Diagnostics

The first `--observe-backend counter` run still exercised the slot path:

```text
observe_backend: counter
mpr_observe_block_offsets_count: 4608
mpr_observe_pre_block_offsets_total_per_decode_step_ms: ~7.49
mpr_observe_block_offsets_total_per_decode_step_ms: ~1.63
```

Added lightweight diagnostic counters to distinguish counter acceptance from
fallback reasons:

```text
mpr_counter_observe_candidate_count
mpr_counter_observe_accepted
mpr_counter_observe_used
mpr_counter_observe_missing_block_table
mpr_counter_observe_missing_query_start_loc_cpu
mpr_counter_observe_not_pure_decode
```

The benchmark script prints these counters, and the step-51 runner carries their
per-run mean into `runtime_summary.csv`.

Validation:

```bash
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  scripts/mpr_benchmark_step_latency.py

bash -n scripts/mpr_run_m5_step51_latency.sh

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py -q

git diff --check
```

Result:

```text
py_compile: passed
bash -n: passed
test_digest.py: 7 passed
git diff --check: passed
```

## 2026-06-10: Step 5.2 Shared Sequence Counter Backend

The counter fallback diagnostics showed that all counter candidates were
rejected because the per-layer FlashAttention metadata did not expose CPU
sequence lengths:

```text
mpr_counter_observe_candidate_count_mean: 4680
mpr_counter_observe_accepted_mean: 0
mpr_counter_observe_missing_seq_lens_cpu_mean: 4680
```

Rather than copying GPU `seq_lens` to CPU, the counter backend now maintains a
sidecar-owned shared sequence counter:

```text
leader layer:
  selected from the first observed layer
  initializes seq_len from single-request prefill query_len
  advances seq_len by 1 on pure decode

other layers:
  do not advance seq_len
  consume the leader's latest seq_len snapshot

all layers:
  create their own layer-local digest when seq_len % block_size == 0
```

The obsolete CPU-seq-lens probe path (`can_observe_kv_write_by_counter` and
`_counter_seq_lens_cpu`) was removed after switching to the sidecar-owned shared
counter. The counter backend now reads attention metadata fields
`num_actual_tokens` and `max_query_len` for the single-request query-length
check, plus the request block table for boundary block-id lookup.

The implementation remains intentionally narrow for the first overhead test:
only single-request rows use the counter backend; multi-request/mixed cases fall
back to the slot backend and reset the counter initialization state.

Validation:

```bash
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/model_executor/layers/attention/attention.py \
  scripts/mpr_benchmark_step_latency.py

bash -n scripts/mpr_run_m5_step51_latency.sh

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

git diff --check
```

Result:

```text
py_compile: passed
bash -n: passed
test_digest.py: 7 passed
test_scoring.py: 48 passed, 2 skipped
test_recovery.py + test_debug_jsonl_validator.py: 60 passed
git diff --check: passed
```

## 2026-06-10: Current Target 1 Status and Next Boundary Profiling Gate

Current working conclusion:

```text
The counter observe backend resolves the MPR enable-only steady-state overhead
for the single-request pure-decode non-boundary path. In that path,
non-boundary decode latency is now close to baseline, so the always-paid slot
scan / CPU-sync observe cost is no longer the primary Target 1 bottleneck.
```

Scope of this conclusion:

```text
resolved:
  MPR enable-only non-boundary observe bookkeeping overhead on the counter path

still in scope:
  block-boundary decode spikes caused by digest creation
  any fallback from counter observe to the slot backend
  multi-request / mixed-batch / broader serving cases not covered by the narrow
    single-request pure-decode counter path
```

Target 1 should therefore be treated as narrowed rather than fully closed:

```text
MPR enable-only steady-state overhead: resolved for the current counter-backend
  benchmark shape
Remaining Target 1 cost: boundary-step digest creation and fallback coverage
```

Next profiling gate:

```text
Profile block-boundary decode steps under --observe-backend counter.

Primary questions:
  1. How much of the boundary spike is digest creation itself?
  2. How much is block-table lookup / counter bookkeeping?
  3. How much is optional Quest metadata append or backup hook early-return work?
  4. Are boundary spikes proportional to number of layers / created digests?
  5. Does any unexpected slot fallback remain in the measurement run?
```

Suggested boundary profiling probes:

```text
observe_kv_write_by_counter total
prepare_counter_kv_write total
counter block-table lookup / block-id extraction
_create_digest_for_full_block total
_key_cache_for_digest total
summarize_key_block total
_to_block_digest total
_append_quest_metadata_digest total
_maybe_backup_kv_block total / early return
counter debug record path, only when logging is enabled
```

Instrumentation rule:

```text
Keep boundary profiling explicitly gated so the measurement machinery does not
become another always-on hot-path cost. Do not change precision policy,
recovery policy, or digest semantics while profiling this boundary cost.
```

## 2026-06-10: Boundary Profiling Instrumentation

Added targeted boundary-step profiling behind:

```text
VLLM_MPR_BOUNDARY_PROFILE=1
```

Runner support:

```bash
bash scripts/mpr_run_m5_step51_latency.sh \
  --modes baseline,mpr_enable_only \
  --observe-backend counter \
  --boundary-profile
```

The profiling flag is disabled by default. When enabled, the benchmark prints
and the step-51 runner summarizes:

```text
mpr_boundary_profile_counter_prepare_*
mpr_boundary_profile_counter_observe_*
mpr_boundary_profile_counter_key_cache_*
mpr_boundary_profile_counter_block_lookup_*
mpr_boundary_profile_counter_create_digest_*
mpr_boundary_profile_counter_summarize_key_block_*
mpr_boundary_profile_counter_to_block_digest_*
mpr_boundary_profile_counter_append_quest_metadata_*
mpr_boundary_profile_counter_backup_*
```

Each probe reports:

```text
count
total_ms
mean_ms
max_ms
```

Validation in the local Windows workspace:

```bash
python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/envs.py \
  scripts/mpr_benchmark_step_latency.py

git -c safe.directory=C:/Lab/mixed_precision_recovery_vllm/vllm diff --check
```

Result:

```text
py_compile: passed
git diff --check: passed
```

## 2026-06-11: Request Block Context Profiling Breakdown

Added temporary `_request_block_context()` breakdown timers under the existing
scoring profiling gate:

```text
VLLM_MPR_SCORING_PROFILE=1
```

New output fields:

```text
mpr_scoring_profile_scoring_request_ctx_seq_lens_*
mpr_scoring_profile_scoring_request_ctx_block_table_lookup_*
mpr_scoring_profile_scoring_request_ctx_block_table_row_*
mpr_scoring_profile_scoring_request_ctx_candidates_*
```

Use these alongside the existing inclusive
`mpr_scoring_profile_scoring_request_block_context_*` fields to distinguish
GPU/CPU metadata materialization from Python candidate-list construction before
applying and later removing the optimization scaffolding.

Follow-up fix: added the same keys to the benchmark stdout printer and the
step-51 runtime summary whitelist, so they appear both in `*_measured_*.log`
and `runtime_summary.csv`.

## 2026-06-11: Request Block Context Step Cache

Remote scoring profile showed `_request_block_context()` at roughly 3 ms per
decode step, dominated by repeated `seq_lens` and `block_table[0]` CPU
materialization across layers.

Added a conservative per-step cache for `_request_block_context()` results.
The cache key includes the per-layer event index used as a decode-step stamp,
block size, recent-token policy, scalar metadata, and tensor identity metadata
for `seq_lens` and the block table. This targets the common single-KV-group
case where all layers in a decode step share request/block metadata while still
missing naturally when KV cache groups or shapes differ.

Expected validation signal:

```text
scoring_request_block_context_total_ms should drop substantially.
scoring_request_ctx_seq_lens_count and scoring_request_ctx_block_table_row_count
should drop from layer-count scale to step/group-count scale.
```

## 2026-06-12: Default Scoring Backend Switched to Quest CUDA

Changed the default `VLLM_MPR_SCORING_BACKEND` from `torch_quest` to
`quest_cuda` in both `MPRConfig` and the vLLM environment registry. The
PyTorch scorer remains available by explicitly setting:

```bash
VLLM_MPR_SCORING_BACKEND=torch_quest
```

Next runtime profile should confirm:

```text
mpr_scoring_backend: quest_cuda
MPR sidecar enabled: ... scoring_backend=quest_cuda ...
scoring_quest_packed_estimate_count > 0 when the persistent-prefix path matches
```

## 2026-06-11: Gate Scoring Debug Work on Logging

Remote scoring-profile results indicated that `scoring_estimate_query_scores`
was dominated by debug/top-k work, especially `scoring_head_topk_debug`, while
the actual scorer/kernel cost was not the obvious end-to-end latency driver.

Change:

```text
Gate score debug field construction on:
  self.config.enable_logging and should_record

When logging is disabled, _estimate_query_scores now skips:
  _score_packing_debug_fields
  _topk_block_scores
  _head_score_debug_fields
  _score_block_debug_fields

_record_score_estimated also avoids JSONL field construction when logging is
disabled, while preserving the score_estimated event counter.
```

Preserved behavior:

```text
score_result.block_scores and physical_block_ids remain available for recovery
and precision tier assignment.

When VLLM_MPR_ENABLE_LOGGING=1 and the layer/step passes dump limits, the same
debug fields are still generated for JSONL validation.
```

Not run in this local workspace:

```text
bash -n scripts/mpr_run_m5_step51_latency.sh:
  blocked because bash.exe failed to launch in this Windows session

pytest:
  blocked because the local Python environment has no pytest module

runtime import smoke:
  blocked because the local Python environment has no torch module
```

## 2026-06-11: Scoring Path Profiling Instrumentation

Added targeted scoring-path wall-time profiling behind:

```text
VLLM_MPR_SCORING_PROFILE=1
```

Runner support:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
bash scripts/mpr_run_m5_step51_latency.sh \
  --modes baseline,scoring_only \
  --observe-backend counter \
  --scoring-profile
```

The profiling flag is disabled by default. When enabled, the benchmark prints
and the step-51 runner summarizes:

```text
mpr_scoring_profile_scoring_estimate_query_scores_*
mpr_scoring_profile_scoring_record_estimated_*
mpr_scoring_profile_scoring_should_record_*
mpr_scoring_profile_scoring_query_clone_*
mpr_scoring_profile_scoring_window_stack_mean_*
mpr_scoring_profile_scoring_block_size_*
mpr_scoring_profile_scoring_request_block_context_*
mpr_scoring_profile_scoring_select_digest_blocks_*
mpr_scoring_profile_scoring_quest_packed_prefix_*
mpr_scoring_profile_scoring_quest_packed_estimate_*
mpr_scoring_profile_scoring_pack_layer_digests_*
mpr_scoring_profile_scoring_backend_estimate_*
mpr_scoring_profile_scoring_score_packing_debug_*
mpr_scoring_profile_scoring_block_topk_*
mpr_scoring_profile_scoring_head_topk_debug_*
mpr_scoring_profile_scoring_block_debug_fields_*
mpr_scoring_profile_scoring_context_build_*
```

Primary attribution questions:

```text
Does scoring_only overhead come from the scorer backend itself, or from
query-window maintenance, request/block CPU metadata conversion, digest packing,
or debug top-k CPU conversions?
```

Validation in the local Windows workspace:

```bash
python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/v1/mixed_precision_recovery/__init__.py \
  vllm/envs.py \
  scripts/mpr_benchmark_step_latency.py

git -c safe.directory=C:/Lab/mixed_precision_recovery_vllm/vllm diff --check
```

Result:

```text
py_compile: passed
git diff --check: passed
```
