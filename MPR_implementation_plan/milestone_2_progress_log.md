# Milestone 2 Progress Log

## 2026-06-02: CPU Backup Implementation Direction

Working decisions from the code analysis and user discussion:

```text
backup layout: semantic CPU fp16 K/V tensor
copy timing: synchronous copy first, hidden behind backup-store interface
capacity: unbounded store for first smoke, no eviction
lifecycle cleanup: GPU KV physical block ids drive release
backup key: layer_name + GPU physical block id, wrapped by an internal key type
validation scope: single request / no preemption
```

Implementation choice:

```text
Use a direct KVCacheManager.free -> MPR sidecar cleanup hook for M2.
```

This is intentionally a dirty-but-isolated prototype hook. It is not claimed to
be the cleanest production integration. The hook should be easy to find and
replace with a connector, event, or lifecycle observer path later.

Baseline requirement:

```text
When VLLM_MPR_ENABLE is unset/false, the KVCacheManager helper returns before
querying block ids or importing mixed_precision_recovery.
```

Follow-up optimization and cleanup items are tracked in:

```text
MPR_implementation_plan/mpr_followup_backlog.md
```

## 2026-06-02: M2 CPU Backup Prototype Implemented

Implemented pieces:

```text
vllm/v1/mixed_precision_recovery/cpu_backup.py
  CPUBackupKey
  SemanticCPUBackupStore
  put/get/release/stats result types

vllm/v1/mixed_precision_recovery/config.py
  VLLM_MPR_CPU_BACKUP flag, default false

vllm/v1/mixed_precision_recovery/sidecar.py
  full-block digest creation now optionally creates semantic CPU fp16 backup
  release_blocks(...) cleans offsets, digests, CPU backups, and invalidates
  append-only Quest metadata stores when released blocks were present

vllm/v1/core/kv_cache_manager.py
  isolated _maybe_release_mpr_blocks(...) helper
  called before coordinator.free(request.request_id)
```

Important caveat:

```text
The KVCacheManager direct hook is intentionally dirty but isolated.
When VLLM_MPR_ENABLE is unset/false, the helper returns before block-id lookup
or mixed_precision_recovery import, so vanilla vLLM should not observe sidecar
initialization or CPU backup behavior.
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/cpu_backup.py \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/v1/core/kv_cache_manager.py \
  tests/v1/mixed_precision_recovery/test_scoring.py

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py

result:
  28 passed, 1 skipped

git diff --check
  passed
```

Observation from step-latency CSV comparison:

```text
MPR write observe only vs vanilla:
  mean delta ~= +13.5 ms/step

CPU backup on vs MPR write observe only:
  mean delta ~= +0.5 ms/step
  boundary-ish steps ~= +5 ms

CUDA scoring + backup vs CUDA scoring only:
  mean delta was within run variance
  boundary-ish steps showed positive backup-related tail deltas
```

Interpretation:

```text
CPU backup copy itself mostly appears as block-boundary/tail latency.
The much larger all-step latency increase comes from MPR observe_kv_write
bookkeeping, likely including CUDA sync/D2H scalar-list conversions such as
item(), unique().cpu().tolist(), and block offset cpu().tolist().
```

Decision:

```text
Do not optimize this before Milestone 3. First complete fp16 recovery plumbing,
then optimize the MPR write-observation path and CPU backup storage/copy path.
```

Note:

```text
Running pytest with base conda Python 3.12 failed before tests loaded because
that environment has a NumPy 2.x / SciPy-sklearn ABI mismatch. The vLLM env
Python 3.10.20 test run passed.
```

## 2026-06-02: CPU Backup Overhead Benchmark Script

Added:

```text
scripts/mpr_benchmark_cpu_backup.py
```

Purpose:

```text
measure synchronous semantic CPU backup overhead separately from e2e inference
compare raw GPU->CPU fp16 copy wall time against SemanticCPUBackupStore.put(...)
measure release_all wall time for the current dict-backed prototype store
```

Columns:

```text
raw_d2h_copy_wall_ms
store_put_wall_ms
store_put_reported_copy_wall_ms
store_put_overhead_wall_ms
release_all_wall_ms
release_entries
release_bytes
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  scripts/mpr_benchmark_cpu_backup.py

git diff --check
  passed
```

Local note:

```text
The benchmark requires CUDA. The current sandbox did not expose an active CUDA
driver, so runtime benchmark numbers were not collected here.
```

## 2026-06-02: Inference Step Latency Benchmark Script

Added:

```text
scripts/mpr_benchmark_step_latency.py
```

Purpose:

```text
measure per-LLMEngine.step wall latency during actual vLLM offline generation
compare vanilla / score-only / score+CPU-backup runs with the same prompt
surface block-boundary latency spikes from synchronous CPU backup
optionally write per-step rows to CSV
```

Notes:

```text
LLM.generate() hides the engine loop, so the script uses LLM.enqueue() and
LLM.wait_for_completion() after wrapping llm.llm_engine.step with a timer.
The script reports predicted block-boundary step indices from prompt length,
generated token count, and block size for single-request/no-chunked-prefill
experiments.
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  scripts/mpr_benchmark_step_latency.py

git diff --check
  passed
```

Update:

```text
Added VLLM_MPR_SCORING_ENABLE, default true.
Use VLLM_MPR_ENABLE=1, VLLM_MPR_CPU_BACKUP=1, and
VLLM_MPR_SCORING_ENABLE=0 to measure step latency with digest/CPU backup copy
enabled but query scoring disabled.
```

## 2026-06-02: Pure CPU Copy Benchmark Script

Added:

```text
scripts/mpr_benchmark_cpu_copy.py
```

Purpose:

```text
measure pure GPU-to-CPU semantic KV block copy latency without backup-store
bookkeeping
compare allocating to(cpu), preallocated pageable CPU copy, and pinned CPU copy
variants
```

Columns:

```text
alloc_to_cpu_wall_ms
pageable_copy_wall_ms
pinned_copy_sync_wall_ms
pinned_copy_nonblocking_wall_ms
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  scripts/mpr_benchmark_cpu_copy.py

git diff --check
  passed
```
