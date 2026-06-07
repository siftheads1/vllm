# Milestone 4 Progress Log

## 2026-06-06: Step 4.1 Precision Tiering Config

Implemented disabled-by-default config/env flags for M4 precision tiering.

Added MPRConfig fields:

```text
precision_tiering_enabled = false
precision_policy = top_ratio
tier_fp16_ratio = 0.25
tier_int8_ratio = 0.50
tier_high_threshold = 0.0
tier_low_threshold = 0.0
backup_storage_mode = eager_fp16_int8
```

Registered matching vLLM environment variables:

```text
VLLM_MPR_PRECISION_TIERING_ENABLE
VLLM_MPR_PRECISION_POLICY
VLLM_MPR_TIER_FP16_RATIO
VLLM_MPR_TIER_INT8_RATIO
VLLM_MPR_TIER_HIGH_THRESHOLD
VLLM_MPR_TIER_LOW_THRESHOLD
VLLM_MPR_BACKUP_STORAGE_MODE
```

Validation rules:

```text
precision_policy in {top_ratio, threshold}
backup_storage_mode in {eager_fp16_int8, fp16_only}
ratio values in [0, 1]
tier_fp16_ratio + tier_int8_ratio <= 1
tier_high_threshold >= tier_low_threshold
```

No tiering behavior is implemented in this step. With
`VLLM_MPR_PRECISION_TIERING_ENABLE` unset/false, the current M3 fp16 recovery
path remains the active recovery behavior.

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/envs.py \
  tests/v1/mixed_precision_recovery/test_scoring.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  39 passed, 2 skipped

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  68 passed, 2 skipped
```

## 2026-06-06: Step 4.2 Precision Policy Module

Implemented the pure score-to-tier assignment module for M4 precision tiering.

Added:

```text
vllm/v1/mixed_precision_recovery/precision_policy.py
  PrecisionTier
  TierAssignment
  PrecisionPolicy
  TopRatioPrecisionPolicy
  ThresholdPrecisionPolicy
```

Policy behavior:

```text
TopRatioPrecisionPolicy:
  stable descending score order
  equal scores preserve original candidate order
  fp16_count = ceil(num_candidates * fp16_ratio), clamped to candidates
  int8_count = ceil(num_candidates * int8_ratio), clamped to remaining candidates
  remaining candidates are assigned to skip

ThresholdPrecisionPolicy:
  score >= high_threshold -> fp16
  score >= low_threshold -> int8
  otherwise -> skip
  candidate order is preserved within each tier
```

Validation rules:

```text
block_scores must be 1D
physical_block_ids length must match block_scores
physical block ids must be unique
every candidate is assigned exactly once to fp16/int8/skip
```

No backup store, recovery materialization, sidecar integration, or runtime MPR
behavior was changed in this step.

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/precision_policy.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py \
  vllm/v1/mixed_precision_recovery/__init__.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_precision_policy.py -q

result:
  13 passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py -q

result:
  81 passed, 2 skipped
```

## 2026-06-06: Step 4.3 Backup Codec Abstraction

Implemented standalone logical backup payload codecs for M4 precision tiering.

Added:

```text
vllm/v1/mixed_precision_recovery/backup_codec.py
  FP16BackupPayload
  INT8BackupPayload
  BackupCodec
  FP16BackupCodec
  INT8BackupCodec
```

Codec boundary:

```text
BackupCodec encodes a semantic KV block into a precision-specific logical
payload and materializes/dequantizes that payload into a target dtype/device.
It does not define CPU backup store layout or lifecycle.
```

Initial payload formats:

```text
fp16:
  tensor: CPU fp16 [2, block_size, num_kv_heads, head_dim]

int8:
  quantized: CPU int8 [2, block_size, num_kv_heads, head_dim]
  scale: CPU fp32 [2, block_size, num_kv_heads]
  scale_granularity: per_token_per_kv_head
```

INT8 reference quantization:

```text
scale = max(abs(vector)) / 127
quantized = clamp(round(vector / scale), -127, 127).to(int8)
zero-vector scale = 1.0, quantized = 0
```

No CPU backup store, recovery materialization, sidecar integration, or runtime
MPR behavior was changed in this step.

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/backup_codec.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  vllm/v1/mixed_precision_recovery/__init__.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_backup_codec.py -q

result:
  7 passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py -q

result:
  88 passed, 2 skipped
```

## 2026-06-06: Step 4.4 Payload-Aware CPU Backup Store

Implemented payload-aware CPU backup storage for M4 precision tiering while
preserving the M3 fp16 recovery contract.

Updated:

```text
vllm/v1/mixed_precision_recovery/cpu_backup.py
  SemanticCPUBackupStore now stores one payload entry per CPUBackupKey
  get(key) still returns the fp16 CPU tensor for M3 recovery
  get_payload(key, backup_format) returns fp16 or int8 logical payloads
  put(..., backup_storage_mode=...) supports fp16_only and eager_fp16_int8
  put/release/stats report fp16, int8, scale, and total actual byte counts

vllm/v1/mixed_precision_recovery/config.py
  rejects cpu_backup_enabled=true + precision_tiering_enabled=false +
  backup_storage_mode=eager_fp16_int8

vllm/v1/mixed_precision_recovery/sidecar.py
  passes config.backup_storage_mode into CPU backup store puts
  emits payload byte accounting on cpu_backup_created, blocks_released,
  recovery_materialized, and recovery_test_mutated debug events

scripts/mpr_validate_debug_jsonl.py
  accepts optional payload byte fields on recovery debug events
```

Compatibility rule:

```text
Existing M3 fp16 recovery tests and smoke commands should set
VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only when VLLM_MPR_CPU_BACKUP=1 and
VLLM_MPR_PRECISION_TIERING_ENABLE is not enabled.
```

Performance note:

```text
Step 4.4 intentionally keeps eager fp16+int8 backup creation as a synchronous
reference path. In the current implementation, backup copy/quantization should
be treated as blocking decode-side work. The eager int8 payload is derived from
the already-created CPU fp16 payload, so it avoids a second GPU->CPU copy but
still pays CPU quantization cost synchronously. Future optimization items are
recorded in ../../backlog.md for Milestone 5, including lazy/on-the-fly
int8 creation, GPU-side quantization, pinned/non_blocking copies, readiness
tracking, and background quantization.
```

Validation added:

```text
config rejects eager fp16+int8 backup when tiering is disabled
store eager mode keeps get(key) fp16-compatible and exposes int8 payload
release removes all payload formats for a block
stats/debug events report payload-format byte accounting
M3 CPU backup/recovery tests explicitly use fp16_only storage mode
```

Validation run locally:

```text
python -m py_compile \
  vllm/v1/mixed_precision_recovery/cpu_backup.py \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  scripts/mpr_validate_debug_jsonl.py \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py

result:
  passed

git diff --check

result:
  passed
```

Local validation not completed:

```text
python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  not run in this Windows environment because pytest is not installed:
  No module named pytest

python import/runtime smoke:
  not run in this Windows environment because the installed torch package is
  incompatible with this vLLM checkout:
  ImportError: cannot import name 'infer_schema' from 'torch.library'
```

Target-server validation:

```text
Focused pytest:
  tests/v1/mixed_precision_recovery/test_scoring.py
  tests/v1/mixed_precision_recovery/test_recovery.py
  tests/v1/mixed_precision_recovery/test_backup_codec.py
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py

result:
  passed

M3 fp16-only recovery smoke:
  passed after scripts/mpr_smoke_recovery_quality.py set
  VLLM_MPR_BACKUP_STORAGE_MODE=fp16_only internally for its MPR child runs

M4 eager fp16+int8 backup smoke:
  passed; cpu_backup_created debug events reported nonzero
  cpu_backup_int8_payload_bytes and cpu_backup_int8_scale_bytes
```

Step 4.4 completion:

```text
Step 4.4 is complete.
Next step is Step 4.5 Recovery Payload Provider.
```

## 2026-06-07: Step 4.5 Recovery Payload Provider

Implemented a provider boundary between precision tier assignment and CPU backup
storage.

Added:

```text
vllm/v1/mixed_precision_recovery/recovery_payload.py
  RecoveryPayloadProvider protocol
  FP16RecoveryPayloadEntry
  INT8RecoveryPayloadEntry
  TieredRecoveryPayloads
  EagerRecoveryPayloadProvider
```

Provider behavior:

```text
fp16 tier -> fetch CPUBackupStore fp16 payload by CPUBackupKey
int8 tier -> fetch CPUBackupStore int8 payload by CPUBackupKey
skip tier -> preserve skipped ids without querying CPUBackupStore
missing payloads -> report missing_fp16_block_ids or missing_int8_block_ids
```

Compatibility boundary:

```text
precision policy code still does not import or query CPUBackupStore
recovery materialization is not changed yet
the M3 fp16 recovery path is preserved
```

Validation:

```text
tests/v1/mixed_precision_recovery/test_recovery_payload.py
  covers fp16/int8 payload group fetch
  covers missing fp16 and missing int8 payload reporting
  covers skip tier avoiding payload fetches
```

Validation run locally:

```text
python -m py_compile \
  vllm/v1/mixed_precision_recovery/recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py

result:
  passed

git diff --check

result:
  passed
```

Local validation not completed:

```text
python -m pytest tests/v1/mixed_precision_recovery/test_recovery_payload.py -q

result:
  not run in this Windows environment because pytest is not installed:
  No module named pytest

manual import/runtime test:
  not run in this Windows environment because torch is not installed:
  No module named 'torch'
```

Next step:

```text
Step 4.6 Tiered Materialization.
```
