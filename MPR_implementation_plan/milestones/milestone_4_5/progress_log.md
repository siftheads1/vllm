# Milestone 4.5 Progress Log

## 2026-06-08: Milestone Created

Created Milestone 4.5 as a separate INT4 integration milestone between
Milestone 4 and Milestone 5.

Rationale:

```text
INT4 support is not just a Post-M5 optimization item
INT4 changes the intended precision tier set to fp16/int8/int4/skip
M5 optimization should run after the full intended tier set is integrated
PyTorch does not provide a general-purpose torch.int4 tensor dtype for this path
therefore M4.5 should implement packed uint8 payloads with explicit pack/unpack
```

Initial scope:

```text
packed INT4 backup codec
top-ratio policy extension
CPU backup/provider integration
recovery materialization into normal GPU KV cache dtype
debug JSONL accounting and validator support
ratio-sweep smoke with generated text visibility
simulated degraded-residency skip validation with INT4 present
```

Out of scope:

```text
direct mixed-dtype attention
GPU low-precision staging buffer
real scheduler-owned offload/eviction
multi-request/preemption correctness
FP8 or vLLM native FP8 KV cache support
```

## 2026-06-08: Step 4.5.0 INT4 Design Review

Completed the pre-implementation INT4 design review.

Added:

```text
MPR_implementation_plan/milestones/milestone_4_5/reports/step_4_5_0_int4_design_review.md
```

Status:

```text
investigation complete
user decisions pending
no implementation changes made
```

Key finding:

```text
The local PyTorch 2.11 environment exposes torch.int4 and torch.uint4 names,
but basic eager operations such as float_tensor.to(torch.int4), fill, copy,
and add are not implemented. M4.5 should not use torch.int4/torch.uint4 as
the backup payload dtype.
```

Recommended M4.5 direction:

```text
represent INT4 payloads as packed torch.uint8 bytes plus explicit scale and
shape metadata
use signed symmetric quantization with emitted values in [-7, 7]
use two's-complement nibble encoding
pack along head_dim with packed shape:
  [2, block_size, num_kv_heads, ceil(head_dim / 2)]
keep per-token-per-kv-head fp32 scales for the reference codec
extend top_ratio first and defer threshold INT4 semantics
add eager_fp16_int8_int4 while keeping eager_fp16_int8 as the default
```

## 2026-06-08: Step 4.5.2 INT4 Backup Codec

Completed the standalone INT4 backup codec step.

Implemented:

```text
INT4BackupPayload
INT4BackupCodec
INT4_BACKUP_FORMAT = "int4"
signed symmetric emitted range [-7, 7]
per-token-per-kv-head fp32 scale
packed torch.uint8 CPU payload
two signed INT4 values per byte using two's-complement nibbles
pack along head_dim with shape [2, block_size, num_kv_heads, ceil(head_dim / 2)]
direct PyTorch materialize/unpack/dequantize path
```

Status:

```text
codec-only implementation complete
CPUBackupStore integration remains Step 4.5.3
runtime recovery integration remains later M4.5 steps
user completed Step 4.5.2 focused tests
```

## 2026-06-08: Step 4.5.3 CPU Backup Store and Payload Provider Integration

Completed the INT4 CPU backup store and eager recovery payload provider
integration step.

Confirmed decisions:

```text
INT4 backup storage is eager for M4.5
add backup_storage_mode=eager_fp16_int8_int4
keep eager_fp16_int8 as the default storage mode
top_ratio remains the main M4.5 runtime smoke policy
threshold INT4 has config/policy surfaces but dedicated threshold tuning and
  threshold-specific smoke tests are deferred
runtime INT4 materialization remains Step 4.5.4/4.5.5
```

Implemented:

```text
MPRConfig accepts VLLM_MPR_BACKUP_STORAGE_MODE=eager_fp16_int8_int4
eager low-precision backup modes require precision_tiering_enabled=True when
  cpu_backup_enabled=True
SemanticCPUBackupStore can create fp16 + int8 + int4 payloads when
  backup_storage_mode=eager_fp16_int8_int4
CPU backup put/release/stats split INT4 packed payload bytes and INT4 scale
  bytes
CPUBackupStore.get_payload(..., "int4") returns INT4BackupPayload when present
EagerRecoveryPayloadProvider fetches int4 tier payloads and reports
  missing_int4_block_ids independently from fp16/int8 missing payloads
TieredRecoveryPayloads carries int4 payload entries and byte accounting
BlockRecoveryManager still rejects non-empty int4 materialization payloads until
  Step 4.5.4
```

Validation run locally:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/cpu_backup.py \
  vllm/v1/mixed_precision_recovery/recovery_payload.py \
  vllm/v1/mixed_precision_recovery/recovery.py \
  vllm/v1/mixed_precision_recovery/__init__.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py -q

result:
  67 passed, 2 skipped

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py -q

result:
  28 passed

git diff --check

result:
  passed
```

Environment note:

```text
python -m pytest from the base Python 3.12 environment is blocked before MPR
tests run by an existing NumPy 2.3.5 / SciPy-sklearn binary compatibility
ImportError while loading tests/conftest.py. The focused tests above were run in
the existing vLLM 20260528 conda environment used by prior MPR validation.
```

Next step:

```text
Step 4.5.4 Recovery Materialization and Debug Events
```

## 2026-06-08: Step 4.5.4 Recovery Materialization and Debug Events

Implemented INT4 tier materialization in the tiered recovery path.

Implemented:

```text
BlockRecoveryManager now owns INT4BackupCodec
materialize_tiered_payloads handles fp16 + int8 + int4 payloads
INT4 payloads unpack/dequantize/materialize into the normal target KV cache
  dtype/device before copy_
RecoveryResult reports recovered_int4_block_ids, missing_int4_block_ids,
  int4_payload_bytes, and int4_scale_bytes
effective_recovery_transfer_bytes includes INT4 packed bytes plus scale bytes
recovered_block_ids metadata is concatenated in tier order:
  fp16 -> int8 -> int4
sidecar no longer rejects INT4 tier assignment as unsupported
eager tiered sidecar recovery now treats missing INT4 payloads as fail-fast
  missing tier payload errors
recovery_materialized debug events report INT4 recovered/missing/byte fields
debug JSONL validation accepts and validates optional INT4 tier fields
```

Tests updated:

```text
recovery manager unit coverage now includes fp16/int8/int4 materialization
INT4 recovered data is checked against the scale / 2 quantization bound
INT4 missing-payload, shape-mismatch, and out-of-range cases are covered
sidecar INT4 runtime recovery replaces the old unsupported-assignment test
sidecar missing INT4 payload fail-fast behavior is covered
tiered recovery debug JSONL fixture now includes INT4 fields
```

Validation attempted in the current Windows environment:

```text
python -m py_compile \
  vllm/v1/mixed_precision_recovery/recovery.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  scripts/mpr_validate_debug_jsonl.py

result:
  passed

python -m py_compile \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py

result:
  passed
```

Focused validation completed in the Linux vLLM runtime environment:

```text
python -m pytest \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  passed

python -m pytest \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  -q

result:
  passed
```

Environment note:

```text
The Windows Python available to this coding session does not have pytest or
torch installed, so only py_compile and git diff --check were run locally here.
The focused pytest commands above were run and reported passing in the
Linux/vLLM runtime environment.
```

## 2026-06-08: Step 4.5.5 Sidecar Integration and Validation Fault Injection

Implemented INT4-aware sidecar validation coverage for the tiered recovery
runtime path.

Implemented:

```text
added a four-block sidecar test fixture for deterministic fp16/int8/int4/skip
  top-ratio assignment
added zero_selected + recover coverage where fp16/int8/int4 blocks are restored
  and the skip-tier block remains degraded
added a regression test showing precision_tiering_enabled=false preserves the
  M3 fp16 recovery path even when INT4 tier config is nonzero
added INT4 CPU backup byte fields to recovery_test_mutated debug events
updated the recovery_test_mutated validator fixture to include INT4 byte fields
```

Validation attempted in the current Windows environment:

```text
python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  scripts/mpr_validate_debug_jsonl.py

result:
  passed

python -c "<validate recovery_test_mutated fixture via mpr_validate_debug_jsonl>"

result:
  passed

git diff --check

result:
  passed
```

Focused validation completed in the Linux vLLM runtime environment on
2026-06-09:

```text
python -m pytest tests/v1/mixed_precision_recovery/test_recovery.py -q

result:
  passed

python -m pytest \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  passed

python -m pytest \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  -q

result:
  passed
```

Status:

```text
Step 4.5.5 complete
```

## 2026-06-09: Step 4.5.7 INT4 Smoke via Existing Scripts

Extended the existing M4 smoke scripts with opt-in INT4 ratio support while
preserving existing two-part fp16/int8 defaults.

Implemented:

```text
scripts/mpr_smoke_tiered_recovery.py accepts both FP16:INT8 and
  FP16:INT8:INT4 ratio entries
scripts/mpr_smoke_tiered_degraded_residency.py reuses the extended ratio parser
existing two-part ratios imply INT4=0 and keep eager_fp16_int8 storage
ratios with INT4 > 0 use eager_fp16_int8_int4 storage and set
  VLLM_MPR_TIER_INT4_RATIO
ratios with INT4 > 0 pass --require-tiered-int4-recovery to the debug JSONL
  validator
smoke summaries now report tier/recovered/missing INT4 counts and INT4
  payload/scale bytes
INT4 opt-in smoke assertions require non-empty INT4 tier/recovery evidence and
  positive INT4 byte accounting
Step 4.5.7 action plan text now documents the existing-script extension
  approach instead of a new script
```

Validation attempted in the current Windows environment:

```text
python -m py_compile \
  scripts/mpr_smoke_tiered_recovery.py \
  scripts/mpr_smoke_tiered_degraded_residency.py \
  scripts/mpr_validate_debug_jsonl.py

result:
  passed

python -c "<validate smoke ratio parser and INT4 env selection>"

result:
  passed

git diff --check

result:
  passed
```

Validation still needed in the Linux vLLM runtime environment:

```text
python -m pytest tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

WORK_DIR=/tmp/mpr_m45_int4_recovery_$(date +%Y%m%d_%H%M%S)
python scripts/mpr_smoke_tiered_recovery.py \
  --work-dir "$WORK_DIR" \
  --tier-ratios 1.00:0.00:0.00,0.50:0.25:0.25,0.25:0.25:0.50,0.00:0.50:0.50,0.00:0.00:1.00 \
  --summary-json "$WORK_DIR/summary.json" \
  --text-preview-chars 500

WORK_DIR=/tmp/mpr_m45_int4_degraded_$(date +%Y%m%d_%H%M%S)
python scripts/mpr_smoke_tiered_degraded_residency.py \
  --work-dir "$WORK_DIR" \
  --tier-ratios 0.25:0.25:0.25,0.25:0.25:0.10,0.25:0.10:0.25,0.10:0.25:0.25 \
  --summary-json "$WORK_DIR/summary.json" \
  --text-preview-chars 500
```
