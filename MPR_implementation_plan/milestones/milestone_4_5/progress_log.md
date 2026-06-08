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
