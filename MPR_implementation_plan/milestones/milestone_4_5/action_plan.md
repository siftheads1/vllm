# Milestone 4.5 Action Plan: INT4 Packed Recovery Tier Integration

Milestone 4.5 exists because INT4 is not merely a codec swap. Adding it turns
M4's tier model from:

```text
fp16 / int8 / skip
```

into:

```text
fp16 / int8 / int4 / skip
```

This should be integrated before Milestone 5 so the optimization phase measures
and cleans up the full intended tier set.

## Summary

Add INT4 as a first-class recovery tier with a packed CPU backup payload. The
first implementation remains correctness-first and materializes recovered INT4
blocks back into the normal GPU KV cache dtype before attention.

PyTorch does not provide a general-purpose `torch.int4` tensor dtype for this
path, so the M4.5 payload should be represented as packed `torch.uint8` bytes
plus explicit metadata and scale tensors.

## Goal

Extend the current M4 dataflow:

```text
DigestScoreResult + physical block ids
  -> PrecisionPolicy
  -> TierAssignment(fp16 / int8 / skip)
  -> RecoveryPayloadProvider
  -> CPUBackupStore
  -> BackupPayload
  -> BackupCodec / Materializer
  -> GPU KV cache
```

to:

```text
DigestScoreResult + physical block ids
  -> PrecisionPolicy
  -> TierAssignment(fp16 / int8 / int4 / skip)
  -> RecoveryPayloadProvider
  -> CPUBackupStore
  -> BackupPayload(fp16 / int8 / int4)
  -> BackupCodec / Materializer
  -> GPU KV cache
```

Success criteria:

```text
precision_tiering_enabled=false still preserves the M3 fp16 recovery path
existing fp16/int8/skip M4 tests keep passing
top-ratio policy can assign fp16, int8, int4, and skip tiers
INT4 payloads are packed into uint8 bytes on CPU
INT4 materialization dequantizes into the target GPU KV cache dtype/device
debug JSONL reports int4 tier, recovery, missing-payload, and byte accounting
ratio-sweep smoke covers fp16/int8/int4 compositions
degraded-residency smoke proves skip remains unrecovered with int4 present
```

## Non-goals

```text
do not add direct mixed-dtype attention
do not rely on a torch.int4 tensor dtype
do not optimize INT4 pack/unpack kernels yet
do not implement real scheduler-owned offload/eviction yet
do not broaden to multi-request/preemption correctness yet
do not add FP8 or vLLM native FP8 KV cache support
```

Potential future solutions that remain out of scope for M4.5:

```text
GPU low-precision staging buffers
direct attention over compressed low-precision KV pages
DiffKV-style direct mixed-dtype attention kernels
```

## Proposed INT4 Payload Format

Default reference quantization:

```text
signed symmetric INT4
semantic quantized range: [-7, 7]
scale = max(abs(vector)) / 7
quantized = clamp(round(vector / scale), -7, 7)
zero-vector scale = 1.0, quantized = 0
scale granularity = per-token-per-kv-head
scale shape = [2, block_size, num_kv_heads]
```

Packed storage:

```text
packed tensor dtype: torch.uint8
packed tensor device: CPU
two signed INT4 values per byte
logical quantized shape: [2, block_size, num_kv_heads, head_dim]
packed length: ceil(numel(logical_quantized) / 2)
odd final nibble: padded with zero
```

Recommended nibble encoding:

```text
store each signed INT4 value as a 4-bit two's-complement nibble
unpack reconstructs int8 values in [-8, 7]
the encoder never emits -8 for the symmetric [-7, 7] quantization range
```

Byte accounting:

```text
int4_payload_bytes = packed_uint8_bytes + scale_bytes
effective_recovery_transfer_bytes includes packed INT4 bytes and scale bytes
recovered_bytes remains the bytes written into the target GPU KV cache dtype
```

## Step 4.5.0: INT4 Design Checkpoint

Before code changes, confirm the remaining INT4-specific decisions.

Decide:

```text
signed symmetric range:
  recommended: [-7, 7]

nibble encoding:
  recommended: 4-bit two's-complement stored in torch.uint8

scale granularity:
  recommended: per-token-per-kv-head, matching INT8

odd head_dim behavior:
  recommended: pad final nibble with zero and track original_shape

threshold policy handling:
  recommended: top_ratio supports INT4 first; threshold INT4 extension is
  either deferred or added only after explicit decision

storage mode:
  recommended: add eager_fp16_int8_int4 while keeping eager_fp16_int8 default
```

Deliverable:

```text
progress_log.md records the confirmed decisions
```

## Step 4.5.1: Config and Precision Policy Extension

Add the config surface needed for INT4 tiering.

Expected changes:

```text
MPRConfig:
  tier_int4_ratio: float = 0.0

env:
  VLLM_MPR_TIER_INT4_RATIO

validation:
  tier_int4_ratio in [0, 1]
  tier_fp16_ratio + tier_int8_ratio + tier_int4_ratio <= 1
```

Policy changes:

```text
PrecisionTier adds int4
TierAssignment adds int4_block_ids
TopRatioPrecisionPolicy assigns in order:
  fp16 -> int8 -> int4 -> skip
TopRatioPrecisionPolicy keeps stable score ordering and candidate-order ties
ceil + clamp behavior remains consistent with the current M4 policy
all candidates still belong to exactly one tier
```

Threshold policy:

```text
do not silently reinterpret the existing high/low thresholds
either keep threshold as fp16/int8/skip for M4.5
or add a separately approved high/mid/low threshold design
```

Tests:

```text
config default keeps tier_int4_ratio = 0.0
env parsing accepts tier_int4_ratio
invalid ratio and ratio sum > 1 are rejected
top-ratio assigns fp16/int8/int4/skip exactly once per candidate
small candidate count uses ceil + clamp without losing all high tiers
candidate order is preserved for threshold/candidate-order paths
```

## Step 4.5.2: INT4 Backup Codec

Add an INT4 codec without changing runtime recovery yet.

Expected changes:

```text
vllm/v1/mixed_precision_recovery/backup_codec.py:
  INT4BackupPayload
  INT4BackupCodec
```

Codec responsibility:

```text
encode semantic KV block into packed INT4 logical payload
materialize/dequantize packed INT4 payload into target dtype/device
leave CPU backup store layout and lifecycle to CPUBackupStore
```

Payload fields:

```text
packed: CPU torch.uint8 tensor
scale: CPU torch.float32 tensor shaped [2, block_size, num_kv_heads]
original_shape: [2, block_size, num_kv_heads, head_dim]
scale_granularity: per_token_per_kv_head
quantized_range: [-7, 7]
```

Tests:

```text
payload dtype/device/shape metadata
packed byte count for even and odd head_dim/numel
round-trip error within INT4 quantization bound
zero-vector scale has no NaN/inf and dequantizes to zero
per-token-per-kv-head scale equals vector max abs / 7
invalid KV block shape raises ValueError
```

## Step 4.5.3: CPU Backup Store and Payload Provider Integration

Wire INT4 payloads into backup storage and payload lookup.

Expected changes:

```text
CPUBackupStore supports INT4 payload entries
RecoveryPayloadProvider can fetch int4 payloads
new storage mode: eager_fp16_int8_int4
existing storage modes remain valid:
  eager_fp16_int8
  fp16_only
```

Behavior:

```text
eager_fp16_int8 remains the M4 default until changed deliberately
eager_fp16_int8_int4 creates fp16, int8, and int4 logical payloads
fp16_only remains a future/on-the-fly provider hook, not a behavior change
missing INT4 payloads are reported separately from missing fp16/int8 payloads
```

Tests:

```text
put/get_payload handles INT4 payloads
stats split fp16/int8/int4/scale/total actual bytes
release removes INT4 payloads with the backup entry
provider returns INT4 payloads for int4 tier ids
missing_int4_payload_block_ids is populated when appropriate
```

## Step 4.5.4: Recovery Materialization and Debug Events

Materialize recovered INT4 payloads into the normal GPU KV cache dtype/layout.

Expected changes:

```text
BlockRecoveryManager or tiered recovery path handles INT4BackupPayload
INT4 payload moves packed bytes and scale to target device
unpack -> dequantize -> target dtype -> write selected physical KV block
```

Debug fields:

```text
tier_int4_block_ids
recovered_int4_block_ids
missing_int4_payload_block_ids
int4_payload_bytes
int4_scale_bytes
effective_recovery_transfer_bytes includes INT4 packed + scale bytes
```

Tests:

```text
INT4 tier materializes nonzero recovered blocks
fp16/int8 behavior is unchanged
missing INT4 payloads do not masquerade as recovered ids
effective transfer bytes distinguish INT4 payload bytes from recovered bytes
```

## Step 4.5.5: Sidecar Integration and Validation Fault Injection

Extend runtime orchestration to pass INT4 tier assignments through recovery.

Expected changes:

```text
RecoverySidecar creates INT4-aware policy from config
tiered recovery receives fp16/int8/int4/skip ids
validation-only zero_selected mutation covers all selected candidates
recover mode restores fp16/int8/int4 ids and leaves skip ids degraded
```

Tests:

```text
zero_selected + recover + fp16/int8/int4/skip restores fp16/int8/int4
skip ids remain zero/degraded
precision_tiering_enabled=false ignores INT4 config and preserves M3 behavior
```

## Step 4.5.6: Debug JSONL Validator Extension

Update the validator so runtime smoke can assert INT4 debug semantics.

Expected changes:

```text
scripts/mpr_validate_debug_jsonl.py:
  optional --require-tiered-int4-recovery
  existing --require-tiered-skip-unrecovered still works with INT4 present
```

Validator requirements when enabled:

```text
at least one tiered recovery event has non-empty tier_int4_block_ids
int4 tier ids are included in recovered_block_ids when payloads are present
missing_int4_payload_block_ids is empty for normal eager storage smoke
int4 payload byte accounting is positive when int4 ids are recovered
```

Tests:

```text
accept valid INT4 tiered recovery event
reject INT4 tier ids that were not recovered
reject missing INT4 payloads in require mode
reject zero/absent INT4 payload byte accounting in require mode
skip-unrecovered requirement still rejects recovered skip ids
```

## Step 4.5.7: INT4 Ratio Sweep Smoke

Extend existing M4 smoke coverage for opt-in fp16/int8/int4 ratio
compositions while preserving existing fp16/int8 defaults.

Updated existing scripts:

```text
scripts/mpr_smoke_tiered_recovery.py
scripts/mpr_smoke_tiered_degraded_residency.py
```

Ratio CLI:

```text
existing FP16:INT8 ratios remain valid and imply INT4=0
new FP16:INT8:INT4 ratios enable M4.5 INT4 smoke validation
ratios with INT4 > 0 use eager_fp16_int8_int4 backup storage
ratios with INT4 > 0 pass --require-tiered-int4-recovery to the JSONL validator
```

M4.5 opt-in no-skip ratio sweep:

```text
1.00:0.00:0.00
0.50:0.25:0.25
0.25:0.25:0.50
0.00:0.50:0.50
0.00:0.00:1.00
```

M4.5 opt-in degraded-residency ratio sweep:

```text
0.25:0.25:0.25
0.25:0.25:0.10
0.25:0.10:0.25
0.10:0.25:0.25
```

Script options:

```text
--tier-ratios
--show-full-text
--text-preview-chars
--summary-json
model/dtype/max-token/debug controls matching M4 smoke scripts
```

Validation:

```text
generation completes
generated token ids are non-empty
tiered recovery events exist
missing fp16/int8/int4 payload lists are empty for eager storage
effective_recovery_transfer_bytes > 0
ratio with int4 > 0 produces tier_int4_block_ids and recovered_int4_block_ids
degraded run mutates skip ids and leaves them unrecovered
existing JSONL validator passes with INT4 require flag
```

Reporting:

```text
generated text preview or full text
first token mismatch vs baseline
text common prefix chars vs baseline
tier fp16/int8/int4/skip counts
recovered fp16/int8/int4 counts
unrecovered skip count for degraded run
fp16/int8/int4 payload bytes
effective transfer bytes and recovered bytes
log/debug/validator paths
summary JSON
```

## Step 4.5.8: Milestone Result Document and M5 Handoff

Record what INT4 changed before starting M5.

Add:

```text
MPR_implementation_plan/milestones/milestone_4_5/results.md
```

Record:

```text
confirmed INT4 encoding and scale decisions
config/env changes
unit and focused regression results
INT4 round-trip error observations
ratio-sweep smoke summaries
fp16/int8/int4 byte accounting
known limitations
M5 optimization targets after INT4 integration
```

M5 handoff should say:

```text
M5 optimizes the fp16/int8/int4/skip recovery skeleton.
M5 does not add new precision tiers by default.
```

## Test Plan

Static checks:

```text
python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/precision_policy.py \
  vllm/v1/mixed_precision_recovery/backup_codec.py \
  vllm/v1/mixed_precision_recovery/cpu_backup.py \
  vllm/v1/mixed_precision_recovery/recovery.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  scripts/mpr_validate_debug_jsonl.py \
  scripts/mpr_smoke_tiered_recovery.py \
  scripts/mpr_smoke_tiered_degraded_residency.py
```

Focused pytest:

```text
python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q
```

Runtime smoke:

```text
WORK_DIR=/tmp/mpr_m45_int4_recovery_$(date +%Y%m%d_%H%M%S) bash -lc \
  'python scripts/mpr_smoke_tiered_recovery.py \
    --work-dir "$WORK_DIR" \
    --tier-ratios 1.00:0.00:0.00,0.50:0.25:0.25,0.25:0.25:0.50,0.00:0.50:0.50,0.00:0.00:1.00 \
    --summary-json "$WORK_DIR/summary.json" \
    --text-preview-chars 500'
```

Runtime degraded skip smoke:

```text
WORK_DIR=/tmp/mpr_m45_int4_degraded_$(date +%Y%m%d_%H%M%S) bash -lc \
  'python scripts/mpr_smoke_tiered_degraded_residency.py \
    --work-dir "$WORK_DIR" \
    --tier-ratios 0.25:0.25:0.25,0.25:0.25:0.10,0.25:0.10:0.25,0.10:0.25:0.25 \
    --summary-json "$WORK_DIR/summary.json" \
    --text-preview-chars 500'
```

## Decision Gate

| Decision | Condition |
|---|---|
| Proceed to Milestone 5 | fp16/int8/int4/skip tiering works semantically, INT4 payloads are packed/materialized, byte accounting is reported, and degraded skip validation still passes |
| Continue Milestone 4.5 | INT4 codec exists but runtime recovery, debug validation, or smoke coverage is incomplete |
| Rework INT4 encoding | signed nibble packing is error-prone or incompatible with intended kernels |
| Rework policy semantics | top-ratio four-tier assignment creates unstable small-candidate behavior |
| Rework storage mode | eager fp16+int8+int4 storage is too costly even for correctness smoke |
| Defer threshold INT4 support | top-ratio is sufficient for M4.5 and threshold four-tier semantics need separate research framing |
