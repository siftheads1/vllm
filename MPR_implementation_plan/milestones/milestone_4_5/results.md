# Milestone 4.5 Results

Milestone 4.5 promoted INT4 from a future optimization idea into a first-class
Mixed-Precision Recovery tier. The implemented tier set is:

```text
fp16 / int8 / int4 / skip
```

## Decision

```text
Decision: Proceed to Milestone 5
```

Rationale:

```text
packed INT4 backup payloads are encoded as CPU uint8 nibbles plus fp32 scales
top-ratio precision tiering assigns fp16/int8/int4/skip blocks
eager fp16+int8+int4 backup storage feeds tiered recovery payload lookup
INT4 recovery materializes back into the normal GPU KV cache dtype/device
debug JSONL reports INT4 tier, recovered, missing, and byte accounting fields
validator support can require recovered INT4 evidence for targeted smoke runs
top-ratio runtime smoke coverage completed for INT4 recovery and degraded skip
remaining concerns are profiling, optimization, and threshold-policy research
```

M5 should optimize the full fp16/int8/int4/skip recovery skeleton rather than
adding another precision tier by default.

## Implemented Surface

Config/env additions:

```text
VLLM_MPR_TIER_INT4_RATIO
VLLM_MPR_TIER_MID_THRESHOLD
VLLM_MPR_BACKUP_STORAGE_MODE=eager_fp16_int8_int4
```

Important defaults and compatibility behavior:

```text
tier_int4_ratio = 0.0 by default
backup_storage_mode = eager_fp16_int8 by default
eager_fp16_int8_int4 is opt-in
precision_tiering_enabled=false preserves the M3 fp16 recovery path
existing FP16:INT8 smoke ratio syntax remains valid and implies INT4=0
```

INT4 encoding decisions:

```text
signed symmetric quantization range: [-7, 7]
storage dtype: packed torch.uint8 CPU tensor
packing: two signed INT4 values per byte using two's-complement nibbles
scale dtype: fp32
scale granularity: per-token-per-kv-head
scale shape: original_shape without head_dim
odd head_dim: final high nibble is zero padding, original_shape is authoritative
```

Runtime behavior:

```text
CPU backup store can eagerly create fp16, int8, and int4 payloads
payload provider fetches int4 payloads independently from fp16/int8 payloads
BlockRecoveryManager materializes int4 via INT4BackupCodec.materialize()
skip tier remains non-materialized
recovered_block_ids metadata is ordered fp16 -> int8 -> int4
effective_recovery_transfer_bytes includes fp16 payload, int8 payload/scale,
  and int4 packed payload/scale bytes
recovered_bytes remains target GPU KV cache dtype bytes actually written
```

Debug/validator behavior:

```text
recovery_materialized reports tier_int4_block_ids
recovery_materialized reports recovered_int4_block_ids
recovery_materialized reports missing_int4_block_ids
recovery_materialized reports int4_payload_bytes and int4_scale_bytes
recovery_test_mutated reports INT4 CPU backup byte fields
scripts/mpr_validate_debug_jsonl.py supports --require-tiered-int4-recovery
scripts/mpr_validate_debug_jsonl.py keeps --require-tiered-skip-unrecovered
  compatible with events that include INT4 fields
```

## Validation Results

Focused unit/regression coverage completed across the milestone:

```text
INT4 codec metadata, packing, odd head_dim padding, signed nibble handling,
  per-token-per-kv-head scaling, round-trip bound, zero vector handling, and
  target dtype/device materialization
INT4 backup store creation, stats, release accounting, and storage mode gating
INT4 eager payload provider fetch and missing-payload reporting
fp16/int8/int4 tiered recovery materialization
INT4 missing payload, shape mismatch, and out-of-range block-id failure modes
sidecar zero_selected + recover across fp16/int8/int4/skip
precision_tiering_enabled=false regression for the M3 fp16 path
debug JSONL validator acceptance and rejection cases for INT4 recovery evidence
```

Focused validation reported passing during M4.5:

```text
python -m pytest tests/v1/mixed_precision_recovery/test_recovery.py -q
python -m pytest tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q
python -m pytest \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  -q
```

Step 4.5.7 top-ratio runtime smoke status:

```text
recover smoke:
  top-ratio fp16/int8/int4 recovery path completed

degraded skip smoke:
  top-ratio fp16/int8/int4 recovery plus skip-unrecovered validation completed

result:
  user reported top-ratio smoke completed successfully
```

The detailed runtime summary JSON paths and per-ratio numeric tables were not
captured in this coding session. Future smoke reruns should preserve
`summary.json`, validator logs, and extracted generated text files for the M5
performance record.

## Known Limitations

M4.5 is correctness-first.

Known limitations:

```text
INT4 materialization uses a reference PyTorch unpack/dequantize path
INT4 pack/unpack kernels are not optimized
direct mixed-dtype attention is not implemented
low-precision payload creation remains eager for M4.5 correctness smoke
materialization still loops through tiered payload entries in Python
effective transfer bytes are debug accounting, not measured PCIe traffic
skip correctness is validated with zero_selected fault injection, not real
  scheduler-owned offload/eviction
single-request/no-preemption assumptions still dominate smoke validation
```

Threshold-policy status:

```text
threshold precision policy config surfaces exist
top-ratio is the validated M4.5 runtime policy
threshold INT4 semantics are deferred until threshold-specific profiling is done
do not silently reinterpret existing threshold behavior as part of M4.5/M5
```

## M5 Handoff

M5 starts from a working fp16/int8/int4/skip recovery skeleton.

Primary M5 targets:

```text
profile backup, scoring, payload lookup, materialization, and debug overhead
measure eager fp16+int8+int4 storage cost and memory pressure
evaluate lazy/on-the-fly low-precision payload generation
evaluate pinned CPU memory and non_blocking transfer paths
evaluate batched or vectorized recovery materialization
separate validation-only mutation/debug cost from production paths
clean up RecoverySidecar and payload/provider ownership boundaries
record real transfer/profiling evidence with external tools where needed
```

Deferred threshold work:

```text
run threshold-specific profiling before adding threshold INT4 smoke as a gate
decide high/mid/low threshold semantics with data rather than by analogy
keep top-ratio as the M4.5 correctness baseline
```
