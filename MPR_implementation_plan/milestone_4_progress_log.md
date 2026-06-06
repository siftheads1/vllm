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
