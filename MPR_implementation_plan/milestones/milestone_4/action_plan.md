# Milestone 4 Action Plan: Mixed-Precision Recovery Tiering

Milestone 4의 목표는 Milestone 3에서 만든 fp16 full-recovery skeleton 위에
score-guided precision tiering을 붙이는 것이다.

이 단계에서는 latency 최적화가 아니라 다음을 먼저 증명한다.

```text
score finalized KV blocks
  -> assign precision tier per physical KV block
  -> recover high-score blocks with fp16 payload
  -> recover medium-score blocks with int8 payload
  -> leave low-score blocks skipped
  -> materialize recovered blocks into the normal GPU KV cache before attention
```

Milestone 4.5는 INT4를 first-class tier로 통합하는 단계로 두고,
Milestone 5는 그 이후 optimization/cleanup 단계로 둔다.

## Goal

vLLM v1 + FlashAttention decode 중 다음 흐름을 구현하고 검증한다.

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

성공 기준은 다음과 같다.

1. Precision tiering flag가 꺼져 있으면 기존 M3 fp16 recovery 동작을 유지한다.
2. Precision tiering flag가 켜져 있으면 score에 따라 block이 fp16/int8/skip
   tier로 나뉜다.
3. fp16 tier는 기존 CPU fp16 backup path로 materialize된다.
4. int8 tier는 compressed int8 payload를 GPU로 옮긴 뒤 GPU에서
   dequantize/materialize된다.
5. skip tier는 recovery write를 하지 않는다.
6. Debug/validation output에서 tier assignment, tier별 recovered/skipped
   block id, byte accounting을 확인할 수 있다.
7. Validation-only degraded/eviction-style smoke에서 skip tier가 실제로
   unrecovered 상태로 attention에 들어가는 것을 확인한다.

## Scope

초기 scope는 의도적으로 좁게 둔다.

```text
vLLM v1
FlashAttention backend
single GPU
single request first
decode path first
no preemption
no real kv_offload connector integration
no scheduler-level eviction policy
whole physical KV block recovery
attention still reads the normal uniform GPU KV cache dtype/layout
```

Milestone 4에서는 direct mixed-dtype attention kernel을 만들지 않는다.
int8 tier도 attention 직전 기존 GPU KV cache dtype으로 materialize한다.

## Design Decisions

M4의 첫 구현 결정은 다음과 같이 둔다.

| Area | M4 Decision |
|---|---|
| First low-precision tier | int8 |
| Long-term tiers | fp16 / int8 / int4 / skip |
| Quantization granularity | per-token-per-kv-head |
| Scale shape | `[2, block_size, num_kv_heads]` |
| Quantized data shape | `[2, block_size, num_kv_heads, head_dim]` for int8 |
| CPU backup storage for first smoke | eager fp16 + int8 payloads |
| Future storage option | fp16-only backup + on-the-fly CPU quantization |
| INT8 materialization | copy compressed int8 payload and scale to GPU, then dequantize/materialize |
| Initial policy default | top-ratio for smoke stability |
| Additional policy | threshold policy for research framing |
| Skip validation | fault injection plus simulated degraded/eviction-style validation |
| Optimization | deferred to Milestone 5 |

The int8 reference quantization should use symmetric per-vector scaling:

```text
vector = kv_block[k_or_v, token_idx, kv_head, :]
scale = max(abs(vector)) / 127
quantized = clamp(round(vector / scale), -127, 127).to(int8)
```

Zero-vector scale handling must be explicit so quant/dequant does not produce
NaN or inf.

## Module Boundary

M4 should keep policy, storage, payload preparation, and materialization
separate.

Recommended shape:

```text
vllm/v1/mixed_precision_recovery/precision_policy.py
  PrecisionTier
  TierAssignment
  TopRatioPrecisionPolicy
  ThresholdPrecisionPolicy

vllm/v1/mixed_precision_recovery/backup_codec.py
  BackupPayload
  BackupCodec protocol
  FP16BackupCodec
  INT8BackupCodec

vllm/v1/mixed_precision_recovery/cpu_backup.py
  payload-aware CPU backup entries
  fp16 + int8 eager storage mode
  stats split by payload format

vllm/v1/mixed_precision_recovery/recovery.py
  tiered recovery/materialization result
  payload provider integration
  GPU dequant/materialization reference path

vllm/v1/mixed_precision_recovery/sidecar.py
  orchestration
  config gating
  debug event recording
  validation-only mutation/fault injection
```

Terminology to define in code docstrings:

```text
Backup codec:
  A precision-specific encoder/materializer for one KV backup format. It
  converts a semantic KV block into a stored backup payload, and later
  materializes that payload into the target GPU KV cache dtype/layout.

RecoveryPayloadProvider:
  The component that turns a TierAssignment into concrete payloads by using the
  CPUBackupStore and available BackupCodecs. It decides whether to use eager
  int8 payloads, fall back to fp16, or later quantize fp16 on the fly.
```

Important dependency rule:

```text
PrecisionPolicy must not know CPUBackupStore or BackupCodec.
CPUBackupStore must not know score policy.
RecoveryPayloadProvider is the boundary between policy decision and stored
payload availability.
```

## Target Files

Likely implementation files:

| File | Role |
|---|---|
| `vllm/v1/mixed_precision_recovery/config.py` | M4 flags and validation |
| `vllm/envs.py` | vLLM env registration |
| `vllm/v1/mixed_precision_recovery/precision_policy.py` | tier assignment |
| `vllm/v1/mixed_precision_recovery/backup_codec.py` | fp16/int8 codec and payload definitions |
| `vllm/v1/mixed_precision_recovery/cpu_backup.py` | payload-aware backup storage |
| `vllm/v1/mixed_precision_recovery/recovery.py` | tiered materialization |
| `vllm/v1/mixed_precision_recovery/sidecar.py` | runtime orchestration/debug |
| `scripts/mpr_validate_debug_jsonl.py` | tiered debug validation |

Likely tests:

```text
tests/v1/mixed_precision_recovery/test_precision_policy.py
tests/v1/mixed_precision_recovery/test_backup_codec.py
tests/v1/mixed_precision_recovery/test_recovery.py
tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py
tests/v1/mixed_precision_recovery/test_scoring.py
```

## Step 4.0: M3 Readiness Check

Before implementation, verify the current M3 state still satisfies:

```text
fp16 CPU backup put/get works
M3 BlockRecoveryManager materializes fp16 backup into kv_cache
score candidate block ids are physical block ids
recovery debug events are still valid
validation-only mutation path is separate from production recovery entrypoint
```

Completion:

```text
focused MPR pytest passes
current M3 recovery smoke command is still documented
no unexpected MPR regressions before tiering changes
```

## Step 4.1: Precision Tiering Config

Add disabled-by-default config flags for M4.

Candidate flags:

```text
VLLM_MPR_PRECISION_TIERING_ENABLE
VLLM_MPR_PRECISION_POLICY
VLLM_MPR_TIER_FP16_RATIO
VLLM_MPR_TIER_INT8_RATIO
VLLM_MPR_TIER_HIGH_THRESHOLD
VLLM_MPR_TIER_LOW_THRESHOLD
VLLM_MPR_BACKUP_STORAGE_MODE
```

Initial defaults:

```text
precision_tiering_enabled = false
precision_policy = top_ratio
tier_fp16_ratio = 0.25
tier_int8_ratio = 0.50
backup_storage_mode = eager_fp16_int8
```

Rules:

```text
VLLM_MPR_ENABLE=0 -> no MPR tiering import or work
VLLM_MPR_RECOVERY_ENABLE=0 -> no recovery materialization
VLLM_MPR_PRECISION_TIERING_ENABLE=0 -> use existing M3 fp16 recovery path
ratio values must be in [0, 1]
tier_fp16_ratio + tier_int8_ratio must be <= 1
threshold policy requires high threshold >= low threshold
```

Completion:

```text
MPRConfig parses new flags
vllm/envs.py recognizes new env vars
config tests cover defaults and invalid values
M3 recovery config tests still pass
```

## Step 4.2: Precision Policy Module

Implement score-to-tier assignment without touching backup storage.

Policy behavior:

```text
TopRatioPrecisionPolicy:
  sort candidates by descending block score
  top fp16_ratio -> fp16
  next int8_ratio -> int8
  remaining candidates -> skip

ThresholdPrecisionPolicy:
  score >= high_threshold -> fp16
  score >= low_threshold -> int8
  otherwise -> skip
```

Completion:

```text
unit tests cover score ordering and candidate id preservation
unit tests cover ratio rounding and empty candidate lists
unit tests cover threshold boundary behavior
TierAssignment stores physical block ids grouped by fp16/int8/skip
```

## Step 4.3: Backup Codec Abstraction

Add codec/payload definitions before changing recovery orchestration.

Initial payload formats:

```text
fp16:
  semantic CPU tensor shaped [2, block_size, num_kv_heads, head_dim]

int8:
  int8 CPU tensor shaped [2, block_size, num_kv_heads, head_dim]
  scale CPU tensor shaped [2, block_size, num_kv_heads]
  original shape and scale granularity metadata
```

Completion:

```text
FP16BackupCodec preserves current semantic backup behavior
INT8BackupCodec implements reference torch quant/dequant
codec classes define docstrings for encode/materialize responsibilities
unit tests check bounded int8 round-trip error
unit tests check zero-vector handling
unit tests check payload byte accounting
```

## Step 4.4: Payload-Aware CPU Backup Store

Extend the CPU backup store to support fp16 + int8 eager payloads.

First implementation behavior:

```text
on full KV block finalized:
  store fp16 payload
  store int8 payload + scale

on release:
  release every payload for matching layer/block key
```

Compatibility rule:

```text
Existing M3 recovery code path should still be able to retrieve the fp16 backup
without knowing about int8 payloads.
```

Stats to expose:

```text
fp16_payload_bytes
int8_payload_bytes
int8_scale_bytes
total_actual_backup_bytes
put_count
release_count
```

Completion:

```text
existing CPU backup tests still pass
new tests cover eager fp16+int8 storage
release removes all payload formats for a block
debug event can report payload-format byte stats
```

## Step 4.5: Recovery Payload Provider

Introduce a provider boundary between TierAssignment and CPUBackupStore.

Initial provider:

```text
EagerRecoveryPayloadProvider:
  fp16 tier -> fetch fp16 payload
  int8 tier -> fetch int8 payload + scale
  skip tier -> no payload request
```

Future provider hook:

```text
OnTheFlyCPUQuantPayloadProvider:
  int8 tier -> fetch fp16 payload, quantize on CPU, send int8 payload
```

Completion:

```text
provider returns per-tier payload groups
missing payloads are reported by tier
policy code does not import or query CPUBackupStore
unit tests cover missing fp16/int8 payload handling
```

## Step 4.6: Tiered Materialization

Extend recovery materialization while preserving the M3 path.

Rules:

```text
precision_tiering_enabled=false:
  use existing M3 selected-block fp16 materialization

precision_tiering_enabled=true:
  assign tiers from existing DigestScoreResult
  materialize fp16 tier from fp16 payload
  transfer int8 payload + scale to GPU
  dequantize int8 payload on GPU with torch reference ops
  copy dequantized tensor into kv_cache[:, physical_block_id]
  leave skip tier untouched
```

Completion:

```text
M3 recovery materialization tests still pass
new tests materialize int8 payload into target kv_cache
new tests verify skip tier does not mutate target block
RecoveryResult records selected/recovered/missing/skipped ids by tier
recovered bytes and effective compressed bytes are reported separately
```

## Step 4.7: Sidecar Integration and Debug Events

Wire tiered recovery into the existing attention-before-forward path.

Rules:

```text
recover_before_attention(...) remains production-style and mutation-free
recover_before_attention_with_test_mutation(...) remains validation-only
scoring is not recomputed only for debug and then recomputed for recovery
large score/digest tensors should not be converted to CPU except where already
needed for small selected id/debug summaries
```

New or extended debug fields:

```text
precision_tiering_enabled
precision_policy
tier_fp16_block_ids
tier_int8_block_ids
tier_skip_block_ids
recovered_fp16_block_ids
recovered_int8_block_ids
missing_fp16_block_ids
missing_int8_block_ids
skipped_block_ids
fp16_payload_bytes
int8_payload_bytes
int8_scale_bytes
effective_recovery_transfer_bytes
actual_cpu_backup_bytes
```

Completion:

```text
debug JSONL validator accepts tiered recovery events
validator preserves compatibility with M3 recovery logs
focused sidecar tests cover tiering disabled vs enabled behavior
```

## Step 4.8: Tiered Recovery Ratio Sweep Smoke

Add an M4 smoke script that runs tiered recovery across several FP16/INT8
ratio compositions and prints/saves generated outputs for inspection.

Default ratio sweep:

```text
1.0:0.0
0.75:0.25
0.5:0.5
0.25:0.75
0.0:1.0
```

Optional skip-accounting case:

```text
--include-skip-ratio adds 0.25:0.50
```

Boundary:

```text
default Step 4.8 sweep avoids skip tier
optional skip case only checks assignment/accounting
true degraded-residency skip correctness remains Step 4.9
```

Completion:

```text
generation completes
generated token ids are non-empty
tiered recovery_materialized events exist
missing fp16/int8 payload lists are empty
effective_recovery_transfer_bytes > 0
no-skip ratios produce zero skip assignments
optional skip ratio produces skip assignments without validating skip correctness
fp16 ratio > 0 produces fp16 tier/recovered ids
int8 ratio > 0 produces int8 tier/recovered ids
debug JSONL validator accepts every ratio run
script reports generated output preview/full text, mismatch vs baseline,
  tier counts, recovered counts, int8 payload bytes, transfer bytes, and paths
```

## Step 4.9: Tiered Degraded-Residency Ratio Sweep Smoke

Add a stronger validation smoke so skip is not only a debug label. This smoke
runs multiple fp16/int8 ratio compositions that intentionally leave a skip tier.

Validation behavior:

```text
after tier assignment:
  degrade candidate KV blocks in validation-only mode
  materialize fp16 tier
  materialize int8 tier
  leave skip tier degraded/unrecovered
  run attention
```

Default ratio sweep:

```text
0.25:0.25
0.25:0.50
0.50:0.25
0.10:0.25
0.25:0.10
```

This simulates the semantic question:

```text
If old finalized blocks were unavailable or degraded, do only the chosen
precision tiers get restored while skipped blocks remain unrecovered?
```

Out of scope for this step:

```text
real vLLM kv_offload connector integration
scheduler-owned eviction
multi-request or preemption correctness
direct mixed-dtype attention
```

Completion:

```text
generation completes for every ratio
generated token ids are non-empty
debug JSONL records degraded candidate ids and unrecovered skip ids
validator checks skip tier remains unrecovered
smoke reports generated text preview/full text, mismatch vs baseline,
  tier/recovered counts, unrecovered skip count, byte accounting, and paths
output divergence is report-only, not a pass/fail condition
```

## Step 4.10: Milestone Result Document

After implementation and validation, write a Milestone 4 result/progress note.

Record:

```text
chosen config values
exact smoke commands
unit/validator test results
tier distribution observed in smoke
fp16 vs int8 payload byte accounting
known limitations and M5 optimization targets
```

## Known Follow-ups After Milestone 4

```text
Milestone 4.5:
  INT4 packed codec and first-class int4 recovery tier

Milestone 5:
batched materialization instead of per-block Python loop
pinned CPU memory and non_blocking transfer
custom CUDA/Triton quant/dequant materialization
CPU fp16-only backup with on-the-fly quant provider
sidecar responsibility split and hot-path debug cleanup
real offload/eviction integration

Potential future / currently out of scope:
  GPU low-precision staging buffer
  direct mixed-dtype attention path
  multi-request/preemption lifecycle correctness
```

## Decision Gate

Milestone 4 끝에서 다음 중 하나를 선택한다.

| Decision | Condition |
|---|---|
| Proceed to Milestone 4.5 | fp16/int8/skip tiering works semantically, compressed int8 transfer/materialization is observed, and simulated skip semantics are validated |
| Continue Milestone 4 | tiering exists but int8 materialization, skip validation, or byte accounting is incomplete |
| Rework quantization granularity | per-token-per-kv-head scale is too costly or too inaccurate |
| Rework payload storage | eager fp16+int8 storage is too awkward for policy experiments |
| Rework validation model | simulated degraded-residency smoke is too far from intended offload semantics |

## Expected Outcome

Milestone 4가 끝나면 우리는 다음 질문에 답할 수 있어야 한다.

```text
Can MPR use digest scores to assign vLLM physical KV blocks to fp16, int8, and
skip tiers, then materialize only the selected precision tiers into the normal
GPU KV cache before attention?
```

이 답이 안정적으로 나오면 Milestone 4.5에서 INT4를 first-class tier로
통합하고, 그 다음 Milestone 5에서 latency, transfer, materialization,
debug overhead, sidecar structure를 정리하고 최적화한다.
