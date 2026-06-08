# Milestone 4 Results

Milestone 4 implemented mixed-precision recovery tiering for MPR. It assigns
scored physical KV blocks to fp16, int8, or skip tiers, fetches logical CPU
backup payloads for the selected precision tiers, and materializes fp16/int8
payloads back into the normal GPU KV cache dtype before attention.

## Decision

```text
Decision: Proceed to Milestone 4.5 before Milestone 5
```

Rationale:

```text
fp16/int8/skip tier assignment works from existing digest score results
fp16 recovery semantics are preserved when precision tiering is disabled
int8 backup payloads are encoded, fetched, transferred, and dequantized
tiered materialization reports fp16/int8 payload bytes and effective transfer bytes
simulated degraded-residency smoke validates skip blocks remain unrecovered
remaining concerns are optimization/cleanup rather than M4 correctness blockers
INT4 should be integrated before optimization so M5 measures the full tier set
```

Milestone 4.5 should add INT4 as a packed recovery tier. After that, Milestone
5 should focus on backup/materialization overhead, async/pinned copy options,
lazy/on-the-fly payload generation, and sidecar cleanup.

## Implemented Surface

Config/env additions:

```text
VLLM_MPR_PRECISION_TIERING_ENABLE
VLLM_MPR_PRECISION_POLICY
VLLM_MPR_TIER_FP16_RATIO
VLLM_MPR_TIER_INT8_RATIO
VLLM_MPR_TIER_HIGH_THRESHOLD
VLLM_MPR_TIER_LOW_THRESHOLD
VLLM_MPR_BACKUP_STORAGE_MODE
```

Current M4 defaults:

```text
precision_tiering_enabled = false
precision_policy = top_ratio
tier_fp16_ratio = 0.25
tier_int8_ratio = 0.50
backup_storage_mode = eager_fp16_int8
```

Important runtime modes:

```text
precision_tiering_enabled=false:
  existing M3 fp16-only recovery path remains active

precision_tiering_enabled=true:
  score finalized candidate blocks
  assign fp16/int8/skip tiers
  fetch fp16 and int8 CPU backup payloads
  materialize fp16 and int8 tiers into the GPU KV cache dtype
  leave skip tier untouched
```

Initial precision/storage decisions:

```text
first low-precision tier: int8
long-term tiers: fp16 / int8 / int4 / skip
quantization granularity: per-token-per-kv-head
int8 scale shape: [2, block_size, num_kv_heads]
first storage mode: eager fp16 + int8 CPU payloads
future storage option: fp16-only backup + on-the-fly quant provider
```

## Validation Results

Focused MPR regression:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  66 passed
```

Step 4.8 tiered recovery ratio sweep:

```text
summary_json:
  /tmp/mpr_m4_tiered_20260608_141246/summary.json

baseline_generated_token_count: 512
ratio_runs: 6
skip_correctness_validated: false
```

Step 4.8 per-ratio observations:

```text
ratio      gen  mismatch  prefix  fp16_tier  int8_tier  skip_tier  fp16_rec  int8_rec  effective/recovered
1.00:0.00  512      none    2628       4590          0          0      4590         0              1.000
0.75:0.25  512      none    2628       3638        952          0      3638       952              0.900
0.50:0.50  512      none    2628       2430       2160          0      2430      2160              0.772
0.25:0.75  512      none    2628       1350       3240          0      1350      3240              0.658
0.00:1.00  512      none    2628          0       4590          0         0      4590              0.516
0.25:0.50  512       305    1596       1350       2398        842      1350      2398              0.690
```

Step 4.8 byte totals:

```text
total_fp16_payload_bytes: 875429888
total_int8_payload_bytes: 450785280
total_effective_recovery_transfer_bytes: 1326215168
total_recovered_bytes: 1749680128
```

Step 4.9 degraded-residency skip sweep:

```text
summary_json:
  /tmp/mpr_m4_degraded_20260608_143758/summary.json

baseline_generated_token_count: 512
ratio_runs: 5
skip_correctness_validated: true
```

Step 4.9 per-ratio observations:

```text
ratio      gen  mismatch  prefix  events  mutated  fp16_tier  int8_tier  skip_tier  fp16_rec  int8_rec  skip_unrec
0.25:0.25  512       305    1596     526     4590       1350       1318       1922      1350      1318        1922
0.25:0.50  512       305    1596     526     4590       1350       2398        842      1350      2398         842
0.50:0.25  512       305    1596     526     4590       2430       1318        842      2430      1318         842
0.10:0.25  512       305    1596     526     4590        732       1318       2540       732      1318        2540
0.25:0.10  512       305    1596     526     4590       1350        700       2540      1350       700        2540
```

Step 4.9 totals:

```text
total_tiered_recovery_events: 2630
total_skip_validating_events: 2182
total_mutated_blocks: 22950

total_fp16_tier / recovered_fp16: 7212 / 7212
total_int8_tier / recovered_int8: 7052 / 7052
total_skip_tier / unrecovered_skip: 8686 / 8686

total_fp16_payload_bytes: 472645632
total_int8_payload_bytes: 238301184
total_effective_recovery_transfer_bytes: 710946816
total_recovered_bytes: 934805504
```

Step 4.9 interpretation:

```text
For every degraded ratio:
  fp16 tier ids were recovered as fp16
  int8 tier ids were recovered as int8
  skip tier ids were validation-mutated and remained unrecovered

Output divergence is report-only:
  all ratio runs first diverged from baseline at token index 305
  all ratio runs shared a 1596-character common text prefix with baseline
```

## Known Limitations and M5 Targets

M4 is correctness-first and intentionally leaves production optimization work
for Milestone 5.

Known limitations:

```text
eager fp16+int8 backup creation is synchronous/blocking
int8 payloads are derived from CPU fp16 payloads, avoiding a second GPU->CPU
  copy but still paying synchronous CPU quantization cost
materialization currently loops through selected payloads in Python
effective_recovery_transfer_bytes is logical debug accounting, not measured
  PCIe traffic
skip semantics are validated through simulated zero_selected degradation, not
  real scheduler/offload-owned eviction
single-request/no-preemption assumptions still dominate validation
direct mixed-dtype attention is not implemented
INT4 is not implemented in M4 and is promoted to Milestone 4.5
```

Milestone 5 targets after Milestone 4.5:

```text
measure backup, scoring, recovery, and mixed-tier overhead separately
optimize CPU backup and int8/int4 payload creation
evaluate pinned CPU memory, non_blocking copy, copy stream, and readiness tracking
evaluate lazy/on-the-fly low-precision payload creation and GPU-side quantization
evaluate batched recovery materialization instead of per-block Python loops
remove validation-only fault injection cost from the production path
split RecoverySidecar responsibilities into cleaner controller boundaries
record PCIe evidence with dmon/nsys only as external profiling evidence
```
