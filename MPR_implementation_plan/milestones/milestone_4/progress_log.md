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
recorded as Milestone 5 Step 5.3, including lazy/on-the-fly int8 creation,
GPU-side quantization, pinned/non_blocking copies, copy-stream readiness
tracking, and background CPU quantization.
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

## 2026-06-07: Step 4.6 Tiered Materialization

Implemented the minimal tiered materialization primitive without wiring it into
the sidecar runtime path.

Updated:

```text
vllm/v1/mixed_precision_recovery/recovery.py
  RecoveryResult keeps the existing M3 fields and adds tier-specific metadata
  BlockRecoveryManager.materialize_tiered_payloads(...)
  fp16 payload materialization via FP16BackupCodec
  int8 payload materialization/dequantization via INT8BackupCodec
```

Tiered materialization behavior:

```text
fp16 payload entries -> materialize to target dtype/device and copy into
  kv_cache[:, physical_block_id]
int8 payload entries -> dequantize/materialize to target dtype/device and copy
  into kv_cache[:, physical_block_id]
skip tier ids -> record as skipped and leave target blocks untouched
missing fp16/int8 payload ids -> report by tier and in the M3-compatible
  missing_backup_block_ids field
out-of-range or shape-mismatched payload targets -> raise ValueError because
  they indicate a broken materialization contract, not a normal skip tier
```

Missing payload note:

```text
missing_fp16_block_ids and missing_int8_block_ids describe provider/store
availability, not target materialization failure.

For the current EagerRecoveryPayloadProvider, missing int8 payloads usually mean
the CPU backup store was populated without int8 payloads, for example
backup_storage_mode=fp16_only. Step 4.7 sidecar integration should decide
whether eager provider + non-empty int8 tier + missing int8 payload is a hard
configuration/contract error.
```

Compatibility boundary:

```text
BlockRecoveryManager.materialize_blocks(...) still uses CPUBackupStore.get(...)
sidecar recovery still calls the M3 fp16-only materialize_blocks(...) path
debug JSONL schema is not changed in this step
```

Validation added:

```text
tests/v1/mixed_precision_recovery/test_recovery.py
  tiered fp16 materialization into target block
  tiered int8 materialization/dequantization with bounded error
  skip tier leaves target block unchanged
  missing fp16/int8 payload ids are reported by tier
  out-of-range and shape-mismatched payload targets raise ValueError
```

Validation run locally:

```text
python -m py_compile \
  vllm/v1/mixed_precision_recovery/recovery.py \
  tests/v1/mixed_precision_recovery/test_recovery.py

result:
  passed

git diff --check

result:
  passed
```

Local validation not completed:

```text
python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py -q

result:
  not run in this Windows environment because pytest is not installed:
  No module named pytest
```

Target-server validation:

```text
Focused M4 regression pytest completed successfully:

python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  passed
```

Next step:

```text
Step 4.7 Sidecar Integration and Debug Events.
```

## 2026-06-07: Step 4.7 Sidecar Integration and Debug Events

Wired the M4 tiered materialization path into the sidecar recovery path while
preserving the M3 fp16-only path when precision tiering is disabled.

Updated:

```text
vllm/v1/mixed_precision_recovery/sidecar.py
  builds TopRatioPrecisionPolicy or ThresholdPrecisionPolicy from MPRConfig
  assigns fp16/int8/skip tiers from the existing QueryScoreContext score result
  fetches tier payloads through EagerRecoveryPayloadProvider
  calls BlockRecoveryManager.materialize_tiered_payloads(...) when
    precision_tiering_enabled=true
  keeps the existing select_recovery_block_ids(...) + materialize_blocks(...)
    path when precision_tiering_enabled=false

scripts/mpr_validate_debug_jsonl.py
  accepts M3 recovery_materialized events without tier fields
  validates M4 tiered recovery fields when precision_tiering_enabled=true
```

Runtime behavior:

```text
precision_tiering_enabled=false:
  existing M3 fp16 recovery behavior is unchanged

precision_tiering_enabled=true:
  score once
  assign tiers from the score result
  fetch eager fp16/int8 payloads
  hard-fail with ValueError if any tier payload is missing
  materialize fp16 and int8 tiers
  leave skip tier blocks untouched
```

Debug JSONL additions for tiered recovery events:

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
fp16_payload_bytes
int8_payload_bytes
effective_recovery_transfer_bytes
```

Validation-only mutation boundary:

```text
recover_before_attention(...) remains mutation-free
recover_before_attention_with_test_mutation(...) remains the only path that can
apply zero_selected or zero_all
tiered zero_selected mutation uses the union of fp16, int8, and skip tier ids
```

Validation added:

```text
tests/v1/mixed_precision_recovery/test_recovery.py
  tiering disabled still exercises the M3 fp16 path
  tiering enabled materializes fp16 and int8 tiers and leaves skip untouched
  top_ratio and threshold precision policies both route through sidecar
  missing fp16 or int8 tier payloads raise ValueError
  tiered validation-only mutation uses all tier ids and skips materialization

tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py
  accepts old M3 recovery_materialized events without tier fields
  accepts M4 tiered recovery_materialized events with tier fields
```

Validation run locally:

```text
python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  scripts/mpr_validate_debug_jsonl.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
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
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  not run in this Windows environment because pytest is not installed:
  No module named pytest
```

Next step:

```text
Step 4.8 Tiered Recovery Ratio Sweep Smoke.
```

## 2026-06-08: Step 4.8 Tiered Recovery Ratio Sweep Smoke

Added an M4 smoke script for ratio-sweep inspection of tiered recovery output.

Added:

```text
scripts/mpr_smoke_tiered_recovery.py
```

Default behavior:

```text
run one MPR-off baseline
run tiered MPR recover mode for:
  1.0:0.0
  0.75:0.25
  0.5:0.5
  0.25:0.75
  0.0:1.0
```

Optional skip-accounting behavior:

```text
--include-skip-ratio adds 0.25:0.50
this observes skip assignment/accounting only
degraded-residency skip correctness remains Step 4.9
```

Per-ratio validation:

```text
generation completes
generated_token_ids is non-empty
tiered recovery_materialized events exist
missing_fp16_block_ids and missing_int8_block_ids are empty
effective_recovery_transfer_bytes is positive
no-skip ratios produce zero skip assignments
optional skip ratio produces skip assignments without validating skip correctness
fp16 ratio > 0 produces fp16 tier/recovered ids
int8 ratio > 0 produces int8 tier/recovered ids
existing debug JSONL validator accepts the ratio debug logs
```

Per-ratio reporting:

```text
generated text preview or full text
first token mismatch vs baseline
text common-prefix chars vs baseline
tier fp16/int8/skip counts
recovered fp16/int8 counts
fp16/int8 payload bytes and effective transfer bytes
generation log, debug dir, and validator log paths
optional --summary-json with generated outputs and accounting summary
```

Validation run locally:

```text
python -m py_compile scripts/mpr_smoke_tiered_recovery.py

result:
  passed

python scripts/mpr_smoke_tiered_recovery.py --help

result:
  passed
```

Runtime smoke not run in this implementation pass because it requires the target
GPU/model environment.

Next step:

```text
Step 4.9 Tiered Degraded-Residency Ratio Sweep Smoke.
```

## 2026-06-08: Step 4.9 Tiered Degraded-Residency Ratio Sweep Smoke

Added an M4 smoke script for ratio-sweep validation of simulated
degraded-residency skip behavior.

Added:

```text
scripts/mpr_smoke_tiered_degraded_residency.py
```

Default behavior:

```text
run one MPR-off baseline
run tiered MPR recover mode with validation-only zero_selected mutation for:
  0.25:0.25
  0.25:0.50
  0.50:0.25
  0.10:0.25
  0.25:0.10
```

Runtime environment per ratio:

```text
VLLM_MPR_BACKUP_STORAGE_MODE=eager_fp16_int8
VLLM_MPR_PRECISION_TIERING_ENABLE=1
VLLM_MPR_PRECISION_POLICY=top_ratio
VLLM_MPR_RECOVERY_TEST_MUTATE=zero_selected
VLLM_MPR_RECOVERY_TEST_MODE=recover
```

Per-ratio validation:

```text
generation completes
generated_token_ids is non-empty
tiered recovery_materialized events exist
recovery_test_mutated_block_ids is non-empty
tier_skip_block_ids is non-empty
skip ids are included in recovery_test_mutated_block_ids
skip ids are included in skipped_block_ids
skip ids are not included in recovered_block_ids
fp16 ratio > 0 produces recovered fp16 ids
int8 ratio > 0 produces recovered int8 ids
missing_fp16_block_ids and missing_int8_block_ids are empty
effective_recovery_transfer_bytes is positive
```

Validator update:

```text
scripts/mpr_validate_debug_jsonl.py
  adds --require-tiered-skip-unrecovered
  existing behavior is unchanged unless the flag is passed
```

Per-ratio reporting:

```text
generated text preview or full text
first token mismatch vs baseline
text common-prefix chars vs baseline
degraded/mutated block count
tier fp16/int8/skip counts
recovered fp16/int8 counts
unrecovered skip count
fp16/int8 payload bytes and effective transfer bytes
generation log, debug dir, and validator log paths
optional --summary-json with baseline and per-ratio outputs
```

Validation run locally:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  scripts/mpr_smoke_tiered_degraded_residency.py \
  scripts/mpr_validate_debug_jsonl.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  43 passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_recovery_payload.py \
  tests/v1/mixed_precision_recovery/test_backup_codec.py \
  tests/v1/mixed_precision_recovery/test_precision_policy.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py -q

result:
  66 passed

git diff --check

result:
  passed
```

Runtime smoke completed on the target GPU/model environment:

```text
WORK_DIR=/tmp/mpr_m4_degraded_20260608_143758 \
bash -lc 'python scripts/mpr_smoke_tiered_degraded_residency.py \
  --work-dir "$WORK_DIR" \
  --summary-json "$WORK_DIR/summary.json" \
  --text-preview-chars 500'

result:
  passed

summary_json:
  /tmp/mpr_m4_degraded_20260608_143758/summary.json
```

Runtime smoke summary:

```text
baseline_generated_token_count: 512
ratio_runs: 5
skip_correctness_validated: true

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

Per-ratio runtime observations:

```text
ratio      gen  mismatch  prefix  events  mutated  fp16_tier  int8_tier  skip_tier  fp16_rec  int8_rec  skip_unrec
0.25:0.25  512       305    1596     526     4590       1350       1318       1922      1350      1318        1922
0.25:0.50  512       305    1596     526     4590       1350       2398        842      1350      2398         842
0.50:0.25  512       305    1596     526     4590       2430       1318        842      2430      1318         842
0.10:0.25  512       305    1596     526     4590        732       1318       2540       732      1318        2540
0.25:0.10  512       305    1596     526     4590       1350        700       2540      1350       700        2540
```

Interpretation:

```text
For every ratio:
  fp16 tier ids were recovered as fp16
  int8 tier ids were recovered as int8
  skip tier ids were validation-mutated and remained unrecovered

Output divergence remains report-only:
  all ratio runs first diverged from baseline at token index 305
  all ratio runs shared a 1596-character common text prefix with baseline
```

Next step:

```text
Step 4.10 Milestone Result Document.
```

## 2026-06-08: Step 4.10 Milestone Result Document

Recorded the Milestone 4 result document.

Added:

```text
MPR_implementation_plan/milestones/milestone_4/results.md
```

Decision:

```text
Proceed to Milestone 4.5 before Milestone 5
```

Recorded:

```text
chosen config/env values
implemented runtime behavior
focused pytest result
Step 4.8 tiered recovery ratio sweep summary
Step 4.9 degraded-residency skip sweep summary
fp16/int8 payload and effective transfer byte accounting
known M4 limitations
M4.5 INT4 integration target
M5 optimization and cleanup targets after INT4
```
