# Milestone 3 Progress Log

## 2026-06-02: M3 Readiness Check

Milestone 3 starts from the approved action plan:

```text
MPR_implementation_plan/milestone_3_action_plan.md
```

Current M3 target:

```text
whole-block synchronous fp16 recovery smoke
single request / no preemption / FlashAttention
CPU fp16 backup -> GPU KV cache materialization before attention
scoring and recovery modules kept separate
```

Readiness checks completed:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  30 passed, 2 skipped
```

The two skipped tests are CUDA-gated in the current environment.

Syntax check completed:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/cpu_backup.py \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/v1/mixed_precision_recovery/scoring.py \
  vllm/v1/core/kv_cache_manager.py \
  vllm/model_executor/layers/attention/attention.py \
  tests/v1/mixed_precision_recovery/test_scoring.py

result:
  passed
```

Code-level readiness observations:

```text
CPU backup put/get stores semantic fp16 CPU K/V blocks.
Sidecar full-block digest creation backs up kv_cache[:, block_id].
Sidecar release_blocks(...) cleans block offsets, digests, CPU backup entries,
and invalidates append-only Quest metadata stores when released blocks exist.
KVCacheManager.free(...) calls the isolated MPR release hook before
coordinator.free(request.request_id).
Score candidate and top-k debug fields are physical block ids.
FlashAttention-side digest/backup path assumes KV cache layout:
  [2, num_blocks, block_size, num_kv_heads, head_dim]
```

Readiness conclusion:

```text
M2 inputs are sufficient to start Step 3.1 recovery config flags.
No implementation changes were made during this readiness check.
```

Known carry-forward limitations:

```text
CUDA runtime backup parity was skipped because CUDA was unavailable here.
M3 remains scoped to single-request/no-preemption first.
The current M2 cleanup key is layer_name + physical GPU block id.
Actual recovery observability still needs M3 fault-injection restore tests,
because normal runs do not evict or corrupt GPU KV blocks.
```

User follow-up:

```text
CUDA-gated readiness checks were manually run by the user and confirmed passing.
```

## 2026-06-02: Step 3.1 Recovery Config Flags

Implemented disabled-by-default recovery config scaffold.

Updated:

```text
vllm/v1/mixed_precision_recovery/config.py
  MPRConfig.recovery_enabled
  MPRConfig.recovery_topk
  MPRConfig.recovery_policy
  MPRConfig.recovery_threshold
  MPRConfig.recovery_test_mutate

vllm/envs.py
  VLLM_MPR_CPU_BACKUP
  VLLM_MPR_SCORING_ENABLE
  VLLM_MPR_RECOVERY_ENABLE
  VLLM_MPR_RECOVERY_TOPK
  VLLM_MPR_RECOVERY_POLICY
  VLLM_MPR_RECOVERY_THRESHOLD
  VLLM_MPR_RECOVERY_TEST_MUTATE

tests/v1/mixed_precision_recovery/test_scoring.py
  default recovery config coverage
  recovery env parsing coverage
  threshold policy parsing coverage
```

Recovery policy choices for the M3 scaffold:

```text
topk_block
threshold_block
```

Threshold policy was included now because it is closer to the eventual
mixed-precision recovery direction than top-k-only selection.

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
  32 passed, 2 skipped

git diff --check -- \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/envs.py \
  tests/v1/mixed_precision_recovery/test_scoring.py

result:
  passed
```

Current status:

```text
Step 3.1 is complete.
Next step is Step 3.2: create recovery.py with recovery target selection and
block materialization unit tests.
```

## 2026-06-02: Step 3.2 Recovery Module v0

Implemented a standalone recovery module.

Added:

```text
vllm/v1/mixed_precision_recovery/recovery.py
  RecoveryResult
  select_recovery_block_ids(...)
  BlockRecoveryManager.recover(...)
  BlockRecoveryManager.materialize_blocks(...)
```

Exported through:

```text
vllm/v1/mixed_precision_recovery/__init__.py
```

Implemented recovery policies:

```text
topk_block
  selects the highest-scoring block ids from score_result.block_scores

threshold_block
  selects all block ids whose score is >= recovery_threshold, preserving
  candidate order
```

Materialization semantics:

```text
backup = cpu_backup_store.get(CPUBackupKey(layer_name, physical_block_id))
target = kv_cache[:, physical_block_id]
target.copy_(backup.to(device=target.device, dtype=target.dtype))
```

The module validates the current M3 FlashAttention KV cache layout:

```text
[2, num_blocks, block_size, num_kv_heads, head_dim]
```

Added focused tests:

```text
tests/v1/mixed_precision_recovery/test_recovery.py
  top-k selection
  threshold selection
  mismatched score/id length validation
  CPU backup -> KV cache block materialization
  missing backup handling
  out-of-range block skip
  shape mismatch skip without mutation
  threshold-policy recover(...) path
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/recovery.py \
  vllm/v1/mixed_precision_recovery/__init__.py \
  tests/v1/mixed_precision_recovery/test_recovery.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  39 passed, 2 skipped

Expanded MPR py_compile:
  passed

git diff --check:
  passed
```

## 2026-06-02: CUDA Fault-Injection Smoke Result

Manual M3 recovery fault-injection smoke completed with:

```text
VLLM_MPR_ENABLE=1
VLLM_MPR_CPU_BACKUP=1
VLLM_MPR_SCORING_ENABLE=1
VLLM_MPR_RECOVERY_ENABLE=1
VLLM_MPR_RECOVERY_TEST_MUTATE=zero_selected
VLLM_MPR_RECOVERY_POLICY=topk_block
VLLM_MPR_RECOVERY_TOPK=1
VLLM_MPR_DEBUG_DIR=/tmp/mpr_debug_recovery_fault
VLLM_MPR_MAX_LAYERS=2
VLLM_MPR_MAX_STEPS=200
```

Important validator note:

```text
In recovery mode, the attention hook calls recover_before_attention rather than
observe_query. Therefore score_estimated events can be zero even when query
scoring was used internally for recovery selection. Recovery validation should
require recovery_materialized, not score_estimated.
```

Correct M3 recovery validator command:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python \
  scripts/mpr_validate_debug_jsonl.py \
  /tmp/mpr_debug_recovery_fault/*.jsonl \
  --min-digest-events 1 \
  --min-score-events 0 \
  --min-recovery-events 1
```

Observed result:

```text
MPR debug JSONL validation passed
events: 400
observe_kv_write: 132
digest_created: 8
score_estimated: 0
recovery_materialized: 14
recovery_skipped: 118
digest_layers: 2
recovery count by layer:
  7 model.layers.0.self_attn.attn
  7 model.layers.1.self_attn.attn
```

Fault-injection evidence:

```text
recovery_test_mutate: zero_selected
recovery_test_mutated_block_ids: [1]
recovery_selected_block_ids: [1]
recovered_block_ids: [1]
missing_backup_block_ids: []
skipped_block_ids: []
```

Generation comparison:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python \
  scripts/mpr_compare_generation_outputs.py \
  /tmp/mpr_m3_baseline_off.log \
  /tmp/mpr_m3_recovery_fault.log

result:
  MPR generation output comparison passed
  generated_token_count: 64
```

Conclusion:

```text
For the current single-request/no-preemption M3 scope, the semantic recovery
smoke passed: selected KV blocks were intentionally zeroed and then restored
from CPU backup before attention, while deterministic generation matched the
MPR-off baseline.
```

## 2026-06-02: Quality-Oriented Recovery Semantic Smoke

User requested a stronger smoke that directly shows:

```text
baseline generation is normal
threshold-selected mutate-only generation becomes wrong
the same threshold-selected mutation followed by CPU recovery returns to normal
```

Added validation-only mode:

```text
VLLM_MPR_RECOVERY_TEST_MODE=recover
  default; zero selected blocks and then materialize from CPU backup

VLLM_MPR_RECOVERY_TEST_MODE=mutate_only
  validation-only fault path; zero selected blocks and intentionally skip
  materialization so generation can visibly degrade
```

Implementation:

```text
vllm/v1/mixed_precision_recovery/config.py
vllm/envs.py
  add VLLM_MPR_RECOVERY_TEST_MODE

vllm/v1/mixed_precision_recovery/sidecar.py
  recover_before_attention remains production-style and mutation-free
  recover_before_attention_with_test_mutation supports recover vs mutate_only
  mutate_only records recovery_test_mutated

scripts/mpr_validate_debug_jsonl.py
  validates recovery_test_mutated
  adds --min-test-mutation-events

scripts/mpr_smoke_recovery_quality.py
  runs baseline, mutate_only, and mutate+recover generations
  checks mutate_only token IDs differ from baseline
  checks mutate+recover token IDs match baseline
  checks mutate_only debug has recovery_test_mutated
  checks recover debug has recovery_materialized with mutated IDs recovered
```

Recommended command:

```text
cd /home/han/KV_cache_quant/proposed_method_develop/vllm

/home/han/anaconda3/envs/20260528_vllm/bin/python \
  scripts/mpr_smoke_recovery_quality.py \
  --model Qwen/Qwen3-8B \
  --dtype half \
  --max-model-len 2048 \
  --max-tokens 512 \
  --gpu-memory-utilization 0.75 \
  --recent-tokens 256
```

The smoke uses:

```text
VLLM_MPR_RECOVERY_POLICY=threshold_block
VLLM_MPR_RECOVERY_THRESHOLD=-1e30
VLLM_MPR_RECOVERY_TEST_MUTATE=zero_selected
VLLM_MPR_RECENT_TOKENS=256
```

For a 512-token generation, `recent_tokens=256` keeps the recent half protected
and makes threshold selection hit roughly the older half of the finalized KV
blocks.

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/model_executor/layers/attention/attention.py \
  scripts/mpr_validate_debug_jsonl.py \
  scripts/mpr_smoke_recovery_quality.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_scoring.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  59 passed, 2 skipped

/home/han/anaconda3/envs/20260528_vllm/bin/python \
  scripts/mpr_smoke_recovery_quality.py --help

result:
  passed

git diff --check:
  passed
```

Coverage clarification:

```text
The smoke does not assume "half" blindly. With threshold=-1e30, selected blocks
are all scored candidate blocks. Candidate blocks are finalized blocks excluding
the recent protected tail from VLLM_MPR_RECENT_TOKENS.

scripts/mpr_smoke_recovery_quality.py now reports and checks the best observed
mutated/finalized block ratio for both mutate_only and mutate+recover runs.
Default minimum:
  --min-mutated-finalized-ratio 0.35

For the default 512-token generation with recent_tokens=256, the expected late
decode shape is roughly:
  finalized blocks ~= 512 / block_size
  protected blocks ~= 256 / block_size
  mutated candidate blocks ~= finalized - protected
so the observed ratio should be close to one half, subject to full-block digest
availability and prompt length.
```

Follow-up diagnosis:

```text
Observed coverage:
  mutate_only_best_mutated_finalized_ratio: 0.531
    17/32 blocks, candidate=17, protected=16
  recover_best_mutated_finalized_ratio: 0.531
    17/32 blocks, candidate=17, protected=16

The first generated_text preview looked identical across all three runs because
the smoke printed only the early prefix. Token-id comparison showed:
  baseline vs mutate_only:
    first_mismatch_index = 251
  baseline vs mutate+recover:
    exact token-id match for 512 generated tokens

Updated scripts/mpr_smoke_recovery_quality.py to print:
  mutate_only_first_token_mismatch_index
  recover_first_token_mismatch_index
  text common-prefix length
  baseline/mutate-only text windows around the divergence
```

## 2026-06-02: Stronger Mutate-only Fault Injection

User requested a more visible mutate-only failure mode by zeroing the whole KV
cache instead of only threshold-selected blocks.

Added:

```text
VLLM_MPR_RECOVERY_TEST_MUTATE=zero_all
  validation-only mutation mode
  zeroes the entire per-layer kv_cache tensor before attention
  records recovery_test_mutated_scope=all_kv_cache

scripts/mpr_smoke_recovery_quality.py
  --test-mutate {zero_selected,zero_all}
  --mutate-only-test-mutate {zero_selected,zero_all}
  --recover-test-mutate {zero_selected,zero_all}
```

Recommended strong-failure smoke:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python \
  scripts/mpr_smoke_recovery_quality.py \
  --model Qwen/Qwen3-8B \
  --dtype half \
  --max-model-len 2048 \
  --max-tokens 512 \
  --gpu-memory-utilization 0.75 \
  --recent-tokens 256 \
  --mutate-only-test-mutate zero_all \
  --recover-test-mutate zero_selected \
  --show-full-text
```

Important caveat:

```text
Use zero_all for mutate_only, but keep recover at zero_selected for the current
M3 semantic smoke. zero_all also corrupts current/recent/possibly non-backed
KV cache blocks, so a normal CPU-backup recovery policy may not restore every
piece needed for exact generation parity.
```

Validation:

```text
py_compile:
  passed

focused pytest:
  61 passed, 2 skipped

script --help:
  passed

git diff --check:
  passed
```

Follow-up error:

```text
The first name_recall prompt was too long for --max-model-len 2048.
Baseline failed before generation:
  prompt contains at least 2049 input tokens
  model maximum context length is 2048 tokens

Reduced the built-in name_recall filler from 24+24 notes to 12+12 shorter
notes so the prompt fits the default 2048 context budget with room for decode.

Also improved scripts/mpr_smoke_recovery_quality.py failure reporting:
  if a generation subprocess fails, print the log path and last log lines.
```

Second follow-up:

```text
The shortened name_recall prompt fit the context window, but baseline did not
emit ailikehuman within 32 generated tokens. It started a natural-language
reasoning completion:
  "The answer is the identity fact from the critical identity note. Okay, ..."

This is a prompt-shape issue for raw completion. Updated the preset to use a
completion-style lookup:
  middle fact:
    USER_NAME = ailikehuman
  final suffix:
    USER_NAME =

Tokenizer check:
  prompt_tokens: 867
```

## 2026-06-02: Name-recall Prompt Probe

User proposed a shorter decode but stronger context-recall probe:

```text
Use a long prompt.
Place "the user's name is ailikehuman" near the middle.
Ask for the name at the end.
Then compare baseline, mutate-only, and mutate+recover.
```

Added:

```text
scripts/mpr_smoke_recovery_quality.py
  --prompt-preset name_recall
  --expected-answer

For --prompt-preset name_recall:
  default expected answer is ailikehuman
  baseline output must contain ailikehuman
  mutate+recover output must contain ailikehuman
  mutate-only is still checked by token divergence from baseline
```

Recommended command:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python \
  scripts/mpr_smoke_recovery_quality.py \
  --model Qwen/Qwen3-8B \
  --dtype half \
  --max-model-len 2048 \
  --max-tokens 32 \
  --gpu-memory-utilization 0.75 \
  --prompt-preset name_recall \
  --recent-tokens 128 \
  --mutate-only-test-mutate zero_all \
  --recover-test-mutate zero_selected \
  --show-full-text
```

Validation:

```text
py_compile:
  passed

script --help:
  passed

git diff --check:
  passed
```

Current status:

```text
Step 3.2 is complete.
Next step is Step 3.3: wire recovery selection/materialization into the sidecar
without duplicating scoring work.
```

## 2026-06-02: Step 3.3 Sidecar Recovery API

Implemented sidecar-level recovery wiring without adding the attention hook yet.

Updated:

```text
vllm/v1/mixed_precision_recovery/sidecar.py
  QueryScoreContext
  RecoverySidecar._estimate_query_scores(...)
  RecoverySidecar._record_score_estimated(...)
  RecoverySidecar._record_recovery_skip(...)
  RecoverySidecar.recover_before_attention(...)
```

Design choice:

```text
Recovery enabled path should call recover_before_attention(...) instead of
observe_query(...). The recovery method performs scoring once, appends the
decode query to the rolling query window once, and passes the resulting
DigestScoreResult to BlockRecoveryManager.
```

This follows the agreed option A:

```text
score-only mode:
  observe_query(...)

recovery mode:
  recover_before_attention(...)
```

This avoids duplicate query-window updates and duplicate scoring when the
attention hook is added in Step 3.4.

Recovery preconditions:

```text
recovery requires config.recovery_enabled
recovery requires config.cpu_backup_enabled
recovery requires config.scoring_enabled
recovery requires kv_cache
```

If a precondition fails, the sidecar records `recovery_skipped` when debug
limits allow it and does not mutate KV cache.

Recovery debug event:

```text
recovery_materialized
  recovery_policy
  recovery_topk
  recovery_threshold
  recovery_selected_block_ids
  recovered_block_ids
  missing_backup_block_ids
  skipped_block_ids
  recovered_bytes
  recovery_copy_wall_ms
  cpu_backup_block_count
  cpu_backup_bytes
  score/block-table debug fields
```

Added tests:

```text
tests/v1/mixed_precision_recovery/test_recovery.py
  sidecar recovery disabled does not mutate KV cache
  sidecar top-k recovery materializes selected block
  sidecar threshold recovery handles selected block with missing backup
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  tests/v1/mixed_precision_recovery/test_recovery.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  42 passed, 2 skipped

Expanded MPR py_compile:
  passed

git diff --check:
  passed
```

Current status:

```text
Step 3.3 is complete.
Next step is Step 3.4: add the attention-before-forward hook that chooses
recover_before_attention(...) when recovery is enabled, otherwise preserving
the existing observe_query(...) score-only path.
```

## 2026-06-02: Step 3.4 Attention Hook Branch

Implemented the attention-before-forward MPR branch.

Updated:

```text
vllm/model_executor/layers/attention/attention.py
  _infer_mpr_block_size(...)
  _maybe_observe_or_recover_mpr_query(...)
  unified_attention_with_output(...)
```

Hook behavior:

```text
VLLM_MPR_ENABLE=0:
  early return, same as previous MPR hook behavior

forward_context.is_dummy_run:
  skip MPR work

sidecar.config.recovery_enabled=false:
  sidecar.observe_query(...)

sidecar.config.recovery_enabled=true:
  sidecar.recover_before_attention(...)
```

This preserves the agreed option A:

```text
score-only mode calls observe_query(...)
recovery mode calls recover_before_attention(...)
```

Therefore the same decode query is not appended twice to the sidecar query
window, and score computation is not duplicated by calling both paths.

Block-size inference:

```text
Use attn_layer.impl.block_size when available.
Fallback to FlashAttention KV cache layout:
  [2, num_blocks, block_size, num_kv_heads, head_dim]
```

Added hook-focused tests:

```text
tests/v1/mixed_precision_recovery/test_recovery.py
  attention hook calls observe_query when recovery is disabled
  attention hook calls recover_before_attention and passes kv_cache/block_size
  when recovery is enabled
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/model_executor/layers/attention/attention.py \
  tests/v1/mixed_precision_recovery/test_recovery.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  44 passed, 2 skipped

Expanded MPR py_compile:
  passed

git diff --check:
  passed
```

Current status:

```text
Step 3.4 is complete.
Next step is Step 3.5/3.7 validation hardening: add or run an end-to-end
single-request recovery smoke when CUDA is available, and add fault-injection
restore validation if needed to prove materialization is observable.
```

## 2026-06-02: Step 3.6 Debug Output Validation

Step 3.5 materialization semantics were already implemented through Steps 3.2
to 3.4:

```text
CPU backup lookup uses CPUBackupKey(layer_name, physical_block_id)
target is kv_cache[:, physical_block_id]
copy uses backup.to(device=target.device, dtype=target.dtype) then target.copy_(...)
layout check requires [2, num_blocks, block_size, num_kv_heads, head_dim]
out-of-range blocks are skipped
missing backups are recorded
shape mismatches are skipped without mutating target
copy byte count and wall time are reported
```

The only strict Step 3.5 checklist item not implemented is an explicit
`backup.dtype == torch.float16` validation in recovery.py. This was intentionally
deferred because future precision policies may store non-fp16 backups, and M2's
current SemanticCPUBackupStore already writes fp16.

Implemented Step 3.6 debug validator support.

Updated:

```text
scripts/mpr_validate_debug_jsonl.py
  --min-recovery-events
  validate_recovery_materialized_event(...)
  validate_recovery_skipped_event(...)
  summary counts for recovery_materialized and recovery_skipped

tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py
  accepts recovery_materialized
  accepts recovery_skipped
  rejects recovered ids outside selected ids
```

Recovery event schema validated:

```text
recovery_materialized
  recovery_policy in {topk_block, threshold_block}
  recovery_topk >= 1
  recovery_threshold finite
  recovery_selected_block_ids
  recovered_block_ids subset of selected ids
  missing_backup_block_ids subset of selected ids
  skipped_block_ids subset of selected ids
  recovered_bytes consistent with recovered_block_ids
  recovery_copy_wall_ms finite and non-negative
  optional kv_cache_shape is [2, num_blocks, block_size, num_kv_heads, head_dim]

recovery_skipped
  skipped_reason
  recovery policy/topk/threshold
  cpu_backup_enabled
  scoring_enabled
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  scripts/mpr_validate_debug_jsonl.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  54 passed, 2 skipped

Expanded MPR py_compile:
  passed

git diff --check:
  passed
```

Current status:

```text
Step 3.6 is complete.
Next step is Step 3.7 validation: run CUDA single-request recovery smoke and/or
add fault-injection restore validation to prove recovery materialization is
observable in an end-to-end path.
```

## 2026-06-02: Step 3.7 Validation Start

User confirmed the first four validation layers passed:

```text
Test 1: CPU-level fault-injection/materialization unit test
Test 2: sidecar-level recovery materialization test
Test 3: attention-hook branch test
Test 4: CUDA single-request recovery smoke
```

Remaining validation hardening:

```text
Implement VLLM_MPR_RECOVERY_TEST_MUTATE=zero_selected so the end-to-end recovery
path can intentionally corrupt selected KV blocks before materialization, then
verify recovery restores them from CPU backup.
```

Implemented `VLLM_MPR_RECOVERY_TEST_MUTATE=zero_selected`.

Updated:

```text
vllm/v1/mixed_precision_recovery/sidecar.py
  RecoverySidecar._maybe_mutate_recovery_targets(...)
  recover_before_attention(...) now selects recovery block ids first, optionally
  zeroes selected KV cache blocks, then materializes from CPU backup

scripts/mpr_validate_debug_jsonl.py
  validates recovery_test_mutate
  validates recovery_test_mutated_block_ids subset of selected ids

tests/v1/mixed_precision_recovery/test_recovery.py
  sidecar zero_selected mutation restores selected block from backup

tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py
  recovery_materialized sample includes mutation fields
```

Fault-injection semantics:

```text
recovery_test_mutate=off
  no mutation

recovery_test_mutate=zero_selected
  zero kv_cache[:, block_id] for selected in-range recovery targets before
  materialization, then recover from CPU backup
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  scripts/mpr_validate_debug_jsonl.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  55 passed, 2 skipped

Expanded MPR py_compile:
  passed

git diff --check:
  passed
```

Next manual CUDA validation command shape:

```text
VLLM_MPR_ENABLE=1
VLLM_MPR_CPU_BACKUP=1
VLLM_MPR_RECOVERY_ENABLE=1
VLLM_MPR_RECOVERY_TEST_MUTATE=zero_selected
VLLM_MPR_RECOVERY_POLICY=topk_block
VLLM_MPR_RECOVERY_TOPK=1
VLLM_MPR_DEBUG_DIR=/tmp/mpr_debug_recovery_fault
```

## 2026-06-02: Test Mutation Boundary Cleanup

User noted that fault injection should not live directly in
`recover_before_attention(...)` because that method may become the production
recovery entrypoint.

Updated boundary:

```text
recover_before_attention(...)
  production-style recovery entrypoint; ignores recovery_test_mutate and never
  mutates KV cache for validation

recover_before_attention_with_test_mutation(...)
  validation-only wrapper; applies VLLM_MPR_RECOVERY_TEST_MUTATE before
  materialization

attention hook
  calls the validation wrapper only when recovery_test_mutate != off
```

Follow-up recorded:

```text
RecoverySidecar is accumulating orchestration/debug/scoring/recovery/fault
injection responsibilities. Plan a larger refactor after M3 semantics settle.
```

Validation:

```text
/home/han/anaconda3/envs/20260528_vllm/bin/python -m py_compile \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/model_executor/layers/attention/attention.py \
  scripts/mpr_validate_debug_jsonl.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_scoring.py

result:
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  tests/v1/mixed_precision_recovery/test_recovery.py \
  tests/v1/mixed_precision_recovery/test_scoring.py -q

result:
  57 passed, 2 skipped

git diff --check:
  passed
```
