# Milestone 3 Action Plan

Milestone 3의 목표는 Milestone 2에서 만든 CPU fp16 backup을 이용해
score로 선택된 KV block을 **attention 실행 전에 GPU KV cache에
materialize/recover**하는 것이다.

이 단계에서는 아직 mixed precision tiering을 하지 않는다. M3는
ArkVale의 binary recall에 가까운 baseline으로, selected physical KV block을
CPU backup에서 GPU KV cache dtype/layout으로 되돌릴 수 있는지 확인한다.

## Goal

vLLM v1 + FlashAttention decode 중 다음 흐름을 구현하고 검증한다.

```text
score finalized KV blocks
  -> select recovery target block ids
  -> lookup CPU fp16 backup
  -> copy backup into GPU KV cache before attention
  -> run normal attention
```

성공 기준은 다음과 같다.

1. Recovery flag가 꺼져 있으면 기존 score/backup path와 동일하게 동작한다.
2. Recovery flag가 켜져 있으면 selected block을 attention 전에 materialize한다.
3. Materialized KV block은 CPU backup payload와 일치한다.
4. Missing backup이나 unsupported shape는 debug event/counter로 보이고,
   first smoke 범위에서는 engine을 불필요하게 깨지 않는다.
5. Normal generation smoke가 완료되고 recovery debug event를 남긴다.

## Scope

초기 scope는 의도적으로 좁게 둔다.

```text
vLLM v1
FlashAttention backend
single GPU
single request
decode path first
no preemption
no sliding-window special handling
whole physical KV block recovery
synchronous CPU -> GPU copy
fp16/full-recovery only
no lower precision tiering
```

Milestone 3에서는 실제 eviction/offload policy를 구현하지 않는다. 현재 GPU
KV cache는 여전히 원본 block을 갖고 있으므로, 일반 실행에서 recovery write는
대부분 no-op에 가깝다. 따라서 M3 validation에는 optional fault-injection
restore test를 포함해 "복구 copy가 실제로 작동한다"는 것을 별도로 증명한다.

## Design Decisions

M3의 첫 구현 결정은 다음과 같이 둔다.

| Area | M3 Decision |
|---|---|
| Recovery unit | whole vLLM physical KV block |
| Selection policy | block-level top-k from existing score result |
| Copy timing | synchronous copy immediately before attention |
| Hook point | `unified_attention_with_output(...)` before `self.impl.forward(...)` |
| Backup key | `CPUBackupKey(layer_name, physical_block_id)` |
| Backup payload | semantic CPU fp16 tensor `[2, block_size, num_kv_heads, head_dim]` |
| Target payload | `kv_cache[:, physical_block_id]` converted to target dtype/device |
| Validation scope | single request / no preemption / FlashAttention |

These are prototype decisions, not production API commitments.

## Module Boundary

Scoring and recovery should be separate modules.

Recommended shape:

```text
vllm/v1/mixed_precision_recovery/scoring.py
  scoring backends
  DigestScoreResult
  TorchQuestScorer
  QuestCudaScorer

vllm/v1/mixed_precision_recovery/recovery.py
  recovery target selection
  CPU backup lookup
  KV cache materialization
  RecoveryResult

vllm/v1/mixed_precision_recovery/sidecar.py
  orchestration
  config gating
  debug event recording
```

Important performance rule:

```text
Do not recompute scores for recovery.
Do not convert large score tensors or digest tensors to CPU just to cross the
scoring/recovery module boundary.
Pass the existing DigestScoreResult and physical block id list to recovery.
Convert only the small selected block id list for debug/materialization.
```

The module split itself should not add meaningful overhead. The avoidable
overhead comes from duplicated scoring, repeated top-k/candidate packing, or
GPU-to-CPU synchronization such as `.cpu().tolist()` on large tensors.

## Target Files

Likely implementation files:

| File | Role |
|---|---|
| `vllm/model_executor/layers/attention/attention.py` | attention-before-forward recovery hook |
| `vllm/v1/mixed_precision_recovery/config.py` | recovery flags |
| `vllm/v1/mixed_precision_recovery/sidecar.py` | orchestration and debug events |
| `vllm/v1/mixed_precision_recovery/recovery.py` | recovery policy/materialization implementation |
| `vllm/v1/mixed_precision_recovery/cpu_backup.py` | backup lookup interface |
| `tests/v1/mixed_precision_recovery/test_scoring.py` | focused unit tests, or split later |

Possible new test file:

```text
tests/v1/mixed_precision_recovery/test_recovery.py
```

## Step 3.0: M2 Recovery Readiness Check

Before implementation, verify the current M2 state still satisfies:

```text
CPU backup put/get returns full semantic K/V block
release_blocks removes CPU backup and stale digest state
score_candidate_block_ids are physical block ids
top-k block ids are physical block ids
kv_cache layout is [2, num_blocks, block_size, num_kv_heads, head_dim]
```

Completion:

```text
focused pytest passes
M2 CPU backup get/release tests pass
current checkout has no unexpected MPR test regressions
```

## Step 3.1: Recovery Config Flags

Add disabled-by-default recovery flags.

Candidate flags:

```text
VLLM_MPR_RECOVERY_ENABLE
VLLM_MPR_RECOVERY_TOPK
VLLM_MPR_RECOVERY_POLICY
VLLM_MPR_RECOVERY_THRESHOLD
VLLM_MPR_RECOVERY_TEST_MUTATE
```

Initial defaults:

```text
recovery_enable = false
recovery_topk = MPR topk or a small explicit default
recovery_policy = topk_block
recovery_threshold = 0.0
recovery_test_mutate = off
```

Rules:

```text
VLLM_MPR_ENABLE=0 -> no MPR recovery import or work
VLLM_MPR_RECOVERY_ENABLE=0 -> scoring/backup can run, recovery write does not
test mutation is always off by default
threshold-based selection is configured but not used until recovery.py consumes
the policy in Step 3.2
```

Completion:

```text
MPRConfig parses recovery flags
default config leaves recovery disabled
env helper tests cover the new flags
```

## Step 3.2: Recovery Module v0

Create a recovery module that consumes scoring output rather than recomputing
scores.

Sketch:

```python
@dataclass(frozen=True)
class RecoveryResult:
    selected_block_ids: list[int]
    recovered_block_ids: list[int]
    missing_backup_block_ids: list[int]
    skipped_block_ids: list[int]
    recovered_bytes: int
    copy_wall_seconds: float


class BlockRecoveryManager:
    def recover(
        self,
        *,
        score_result,
        physical_block_ids: list[int],
        kv_cache,
        cpu_backup_store,
        layer_name: str,
        topk: int,
    ) -> RecoveryResult:
        ...
```

Initial selection:

```text
Use score_result.block_scores.
Select top-k block ids from physical_block_ids.
Recover whole blocks only.
```

Completion:

```text
unit test selects expected block ids
unit test copies backup payload into target kv_cache block
unit test missing backup is counted and skipped
```

## Step 3.3: Sidecar Recovery API

Add a sidecar method that can be called before attention.

Sketch:

```python
RecoverySidecar.recover_before_attention(
    layer_name,
    query,
    attn_metadata,
    kv_cache,
    block_size,
) -> None
```

This method should:

```text
respect config.enabled and config.recovery_enabled
reuse the existing query/scoring candidate path
produce or reuse DigestScoreResult
call recovery.py with score_result and physical block ids
record recovery debug event
```

Open implementation choice:

```text
Option A: make observe_query return an internal score/debug result
Option B: keep observe_query score-only and add a shared helper used by both
          observe_query and recover_before_attention
```

Recommended first choice:

```text
Option B, because it preserves the score-only API while avoiding duplicate
scoring logic.
```

Completion:

```text
score-only behavior remains available
recovery path does not duplicate scoring implementation
debug counters distinguish score_estimated from recovery_materialized
```

## Step 3.4: Attention Hook

Hook location:

```text
vllm/model_executor/layers/attention/attention.py
  unified_attention_with_output(...)
```

Current order:

```text
_maybe_observe_mpr_query(...)
self.impl.forward(...)
```

M3 order:

```text
_maybe_observe_mpr_query_or_recover(...)
self.impl.forward(...)
```

The hook must pass:

```text
layer_name
query
attn_metadata
kv_cache
block_size
```

Completion:

```text
recovery materialization happens before attention forward
dummy run still skips MPR work
env-off path remains early-return
```

## Step 3.5: Materialization Semantics

Initial copy operation:

```text
backup_cpu = cpu_backup_store.get(CPUBackupKey(layer_name, block_id))
target = kv_cache[:, block_id]
target.copy_(backup_cpu.to(device=target.device, dtype=target.dtype))
```

Validation checks:

```text
kv_cache.ndim == 5
kv_cache.shape[0] == 2
block_id is in range
backup shape matches target shape
backup dtype is fp16
```

Unsupported or missing cases should be recorded as skipped/missing instead of
silently corrupting KV cache.

Completion:

```text
copy writes exactly the selected target block
other blocks remain unchanged
copy byte count and wall time are reported
```

## Step 3.6: Debug Output

Add recovery debug events.

Candidate event:

```text
recovery_materialized
```

Fields:

```text
layer_name
recovery_policy
recovery_topk
score_candidate_block_ids
recovery_selected_block_ids
recovered_block_ids
missing_backup_block_ids
skipped_block_ids
recovered_bytes
recovery_copy_wall_ms
kv_cache_shape
cpu_backup_block_count
cpu_backup_bytes
```

Possible skip event:

```text
recovery_skipped
```

Skip reasons:

```text
recovery_disabled
no_score_candidates
no_digest_blocks
missing_backup
unsupported_kv_cache_layout
shape_mismatch
```

Completion:

```text
JSONL validator accepts recovery events
generation smoke can show recovery_materialized count > 0
```

## Step 3.7: Validation

Validation should include unit tests and at least one smoke path.

Unit tests:

```text
Recovery module selects top-k block ids from block_scores.
CPU backup payload is copied into kv_cache[:, block_id].
Missing backup is counted and does not mutate target.
Recovery disabled does not mutate kv_cache.
Sidecar release still removes recovered block backup state.
```

Fault-injection restore test:

```text
1. Create CPU backup for a known block.
2. Mutate the target kv_cache block, for example zero it.
3. Run recovery.
4. Assert kv_cache[:, block_id] equals backup payload converted to target dtype.
```

Generation smoke:

```text
VLLM_MPR_ENABLE=1
VLLM_MPR_CPU_BACKUP=1
VLLM_MPR_RECOVERY_ENABLE=1
single request
FlashAttention
fixed decode length
debug JSONL contains recovery_materialized
```

Completion:

```text
focused pytest passes
py_compile passes
git diff --check passes
single-request smoke completes when CUDA is available
```

## Step 3.8: Milestone 3 Completion Report

When M3 is complete, update:

```text
milestone_3_progress_log.md
vllm_integration_notes.md
milestone_1_results.md or a new milestone_3_results.md if useful
```

Record:

```text
implemented recovery API
hook point
recovery policy
debug schema
validation commands
known unsupported cases
decision: proceed to M4 / extend M3
```

## Proposed Implementation Order

1. Run focused M2 readiness tests.
2. Add recovery config flags.
3. Add `recovery.py` with `RecoveryResult` and block materialization tests.
4. Refactor sidecar scoring path enough to share score results without
   recomputing.
5. Add `RecoverySidecar.recover_before_attention(...)`.
6. Add attention-before-forward hook.
7. Add debug event fields and validator support.
8. Add fault-injection restore test.
9. Run focused tests and CUDA smoke when available.
10. Record progress and validation results in markdown.

## Out of Scope for Milestone 3

```text
mixed precision tier policy
int8/fp8/lower precision recovery
per-KV-head or token-level recovery
async H2D copy
pinned CPU memory readiness tracking
DMA/cuMemcpyBatchAsync path
fixed CPU capacity policy
multi-request correctness
preemption cleanup
sliding-window skipped-block cleanup
real eviction/offload integration
mixed-dtype attention kernel support
production CUDA Graph compatibility
```

## Main Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Recovery appears to do nothing | M2 does not evict or corrupt GPU KV, so copying backup back may be no-op | Add fault-injection restore test |
| Duplicate scoring overhead | Recovery could accidentally recompute score after debug scoring | Shared score helper and `DigestScoreResult` handoff |
| CPU/GPU sync overhead | Top-k/debug conversions can synchronize hot path | Keep tensors on GPU; convert only selected ids/debug fields |
| Stale physical block ids | M2 keys are physical-block based | Keep single-request/no-preemption scope; release before reuse |
| Shape/layout mismatch | M3 assumes FlashAttention KV layout | Explicit layout checks and skip/debug unsupported cases |
| Attention correctness regression | KV cache is mutated before attention | Recovery disabled by default; unit test exact target mutation |

## Decision Gate

Milestone 3 끝에서 다음 중 하나를 선택한다.

| Decision | Condition |
|---|---|
| Proceed to Milestone 4 | fp16 block recovery materializes before attention and validation passes; next work is cleanup/optimization before mixed precision |
| Extend Milestone 3 | unit tests pass but generation smoke/debug recovery is incomplete |
| Rework M2 backup layout | semantic CPU fp16 backup is insufficient or too slow for recovery |
| Rework lifecycle/keying | physical block id reuse causes stale recovery risk |
| Rework hook point | attention-before-forward hook cannot safely materialize KV |

## Expected Outcome

Milestone 3이 끝나면 우리는 아직 mixed precision을 하지 않는다. 대신 다음
질문에 답할 수 있어야 한다.

```text
At each decode step, can selected vLLM physical KV blocks be materialized from
the MPR CPU fp16 backup into the GPU KV cache before attention?
```

이 답이 안정적으로 나오면 Milestone 4에서 현재 recovery skeleton의 overhead를
덜어내고 측정/구조를 정리한다. 그 다음 Milestone 5에서 score를 precision tier로
매핑하고 lower-precision backup/materialization policy를 붙일 수 있다.
