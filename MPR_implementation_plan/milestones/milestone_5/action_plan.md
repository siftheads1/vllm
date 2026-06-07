# Milestone 5 Action Plan: Recovery Cleanup and Optimization

Milestone 5 is the optimization/cleanup phase after mixed-precision tiering.

## Goal

Keep the Milestone 4 mixed-precision recovery skeleton, but remove avoidable
overhead and clarify the code structure before broadening the serving scope.

Expected Milestone 4 conclusion:

```text
mixed-precision tiering: pass
fp16 recovery semantics: preserved
low-precision backup/materialization: observed
overhead: likely too high for production direction
```

## Non-goals

```text
do not add a mixed-dtype attention kernel unless it is explicitly chosen
do not broaden to full multi-request/preemption serving yet unless needed for cleanup
do not change precision policy semantics while measuring overhead
```

## Step 5.1: Measurement Baseline

Record current M4 overhead before changing the implementation.

Measure:

```text
baseline generation latency
MPR backup-only latency
MPR scoring-only latency
MPR fp16 recovery latency
MPR mixed-tier recovery latency
recovered_bytes / generated token
recovery_copy_wall_ms
optional dmon/nsys PCIe evidence
```

Success criteria:

```text
one repeatable command set exists
results separate backup/scoring/fp16-recovery/mixed-tier costs
mixed-precision semantic smoke still passes
```

## Step 5.2: Remove Validation-only Cost from Production Path

Separate test/fault-injection machinery from the normal recovery API.

Targets:

```text
production recover_before_attention remains mutation-free
validation-only mutation code is isolated
debug fields for fault injection do not add work when disabled
attention hook branch remains simple
```

Success criteria:

```text
unit tests prove production entrypoint ignores test mutation flags
fault-injection smoke still works via explicit validation wrapper
```

## Step 5.3: Recovery Materialization Optimization

Current M3/M4 paths copy selected blocks one by one in Python.

Investigate:

```text
small top-k path: per-block loop may be acceptable
large threshold path: batched transfer likely needed
stack selected CPU blocks -> one H2D transfer -> index_copy_(dim=1)
pinned CPU memory and non_blocking copy
cuMemcpyBatchAsync / DMA-friendly layout feasibility
```

Success criteria:

```text
materialization remains semantically identical
large threshold recovery reduces Python-loop overhead
debug reports bytes/block counts consistently
```

## Step 5.4: Scoring and Debug Hot-path Cleanup

Reduce avoidable synchronization and CPU conversion.

Targets:

```text
avoid item()/cpu().tolist() on hot path except debug-limited events
avoid duplicate score estimation
avoid packing/repacking digest metadata when persistent packed metadata applies
separate debug sampling from required recovery state
```

Success criteria:

```text
score/recovery behavior unchanged
focused tests pass
latency breakdown improves or clearly explains remaining overhead
```

## Step 5.5: Sidecar Refactor Boundary

RecoverySidecar is currently carrying too many responsibilities.

Candidate split:

```text
ScoringController
CPUBackupController
RecoveryController
PrecisionPolicyController
DebugEventEmitter
ValidationFaultInjector
```

Do this conservatively. The goal is to make later serving integration easier,
not to rewrite the whole sidecar.

Success criteria:

```text
public hook APIs stay stable
tests remain focused
fault injection cannot accidentally enter production path
precision tiering remains easy to inspect
```

## Step 5.6: Exit Criteria

Milestone 5 is complete when:

```text
M4 mixed-precision semantic smoke passes
recovery copy/latency metrics are reported
major avoidable Python/debug overhead has been removed or explicitly backlogged
materialization optimization path is either implemented or measured and deferred
code boundaries are clean enough to broaden serving integration
```

## Decision Gate

| Decision | Condition |
|---|---|
| Proceed beyond M5 | mixed-precision recovery remains correct and overhead/structure is acceptable |
| Continue M5 | correctness is fine but overhead is still dominated by obvious Python/debug work |
| Rework CPU backup layout | semantic tensor backup blocks pinned/batched copy or capacity policy |
| Rework recovery policy | threshold recovery selects too many blocks for practical decode |
