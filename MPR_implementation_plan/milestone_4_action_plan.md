# Milestone 4 Action Plan: Recovery Cleanup and Optimization

Milestone 4 is the optimization/cleanup phase inserted after Milestone 3.
Mixed-precision recovery is moved to Milestone 5.

## Goal

Keep the Milestone 3 semantic recovery skeleton, but remove avoidable overhead
and clarify the code structure before adding mixed-precision policies.

Current Milestone 3 conclusion:

```text
semantic recovery: pass
fault-injection recovery smoke: pass
CPU backup -> GPU KV materialization: observed
overhead: too high for production direction
```

## Non-goals

```text
do not add mixed-precision tiers yet
do not implement a mixed-dtype attention kernel yet
do not broaden to full multi-request/preemption serving yet unless needed for cleanup
```

## Step 4.1: Measurement Baseline

Record current M3 overhead before changing the implementation.

Measure:

```text
baseline generation latency
MPR backup-only latency
MPR scoring-only latency
MPR recovery latency
recovered_bytes / generated token
recovery_copy_wall_ms
optional dmon/nsys PCIe evidence
```

Success criteria:

```text
one repeatable command set exists
results separate backup/scoring/recovery costs
name_recall semantic smoke still passes
```

## Step 4.2: Remove Validation-only Cost from Production Path

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

## Step 4.3: Recovery Materialization Optimization

Current M3 v0 copies selected blocks one by one in Python.

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

## Step 4.4: Scoring and Debug Hot-path Cleanup

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

## Step 4.5: Sidecar Refactor Boundary

RecoverySidecar is currently carrying too many responsibilities.

Candidate split:

```text
ScoringController
CPUBackupController
RecoveryController
DebugEventEmitter
ValidationFaultInjector
```

Do this conservatively. The goal is to make M5 easier, not to rewrite the
whole sidecar.

Success criteria:

```text
public hook APIs stay stable
tests remain focused
fault injection cannot accidentally enter production path
```

## Step 4.6: Exit Criteria

Milestone 4 is complete when:

```text
M3 name_recall semantic smoke passes
recovery copy/latency metrics are reported
major avoidable Python/debug overhead has been removed or explicitly backlogged
materialization optimization path is either implemented or measured and deferred
code boundaries are clean enough to add mixed-precision tiers
```

## Decision Gate

| Decision | Condition |
|---|---|
| Proceed to Milestone 5 | M3 recovery remains correct and M4 overhead/structure is acceptable |
| Continue M4 | correctness is fine but overhead is still dominated by obvious Python/debug work |
| Rework CPU backup layout | semantic tensor backup blocks pinned/batched copy or capacity policy |
| Rework recovery policy | threshold recovery selects too many blocks for practical decode |
