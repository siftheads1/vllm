# Milestone 5 Action Plan: Recovery Cleanup and Optimization

Milestone 5 is the optimization/cleanup phase after Milestone 4 tiering and
Milestone 4.5 INT4 integration.

## Goal

Keep the Milestone 4/4.5 mixed-precision recovery skeleton, but remove
avoidable overhead and clarify the code structure before broadening the serving
scope.

Expected Milestone 4.5 conclusion:

```text
mixed-precision tiering: pass
INT4 packed recovery tier: pass
fp16 recovery semantics: preserved
low-precision backup/materialization: observed for int8 and int4
overhead: likely too high for production direction
```

## Non-goals

```text
do not add a mixed-dtype attention kernel unless it is explicitly chosen
do not broaden to full multi-request/preemption serving yet unless needed for cleanup
do not change precision policy semantics while measuring overhead
```

## Step 5.1: Measurement Baseline

Record current M4/M4.5 overhead before changing the implementation.

Measure:

```text
baseline generation latency
MPR backup-only latency
MPR scoring-only latency
MPR fp16 recovery latency
MPR mixed-tier recovery latency
MPR mixed-tier recovery latency with INT4
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

## Step 5.3: CPU Backup and Low-Precision Payload Creation Optimization

Step 4.4 deliberately used a synchronous eager fp16+int8 reference path for
correctness. Milestone 4.5 extends that correctness-first path to INT4. Optimize
or explicitly defer that backup-time cost before treating M4/M4.5 as a
production direction.

Current M4/M4.5 behavior:

```text
one GPU->CPU semantic fp16 copy creates the CPU fp16 backup payload
eager INT8BackupPayload is derived from that CPU fp16 payload
eager INT4BackupPayload is derived from that CPU fp16 payload when enabled
this avoids a second GPU->CPU copy
CPU int8 quantization still runs synchronously on the decode-side backup path
CPU int4 quantization and packing also run synchronously in the reference path
```

Investigate:

```text
keep/verify GPU->CPU copy duplication removal
pinned CPU memory and non_blocking GPU->CPU copy
dedicated copy stream plus payload readiness tracking
background CPU quantization worker for int8/int4 payload creation
GPU-side quantization with compressed low-precision payload + scale CPU transfer
lazy/on-the-fly int8/int4 payload creation instead of eager backup
```

Design options:

```text
eager_fp16_int8:
  M4 correctness-first reference mode
  simplest int8 recovery provider
  highest backup-time CPU quantization cost

eager_fp16_int8_int4:
  M4.5 correctness-first reference mode
  creates fp16, int8, and packed int4 payloads eagerly
  highest backup-time CPU quantization/packing cost

fp16_only + on-the-fly CPU quantization:
  stores one fp16 payload on CPU
  creates int8/int4 only when the recovery policy asks for that tier
  shifts quantization cost from backup time to recovery/fetch time

GPU quantization before CPU transfer:
  creates int8/int4+scale near the source KV block
  can reduce CPU transfer bytes for low-precision tiers
  requires GPU quant kernels and copy/readiness integration

background CPU quantization:
  keeps fp16 backup immediately available
  fills int8/int4 payloads asynchronously
  requires readiness state and fallback behavior when low-precision payloads
  are not ready
```

Success criteria:

```text
backup copy cost and int8/int4 quantization/packing cost are measured separately
default correctness behavior remains unchanged until a faster path is proven
fp16 recovery remains available even when low-precision payload creation is lazy/async
missing/not-ready int8/int4 payload behavior is explicit
debug stats distinguish fp16 payload bytes, int8 payload bytes, int4 packed
  bytes, scale bytes, actual CPU bytes, and effective transfer bytes
```

## Step 5.4: Recovery Materialization Optimization

Current M3/M4/M4.5 paths copy or materialize selected blocks one by one in
Python.

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

## Step 5.5: Scoring and Debug Hot-path Cleanup

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

## Step 5.6: Sidecar Refactor Boundary

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

## Step 5.7: Exit Criteria

Milestone 5 is complete when:

```text
M4/M4.5 mixed-precision semantic smoke passes
recovery copy/latency metrics are reported
backup copy/int8/int4 payload creation costs are measured and either optimized or deferred
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
