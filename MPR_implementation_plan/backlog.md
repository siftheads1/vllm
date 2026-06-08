# MPR Follow-up Backlog

This file tracks optimization and cleanup items that should survive milestone
progress logs and conversation compaction.

## Milestone 1 / Scoring Follow-ups

```text
CUDA Graph support for MPR scoring paths
remove or relax eager-mode-only assumptions
Quest CUDA scorer bf16 support instead of fp16-only operation
ArkVale-style scoring backend for ablation/evaluation
top-k/recovery policy integration after score-only validation
```

## Milestone 2 / CPU Backup Follow-ups

```text
replace semantic tensor backup with raw/pinned/DMA-friendly layout if needed
support pinned CPU memory and non_blocking copy with readiness tracking
evaluate cuMemcpyBatchAsync / DMA copy path for production-style backup
make backup dtype configurable or preserve source dtype for bf16 sources
add fixed CPU capacity policy: max blocks, max bytes, pool, skip/fail behavior
replace direct KVCacheManager -> MPR hook with cleaner connector/event/lifecycle observer
cover preemption cleanup
cover sliding-window remove_skipped_blocks cleanup
add generation/request/logical/block-hash keying for block reuse safety
expand validation beyond single-request/no-preemption
optimize observe_kv_write per-step bookkeeping; current Python path performs
  CUDA scalar/list transfers such as item(), cpu().tolist(), and unique()
separate CPU backup copy cost from MPR write-observation overhead in future
  benchmarks
```

## Milestone 5 / Recovery Cleanup and Optimization

```text
INT4 packed codec is not a Post-M5 backlog item anymore. It is promoted to
  Milestone 4.5 so M5 can optimize the full fp16/int8/int4/skip tier set.
optimize BlockRecoveryManager.materialize_blocks(...): current M3 v0 copies one
  selected block at a time for correctness/debug simplicity; threshold-based
  recovery may select many blocks and make the per-block Python loop costly
evaluate batched materialization:
  stack selected CPU backup tensors
  transfer one batched tensor to GPU
  write into kv_cache with index_copy_(dim=1, ...)
measure per-block loop vs batched copy for small top-k and large threshold
  selections before changing the default path
refactor RecoverySidecar orchestration once Milestone 3 semantics stabilize:
  split scoring context, recovery materialization, debug event emission, CPU
  backup lifecycle, and validation-only fault injection into clearer modules
keep production recovery entrypoints separate from test-only mutation/fault
  injection paths
separate MPR debug-event measurement from real PCIe counter measurement:
  use recovered_bytes/recovery_copy_wall_ms for code-level copy payload
  use dmon/nsys only as external traffic/profiling evidence
reduce validation-only mutation machinery from normal production hot path
make latency benchmark report backup/scoring/recovery components separately
Step 4.4/4.5 CPU backup / eager low-precision payload creation optimization is
  now a first-class Milestone 5 action item. See:
  MPR_implementation_plan/milestones/milestone_5/action_plan.md
  Step 5.3: CPU Backup and Low-Precision Payload Creation Optimization
```

## Production Integration Questions

```text
Should MPR remain a sidecar, become a vLLM connector, or integrate with kv_offload?
Should final recovery keys be physical, generation-based, request/logical, or hash-based?
Should CPU backup be semantic fp16, source-dtype preserving, or raw bytes?
Which scoring backends are needed for ablation: Quest, ArkVale, DiffKV-inspired, learned?
```
