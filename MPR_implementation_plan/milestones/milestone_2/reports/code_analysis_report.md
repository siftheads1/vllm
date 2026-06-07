# Milestone 2 Code Analysis Report

Milestone 2 is about CPU fp16 backup, not recovery yet.

The target from the mixed precision plan is:

```text
GPU KV block finalized
  -> save CPU fp16 backup in sidecar
  -> map vLLM block id to CPU backup location
  -> release backup when request/block lifecycle releases the block
```

This report summarizes the relevant vLLM code paths before choosing the M2
design.

## Files Read

Simple CPU offload:

```text
vllm/v1/simple_kv_offload/worker.py
vllm/v1/simple_kv_offload/manager.py
vllm/v1/simple_kv_offload/metadata.py
vllm/v1/simple_kv_offload/copy_backend.py
vllm/v1/simple_kv_offload/cuda_mem_ops.py
```

General KV offload:

```text
vllm/v1/kv_offload/base.py
vllm/v1/kv_offload/cpu/manager.py
vllm/v1/kv_offload/cpu/gpu_worker.py
vllm/v1/kv_offload/cpu/common.py
```

Block lifecycle:

```text
vllm/v1/core/block_pool.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/sched/scheduler.py
```

## SimpleCPUOffload Worker Layout

`SimpleCPUOffloadWorker.register_kv_caches(...)` receives the per-layer GPU KV
caches from the worker and builds block-level int8 views over their raw storage.

Key behavior:

```text
input:
  kv_caches: dict[str, torch.Tensor]

dedup:
  multiple layers may share one raw storage
  worker deduplicates by untyped_storage().data_ptr()

GPU view:
  raw storage is converted to int8
  each unique storage becomes [num_blocks, block_bytes]

CPU view:
  one CPU tensor per unique GPU storage
  shape = [num_cpu_blocks, block_bytes]
  dtype = int8
  optionally pinned with cudaHostRegister
```

Important layout detail:

```text
FlashAttention KV cache shape is usually:
  [2, num_blocks, block_size, num_kv_heads, head_dim]

The worker detects outer segment dimensions whose byte stride exceeds one
block's byte size. For FlashAttention, this splits K and V into separate
unique GPU views such as:
  layer_name.0
  layer_name.1
```

Implication for MPR:

```text
The existing simple offload path copies whole raw KV block bytes, not a semantic
[K,V,block,num_heads,head_dim] object.

This is useful for transfer machinery, but MPR Milestone 2 may prefer a simpler
sidecar-owned semantic fp16 backup first:
  backup[(layer_name, physical_block_id)] = kv_cache[:, block_id].cpu()
```

That semantic path is easier to validate and avoids immediately adopting
connector-level scheduler semantics.

## SimpleCPUOffload Copy Path

The copy backend is asynchronous:

```text
DmaCopyBackend
  background Python thread
  queues copy jobs
  calls cuMemcpyBatchAsync / hipMemcpyBatchAsync
  records torch.Event on load/store stream
```

`cuda_mem_ops.copy_blocks(...)` builds a batched copy:

```text
for each tensor view and each block id:
  src = src_base + src_block_id * bytes_per_block
  dst = dst_base + dst_block_id * bytes_per_block
  size = bytes_per_block
```

Implication for MPR:

```text
For production-quality CPU backup, pinned CPU memory and async DMA are the
right direction.

For the first M2 implementation, synchronous or simple non_blocking Tensor.copy_
is probably easier to reason about. We can benchmark and replace it later with
the existing DMA machinery if decode critical path overhead is too high.
```

## SimpleCPUOffload Scheduler Behavior

`SimpleCPUOffloadScheduler` is scheduler-side and owns CPU block allocation,
load/store metadata, and completion processing.

Important state:

```text
cpu_kv_cache_config
cpu_block_pool
_reqs_to_load
_reqs_to_store
_store_event_to_blocks
_store_event_to_reqs
_in_flight_store_gpu_blocks
```

Store preparation:

```text
build_connector_meta(...)
  -> prepare_store_specs(...)
  -> _prepare_eager_store_specs(...) or _prepare_lazy_store_specs(...)
  -> returns store_gpu_blocks and store_cpu_blocks
```

Eager store mode:

```text
tracks request block ids from scheduler_output via yield_req_data(...)
stores only blocks with confirmed KV data
confirmed_tokens = request.num_computed_tokens - request.num_output_placeholders
aligned_tokens = confirmed_tokens // block_size * block_size
does not store current-step data until the next step
may miss the last full block if a request finishes in the same step
```

Store completion:

```text
worker reports completed store events
scheduler waits for all workers
_process_store_completion(...)
  inserts CPU blocks into prefix-cache map
  frees CPU/GPU touch refs
```

Load path:

```text
CPU block hits are found by block hash
load blocks are touched/pinned until load completes
GPU blocks are touched/pinned during async load
```

Implication for MPR:

```text
simple_kv_offload is more than a raw copy helper. It is a prefix-cache/offload
connector with hash-based CPU cache semantics.

MPR Milestone 2's stated need is simpler:
  save fp16 backup for the physical blocks observed by the sidecar

Directly reusing SimpleCPUOffloadScheduler would pull in request hash/cache-hit
semantics we do not need yet.
```

## General kv_offload Abstraction

`vllm/v1/kv_offload/base.py` defines a more general abstraction:

```text
OffloadKey = block_hash + group_idx
OffloadingManager
  lookup
  prepare_load
  touch
  complete_load
  prepare_store
  complete_store
```

The CPU implementation manages:

```text
CPUOffloadingManager
  cache policy: LRU or ARC
  key -> BlockStatus
  block ref counts
  ready/not-ready store states
  eviction events
```

The worker handler supports:

```text
GPULoadStoreSpec
CPULoadStoreSpec
block_size_factor for CPU blocks larger than GPU blocks
mmap-backed shared offload regions
async stream/event transfers
```

Implication for MPR:

```text
The general kv_offload stack is the better long-term integration point if MPR
becomes a first-class offload/recovery system.

For M2, it is probably too large to adopt directly because MPR currently keys
state by (layer_name, physical_block_id), not block hash/offload key.
```

## vLLM Block Lifecycle

### BlockPool

`BlockPool` owns all `KVCacheBlock` objects:

```text
blocks: list[KVCacheBlock]
free_block_queue
cached_block_hash_to_block
```

Allocation:

```text
get_new_blocks(num_blocks)
  pops from free queue
  maybe evicts cached hash metadata
  increments ref_cnt
```

Free:

```text
free_blocks(ordered_blocks)
  decrements ref_cnt
  appends ref_cnt == 0 blocks to free queue
```

Touch:

```text
touch(blocks)
  removes free blocks from free queue if needed
  increments ref_cnt
```

Eviction/reuse:

```text
_maybe_evict_cached_block(block)
  removes hash mapping
  block.reset_hash()
```

Implication for MPR:

```text
Physical block ids are reusable.
Any MPR backup keyed only by physical_block_id becomes stale when that block id
is reused for a different request/content.

M2 must either:
  observe/request lifecycle cleanup before reuse, or
  associate backup entries with a generation/version/hash in addition to
  physical_block_id.
```

### KVCacheManager / SingleTypeKVCacheManager

Request finish/preemption path:

```text
KVCacheManager.free(request)
  -> coordinator.free(request_id)
  -> SingleTypeKVCacheManager.free(request_id)
  -> block_pool.free_blocks(reversed(req_blocks))
```

Skipped blocks:

```text
remove_skipped_blocks(...)
  frees blocks outside the attention window
  replaces request block entries with null_block
```

New block ids:

```text
KVCacheManager.take_new_block_ids()
  drains newly allocated block ids for zeroing
```

Implication for MPR:

```text
The cleanest lifecycle signal is close to the KV manager / scheduler free path,
not the attention hook.

The current M1 sidecar sees writes and queries, but it does not know when vLLM
frees or reuses blocks. M2 needs a lifecycle hook or a conservative validation
scope that avoids reuse/preemption.
```

### Scheduler

Finish path:

```text
Scheduler._free_request(...)
  -> _connector_finished(request)
  -> encoder_cache_manager.free(request)
  -> maybe _free_blocks(request)

Scheduler._free_blocks(request)
  -> kv_cache_manager.free(request)
  -> del self.requests[request_id]
```

Connector finish hook:

```text
_connector_finished(request)
  -> kv_cache_manager.remove_skipped_blocks(...)
  -> block_ids = kv_cache_manager.get_block_ids(request_id)
  -> connector.request_finished(...) or request_finished_all_groups(...)
```

Preemption path:

```text
Scheduler._preempt_request(...)
  -> kv_cache_manager.free(request)
  -> encoder_cache_manager.free(request)
  -> request.status = PREEMPTED
  -> request.num_computed_tokens = 0
```

Implication for MPR:

```text
Connectors already get a request_finished hook with block ids before free.
MPR sidecar is not currently a connector, so it does not get this signal.

For M2, either add a small MPR lifecycle hook near Scheduler._free_request /
_preempt_request / KVCacheManager.free, or integrate MPR backup with a connector
path. The small hook is likely lower risk for the first CPU backup prototype.
```

## M2 Design Options Exposed by Code

### Option A: Sidecar Semantic Backup Store

Backup directly in the MPR sidecar:

```text
on full block digest creation:
  backup[(layer_name, physical_block_id)] =
      kv_cache[:, physical_block_id].detach().to("cpu", dtype=torch.float16)
```

Pros:

```text
simple to implement
same timing as digest creation
easy to inspect and validate
does not require scheduler connector adoption
works for current single-request/no-preemption scope
```

Cons:

```text
synchronous GPU->CPU copy may hit decode critical path
no CPU pool/capacity policy unless we add one
needs explicit lifecycle hook to avoid stale entries
not directly using vLLM's mature async DMA path
```

### Option B: Sidecar Raw Block Pool Inspired by SimpleCPUOffloadWorker

Build MPR-owned CPU tensors with raw `[num_cpu_blocks, block_bytes]` layout and
copy block bytes.

Pros:

```text
closer to vLLM offload mechanics
predictable CPU memory usage
can later use pinned memory and batch DMA
```

Cons:

```text
more code up front
must map layer/segment names carefully
semantic validation is harder
still needs lifecycle hook
```

### Option C: Reuse kv_offload / SimpleCPUOffload Connector

Adopt vLLM connector/offloading abstractions.

Pros:

```text
best long-term fit for scheduler-aware offload/reload
already handles async transfer, events, ref counts, CPU pool, preemption flush
```

Cons:

```text
large integration surface
hash/prefix-cache semantics do not match MPR's current physical-block sidecar
state
probably too much for first M2 step
```

## Preliminary Recommendation

Start M2 with Option A:

```text
sidecar-owned semantic CPU fp16 backup
single-request/no-preemption validation scope
backup at the same moment full-block digest is created
debug counters and memory accounting
explicit lifecycle hook added before claiming M2 complete
```

Then, after correctness and lifecycle are visible, decide whether to:

```text
keep semantic backup for M3 fp16 recovery smoke
or replace the storage backend with a raw block pool / async DMA path
```

This keeps M2 aligned with the milestone goal while avoiding premature adoption
of vLLM's full offload connector stack.

## Decisions Needed Next

Before implementation, decide:

```text
1. Backup layout:
   semantic tensor copy [2, block_size, num_kv_heads, head_dim]
   vs raw int8 block-byte copy

2. Copy timing:
   synchronous copy at full-block digest creation
   vs async/pinned copy with stream/event tracking

3. CPU capacity policy:
   unbounded dict for first smoke
   vs fixed CPU block pool with eviction/failure behavior

4. Lifecycle hook:
   where to call sidecar.release_blocks(...)
   Scheduler._free_request / _preempt_request
   vs KVCacheManager.free / BlockPool reuse

5. Backup key:
   (layer_name, physical_block_id)
   vs including request_id/logical_block_idx/block generation/hash

6. Validation target:
   single request only with no preemption/reuse
   vs trying to cover block free/reuse immediately
```

## M2 Working Decisions

These decisions are for the first Milestone 2 implementation. They are local
prototype decisions, not production API commitments.

### 1. Backup Layout

Use semantic CPU fp16 backup for the first implementation:

```text
CPU backup payload:
  kv_cache[:, physical_block_id].detach().to("cpu", dtype=torch.float16)

expected semantic shape:
  [2, block_size, num_kv_heads, head_dim]
```

This preserves K/V block meaning and makes validation easier than raw byte
storage. The fp16 conversion is an explicit prototype policy, not a lossless
claim for bf16 sources. Future work may preserve source dtype or make backup
dtype configurable.

The storage method should be hidden behind a small internal interface so the
implementation can later become raw, pinned, async, or DMA-backed without
spreading storage details through the sidecar.

### 2. Copy Timing

Start with synchronous copy at full-block creation time:

```text
full block detected
  -> create digest
  -> synchronously copy K/V block to CPU backup store
  -> continue
```

This is easiest to validate and avoids async readiness semantics in the first
M2 step. The copy policy should remain behind the backup-store implementation
so later pinned/non_blocking/DMA copies can replace it.

Known caveat:

```text
Synchronous GPU-to-CPU copy may be on the decode critical path.
M2 should record put count, backup bytes, and copy wall time so the overhead
can be measured before changing copy strategy.
```

### 3. Capacity Policy

Use an unbounded store for the first smoke implementation. The concrete
semantic store may use a dict internally, but the sidecar must not depend on
that dict shape.

```text
RecoverySidecar
  -> CPUBackupStore interface
      -> SemanticCPUBackupStore internal implementation
```

No eviction should be introduced in the first M2 path. Optional max-blocks or
max-bytes guardrails can be added later, with skip/fail counters rather than
silent eviction.

### 4. Lifecycle Cleanup

The lifecycle id passed into MPR cleanup is the GPU KV physical block id, not a
CPU block id.

```text
vLLM free/reuse signal
  -> GPU physical block ids
  -> RecoverySidecar.release_blocks(...)
  -> cleanup MPR state for those block ids
```

The sidecar should release all block-id keyed state:

```text
_block_offsets
_digest_cache
_quest_metadata_stores or their stale-safe equivalent
_cpu_backup_store
```

The CPU backup store may map the external GPU block id to any internal CPU
handle, tensor, or pool slot. That internal CPU id is a backend detail.

### 5. Backup Key

Use an internal key wrapper to avoid spreading raw tuple keys:

```text
CPUBackupKey:
  layer_name
  physical_block_id
  optional request_id
  optional logical_block_idx
  optional generation/hash
```

For the first M2 implementation, the actual identity is:

```text
layer_name + GPU physical block id
```

Correctness therefore depends on cleanup before physical block id reuse. The
wrapper is local scaffolding so future production code can switch to generation,
request/logical, block-hash, or vLLM offload keys without rewriting every call
site.

### 6. Validation Scope

Validate M2 on single-request, no-preemption decode only.

Expected validation signals:

```text
during decode:
  cpu_backup_put_count > 0
  cpu_backup_block_count > 0
  cpu_backup_bytes > 0

after request/block cleanup:
  cpu_backup_release_count > 0
  released backup entries disappear
```

Multi-request ownership, preemption, sliding-window skipped-block cleanup,
capacity eviction, and async copy readiness are follow-up work.

## Lifecycle Hook Target Narrowing

The M2 cleanup signal needs to fire before GPU KV physical block ids can be
reused:

```text
GPU physical block ids about to be freed
  -> RecoverySidecar.release_blocks(...)
  -> existing vLLM free path continues
```

The relevant code paths are:

```text
normal finish:
  Scheduler._free_request(...)
    -> Scheduler._free_blocks(request)
      -> KVCacheManager.free(request)
        -> KVCacheCoordinator.free(request_id)
          -> SingleTypeKVCacheManager.free(request_id)
            -> BlockPool.free_blocks(...)

preemption:
  Scheduler._preempt_request(...)
    -> KVCacheManager.free(request)

connector-delayed free:
  Scheduler._free_request(...)
    -> connector may delay block free
    -> later Scheduler._free_blocks(request)

sliding-window skipped blocks:
  SingleTypeKVCacheManager.remove_skipped_blocks(...)
    -> BlockPool.free_blocks(...)
```

`Scheduler._free_blocks(request)` is the best first M2 target for finished
requests because:

```text
it runs immediately before KVCacheManager.free(request)
it is not called when a connector delays KV block release
it is also used later when delayed connector send/recv completes
it can query KVCacheManager.get_block_ids(request.request_id) before free
it keeps the MPR-specific hook out of BlockPool and KVCacheManager core code
```

For preemption, `_preempt_request(...)` bypasses `_free_blocks(...)` and calls
`KVCacheManager.free(request)` directly. If M2 wants to install the preemption
cleanup path now, the least intrusive design is a scheduler-local helper:

```text
Scheduler._release_mpr_blocks(request, reason)
  -> block_ids = KVCacheManager.get_block_ids(request.request_id)
  -> flatten KV cache groups
  -> get_mpr_sidecar().release_blocks(block_ids, reason=reason)
```

Then call it before the existing free operation in:

```text
Scheduler._free_blocks(request)
Scheduler._preempt_request(request, timestamp)
```

For the current M2 validation scope, single-request/no-preemption only,
`Scheduler._free_blocks(request)` is enough to validate request-finish cleanup.
Preemption and sliding-window `remove_skipped_blocks(...)` cleanup remain
follow-up unless explicitly pulled into the M2 validation target.
