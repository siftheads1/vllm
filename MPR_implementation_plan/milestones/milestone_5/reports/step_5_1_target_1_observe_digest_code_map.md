# Step 5.1 Target 1 Code Map: Observe/Digest Base Overhead

Target 1 explains the overhead observed in `mpr_enable_only`, where
`VLLM_MPR_ENABLE=1` but CPU backup, scoring, recovery, and debug JSONL writing
are disabled.

Current status after the Step 5.2 counter-backend work:

```text
This report is the initial slot-path code map for Target 1. The later
counter-based observe path removes the always-paid slot scan / CPU-sync cost
for the single-request pure-decode non-boundary benchmark shape. The remaining
Target 1 profiling focus is now block-boundary digest creation and any fallback
from the counter path to the slot path.
```

Measured evidence from the first runtime summary:

```text
baseline decode mean:        25.00 ms
mpr_enable_only decode mean: 43.06 ms
increment:                  +18.07 ms
ratio:                       1.72x
```

This means the MPR observe path is not a cheap no-op. It still performs KV
write observation, slot/block bookkeeping, digest creation, and some debug
record preparation.

## Entry Points

KV cache update hook:

```text
vllm/model_executor/layers/attention/attention.py
  unified_kv_cache_update(...)
    attn_layer.impl.do_kv_cache_update(...)
    _maybe_observe_mpr_kv_write(...)
```

Important lines:

```text
attention.py:691  _maybe_observe_mpr_kv_write(...)
attention.py:699  if not envs.VLLM_MPR_ENABLE: return
attention.py:705  block_size = _infer_mpr_block_size(...)
attention.py:709  get_mpr_sidecar().observe_kv_write(...)
attention.py:787  unified_kv_cache_update(...)
attention.py:809  _maybe_observe_mpr_kv_write(...) after do_kv_cache_update
```

Query hook:

```text
vllm/model_executor/layers/attention/attention.py
  unified_attention_with_output(...)
    _maybe_observe_or_recover_mpr_query(...)
```

Important lines:

```text
attention.py:745  _maybe_observe_or_recover_mpr_query(...)
attention.py:760  sidecar = get_mpr_sidecar()
attention.py:761  recovery path if recovery_enabled
attention.py:780  sidecar.observe_query(...)
attention.py:837  unified_attention_with_output(...)
attention.py:854  _maybe_observe_or_recover_mpr_query(...) before attention forward
```

In `mpr_enable_only`, `observe_query()` enters but returns quickly because
`scoring_enabled=False`. The main Target 1 cost is expected to be the KV-write
hook, not query scoring.

## Main Hot Path: RecoverySidecar.observe_kv_write

File:

```text
vllm/v1/mixed_precision_recovery/sidecar.py
```

Important lines:

```text
sidecar.py:307  observe_kv_write(...)
sidecar.py:337  _should_record_layer_event(...)
sidecar.py:351  flat_slots = slot_mapping.detach().reshape(-1)
sidecar.py:360  (flat_slots < PAD_SLOT_ID).any().item()
sidecar.py:372  valid_slots = flat_slots[flat_slots != PAD_SLOT_ID]
sidecar.py:382  block_ids = valid_slots // block_size
sidecar.py:386  block_offsets = valid_slots % block_size
sidecar.py:387  block_ids.unique().detach().cpu().tolist()
sidecar.py:391  block_offsets.min().item()
sidecar.py:392  block_offsets.max().item()
sidecar.py:393  _observe_block_offsets(...)
sidecar.py:402  if should_record: _record(...)
```

Likely overhead sources:

```text
slot_mapping detach/reshape
PAD-slot filtering
GPU tensor operations on slot ids
GPU->CPU sync points:
  any().item()
  min().item()
  max().item()
  unique().detach().cpu().tolist()
Python-side list construction for unique block ids
debug record field preparation when should_record is true
```

Important nuance:

```text
VLLM_MPR_DEBUG_DIR unset prevents JSONL file writes, but it does not prevent
observe_kv_write from computing fields before _record() calls.
```

## Block Offset And Digest Path

File:

```text
vllm/v1/mixed_precision_recovery/sidecar.py
```

Important lines:

```text
sidecar.py:137   _block_offsets: dict[layer, dict[block, set[offset]]]
sidecar.py:138   _digest_cache: dict[layer, dict[block, BlockDigest]]
sidecar.py:1444  _observe_block_offsets(...)
sidecar.py:1486  key_cache = kv_cache[0]
sidecar.py:1490  layer_offsets = self._block_offsets.setdefault(...)
sidecar.py:1494  layer_digests = self._digest_cache.setdefault(...)
sidecar.py:1499  block_ids.detach().cpu().tolist()
sidecar.py:1500  block_offsets.detach().cpu().tolist()
sidecar.py:1501  Python loop over block id / offset pairs
sidecar.py:1518  offsets = layer_offsets.setdefault(block_id, set())
sidecar.py:1519  offsets.add(block_offset)
sidecar.py:1520  if len(offsets) == block_size and block_id not in layer_digests
sidecar.py:1523  summarize_key_block(key_cache[block_id], ...)
sidecar.py:1527  layer_digests[block_id] = self._to_block_digest(...)
sidecar.py:1531  _append_quest_metadata_digest(...)
sidecar.py:1536  _maybe_backup_kv_block(...)
```

Likely overhead sources:

```text
second GPU->CPU list conversion for all valid block ids and offsets
per-token Python loop
per-block Python set insertion
dict lookups by layer and physical block id
full-block digest creation
digest cache insertion
optional Quest metadata append when scoring_backend == quest_cuda
optional backup call, which returns early in mpr_enable_only
```

For `mpr_enable_only`, `_maybe_backup_kv_block()` returns because
`cpu_backup_enabled=False`, but all earlier offset and digest work still runs.

## Digest Computation

File:

```text
vllm/v1/mixed_precision_recovery/digest.py
```

Important lines:

```text
digest.py:53  summarize_key_block(...)
digest.py:77  _validate_key_block(...)
digest.py:79  raw_max = key_block.amax(dim=0)
digest.py:80  raw_min = key_block.amin(dim=0)
digest.py:93  centers = (raw_max + raw_min) / 2
digest.py:98  dists = (centers.unsqueeze(0) - key_block).abs().mean(dim=0)
```

Likely overhead sources:

```text
GPU reductions over a full key block
temporary tensors for centers and distances
ArkVale-style digest path performs more work than raw min/max
```

If scoring/recovery are disabled, digest creation may be unnecessary for
production-like `mpr_enable_only`. That is a strong optimization candidate.

## Debug And Counter Path

Files:

```text
vllm/v1/mixed_precision_recovery/sidecar.py
vllm/v1/mixed_precision_recovery/debug.py
```

Important lines:

```text
sidecar.py:1395  _should_record_layer_event(...)
sidecar.py:1414  layer index bookkeeping
sidecar.py:1423  event_counts[layer_name] += 1
sidecar.py:1431  dump_every gating
sidecar.py:1437  _shape_of(...)
sidecar.py:1796  _num_digest_blocks()
sidecar.py:1846  _record(...)
sidecar.py:1848  self.counters[event] += 1
sidecar.py:1849  self._debug_writer.write(...)
debug.py:31     MPRDebugWriter.write(...)
debug.py:32     if self._file is None: return
```

Likely overhead sources:

```text
counter bookkeeping even with debug_dir unset
layer index and event count bookkeeping
shape list construction
total digest count summation when record fields are built
dict(self.counters) construction before debug writer returns
```

Debug file I/O is not active when `VLLM_MPR_DEBUG_DIR` is unset, but record
preparation can still matter.

## Immediate Profiling Plan

Add targeted internal timing only after deciding to instrument Target 1.
Suggested probes:

```text
observe_kv_write total
_should_record_layer_event
slot_mapping flatten/filter
invalid negative slot check
block id / offset derivation
unique/min/max CPU sync conversions
_observe_block_offsets total
block_ids/block_offsets CPU list conversion
Python offset bookkeeping loop
summarize_key_block total
_append_quest_metadata_digest
_maybe_backup_kv_block early return / total
_record field construction/write path
```

Instrumentation should be behind an explicit debug/profiling flag so it does
not become another always-on hot-path cost.

## Optimization Candidates

High-confidence candidates:

```text
skip digest creation when scoring_enabled=False and recovery_enabled=False
skip Quest metadata append when scoring is disabled, even if scoring_backend is
  quest_cuda
avoid building observe_kv_write debug fields unless debug writer is active
avoid _num_digest_blocks() hot-path summation unless JSONL output needs it
```

Medium-risk candidates:

```text
replace Python set-per-block offset tracking with cheaper boundary-aware state
reduce duplicate GPU->CPU conversions by sharing one CPU slot/block snapshot
avoid unique().cpu().tolist() when the result is used only for debug output
use vLLM-owned block/slot metadata instead of recomputing from slot_mapping
```

Higher-risk candidates:

```text
move digest creation to an asynchronous/background path
create digests lazily only when scoring first needs them
replace ArkVale digest with cheaper raw_minmax for measurement modes
```

These candidates need explicit design confirmation before implementation
because they may affect scoring/recovery availability and correctness timing.

## Current Boundary Profiling Focus

After the counter-backend implementation, the next profiling pass should focus
on boundary-only work instead of the full slot observe path.

Enable the targeted probes with:

```text
VLLM_MPR_BOUNDARY_PROFILE=1
```

Suggested counter-boundary probes:

```text
prepare_counter_kv_write total
observe_kv_write_by_counter total
counter block-table lookup / block-id extraction
_create_digest_for_full_block total
_key_cache_for_digest total
summarize_key_block total
_to_block_digest total
_append_quest_metadata_digest total
_maybe_backup_kv_block total / early return
counter debug record path, only when logging is enabled
```

The profiling result should separate:

```text
non-boundary steady-state counter overhead
boundary digest creation overhead
fallback-to-slot overhead, if any
```
