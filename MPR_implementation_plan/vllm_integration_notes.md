# vLLM Integration Notes for Mixed-Precision Recovery

Milestone 0 result: vLLM is a reasonable base for the first prototype. The
lowest-risk path is to start with vLLM v1 + FlashAttention, add scoring-only
instrumentation first, and defer CPU KV backup layout decisions until recovery
actually moves KV pages between CPU and GPU.

## Scope Assumptions

- Target tree: `/workspace/ArkVale/vllm`
- Reference tree: `/workspace/ArkVale/source/arkvale`
- Initial backend: vLLM v1 FlashAttention
- Initial mode: single GPU, decoder-only model, no MLA, no DCP/CP, no speculative
  decoding, no sliding-window special case
- Initial objective: observe/query-score/select candidate pages without changing
  the attention result

These assumptions keep Milestone 1 small enough to validate the idea before
committing to a full recovery/offload implementation.

## Main vLLM Execution Path

The relevant vLLM v1 path is:

1. `GPUModelRunner.execute_model`
2. `GPUModelRunner.prepare_attn`
3. `DefaultModelState.prepare_attn`
4. `build_attn_metadata`
5. `Attention.forward`
6. `unified_kv_cache_update`
7. `unified_attention_with_output`
8. `FlashAttentionImpl.forward`

Key files:

- `/workspace/ArkVale/vllm/vllm/v1/worker/gpu/model_runner.py`
- `/workspace/ArkVale/vllm/vllm/v1/worker/gpu/model_states/default.py`
- `/workspace/ArkVale/vllm/vllm/model_executor/layers/attention/attention.py`
- `/workspace/ArkVale/vllm/vllm/v1/attention/backends/flash_attn.py`

`DefaultModelState.prepare_attn` builds the backend metadata from vLLM's common
request state. The important fields are:

- `query_start_loc`
- `seq_lens`
- `block_tables`
- `slot_mappings`
- `num_reqs`
- `num_tokens`
- `max_query_len`
- `max_seq_len`

For FlashAttention, this becomes `FlashAttentionMetadata`, which contains:

- `block_table`
- `slot_mapping`
- `seq_lens`
- `query_start_loc`
- `num_actual_tokens`
- `max_query_len`
- `max_seq_len`

## Attention/KV Hook Points

### 1. `unified_kv_cache_update`

Location:

- `/workspace/ArkVale/vllm/vllm/model_executor/layers/attention/attention.py`

This receives:

- `key`: `[num_tokens, num_kv_heads, head_size]`
- `value`: `[num_tokens, num_kv_heads, head_size_v]`
- `layer_name`

It resolves from forward context:

- `attn_layer`
- `kv_cache`
- per-layer `slot_mapping`

Then it calls the backend's `do_kv_cache_update`.

This is the best hook for digest creation and for observing which newly computed
tokens land in which physical KV slots. For FlashAttention, slot ids are physical
cache slots:

```text
slot_id = physical_block_id * block_size + block_offset
```

For Milestone 1, this hook can build or update a sidecar digest table. It should
not alter the KV cache or scheduling yet.

### 2. `unified_attention_with_output`

Location:

- `/workspace/ArkVale/vllm/vllm/model_executor/layers/attention/attention.py`

This receives:

- `query`: `[num_tokens, num_heads, head_size]`
- `key`
- `value`
- `output`
- `layer_name`

It resolves:

- backend-specific `attn_metadata`
- `kv_cache`
- attention layer object

Then it calls `self.impl.forward(...)`.

This is the best hook for score-only instrumentation because it sees the query,
the current request metadata, and the KV cache before the real attention kernel
runs. A guarded prototype can compute ArkVale-style page scores here and log or
stash them without changing the output.

### 3. `FlashAttentionImpl.do_kv_cache_update`

Location:

- `/workspace/ArkVale/vllm/vllm/v1/attention/backends/flash_attn.py`

This unbinds:

```python
key_cache, value_cache = kv_cache.unbind(0)
```

and calls:

```python
reshape_and_cache_flash(...)
```

This is a lower-level hook than `unified_kv_cache_update`. It is useful if the
sidecar needs exact backend layout semantics, but for Milestone 1 it is probably
cleaner to stay one level above it.

## FlashAttention KV Layout

FlashAttention's KV cache shape is:

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
```

The first dimension is K/V. vLLM can choose NHD/HND stride order, but the
logical page unit remains the physical block id. This matters because our sidecar
should use vLLM block ids as the stable page identity, not raw tensor offsets.

The existing vLLM simple KV offload worker already turns backend-specific KV
storage into per-block int8 views. It infers `page_size_bytes` from the raw
storage and `num_blocks`, then builds CPU tensors shaped like:

```text
[num_cpu_blocks, block_bytes]
```

Reference:

- `/workspace/ArkVale/vllm/vllm/v1/simple_kv_offload/worker.py`

This is a strong hint for Milestone 2, but not required for Milestone 1.

## Block Table and Slot Mapping

Location:

- `/workspace/ArkVale/vllm/vllm/v1/worker/gpu/block_table.py`

`BlockTables.append_block_ids` stages newly allocated physical block ids per
request. `apply_staged_writes` commits them. `gather_block_tables` produces the
per-forward block table tensors used by attention metadata.

`compute_slot_mappings` maps token positions to physical KV slots. In the simple
non-CP case:

```text
block_index = position // block_size
block_offset = position % block_size
block_number = block_table[request_index, block_index]
slot_id = block_number * block_size + block_offset
```

This gives us enough information to associate incoming K/V tensors with vLLM
physical blocks.

## KV Block Lifecycle

Locations:

- `/workspace/ArkVale/vllm/vllm/v1/core/kv_cache_manager.py`
- `/workspace/ArkVale/vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `/workspace/ArkVale/vllm/vllm/v1/core/block_pool.py`
- `/workspace/ArkVale/vllm/vllm/v1/core/sched/scheduler.py`

Important functions:

- `KVCacheManager.allocate_slots`
- `KVCacheManager.free`
- `KVCacheManager.get_block_ids`
- `KVCacheManager.take_new_block_ids`
- `SingleTypeKVCacheManager.allocate_new_blocks`
- `SingleTypeKVCacheManager.free`
- `BlockPool.get_new_blocks`
- `BlockPool.free_blocks`

For Milestone 1, we can avoid changing allocation and eviction. The sidecar only
needs to observe block ids and per-layer digest state. For Milestone 2+, recovery
and CPU backup must track the lifecycle of physical GPU block ids carefully,
especially when blocks are freed, reused, or prefix-cached.

## ArkVale Contract Notes

Reference files:

- `/workspace/ArkVale/source/arkvale/kernels.py`
- `/workspace/ArkVale/source/arkvale_cpp/src/estimate.cu`
- `/workspace/ArkVale/source/arkvale_cpp/src/select.cu`

`estimate_scores` expects:

```text
q:                [batch, 1, num_heads, head_dim]
dg_data:          paged digest tensor
dg_indices:       [batch, num_pages]
dg_indptr:        [batch + 1]
dg_last_page_len: [batch]
dg_seq_len:       scalar
layout:           HND or NHD
n_groups:         GQA grouping
```

The CUDA kernel computes the real sequence length from:

```text
kv_chunk_len =
  (indptr[i + 1] - indptr[i] - 1) * page_size + last_page_len[i]
```

and writes scores using `kv_chunk_len` as the stride. Therefore, the metadata
passed to ArkVale must be packed to real pages/tokens. A padded 2D page table is
not safe unless `indptr` and `last_page_len` describe only the real pages.

`select_topk` is more ArkVale-specific. It expects `incache`, `new_in`,
`pos_ids`, and `recall_ids` structures tied to ArkVale's CPU/GPU cache manager.
For vLLM Milestone 1, reuse `estimate_scores` first and implement selection in a
small sidecar layer or PyTorch code before considering ArkVale's `select_topk`.

## Milestone 1 Recommendation

Start with a score-only prototype in vLLM:

1. Add a small, disabled-by-default sidecar module under vLLM v1.
2. Hook `unified_kv_cache_update` to observe K/V and slot mapping.
3. Build per-layer, per-physical-block digest entries.
4. Hook `unified_attention_with_output` to score current query against digest
   pages.
5. Emit debug stats only:
   - layer name
   - request count
   - sequence length
   - candidate page ids
   - score distribution/top-k
6. Do not modify attention output, KV cache contents, scheduling, or offload.

The first implementation can use PyTorch scoring even if it is slower. Once the
metadata contract is correct, replace the scoring path with ArkVale's
`estimate_scores` kernel.

## What Is Already Available

- vLLM already has physical block ids, block tables, slot mappings, and per-layer
  KV cache tensors.
- vLLM already has a simple CPU KV offload path with block-granular CPU/GPU copy
  machinery.
- ArkVale already has digest scoring kernels and an implementation pattern for
  page-score/select/recall.
- ArkVale's HuggingFace adapter is useful as a conceptual reference, but it is
  not directly vLLM-native.

## What Is Not Yet Available

- A vLLM-native sidecar digest table.
- A vLLM-native mapping from physical GPU block id to digest page id.
- A lifecycle policy for clearing/reusing sidecar entries when vLLM frees or
  reuses blocks.
- A CPU backup layout for mixed-precision recovered KV pages.
- A recovery path that swaps or overlays higher-precision KV into attention.
- A scheduler policy that decides when recovery is worth the transfer/compute
  cost.

## CPU Layout Status

CPU KV backup layout should not be treated as fixed yet.

It becomes necessary around Milestone 2, when we need a real CPU fp16 backup and
stable CPU/GPU block mapping. It becomes correctness-critical in Milestone 3,
when attention output starts depending on recovered CPU-backed pages.

For Milestone 1, we only need:

- vLLM physical GPU block ids
- block size
- layer name
- digest page metadata
- query-time block table and sequence length

## Risks

- `unified_kv_cache_update` and `unified_attention_with_output` are registered
  custom ops. Any hook must be guarded and benchmarked because this area is
  sensitive to `torch.compile` and CUDA graph behavior.
- FlashAttention has cascade, sliding-window, DCP, and speculative paths. These
  should be out of scope until the basic path works.
- ArkVale's score kernel is strict about `indptr`, `last_page_len`, and
  `dg_seq_len`. Padded metadata can silently produce incorrect or uninitialized
  scores.
- K/V tensors at the attention layer should be verified for the chosen model:
  the sidecar should confirm whether keys are already RoPE-applied at the hook
  point before treating digest scores as semantically meaningful.

## Go/Pivot Decision

Proceed with vLLM for Milestone 1.

The implementation should remain shallow and reversible: score-only, no output
changes, no CPU layout commitment. If that path becomes too invasive because of
compile/CUDA graph constraints, switch to InfiniGen for algorithmic prototyping
while keeping this vLLM map for the later systems implementation.
