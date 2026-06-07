# Milestone 1 Step 1.7 Quest Kernel Dependency Slicing Report

## Scope

This report analyzes what must be copied, ported, or rewritten before replacing
the current MPR `mpr_estimate_attn_score` stub with Quest's CUDA estimate path.

The question here is narrower than the previous Quest estimate-kernel report:

```text
Can we bring in only the scoring-critical Quest kernel pieces, without importing
Quest's full inference controller, full decode attention stack, or top-k path?
```

## Short Verdict

Use an estimate-only vendored subset, not a direct full-file import.

Quest's lower-level estimate kernel is still the right first CUDA target for the
current MPR default:

```text
digest_kind        = raw_minmax
score_granularity = kv_head
score_agg         = max
```

However, directly copying Quest's full `decode_attn.cuh` pulls in far more than
MPR needs. The full file includes full decode attention, cascade/state helpers,
RoPE helpers, random/partition code, and append/prefill-adjacent utilities. For
MPR Step 1.7, the required scoring surface is much smaller:

```text
estimate launcher
estimate CUDA kernel
compute_max_possible score math
minimal paged metadata-cache accessor
small FlashInfer utility subset
MPR-facing torch binding
```

## Source Files Reviewed

```text
/home/han/KV_cache_quant/quest/quest/ops/csrc/estimate.cu
/home/han/KV_cache_quant/quest/quest/ops/csrc/bsk_ops.h
/home/han/KV_cache_quant/quest/quest/ops/csrc/pytorch_extension_utils.h
/home/han/KV_cache_quant/quest/kernels/include/decode/decode_attn.cuh
/home/han/KV_cache_quant/quest/kernels/include/decode/decode_page.cuh
/home/han/KV_cache_quant/quest/kernels/3rdparty/flashinfer/include/flashinfer/layout.cuh
/home/han/KV_cache_quant/quest/kernels/3rdparty/flashinfer/include/flashinfer/utils.cuh
/home/han/KV_cache_quant/quest/kernels/3rdparty/flashinfer/include/flashinfer/math.cuh
/home/han/KV_cache_quant/quest/kernels/3rdparty/flashinfer/include/flashinfer/cp_async.cuh
/home/han/KV_cache_quant/quest/kernels/3rdparty/flashinfer/include/flashinfer/vec_dtypes.cuh
/home/han/KV_cache_quant/quest/kernels/3rdparty/flashinfer/include/flashinfer/rope.cuh
/home/han/KV_cache_quant/quest/quest/utils/utils.py

/home/han/KV_cache_quant/proposed_method_develop/vllm/CMakeLists.txt
/home/han/KV_cache_quant/proposed_method_develop/vllm/csrc/mpr/quest_estimate_stub.cpp
/home/han/KV_cache_quant/proposed_method_develop/vllm/csrc/torch_bindings.cpp
/home/han/KV_cache_quant/proposed_method_develop/vllm/csrc/ops.h
/home/han/KV_cache_quant/proposed_method_develop/vllm/vllm/_custom_ops.py
/home/han/KV_cache_quant/proposed_method_develop/vllm/vllm/v1/mixed_precision_recovery/scoring.py
/home/han/KV_cache_quant/proposed_method_develop/vllm/vllm/v1/mixed_precision_recovery/quest_packing.py
```

## Dependency Shape

Quest's exposed wrapper is small:

```cpp
estimate_attn_score(
    q,
    o,
    metadata_data,
    metadata_indices,
    metadata_indptr,
    metadata_last_page_len,
    metadata_last_page_idx,
    layout)
```

But the wrapper includes:

```text
bsk_ops.h
pytorch_extension_utils.h
decode/decode_attn.cuh
decode/decode_page.cuh
flashinfer/*
```

The important split is:

```text
Quest exposed wrapper:
  batch size 1
  q_heads == metadata_heads
  uses Quest's dtype/check macros

Quest lower-level launcher:
  accepts num_qo_heads separately
  derives num_kv_heads from paged_kv.num_heads
  supports GQA-shaped dispatch when group size is supported
```

Therefore the MPR binding should not preserve the original wrapper contract.
It should keep:

```text
num_q_heads = q.size(1)
num_kv_heads = metadata_data.size(3)  # NHD
```

separate, then call the lower-level launcher with:

```text
paged_kv.num_heads = num_kv_heads
num_qo_heads = num_q_heads
```

## Estimate-Only Kernel Pieces

The actual scoring math lives in `compute_max_possible`:

```text
score(entry, query_head) = sum_i max(q_i * digest_max_i, q_i * digest_min_i)
```

The estimate kernel then writes:

```text
out[query_head, metadata_entry]
```

for each query head and each scored metadata entry.

Required pieces from `decode_attn.cuh`:

```text
compute_max_possible
MaxPossibleSampleWithPagedKVCacheKernel
MaxPossibleSampleWithPagedKVCache
```

Required behavior:

```text
partition_kv = false
rotary_mode = RotaryMode::kNone
PageStorage = kIndices
metadata layout = NHD for the current MPR packer
```

Not required for Step 1.7:

```text
full BatchDecodeWithPagedKVCache kernels
partition-kv decode path
cascade/state helpers
RoPE computation
top-k filtering
append/prefill kernels
Quest InferenceController/KvCache Python classes
```

## Minimal Paged Cache Pieces

The estimate kernel treats the digest cache as a paged KV cache:

```text
metadata_data:
  [num_metadata_pages, 2, page_size, num_kv_heads, head_dim]  # NHD

plane 0:
  digest max

plane 1:
  digest min
```

From `decode_page.cuh`, the estimate-only path needs:

```text
PageStorage
paged_kv_t fields
PageStorage::kIndices constructor
get_k_elem_offset
get_v_elem_offset
get_k_elem_offset_in_page
get_v_elem_offset_in_page
kv_offset_delta
protective_get_k_ptr
```

The append kernels and pointer-storage path are not needed for the first MPR
CUDA score backend.

## FlashInfer Utility Subset

The smallest useful compatibility subset is:

```text
layout.cuh
  QKVLayout
  get_elem_offset_impl
  get_n_stride_impl
  get_h_stride_impl

utils.cuh
  FLASHINFER_CUDA_CALL
  SWITCH_LAYOUT
  SWITCH_GQA_GROUP_SIZE
  SWITCH_HEAD_DIM
  ceil_div

math.cuh
  math::shfl_xor_sync for float

cp_async.cuh
  PrefetchMode
  SharedMemFillMode
  pred_load
  commit_group
  wait_group

vec_dtypes.cuh
  vec_t<float, N>
  vec_t<half, N>
  cast_load/cast_from for half <-> float
```

`rope.cuh` is only needed for the `RotaryMode` enum if we keep the original
launcher signature. For the MPR-only port, it is cleaner to define a tiny local
enum or remove the runtime `rotary_mode` parameter and hard-code no-RoPE in the
MPR launcher.

## Direct Copy vs Sliced Port

### Direct Full Copy

Pros:

```text
least initial code surgery
closest to Quest source
```

Cons:

```text
pulls full decode attention dependencies into vLLM _C
imports code paths MPR will not call
larger compile surface
more external FlashInfer header assumptions
harder to reason about future local modifications
```

This is not recommended for the first MPR integration.

### Estimate-Only Sliced Port

Pros:

```text
keeps the binding focused on score-only MPR
reduces build/debug surface
lets us remove Quest wrapper assumptions
keeps Python aggregation/top-k unchanged
```

Cons:

```text
requires careful attribution and local comments for derived code
requires parity tests against the PyTorch reference
may need small follow-up edits if FlashInfer utility assumptions are missed
```

This is the recommended path.

## Proposed vLLM File Layout

Recommended first layout:

```text
csrc/mpr/quest_estimate.cu
  MPR torch binding implementation
  input validation
  dtype dispatch
  call into estimate-only launcher

csrc/mpr/quest_estimate_kernel.cuh
  compute_max_possible
  MaxPossibleSampleWithPagedKVCacheKernel
  MPR estimate-only launcher

csrc/mpr/quest_paged_kv.cuh
  minimal PageStorage + paged_kv_t subset

csrc/mpr/flashinfer_compat/layout.cuh
csrc/mpr/flashinfer_compat/utils.cuh
csrc/mpr/flashinfer_compat/math.cuh
csrc/mpr/flashinfer_compat/cp_async.cuh
csrc/mpr/flashinfer_compat/vec_dtypes.cuh
  minimal FlashInfer-derived compatibility headers
```

If compile iteration becomes too slow, the first PoC may temporarily put the
kernel and paged-cache subset in a single `quest_estimate.cu`, but keeping the
headers separated is easier to audit.

## Binding Contract

The existing vLLM op schema is already enough for the first pass:

```text
torch.ops._C.mpr_estimate_attn_score(
    q,
    out,
    metadata_data,
    metadata_indices,
    metadata_indptr,
    metadata_last_page_len,
    metadata_last_page_idx,
    layout,
)
```

The implementation should derive:

```text
q:
  [1, num_q_heads, head_dim]

out:
  [num_q_heads, num_score_entries]

metadata_data, NHD:
  [num_metadata_pages, 2, page_size, num_kv_heads, head_dim]

metadata_indices:
  int32 [num_metadata_pages]

metadata_indptr:
  int32 [2] for the first single-request PoC
```

Validation should enforce:

```text
q.is_cuda
out.is_cuda
metadata_data.is_cuda
metadata_indices.is_cuda
metadata_indptr.is_cuda

q.dim == 3
q.size(0) == 1
metadata_data.dim == 5
metadata_indices.dim == 1
metadata_indptr.dim == 1
metadata_indptr.numel == 2 for Step 1.7

layout == NHD for the first MPR path
metadata_data.size(1) == 2
metadata_data.size(3) == num_kv_heads
metadata_data.size(4) == head_dim
num_q_heads % num_kv_heads == 0
```

For the compact packer currently used by `QuestCudaScorer`, the expected score
entry count can be validated without reading CUDA `metadata_indptr` on the host:

```text
expected_score_entries =
  (metadata_indices.numel() - 1) * page_size
  + metadata_last_page_len
  - 1
```

This works because the first PoC uses batch size 1 and compact metadata pages.
A batched future version should avoid assuming `metadata_indices.numel()` equals
the request page count.

## Current MPR Compatibility Notes

The current Python packer already produces the desired NHD shape:

```text
[num_metadata_pages, 2, page_size, num_kv_heads, head_dim]
```

It also appends one guard metadata entry because Quest's estimate kernel
unconditionally excludes the final logical metadata entry:

```text
cur_last_page_len = paged_kv.last_page_len - 1
```

This is a deliberate compatibility shim for the unmodified Quest estimate
contract. It should remain documented in code and progress logs. If the MPR
binding later exposes an explicit `num_score_entries` or exclusion count, the
guard entry should be removed.

Important pre-implementation fix:

```text
Quest/FlashInfer TensorLayout.NHD == 0
Quest/FlashInfer TensorLayout.HND == 1

Current MPR Python code has:
  QUEST_NHD_LAYOUT = 1
```

That constant must be changed to `0` before the real CUDA kernel is connected.
The current stub does not use the value, so tests did not expose this yet.

## Supported Shapes for the First Kernel

The lower-level Quest launcher dispatches only the group/head-dim cases covered
by FlashInfer macros:

```text
GQA group_size:
  1, 4, 8

head_dim:
  64, 128, 256
```

This matches the current Python-side `QuestCudaScorer` guard for group size.
The C++ binding should also check these limits and raise a clear `TORCH_CHECK`
error before dispatch.

The original Quest wrapper dispatches only fp16 through
`DISPATCH_PYTORCH_DTYPE_TO_CTYPE`. For the first MPR PoC, fp16-only is acceptable
and closest to the reference source. BF16 support can be added after fp16 parity
is established because the FlashInfer `vec_t` header already has BF16 vector
support, but the MPR binding will need explicit dtype dispatch and tests.

## Build Integration

Current vLLM source list contains:

```text
csrc/mpr/quest_estimate_stub.cpp
```

Recommended build edit:

```text
replace:
  csrc/mpr/quest_estimate_stub.cpp

with:
  csrc/mpr/quest_estimate.cu
```

No schema change is needed in:

```text
csrc/torch_bindings.cpp
csrc/ops.h
vllm/_custom_ops.py
```

unless we decide to pass `num_score_entries` explicitly. For the first PoC, keep
the schema stable and validate `out.size(1)` from the packed metadata shape.

## License And Attribution

Quest's repository license is MIT:

```text
Copyright (c) 2024 MIT HAN Lab
```

The kernel files used here are derived from FlashInfer headers and carry
Apache-2.0 notices:

```text
Copyright (c) 2023 by FlashInfer team.
Licensed under the Apache License, Version 2.0
```

When copying or deriving the sliced files, preserve the original copyright and
license headers in the new `csrc/mpr/...` files. For files that mix vLLM binding
code with derived kernel code, add a short comment identifying which sections
are derived from Quest/FlashInfer.

## Validation Plan

Minimum validation before enabling `quest_cuda` in an actual run:

```text
1. Build vLLM _C with quest_estimate.cu.
2. Import smoke:
   import vllm._custom_ops as ops
   hasattr(ops, "mpr_estimate_attn_score")
3. Tiny CUDA parity test:
   random q/digest_min/digest_max
   pack_quest_metadata_cache(add_guard_entry=True)
   CUDA output [num_q_heads, num_score_entries]
   compare to TorchQuestScorer.per_query_head_scores.T
4. GQA cases:
   group_size = 1
   group_size = 4
   group_size = 8
5. Shape/error tests:
   unsupported group_size
   unsupported head_dim
   wrong layout
   wrong dtype
   wrong output length
6. Focused MPR pytest suite.
7. One short generation smoke with:
   VLLM_MPR_ENABLE=1
   VLLM_MPR_SCORING_BACKEND=quest_cuda
```

## Recommended Next Step

Implement the sliced estimate-only CUDA backend:

```text
1. Add local FlashInfer-compatible minimal headers.
2. Add local minimal paged metadata-cache accessor.
3. Add local estimate-only launcher/kernel.
4. Replace quest_estimate_stub.cpp with quest_estimate.cu in CMake.
5. Fix QUEST_NHD_LAYOUT from 1 to 0.
6. Add CUDA parity tests against TorchQuestScorer.
7. Build and run focused tests.
```

The main implementation risk is not the score math. The score math is already
aligned with the PyTorch reference. The main risk is preserving Quest's paged
metadata indexing and GQA head mapping while removing unrelated decode-attention
code.
