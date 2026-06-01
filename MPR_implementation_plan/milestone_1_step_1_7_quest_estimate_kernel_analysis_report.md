# Milestone 1 Step 1.7 Quest Estimate Kernel Analysis Report

## Scope

This report analyzes Quest's `estimate.cu` path as the next candidate scoring
kernel for MPR after the ArkVale compatibility review.

The question is not whether Quest can be imported directly into vLLM. The
question is whether its estimate kernel contract is a better target for the MPR
adapter than ArkVale's exposed `estimate_scores` path.

## Short Verdict

Quest's estimate kernel is a better conceptual fit for the current MPR default:

```text
digest_kind       = raw_minmax
score_granularity = kv_head
score_agg         = max
```

Reasons:

1. Quest metadata is raw per-page min/max, matching `raw_minmax`.
2. The estimate kernel computes per-query-head page scores.
3. The underlying kernel has a GQA-shaped dispatch path.
4. It does not impose ArkVale's exposed mean aggregation contract.

However, Quest's exposed Python/C++ wrapper is narrow:

1. It hardcodes batch size 1.
2. It assumes `q_heads == metadata_heads` at the wrapper level.
3. It uses Quest's own `InferenceController` and `KvCache` metadata layout.
4. It returns scores for Quest's metadata-entry sequence, excluding the current
   last metadata entry.

So the recommendation is:

```text
Do not directly call Quest's current Python wrapper.
Use Quest's underlying estimate kernel design as the adapter target.
Write or modify a small binding that accepts separate num_q_heads and
num_kv_heads, then returns per-query-head scores.
Aggregate query-head scores to KV-head scores in Python first.
```

## Source Files Reviewed

```text
/home/han/KV_cache_quant/quest/quest/ops/csrc/estimate.cu
/home/han/KV_cache_quant/quest/quest/utils/__init__.py
/home/han/KV_cache_quant/quest/quest/utils/controller.py
/home/han/KV_cache_quant/quest/quest/utils/kv_cache.py
/home/han/KV_cache_quant/quest/kernels/include/decode/decode_attn.cuh
/home/han/KV_cache_quant/quest/kernels/include/decode/decode_page.cuh
/home/han/KV_cache_quant/quest/quest/tests/test_estimate.py
/home/han/KV_cache_quant/quest/quest/ops/csrc/topk.cu
```

## Exposed Wrapper Contract

Quest exposes:

```cpp
void estimate_attn_score(
    torch::Tensor q,
    torch::Tensor o,
    torch::Tensor metadata_data,
    torch::Tensor metadata_indices,
    torch::Tensor metadata_indptr,
    unsigned int metadata_last_page_len,
    unsigned int metadata_last_page_idx,
    unsigned int layout)
```

The wrapper expects:

```text
q:
  [1, num_heads, head_dim]

metadata_data, NHD layout:
  [num_max_pages, 2, page_size, num_heads, head_dim]

metadata_indices:
  int32 physical metadata page ids

metadata_indptr:
  int32 paged-cache indptr, batch size 1 in the exposed wrapper

o:
  allocated by Python as [num_heads, output_len]
```

`estimate.cu` sets:

```cpp
constexpr size_t batch_size = 1;
size_t num_heads = q.size(1);
...
CHECK_EQ(metadata_data.size(3), num_heads);
...
paged_kv_t(..., num_heads, page_size, head_dim, batch_size, ...)
...
MaxPossibleSampleWithPagedKVCache(..., q, paged_kv, o, num_heads, ...)
```

This means the exposed wrapper treats one `num_heads` value as both:

```text
num_q_heads
num_kv_heads stored in metadata
```

Therefore the wrapper is effectively MHA-only, even though the lower-level
kernel can represent GQA.

## Score Math

Quest metadata stores channel-wise key extrema:

```text
metadata K plane = page max key vector
metadata V plane = page min key vector
```

For query vector `q`, page max vector `M`, and page min vector `m`, the estimate
is:

```text
score(page, q) = sum_i max(q_i * M_i, q_i * m_i)
```

This is the same upper-bound style score currently implemented by the MPR
`raw_minmax` PyTorch scorer.

The CPU reference in `quest/tests/test_estimate.py` computes the same value by
flipping keys according to the query sign and taking a page max:

```text
sign = +1 where q > 0 else -1
score = positive_query @ page_max(k * sign)
```

So, at the scoring-math level:

```text
MPR raw_minmax scorer == Quest estimate scorer
```

## Metadata Cache Semantics

Quest keeps two paged caches:

```text
kv_cache:
  real token K/V pages

metadata_cache:
  one min/max metadata entry per real KV page
```

Both use the same `KvCache` abstraction. In NHD layout, the backing tensor is:

```text
[num_layers, capacity, 2, page_size, num_heads, head_dim]
```

For `metadata_cache`, this tensor is semantically different from a real KV
cache:

```text
the K plane stores max vectors
the V plane stores min vectors
one metadata entry corresponds to one original KV page
metadata entries are themselves packed into metadata pages
```

Important consequence:

```text
metadata_cache.seqlen == number of original KV pages summarized so far
metadata_cache.page_size == page_size used to pack metadata entries
```

In `AppendPagedKVCachePrefillKernel`, the mapping is:

```cpp
candidate_page_idx = page_offset_bdx / candidate_kv.page_size;
candidate_entry_idx = page_offset_bdx % candidate_kv.page_size;
```

So metadata entry index `i` summarizes original KV page `i`, and these metadata
entries are paged into the candidate/metadata cache.

## Estimate Output Semantics

Quest's Python helper allocates:

```python
o = torch.empty(
    (iController.num_heads, iController.metadata_cache.seqlen - 1),
    dtype=q.dtype,
    device=q.device,
)
```

The `-1` excludes the current last metadata entry. Since one metadata entry
summarizes one original KV page, this means:

```text
estimate all previous KV pages, excluding the current last KV page
```

The underlying kernel computes:

```text
kv_chunk_len =
  (indptr_end - indptr_begin - 1) * page_size
  + metadata_last_page_len
  - 1
```

and writes:

```text
o[(batch_idx, qo_head_idx), 0:kv_chunk_len]
```

For Quest's exposed batch-size-1 wrapper, this becomes:

```text
o: [num_q_heads, num_metadata_entries_excluding_last]
```

That is exactly the artifact we want for a correctness-first MPR kernel path:

```text
per query head, per digest entry/block score
```

## Head And GQA Behavior

The lower-level function is:

```cpp
MaxPossibleSampleWithPagedKVCache(
    q,
    paged_kv,
    o,
    uint32_t num_qo_heads,
    RotaryMode rotary_mode)
```

It derives:

```cpp
num_kv_heads = paged_kv.num_heads;
```

and checks:

```cpp
num_qo_heads % num_kv_heads == 0
```

Then it dispatches:

```cpp
GROUP_SIZE = num_qo_heads / num_kv_heads
grid       = (batch_size, num_kv_heads)
threads.y = GROUP_SIZE
```

Inside the kernel:

```cpp
kv_head_idx = blockIdx.y
qo_head_idx = kv_head_idx * GROUP_SIZE + threadIdx.y
```

So the kernel is structurally GQA-aware. For each KV head, it can score multiple
query heads in that KV group.

But the exposed `estimate.cu` wrapper calls:

```cpp
MaxPossibleSampleWithPagedKVCache(..., num_heads, ...)
```

where `num_heads` is both `q.size(1)` and `metadata_data.size(head_axis)`.
Therefore current Quest wrapper dispatches:

```text
GROUP_SIZE = 1
```

unless the wrapper is changed to pass:

```text
num_qo_heads = q.size(1)
num_kv_heads = metadata_data.size(head_axis)
```

Practical interpretation:

```text
Underlying kernel: GQA-capable shape.
Current wrapper/controller: MHA-only shape.
```

For MPR, that means we should not preserve the wrapper contract as-is. We should
create an MPR-facing wrapper that accepts separate query-head and KV-head counts.

## Aggregation Behavior

Quest estimate does not reduce query heads into KV heads. It writes one score
row per query head:

```text
[num_q_heads, num_scored_digest_entries]
```

Quest then runs a separate top-k path using this per-head score matrix.

This is different from ArkVale's exposed kernel, which mean-reduces query heads
inside each KV group before returning scores.

For the current MPR default:

```text
score_granularity = kv_head
score_agg         = max
```

the cleanest first kernel path is:

```text
Quest estimate kernel:
  produce per_query_head_scores

MPR Python aggregation:
  group query heads by KV head
  per_kv_head_score = max(query_head_scores in group)
  per_kv_head top-k = topk(per_kv_head_score)
```

This keeps the kernel correctness oracle simple and lets us compare:

```text
PyTorch per_query_head_scores
CUDA per_query_head_scores
Python aggregation result
```

Later, if overhead matters, we can add a kernel mode that directly returns:

```text
[num_kv_heads, num_scored_digest_entries]
```

using group max/union aggregation inside the kernel.

## Direct Reuse Feasibility

Directly using Quest's Python API is not a good target:

```text
decode_estimate(q, iController, layer_idx)
```

requires Quest's:

```text
InferenceController
KvCache
metadata_cache
metadata_indices
metadata_indptr_for_append
metadata_last_page_idx
layout enum
```

It also assumes:

```text
batch size 1
q_heads == metadata_heads
output length = metadata_cache.seqlen - 1
```

MPR currently stores per-layer digest entries in a Python sidecar and does not
already own a Quest-style paged metadata cache tensor.

So direct reuse is not the right interface.

## Adapter Feasibility

A Quest-style adapter is feasible and cleaner than the ArkVale exposed wrapper
for the current default.

The adapter needs to pack MPR digests into:

```text
metadata_data:
  [num_metadata_pages, 2, page_size, num_kv_heads, head_dim]

metadata_indices:
  int32 physical metadata page ids

metadata_indptr:
  int32 request indptr

metadata_last_page_len:
  number of valid metadata entries in the final metadata page

metadata_last_page_idx:
  physical id of the final metadata page
```

Then call a modified binding:

```text
estimate_attn_score_mpr(
    q,                 # [1, num_q_heads, head_dim]
    o,                 # [num_q_heads, num_digest_entries_to_score]
    metadata_data,      # [metadata_pages, 2, page_size, num_kv_heads, head_dim]
    metadata_indices,
    metadata_indptr,
    metadata_last_page_len,
    metadata_last_page_idx,
    layout,
    num_q_heads,
    num_kv_heads)
```

Implementation note:

```text
The binding may not need both num_q_heads and num_kv_heads as explicit args if
it derives them from q and metadata_data, but the implementation must keep them
separate.
```

## Top-k Kernel Note

Quest's `topk_filtering` is not required for MPR Step 1.7.

Reasons:

1. MPR already has a Python/top-k debug policy path.
2. We want to validate score tensors before optimizing selection.
3. Quest's top-k binding is tied to Quest's per-head page index layout.
4. The source path contains assumptions around fixed head count in the top-k
   template path, so it is a worse first integration target than estimate.

Therefore the first CUDA integration target should be:

```text
estimate only
```

not:

```text
estimate + Quest topk_filtering
```

## Compatibility Table

| Area | MPR Current Default | Quest Estimate Kernel | Compatibility |
| --- | --- | --- | --- |
| Digest semantics | raw min/max | raw min/max | good |
| Score math | sum max(q*max, q*min) | same | good |
| Output granularity | KV-head default, query-head available | query-head output | good after Python aggregation |
| GQA | conservative query-head union per KV head | lower-level GQA shape exists | good after wrapper change |
| Wrapper | vLLM sidecar metadata | Quest `InferenceController` metadata | needs adapter |
| Batch | vLLM may batch requests | exposed wrapper batch size 1 | Step 1.7 can start single request |
| Output length | digest blocks selected by sidecar | metadata entries excluding current last entry | compatible if packed carefully |
| Top-k | MPR policy/debug path | separate Quest top-k op | do not reuse initially |

## Recommended Next Implementation Shape

1. Keep the current PyTorch scorer as the reference backend.
2. Add a new backend name later, for example:

```text
quest_cuda
```

3. Implement a packing helper that converts MPR layer digests to Quest-style
   metadata pages.
4. Add or modify a C++ binding that separates `num_q_heads` and `num_kv_heads`.
5. Return per-query-head scores from CUDA.
6. Reuse the existing MPR Python aggregation:

```text
per_query_head_scores -> per_kv_head_scores -> topk_block_ids_by_head
```

7. Compare PyTorch and CUDA per-query-head score tensors before optimizing any
   aggregation or top-k logic.

## Open Questions

1. Should the first CUDA adapter score all historical digest blocks, or should it
   mirror Quest and always exclude the current last block?
2. Should the first CUDA backend be single-request only, matching Quest's exposed
   wrapper, or should the binding be shaped for batched requests from day one?
3. Should GQA support initially be limited to group sizes supported by the
   underlying FlashInfer macro path, then widened only if needed?
4. Should the kernel output dtype remain `q.dtype`, or should we expose fp32
   scores for easier debugging?

## Final Assessment

The detailed kernel review supports switching the kernel-integration target from
ArkVale's exposed estimate wrapper to a Quest-style estimate wrapper.

The important distinction is:

```text
Quest current wrapper:
  not directly reusable for vLLM MPR

Quest underlying estimate kernel design:
  strong fit for MPR raw_minmax + head-level scoring
```

The safest next step is an estimate-only CUDA backend that returns
per-query-head scores and leaves KV-head aggregation/top-k in the already-tested
Python path.
