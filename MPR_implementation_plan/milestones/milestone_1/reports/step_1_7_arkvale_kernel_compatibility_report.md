# Milestone 1 Step 1.7 ArkVale Kernel Compatibility Report

Last updated: 2026-06-01

## Scope

This report covers the first half of Step 1.7:

1. ArkVale `estimate_scores` kernel analysis
2. First-pass compatibility analysis against the current vLLM MPR score-only
   sidecar

It does not implement an adapter yet. It also does not analyze ArkVale
`select_topk` beyond the context needed to understand score output semantics.

Primary source trees:

```text
ArkVale: /home/han/KV_cache_quant/ArkVale
vLLM MPR: /home/han/KV_cache_quant/proposed_method_develop/vllm
```

## High-Level Result

ArkVale's `estimate_scores` kernel is conceptually compatible with the current
MPR score-only objective: both compute page/block importance from a query vector
and ArkVale-style digest min/max bounds.

However, it is not directly callable with the current vLLM sidecar state.
The current vLLM sidecar stores one digest entry per physical KV block in Python
dicts:

```text
layer_name -> physical_block_id -> BlockDigest(digest_min, digest_max, ...)
```

ArkVale's kernel expects a FlashInfer-style paged digest cache:

```text
dg_data
dg_indices
dg_indptr
dg_last_page_len
dg_seq_len
```

Therefore the preliminary feasibility is:

```text
direct_reuse = no
single_request_reuse_after_packing = likely yes
multi_request_reuse_after_packing = not yet; kernel/output contract needs more care
```

## ArkVale Kernel Analysis

### Python Entry Point

File:

```text
/home/han/KV_cache_quant/ArkVale/source/arkvale/kernels.py
```

The wrapper signature is:

```python
def estimate_scores(
    q,                 # [bsz, 1, num_heads, head_dim]
    dg_data,           # [n_max_pages, 2, page_size, n_kv_heads, head_dim]
    dg_indices,        # [bsz, num_pages]
    dg_indptr,         # [bsz + 1]
    dg_last_page_len,  # [bsz]
    dg_seq_len: int,
    layout: str = "NHD",
    n_groups: int = 1,
) -> torch.Tensor:
    ...
```

Expected output:

```text
[bsz, n_groups, dg_seq_len]
```

The source comment says `[bsz, n_groups, n_kv_pages - 1]`; in practice
`dg_seq_len` is the output length passed into the C++ extension.

### C++/CUDA Entry Point

File:

```text
/home/han/KV_cache_quant/ArkVale/source/arkvale_cpp/src/estimate.cu
```

The C++ wrapper checks:

```text
q.ndim == 4
dg_data.ndim == 5
dg_indices.ndim == 2
dg_indptr.ndim == 1
q.size(1) == 1
dg_indices dtype == int32
dg_indptr dtype == int32
```

For `layout == NHD`, it interprets:

```text
dg_data.shape = [n_max_pages, 2, page_size, n_kv_heads, head_dim]
page_size     = dg_data.size(2)
num_kv_heads  = dg_data.size(3)
head_dim      = dg_data.size(4)
```

The output buffer is allocated as:

```text
o = empty([batch_size, num_qo_heads, dg_seq_len], q.options())
```

The kernel then returns:

```text
o.reshape([batch_size, n_groups, num_qo_heads / n_groups, dg_seq_len]).mean(2)
```

So ArkVale's exposed kernel always mean-reduces query heads within each
`n_groups` bucket. It does not expose the per-query-head score tensor.

### Paged Digest Cache Contract

ArkVale uses FlashInfer's `paged_kv_t` abstraction. For `PageStorage::kIndices`
and `NHD` layout, the internal data layout is:

```text
[max_num_pages, 2, page_size, num_heads, head_dim]
```

`dg_indices` is an array of digest-cache page ids, not original KV page ids.
The kernel reads `dg_indices[page_iter]` and uses that value to index `dg_data`.

`dg_indptr` and `dg_last_page_len` define how many digest entries are valid per
batch row:

```text
cur_page_indptr_begin = dg_indptr[batch_idx]
cur_page_indptr_end   = dg_indptr[batch_idx + 1]
cur_last_page_len     = dg_last_page_len[batch_idx]

kv_chunk_len =
  (cur_page_indptr_end - cur_page_indptr_begin - 1) * page_size
  + cur_last_page_len
```

In the digest-score path, `kv_chunk_len` should be understood as the number of
digest entries to score for that batch row.

### Score Formula

The device function `compute_cuboid_scores` loads one digest K vector and one
digest V vector, then computes:

```text
score = sum_i max(q_i * k_i, q_i * v_i)
```

In ArkVale's digest cache, the K plane stores one digest bound and the V plane
stores the other digest bound. ArkVale's `_summarize_keys` returns:

```text
maxs, mins
```

and `save_digests` appends them through `append_paged_kv_cache(*digest, ...)`.
Therefore the score formula is effectively:

```text
score = sum_i max(q_i * digest_max_i, q_i * digest_min_i)
```

This matches the current MPR PyTorch scoring formula.

### Digest Generation in ArkVale

File:

```text
/home/han/KV_cache_quant/ArkVale/source/arkvale/infer_state.py
```

ArkVale summarizes full KV pages with:

```text
raw_max = filled_keys.max(dim=2)
raw_min = filled_keys.min(dim=2)
center  = (raw_max + raw_min) / 2
dist    = mean(abs(center - filled_keys), dim=page_tokens)
maxs    = center + dist
mins    = center - dist
```

For prefill, ArkVale computes digests for filled pages and excludes the partial
last KV page:

```text
n_filled_pages = ceil(q_len / page_size) - 1
```

This agrees with the current MPR Step 1.3/1.4 policy of scoring finalized full
blocks only.

### Dispatch Constraints

The current generated dispatch table is narrow:

```text
head_dim: 128 only
group_size: 1, 4, 8
kv_layout: NHD only
pos_encoding: None only
```

The dtype dispatch supports fp16 by default and bf16 only when ArkVale's
extension is compiled with `FLASHINFER_ENABLE_BF16`.

For the current Qwen3-8B smoke target, `head_dim=128` is likely compatible.
The dtype still needs to be checked against the target ArkVale extension build.

### Multi-Request Caveat

The kernel allocates output with a fixed `dg_seq_len`, but the internal write
offset uses each row's local `kv_chunk_len`:

```text
o + (batch_idx * num_qo_heads + qo_head_idx) * kv_chunk_len
```

This is safe when the batch has one row, or when every row has the same
`kv_chunk_len == dg_seq_len`. It is risky for variable-length multi-request
batches because different rows can have different digest counts.

This matches the current Milestone 1 scope, which is single-request only.
It is not sufficient for general vLLM serving.

## Current vLLM MPR Score-Only Contract

Relevant files:

```text
vllm/v1/mixed_precision_recovery/digest.py
vllm/v1/mixed_precision_recovery/scoring.py
vllm/v1/mixed_precision_recovery/sidecar.py
vllm/model_executor/layers/attention/attention.py
```

### KV Write Observation

The vLLM hook observes KV writes after `do_kv_cache_update`.

For FlashAttention, it assumes:

```text
kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
key_cache = kv_cache[0]
slot_id = physical_block_id * block_size + block_offset
```

The sidecar tracks observed offsets per:

```text
(layer_name, physical_block_id)
```

When all offsets `0..block_size-1` have been observed, it summarizes:

```text
key_cache[physical_block_id]
```

with the same ArkVale-style digest formula.

### Digest Storage

The current sidecar stores:

```text
_digest_cache[layer_name][physical_block_id] = BlockDigest(
    digest_min: [num_kv_heads, head_dim],
    digest_max: [num_kv_heads, head_dim],
    valid_token_count = block_size,
    block_size = block_size,
)
```

This is simple and correct for the single-request smoke scope, but it is not a
kernel-facing layout. It has no paged digest-cache tensor, no digest page ids,
and no indptr/last-page metadata.

### Query Observation

The vLLM score hook observes query before the real attention backend forward.
It scores only:

```text
max_query_len == 1
num_actual_tokens == 1
```

The current query window is keyed only by `layer_name`:

```text
_query_windows[layer_name] = deque(...)
```

and stores decode queries shaped:

```text
[num_q_heads, head_dim]
```

The scoring query is:

```text
window_query = mean(last up to VLLM_MPR_WINDOW_SIZE decode queries)
```

### Current PyTorch Scoring

Current tensor-only scoring inputs:

```text
query_window: [num_q_heads, head_dim]
digest_min:   [num_blocks, num_kv_heads, head_dim]
digest_max:   [num_blocks, num_kv_heads, head_dim]
```

Current output:

```text
scores: [num_blocks]
```

The formula matches ArkVale's cuboid score:

```text
per_query_head_score =
  sum_i max(q_i * digest_max_i, q_i * digest_min_i)
```

Current aggregation is configurable:

```text
VLLM_MPR_SCORE_AGG=max|mean
default=max
```

ArkVale's exposed kernel supports the `mean` behavior, not the current default
`max` behavior.

### Request/Block Debug Alignment

Step 1.5/1.6 debug fields compute:

```text
valid_block_ids     = block_table[:ceil(seq_len / block_size)]
finalized_block_ids = block_table[:floor(seq_len / block_size)]
observed_digest_block_ids
missing_digest_blocks
extra_digest_blocks
```

Strict validation currently requires:

```text
set(observed_digest_block_ids) == set(finalized_block_ids)
missing_digest_blocks == []
extra_digest_blocks == []
```

This makes the current single-request smoke compatible with request-local
scoring, but the implementation still packs all cached layer digests. The strict
validation is the guard that makes those equivalent in the smoke scope.

## First-Pass Compatibility Matrix

| Area | vLLM MPR current state | ArkVale kernel expectation | Compatibility |
|---|---|---|---|
| Query shape | `[num_q_heads, head_dim]` rolling `window_query` | `[bsz, 1, num_q_heads, head_dim]` | Compatible after `unsqueeze(0).unsqueeze(1)` for single request |
| Query semantics | Rolling average of recent decode queries | Current query tensor supplied by caller | Compatible if we intentionally treat `window_query` as `q`; not identical to ArkVale runtime policy |
| Digest formula | ArkVale-style min/max over full vLLM block | ArkVale-style max/min digest in K/V planes | Compatible |
| Digest storage | Python dict by physical block id | Paged tensor `dg_data` | Not directly compatible |
| Digest unit | One vLLM physical block | One digest entry, usually representing one KV page | Compatible under Milestone 1 block == page assumption |
| Partial last block | Excluded from score candidates | Prefill excludes partial last KV page | Compatible |
| Block ids | vLLM physical block ids | `dg_indices` are digest-cache page ids | Not directly compatible; output index to physical block id mapping is required |
| Metadata | Debug fields derive finalized block ids | `dg_indptr`, `dg_last_page_len`, `dg_seq_len` | Compatible after packing for single request |
| Output | `[num_blocks]` | `[bsz, n_groups, dg_seq_len]` | Compatible only for `bsz=1`, `n_groups=1`, then squeeze |
| Aggregation | `max` or `mean`, default `max` | Mean within each `n_groups` bucket | Compatible with `score_agg=mean`; not compatible with default `max` |
| Layout | FlashAttention KV cache `[2, blocks, block_size, heads, dim]`; sidecar digest `[blocks, heads, dim]` | NHD paged digest cache `[pages, 2, page_size, heads, dim]` | Requires packing |
| Dtype | vLLM model/query dtype, likely fp16 or bf16 depending config | fp16; bf16 only if compiled with BF16 | Needs target build check or cast policy |
| Head dim | Qwen3-8B likely 128 | generated dispatch supports 128 only | Likely compatible for current target |
| Multi-request | Explicitly skipped | Kernel accepts batch metadata but output stride is risky for variable lengths | Not compatible for general serving yet |

## What A Minimal Packing Layer Would Need To Produce

This is not yet the adapter design. It is the minimum data-contract implication
from the compatibility analysis.

For one layer and one request:

```text
candidate_block_ids = finalized_block_ids
N = len(candidate_block_ids)
dg_page_size = block_size or another chosen digest-cache page size
n_dg_pages = ceil(N / dg_page_size)
```

The packing layer would need to build:

```text
q:
  [1, 1, num_q_heads, head_dim]

dg_data:
  [n_dg_pages, 2, dg_page_size, num_kv_heads, head_dim]

dg_indices:
  [1, n_dg_pages]

dg_indptr:
  [0, n_dg_pages]

dg_last_page_len:
  [((N - 1) % dg_page_size) + 1]

dg_seq_len:
  N

score_index_to_physical_block_id:
  list[int] of length N
```

The digest tensor packing must place:

```text
dg_data[digest_page, 0, digest_offset] = digest_max
dg_data[digest_page, 1, digest_offset] = digest_min
```

because ArkVale scores `max(q * K, q * V)` and ArkVale stores `(maxs, mins)` as
the digest K/V pair.

Important: `dg_indices` should point to rows in `dg_data`. It should not contain
vLLM physical block ids unless `dg_data` itself is allocated with those physical
block ids as digest page ids, which would be wasteful and would still not solve
the output-position mapping.

## Preliminary Feasibility Judgment

### Direct Reuse

```text
No.
```

The current sidecar has the right mathematical ingredients, but not the right
kernel input layout.

### Single-Request Reuse After Packing

```text
Likely yes, with score_agg=mean or an ArkVale wrapper change.
```

For the current Milestone 1 smoke scope:

```text
single GPU
single request
FlashAttention
head_dim=128
full-block digest candidates
strict current-request score validation
```

the kernel can probably replace the PyTorch scoring core after a packing layer
is added.

### Main Blockers Before Implementation

1. **Aggregation mismatch**

   Current MPR default is `VLLM_MPR_SCORE_AGG=max`. ArkVale's exposed
   `estimate_scores` returns mean-reduced group scores. We must either:

   ```text
   use score_agg=mean for ArkVale-kernel mode
   or expose per-query-head scores from the C++ wrapper and aggregate in Python
   or add max aggregation support to the C++ wrapper
   ```

2. **Digest page id mapping**

   vLLM physical block ids cannot be passed as `dg_indices` directly. The
   packing layer must maintain:

   ```text
   score index -> physical block id
   ```

3. **Request-local filtering**

   The current `_pack_layer_digests` packs every cached layer digest. The kernel
   path should pack only request-owned finalized blocks, using the block table.
   The current strict single-request validation confirms equivalence only for
   smoke runs.

4. **Dtype/build compatibility**

   If the target vLLM run uses bf16 queries/digests, the ArkVale extension must
   have BF16 enabled or the packing layer must cast to fp16. Casting changes
   score numerics and should be explicit.

5. **Multi-request output layout**

   The current ArkVale kernel should not be treated as general variable-length
   vLLM batch support. Multi-request support needs either equal digest lengths,
   one-request-at-a-time kernel calls, or a kernel/wrapper fix.

6. **Lifecycle/ownership**

   This is not a kernel-input blocker for Step 1.7, but it remains a correctness
   blocker before CPU backup/recovery. Persistent sidecar state keyed only by
   `(layer_name, physical_block_id)` can become stale after vLLM block reuse.

## Recommended Next Step

Before implementing an adapter, decide two policy points:

1. Should ArkVale-kernel mode initially force `VLLM_MPR_SCORE_AGG=mean`, or
   should we modify/expose the kernel path to support `max` aggregation?
2. Should the first packing prototype allocate a temporary compact `dg_data`
   per score call, or should it introduce a persistent paged digest tensor in
   the sidecar?

After those decisions, Step 1.7 can move from compatibility analysis to an
adapter design sketch.
