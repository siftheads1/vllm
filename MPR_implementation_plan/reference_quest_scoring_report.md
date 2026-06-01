# Quest Scoring Reference Report

Last updated: 2026-06-01

## Scope

This note summarizes Quest's scoring and page-selection mechanism as a
reference for MPR importance scoring and future precision allocation.

Source tree:

```text
/home/han/KV_cache_quant/quest
```

Primary files:

```text
quest/models/QuestAttention.py
quest/utils/__init__.py
quest/utils/controller.py
quest/utils/kv_cache.py
quest/ops/csrc/estimate.cu
quest/ops/csrc/page.cu
kernels/include/decode/decode_attn.cuh
kernels/include/decode/decode_page.cuh
quest/tests/test_estimate.py
quest/tests/test_approx_attention.py
README.md
assets/quest_paper.pdf
```

## High-Level Summary

Quest is a query-aware page-sparse attention method. It does not assign a
static importance to KV tokens. Instead, for each decode query, it estimates
which KV pages could contain high-attention tokens, selects top-k pages, and
then runs sparse attention over only those pages plus the current last page.

The decode flow in `QuestAttention.forward` is:

```text
query/key/value projection
apply RoPE
append new K/V and update metadata cache

if prefill:
  full prefill attention
else if no page reduction is needed:
  full decode attention over pages except manually handled last page
else:
  decode_estimate(...)
  decode_topk(...)
  decode_sparse_attn(..., topk pages)
```

The important part for MPR is `decode_estimate`: it produces query-dependent
per-page scores.

## Scoring Formula

Quest keeps per-page channel-wise key extrema:

```text
M_i = max key value in page at channel i
m_i = min key value in page at channel i
```

For a decode query vector `q`, Quest estimates a page's criticality as:

```text
score(page) = sum_i max(q_i * M_i, q_i * m_i)
```

This is an upper bound on the largest possible `q · k` score for any key in the
page. If `q_i` is positive, the max key value gives the largest contribution.
If `q_i` is negative, the min key value gives the largest contribution.

The Quest paper describes this as a page criticality estimate from the
channel-wise minimal and maximal key values. The local PDF states that Quest
computes:

```text
U_i = max(Q_i * m_i, Q_i * M_i)
score = sum_i U_i
```

The implementation matches this exactly:

```text
compute_max_possible:
  max_possible += max(q_vec[i] * max_vec[i], q_vec[i] * min_vec[i])
```

## CPU Reference

`quest/tests/test_estimate.py` is the clearest reference implementation.

It transforms keys based on the query sign:

```python
sign = (q > 0) + (~(q > 0)) * -1
max_key = k * sign
positive_query = q * sign
```

Then it computes page maxima:

```python
page_max_key = max_key.reshape(
    num_heads,
    num_pages,
    page_size,
    head_dim,
).amax(dim=-2)
```

And finally:

```python
approx_attn_paged =
    positive_query @ page_max_key.transpose(1, 2)
```

This is equivalent to:

```text
sum_i max(q_i * page_max_i, q_i * page_min_i)
```

The reference removes the last page from the estimate:

```python
approx_attn_paged = approx_attn_paged[:, :, :cur_num_pages - 1]
```

## Metadata Cache

Quest maintains two paged caches:

```text
kv_cache:
  real K/V pages

metadata_cache:
  page-level key max/min metadata
```

Both are instances of the same `KvCache` abstraction. The metadata cache has
sequence length equal to the number of KV pages, not the number of tokens.

The metadata page layout is still paged-KV-like:

```text
[num_layers, capacity, 2, page_size, num_heads, head_dim]
```

but semantically:

```text
metadata K plane = page max key vector
metadata V plane = page min key vector
```

During KV append, Quest updates both:

```text
append_kv_cache_prefill / append_kv_cache_decode
  writes real K/V into kv_cache
  updates page max/min into metadata_cache
```

The append kernels use channel-wise reductions:

```text
local_max = max(local_max, local_k)
local_min = min(local_min, local_k)
```

This means score-time estimation reads compact metadata instead of scanning
real KV pages.

## Estimate Kernel Contract

Python wrapper:

```python
decode_estimate(q, iController, layer_idx)
```

Input:

```text
q: [1, num_heads, head_dim]
metadata_data: paged cache for max/min metadata
metadata_indices: active metadata page ids
metadata_indptr
metadata_last_page_len
metadata_last_page_idx
layout: NHD
```

Output:

```text
[num_heads, metadata_cache.seqlen - 1]
```

The `-1` excludes the current last metadata entry. Quest manually handles the
current last KV page during sparse attention.

The C++ entry point is:

```text
estimate_attn_score(...)
```

It calls:

```text
MaxPossibleSampleWithPagedKVCache(...)
```

The kernel computes page scores independently per query head. It does not reduce
scores across heads.

## Top-K Selection

After estimation, Quest calls:

```python
decode_topk(estimated_attn_score, iController)
```

`estimated_attn_score` is shaped:

```text
[num_heads, num_candidate_pages]
```

`topk_filtering` selects top pages per head:

```text
topk_dindices_buffer: [num_heads, page_budget - 1]
```

The selected page ids are then passed to decode sparse attention. The last page
is excluded from top-k selection and handled separately.

Therefore Quest's basic artifact is:

```text
layer, head, page -> score
```

not a single layer-level page score.

## Relation To ArkVale And Current MPR

Quest, ArkVale, and the current MPR prototype all use a cuboid-style scoring
shape:

```text
sum_i max(q_i * upper_i, q_i * lower_i)
```

The difference is what `upper/lower` mean.

| System | Metadata | Score Meaning |
|---|---|---|
| Quest | raw per-page key max/min | upper bound on the largest possible `q · k` in the page |
| ArkVale | summarized digest max/min | approximate page digest score |
| Current MPR | ArkVale-style digest per full vLLM block | approximate block/page score |

Quest's score has a clearer interpretation for precision allocation because it
directly estimates the maximum possible attention logit in a page for the
current query.

## MPR Implications

Quest is a strong reference baseline for MPR precision allocation:

```text
high Quest score   -> fp16 / high precision recovery
medium Quest score -> low precision recovery
low Quest score    -> skip
```

Potential options:

1. Keep the current ArkVale-style digest score.

   This preserves the existing MPR Step 1 implementation path and stays close
   to ArkVale.

2. Add Quest-style raw min/max metadata.

   This would make the score an actual upper bound on page attention logits and
   may be more interpretable for precision allocation.

3. Record both scores in an experiment.

   This is probably the cleanest research path: compare ArkVale-style digest
   scores, Quest-style upper-bound scores, and eventually DiffKV-style scores
   against attention/top-k recall or recovery quality.

## Caveats

Quest's open-source implementation is intentionally narrow:

```text
batch size 1
decode-first optimization
page-level selection
per-head top-k
NHD layout
first two layers skipped by default in the end-to-end model wrapper
```

The current implementation also assumes a page budget rather than assigning
precision tiers. For MPR, the score would need to be converted into tier
thresholds or per-step budget allocation.

## Current Takeaway

For MPR, Quest should be treated as the cleanest reference for query-aware page
importance scoring:

```text
score(page | query) = upper bound on page attention logit
```

It is more directly interpretable than the current ArkVale-style digest score,
but it requires maintaining raw page max/min metadata rather than the current
center-distance digest.
