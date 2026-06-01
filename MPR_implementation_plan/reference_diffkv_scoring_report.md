# DiffKV Scoring Reference Report

Last updated: 2026-06-01

## Scope

This note summarizes DiffKV's token-importance scoring and high/low precision
allocation policy as a reference for MPR precision allocation.

Source tree:

```text
/home/han/KV_cache_quant/DiffKV
```

Note: the local DiffKV tree has uncommitted changes. This report reflects the
current local files at analysis time.

Primary files:

```text
README.md
vllm/model_executor/layers/sparse_attention_big_kernel.py
vllm/model_executor/layers/sparse_attention_small_kernel.py
vllm/model_executor/layers/triton_flash_attention.py
vllm/model_executor/layers/triton_fused_softmax_sum.py
csrc/cache_kernels.cu
csrc/long_prompt_cache_kernels.cu
csrc/attention/sparse_attention_kernels.cu
csrc/attention/cache_utils.h
vllm/config.py
vllm/core/block_manager.py
vllm/worker/cache_engine.py
vllm/sequence.py
```

## High-Level Summary

DiffKV is not a Quest/ArkVale-style digest scorer. Quest and ArkVale estimate
page criticality from query and page metadata. DiffKV instead uses actual
attention probability mass as the importance signal.

At a high level:

```text
importance(token, kv_head) =
  accumulated post-softmax attention probability assigned to that token

GQA aggregation =
  max over query heads that share the KV head

allocation =
  keep recent kv_buffer_size tokens at high precision
  prune very low-score old tokens
  store medium-score old tokens at low precision
  store remaining tokens at high precision
```

The implemented cache has two precision regions per KV head:

```text
left side of block table:
  high-precision pages

right side of block table:
  low-precision pages
```

This is a two-tier policy plus pruning, not a general multi-bit allocator.

## Prompt-Phase Scoring

In the big-kernel path, `SparsePagedAttention.forward` computes prompt attention
with a modified Triton FlashAttention path:

```text
triton_attention(...)
  returns triton_out
  returns qk_products for the last NUM_TOKENS_SCORE query tokens

triton_fused_softmax_sum(qk_products)
  converts those qk rows to softmax probabilities
  sums them over the sampled query rows
```

`NUM_TOKENS_SCORE` is 64. This avoids storing the full prompt attention matrix
while still obtaining a recent-query attention-mass estimate for every prompt
token.

The resulting score tensor is:

```text
triton_score_sum: [batch_size, num_heads, max_prompt_len]
```

For the normal prompt path, `compress_and_append_cache_prompt_phase` receives
the per-query-head score tensor and performs the GQA reduction in CUDA:

```text
score_ = max_j score[query_head_j, token_idx]
```

For the long-prompt path, Python first reshapes:

```text
[batch_size, num_heads, max_prompt_len]
  -> [batch_size, num_kv_heads, num_queries_per_kv, max_prompt_len]
```

and computes:

```text
max_score_sum = max over num_queries_per_kv
```

Then the long-prompt CUDA kernel consumes `[batch_size, num_kv_heads,
max_prompt_len]`.

## Prompt Allocation Rule

DiffKV computes a per-token mean score:

```text
ideal_num_queries = prompt_len - token_idx
real_num_queries = min(ideal_num_queries, NUM_TOKENS_SCORE)
mean_score = accumulated_score / real_num_queries
```

Thresholds are request-specific values from `compress_config_tables` scaled by
sequence length:

```text
base_threshold = 1.0 / prompt_len
prune_threshold = base_threshold * prune_alpha
quant_threshold = base_threshold * quant_alpha
```

The current sequence object names these values `compress_configs` and requires:

```text
[prune_alpha, quant_alpha]
0 <= prune_alpha <= 1
quant_alpha >= prune_alpha
```

Decision:

```text
if token is within the recent kv_buffer_size:
  keep high precision
else if mean_score < prune_threshold:
  prune
else if mean_score < quant_threshold:
  store low precision
else:
  store high precision
```

Prompt tokens are sorted by mean score. Pruned tokens are skipped, the next
lowest-scoring tokens are written to low-precision pages, and the remaining
tokens are written to high-precision pages.

The cache metadata records the resulting lengths:

```text
kv_len_tables[..., 0] = high_precision_token_count
kv_len_tables[..., 1] = low_precision_token_count
```

## Decode-Phase Scoring

DiffKV's decode path has two stages inside `SparsePagedAttention.forward`:

```text
1. compress_and_append_cache_decode_phase(...)
   decide whether to prune, demote, or append before the current attention

2. sparse_paged_attention(...)
   attend over high/low cache pages
   update each cached token's accumulated score with current softmax mass
```

The CUDA attention kernel writes current-token attention probabilities into
`tmp_scores`. The reduce kernel then updates the in-cache score field:

```text
score(token) += max_j tmp_scores[query_head_j, token]
```

where `j` ranges over query heads in the same GQA group.

The decode compression kernel reads cached scores and positions, computes:

```text
num_queries = current_position - cached_position
mean_score = cached_score / num_queries
```

and finds the lowest-score eligible high-precision token and lowest-score
eligible low-precision token. Recent tokens are protected by:

```text
num_queries >= kv_buffer_size
```

The decode thresholds are:

```text
base_threshold = 1.0 / current_position
prune_threshold = base_threshold * prune_alpha
quant_threshold = base_threshold * quant_alpha
```

The policy has four practical cases:

```text
if min_high_score < prune_threshold and min_high_score < min_low_score:
  prune the high-precision victim
else if min_high_score < quant_threshold and min_low_score < prune_threshold:
  demote the high-precision victim into the low-precision victim's slot
else if min_high_score < quant_threshold:
  demote the high-precision victim into a new low-precision slot
else:
  append the new token at high precision
```

The new token's score is initialized to zero when appended and is updated by the
subsequent `sparse_paged_attention` stage. Therefore, precision decisions use
scores accumulated up to the previous attention step, then the current step's
attention mass is added.

## Cache Layout

DiffKV uses a unified `torch.int16` cache block, interpreted as `uint16_t` in
CUDA. Each physical block stores packed data and metadata:

```text
key payload
key quant metadata: scale, zero_point
value payload
value quant metadata: scale, zero_point
score: float32
position: int32
```

This is materially different from standard vLLM KV layout and from MPR's current
sidecar digest representation.

The block size is selected in `CacheConfig.compute_cache_block_size` by scanning
candidate byte sizes and minimizing worst-case residual fragmentation across the
supported heterogeneous KV quant configurations. The resulting block byte size
is shared, while each quant config has a different number of tokens per block.

Supported KV cache bit/group tuples include:

```text
(8, 8, 1, 1)
(8, 4, 1, 2)
(8, 4, 1, 1)
(8, 2, 1, 1)
(4, 4, 1, 1)
(4, 2, 2, 4)
(4, 2, 1, 1)
(4, 1, 1, 1)
```

For a two-tier request, `quant_configs` and `quant_groups` specify:

```text
[kbits_high, vbits_high, kbits_low, vbits_low]
[kgroups_high, vgroups_high, kgroups_low, vgroups_low]
```

If only one precision tuple is provided, DiffKV duplicates it and requires the
prune and quant thresholds to match.

## Relation To Quest And ArkVale

Quest/ArkVale scoring:

```text
query-aware page score
score(page) = sum_i max(q_i * page_max_i, q_i * page_min_i)
input is compact page metadata/digest
score is an estimate or upper bound before attention
```

DiffKV scoring:

```text
attention-mass token score
score(token) += post-softmax attention probability
input is real attention computation over retained KV
score is accumulated after attention
```

The biggest conceptual difference is timing. Quest/ArkVale use the current query
to choose pages before sparse attention. DiffKV updates token importance after
the attention computation and uses the accumulated score to make future cache
precision/pruning decisions.

## Implications For MPR

Useful pieces for MPR:

```text
1. DiffKV is a strong reference for using actual attention mass as an
   importance/oracle signal.

2. The GQA aggregation is max over query heads, matching MPR's current default
   more closely than ArkVale's exposed mean aggregation.

3. The alpha / sequence_length threshold heuristic is a simple baseline for
   precision allocation after MPR has a stable score source.

4. The recent-token buffer rule is practical and should be considered for MPR:
   recent blocks/tokens may be excluded from demotion regardless of score.

5. DiffKV's separate K/V bit choices are relevant to MPR if precision allocation
   later distinguishes K precision from V precision.
```

Parts that are not directly portable:

```text
1. DiffKV's score requires actual softmax attention probabilities, not only a
   digest cache.

2. Its score is token-level, while current MPR digest scoring is block-level.
   A token-to-block aggregation rule would be required before using DiffKV-style
   scores for MPR block precision.

3. Its CUDA kernels assume DiffKV's unified packed cache layout with score and
   position embedded in each block.

4. Its allocation policy is tightly coupled to two precision regions plus prune.
   It is not immediately a general adapter for arbitrary mixed precision tiers.
```

## Working Conclusion

DiffKV should be treated as a precision-allocation reference, not as a kernel
reuse candidate for the current MPR digest-scoring path.

For MPR, the likely adaptation path is:

```text
Quest/ArkVale-style digest score:
  candidate runtime score source

DiffKV-style accumulated attention mass:
  calibration signal or optional oracle/debug score source

DiffKV-style threshold policy:
  candidate baseline allocation rule after score semantics are fixed
```

Open decisions before using DiffKV ideas in MPR:

```text
1. Should MPR precision allocation be block-level only, or maintain token-level
   auxiliary scores inside each block?

2. If using token-level attention mass as a reference, how should token scores
   be aggregated to a block precision decision: max, mean, sum, or percentile?

3. Should MPR initially keep a recent-token or recent-block high-precision
   buffer independent of digest score?

4. Should DiffKV's alpha / sequence_length thresholds be used as the first
   allocation baseline, or should MPR start with top-k/top-ratio allocation?
```
