# Milestone 1 Step 1.7 Modular Scoring Implementation Report

Last updated: 2026-06-01

## Scope

This note records the first implementation pass after the ArkVale/Quest/DiffKV
scoring review. The goal is not to add ArkVale CUDA kernel packing yet. The goal
is to make the current MPR score-only path easier to switch, ablate, and extend.

## Implemented Defaults

The default behavior remains compatible with the existing Milestone 1 smoke path:

```text
VLLM_MPR_SCORING_BACKEND=torch_quest
VLLM_MPR_DIGEST_KIND=raw_minmax
VLLM_MPR_SCORE_AGG=max
VLLM_MPR_SCORE_GRANULARITY=kv_head
```

`torch_quest` means the PyTorch reference scorer computes the Quest/ArkVale
cuboid score:

```text
score = sum_i max(q_i * digest_max_i, q_i * digest_min_i)
```

`raw_minmax` digest kind is the default for Quest-style metadata:

```text
digest_max = raw_max
digest_min = raw_min
```

`arkvale` digest kind remains available for comparison with the previous
Milestone 1 tightened bounds:

```text
raw_max = key_block.max(dim=0)
raw_min = key_block.min(dim=0)
center = (raw_max + raw_min) / 2
dist = mean(abs(center - key_block), dim=block_tokens)
digest_max = center + dist
digest_min = center - dist
```

## Module Structure

Updated modules:

```text
vllm/v1/mixed_precision_recovery/config.py
  MPRConfig.scoring_backend
  MPRConfig.digest_kind
  MPRConfig.score_granularity

vllm/v1/mixed_precision_recovery/digest.py
  summarize_key_block(..., digest_kind=...)
  arkvale digest
  raw_minmax digest

vllm/v1/mixed_precision_recovery/scoring.py
  DigestScoringBackend protocol
  TorchQuestScorer
  DigestScoreResult
  estimate_query_head_digest_scores(...)
  aggregate_query_head_scores(...)
  estimate_digest_score_result(...)
  estimate_digest_scores(...) legacy wrapper

vllm/v1/mixed_precision_recovery/sidecar.py
  RecoverySidecar resolves configured scoring backend in __post_init__
  observe_kv_write creates configured digest kind
  observe_query calls backend.estimate(...)

vllm/envs.py
  VLLM_MPR_SCORING_BACKEND
  VLLM_MPR_DIGEST_KIND
  VLLM_MPR_SCORE_GRANULARITY
```

## Call Chain

KV digest path:

```text
unified_kv_cache_update(...)
  -> _maybe_observe_mpr_kv_write(...)
  -> RecoverySidecar.observe_kv_write(...)
  -> RecoverySidecar._observe_block_offsets(...)
  -> summarize_key_block(key_cache[physical_block_id], digest_kind=...)
  -> _digest_cache[layer_name][physical_block_id] = BlockDigest(...)
```

Query scoring path:

```text
unified_attention_with_output(...)
  -> _maybe_observe_mpr_query(...)
  -> RecoverySidecar.observe_query(...)
  -> _pack_layer_digests(...)
  -> TorchQuestScorer.estimate(...)
  -> estimate_query_head_digest_scores(...)
  -> aggregate_query_head_scores(...)
  -> torch.topk(block_scores)
  -> optional head-local top-k when score_granularity is kv_head/query_head
  -> score_estimated JSONL
```

## GQA Policy

For GQA/MQA, query heads are mapped to KV heads as contiguous groups:

```text
group_size = num_q_heads // num_kv_heads
kv_head = q_head // group_size
```

The scorer now exposes:

```text
per_query_head_scores: [num_blocks, num_q_heads]
per_kv_head_scores:    [num_blocks, num_kv_heads]
block_scores:          [num_blocks]
```

With `score_agg=max`, aggregation is conservative:

```text
per_kv_head_score = max over query heads in the KV group
block_score = max over KV groups
```

This implements the agreed ad-hoc GQA union policy for the current block-level
top-k path.

With `score_agg=mean`, aggregation remains ArkVale-comparison friendly:

```text
per_kv_head_score = mean over query heads in the KV group
block_score = mean over KV groups
```

## Score Granularity

The runtime debug output can now expose top-k candidates at three granularities:

```text
VLLM_MPR_SCORE_GRANULARITY=block
  topk_block_ids/topk_scores use block_scores: [num_blocks]

VLLM_MPR_SCORE_GRANULARITY=kv_head
  topk_block_ids/topk_scores still use block_scores for backward compatibility
  topk_block_ids_by_head/topk_scores_by_head use:
    per_kv_head_scores: [num_blocks, num_kv_heads]

VLLM_MPR_SCORE_GRANULARITY=query_head
  topk_block_ids/topk_scores still use block_scores for backward compatibility
  topk_block_ids_by_head/topk_scores_by_head use:
    per_query_head_scores: [num_blocks, num_q_heads]
```

The current default Quest-style head-level smoke is:

```text
VLLM_MPR_DIGEST_KIND=raw_minmax
VLLM_MPR_SCORE_GRANULARITY=kv_head
VLLM_MPR_SCORE_AGG=max
```

`kv_head` is closer to the eventual KV cache precision/recovery unit under GQA.
`query_head` is useful for analysis, but it is not directly a KV storage unit.

## Debug Schema Additions

New optional JSONL fields:

```text
observe_kv_write:
  digest_kind

digest_created:
  digest_kind

score_skipped:
  scoring_backend
  digest_kind

score_estimated:
  scoring_backend
  digest_kind
  num_q_heads
  num_kv_heads
  gqa_group_size
  score_granularity
  num_score_heads
  head_score_count
  topk_block_ids_by_head
  topk_scores_by_head
```

The head-level fields are present only for `score_granularity=kv_head` or
`query_head`. The validator accepts and checks these optional fields while
preserving compatibility with older JSONL files.

## Validation

Completed locally:

```text
python -m py_compile ... selected MPR source/test/validator files
git diff --check
direct importlib smoke for digest.py and scoring.py
```

Full pytest in this local base environment is blocked before MPR tests run:

```text
tests/conftest.py imports transformers/sklearn/scipy
local NumPy is 2.3.5
installed SciPy/sklearn extension was built against NumPy 1.x
```

Running MPR tests with parent conftest disabled gets past that issue, but this
base environment is missing `cbor2`, which is required by the local vLLM import
chain. The target vLLM environment should run:

```bash
python -m pytest \
  tests/v1/mixed_precision_recovery/test_digest.py \
  tests/v1/mixed_precision_recovery/test_scoring.py \
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py \
  -q
```

## Next Adapter Work

The next Step 1.7 implementation decision is ArkVale kernel adapter shape:

```text
temporary per-score packing into ArkVale dg_data/dg_indices
vs.
persistent sidecar paged digest storage
```

For the current codebase, temporary single-request packing is still the safer
first adapter experiment.
