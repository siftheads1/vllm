# Milestone 1 Results

Milestone 1 implemented a score-only Mixed-Precision Recovery sidecar in vLLM.
It observes KV writes, builds per-block key digests, scores finalized KV blocks
during decode, and records debug/validation artifacts without changing the KV
cache, attention output, scheduler, CPU offload, or recovery behavior.

## Decision

```text
Decision: Proceed to Milestone 2
```

Rationale:

```text
score-only sidecar works in the vLLM v1 FlashAttention decode path
slot/block metadata is visible and validated through JSONL
digest entries are created per finalized vLLM physical KV block
Quest-style scoring works through both PyTorch and CUDA backends
quest_cuda per-call packing overhead was removed for the single-request prefix path
attention/KV mutation remains out of scope and untouched
```

Milestone 2 can start from the current score-only output and design the CPU
fp16 backup layout, lifecycle, and recovery materialization policy.

## Implemented Hooks

KV write observation:

```text
vllm/model_executor/layers/attention/attention.py
  unified_kv_cache_update(...)
```

This hook observes:

```text
layer_name
key/value shapes
kv_cache
slot_mapping
block_size
```

It converts valid slot ids into:

```text
physical_block_id = slot_id // block_size
block_offset      = slot_id % block_size
```

Decode query scoring:

```text
vllm/model_executor/layers/attention/attention.py
  unified_attention_with_output(...)
```

This hook observes:

```text
layer_name
query
attn_metadata
block_table
seq_lens
```

It records scores only. The real attention call and output tensor are not
modified.

## Sidecar API

Main module:

```text
vllm/v1/mixed_precision_recovery/
```

Important files:

```text
config.py          environment-backed MPR config
sidecar.py         RecoverySidecar state and hook handlers
digest.py          per-block digest construction
scoring.py         torch_quest and quest_cuda scoring backends
quest_packing.py   Quest metadata packing and persistent metadata store
debug.py           JSONL debug writer
```

Main runtime calls:

```python
get_mpr_sidecar().observe_kv_write(...)
get_mpr_sidecar().observe_query(...)
```

Feature flags:

```text
VLLM_MPR_ENABLE
VLLM_MPR_DEBUG_DIR
VLLM_MPR_TOPK
VLLM_MPR_MAX_LAYERS
VLLM_MPR_MAX_STEPS
VLLM_MPR_DUMP_EVERY
VLLM_MPR_WINDOW_SIZE
VLLM_MPR_RECENT_TOKENS
VLLM_MPR_SCORE_AGG
VLLM_MPR_SCORING_BACKEND
VLLM_MPR_DIGEST_KIND
VLLM_MPR_SCORE_GRANULARITY
```

Current defaults:

```text
enabled = false
topk = 8
window_size = 64
recent_tokens = 64
score_agg = max
scoring_backend = torch_quest
digest_kind = raw_minmax
score_granularity = kv_head
```

## Digest State

The sidecar creates one digest entry per finalized vLLM physical KV block:

```text
(layer_name, physical_block_id)
  -> digest_min [num_kv_heads, head_dim]
  -> digest_max [num_kv_heads, head_dim]
  -> valid_token_count
  -> block_size
```

Two digest kinds are available:

```text
raw_minmax
  Quest-style raw per-block key min/max

arkvale
  ArkVale-style tightened cuboid bounds
```

The current scoring path defaults to `raw_minmax` because it aligns directly
with Quest's estimate kernel and the current CUDA backend.

## Scoring Backends

Implemented backends:

```text
torch_quest
  PyTorch reference implementation of Quest-style raw_minmax scoring

quest_cuda
  vLLM custom-op binding to an estimate-only Quest CUDA kernel subset
```

Score semantics:

```text
score(q, block) = sum_i max(q_i * digest_max_i, q_i * digest_min_i)
```

For GQA/MQA, query heads are grouped under their KV head. The default
aggregation is conservative:

```text
query heads -> KV head: max
KV heads -> block score: max
```

The debug path can expose:

```text
block_scores
per_kv_head_scores
per_query_head_scores
```

## Quest CUDA Kernel Path

Vendored/derived CUDA pieces:

```text
csrc/mpr/quest_estimate.cu
csrc/mpr/quest_estimate_kernel.cuh
csrc/mpr/quest_paged_kv.cuh
csrc/third_party/flashinfer/include/flashinfer/*
```

The first binding mirrors Quest's exposed estimate path closely:

```text
single-request compact metadata
NHD layout
group_size in {1, 4, 8}
head_dim in {64, 128, 256}
fp16 only for now
```

Known dtype limitation:

```text
quest_cuda currently supports fp16 query/metadata tensors only.
Qwen3-8B with dtype=auto may produce bf16 queries, so current quest_cuda smoke
uses --dtype half.

BF16 is future work. The lower FlashInfer vector headers include nv_bfloat16
support, but the original Quest exposed wrapper only dispatches fp16. MPR needs
explicit torch::kBFloat16 -> nv_bfloat16 dispatch and parity tests before
enabling bf16.
```

## Persistent Quest Metadata Fast Path

Initial `quest_cuda` microbench showed the estimate kernel was fast, but the
backend spent most time rebuilding Quest metadata on every score call.

Added:

```text
QuestMetadataStore
  append-only per-layer Quest metadata cache
  stores [num_pages, 2, page_size, num_kv_heads, head_dim]
  appends a zero guard entry because Quest estimate drops the final logical entry

QuestCudaScorer.estimate_packed(...)
  scores from already packed Quest metadata

RecoverySidecar fast path
  uses persistent packed metadata when score candidates match the store prefix
  falls back to compact packing if prefix/dtype/device/page-size checks fail
```

Runtime JSONL confirmed:

```text
quest_packed_fast_path counts: Counter({True: 910})
fallback_reason: None
```

This means the tested single-request run used the persistent metadata fast path
for every score event.

## Debug Output Format

The JSONL debug stream records event types including:

```text
init
observe_kv_write
digest_created
score_skipped
score_estimated
```

Important `score_estimated` fields:

```text
layer_name
query_shape
window_query_shape
window_query_len
num_digest_blocks
score_count
score_agg
scoring_backend
digest_kind
num_q_heads
num_kv_heads
gqa_group_size
score_granularity
topk
topk_block_ids
topk_scores
num_score_heads
head_score_count
topk_block_ids_by_head
topk_scores_by_head
num_reqs
seq_lens
block_size
block_table_row
valid_block_ids
finalized_block_ids
recent_tokens
protected_tail_entries
protected_block_ids
score_candidate_block_ids
observed_digest_block_ids
missing_digest_blocks
extra_digest_blocks
quest_packed_fast_path
quest_packed_fallback_reason
```

Sample fast-path excerpt:

```text
{
  "event": "score_estimated",
  "score_count": 1,
  "scoring_backend": "quest_cuda",
  "digest_kind": "raw_minmax",
  "score_granularity": "kv_head",
  "quest_packed_fast_path": true,
  "quest_packed_fallback_reason": null
}
```

## Validation Commands

Unit tests:

```bash
/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py
```

Observed result:

```text
21 passed, 1 skipped
```

CUDA debug smoke:

```bash
rm -rf /tmp/mpr_debug_cuda

env -u VLLM_MPR_MAX_STEPS \
VLLM_MPR_ENABLE=1 \
VLLM_MPR_SCORING_BACKEND=quest_cuda \
VLLM_MPR_DEBUG_DIR=/tmp/mpr_debug_cuda \
VLLM_MPR_DUMP_EVERY=1 \
VLLM_MPR_MAX_LAYERS=2 \
python scripts/mpr_baseline_qwen3_8b.py \
  --dtype half \
  --max-tokens 512 \
  --ignore-eos
```

JSONL validation:

```bash
python scripts/mpr_validate_debug_jsonl.py \
  /tmp/mpr_debug_cuda/*.jsonl \
  --min-score-events 1
```

Fast-path check:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

counts = Counter()
for path in sorted(Path("/tmp/mpr_debug_cuda").glob("*.jsonl")):
    for line in path.open():
        e = json.loads(line)
        if e.get("event") == "score_estimated":
            counts[e.get("quest_packed_fast_path", "MISSING")] += 1
print(counts)
PY
```

Expected:

```text
Counter({True: ...})
```

Scorer microbench:

```bash
python scripts/mpr_benchmark_scoring.py \
  --dtype float16 \
  --candidates 8,16,32,64,128 \
  --warmup 50 \
  --iters 1000
```

Representative wall-time result:

```text
torch_quest total:              ~0.071 ms / score call
quest_cuda with packing:        ~0.163 ms / score call
quest_cuda persistent metadata: ~0.068 ms / score call
pack only:                      ~0.092 ms / call
kernel only:                    ~0.012 ms / call
```

Interpretation:

```text
Quest CUDA kernel itself is fast.
Per-call metadata packing was the major CUDA backend overhead.
Persistent metadata fast path removes that overhead for the tested prefix case.
```

## Known Unsupported Cases

Milestone 1 intentionally does not support:

```text
recovery or precision mutation
CPU fp16 backup layout
CPU/GPU KV copy
attention kernel changes
scheduler/block eviction policy changes
multi-request query-window correctness
preemption/reuse lifecycle cleanup
DCP/CP
speculative decoding
sliding-window special handling
production CUDA Graph compatibility validation
bf16 quest_cuda scoring
```

Current `quest_cuda` fast path is valid for the tested single-request,
append-only, prefix-candidate case. Non-prefix cases fall back to the previous
compact packing path.

## ArkVale vs Quest Decision

ArkVale analysis found important integration friction:

```text
ArkVale's exposed kernel expects ArkVale's digest-cache metadata contract
ArkVale uses mean aggregation in the exposed scoring path
the current MPR default wants raw_minmax and head/KV-head level score visibility
adapter work is possible but less direct
```

Quest analysis found a cleaner Milestone 1 match:

```text
Quest metadata is raw per-page min/max
Quest estimate produces per-query-head page scores
GQA can be handled by conservative group union in MPR
the estimate-only CUDA kernel can be sliced and embedded without importing the
full Quest runtime/controller
```

Milestone 1 therefore uses Quest-style scoring as the current default. ArkVale
style scoring/backend remains future work and is not blocking Milestone 2.

## Milestone 2 Inputs

Milestone 2 should start from the following artifacts:

```text
score_candidate_block_ids
observed_digest_block_ids
block_scores
per_kv_head_scores
protected_recent_blocks
layer_name
physical_block_id
block_size
num_kv_heads
head_dim
```

The next design tasks are:

```text
CPU fp16 backup layout
physical block lifecycle and cleanup
request/block ownership under reuse/preemption
precision/recovery policy consuming per-KV-head scores
materialization path for recovered KV before attention
eventual graph/eager compatibility and performance cleanup
```

Milestone 1 answers the question:

```text
At each decode step, for each target layer, which vLLM physical KV blocks would
we recover if mixed-precision recovery were enabled?
```
