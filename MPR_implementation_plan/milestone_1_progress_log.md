# Milestone 1 Progress Log

Last updated: 2026-06-01

## Current State

We are implementing Milestone 1: a score-only prototype for mixed-precision
recovery on vLLM.

The current target is:

```text
vLLM tree: /workspace/vllm
ArkVale reference: /workspace/ArkVale/source/arkvale
model: Qwen/Qwen3-8B
GPU: NVIDIA GeForce RTX 5090
conda env: vllm
```

Important note: `/workspace/ArkVale/vllm` and `/workspace/vllm` are currently
the same commit, but the editable install points to `/workspace/vllm`. Future
code changes should target `/workspace/vllm`.

## Completed

### Milestone 0

- Mapped vLLM v1 attention path.
- Identified FlashAttention as the initial backend.
- Identified hook candidates:
  - `unified_kv_cache_update`
  - `unified_attention_with_output`
- Confirmed CPU KV backup layout is not needed until Milestone 2.
- Wrote:
  - `/workspace/vllm_arkvale_mixed_precision_plan.md`
  - `/workspace/milestone_0_action_plan.md`
  - `/workspace/vllm_integration_notes.md`

### Milestone 1 Planning

- Wrote `/workspace/milestone_1_action_plan.md`.
- Clarified that `MPR` means `Mixed-Precision Recovery`.
- Clarified initial page/block assumption:

```text
vLLM logical block size
== vLLM physical KV block size
== digest scoring unit
== future recall unit
```

- Replaced temporary mean-key digest idea with ArkVale-style digest.
- ArkVale digest summary:

```text
raw_max = filled_keys.max(dim=2)
raw_min = filled_keys.min(dim=2)
center = (raw_max + raw_min) / 2
dist = mean(abs(center - filled_keys), dim=page_tokens)
digest_max = center + dist
digest_min = center - dist
```

- Decided Step 1.4 scoring should use `window_query`, not just the current
  decode query:

```text
window_query = mean(last up to 64 decode queries)
```

- Chunked prefill and speculative decoding are out of scope for Milestone 1.

### Step 1.0 Baseline

- User created conda environment `vllm` and installed vLLM editable.
- Verified:

```text
torch: 2.11.0+cu130
CUDA available: true
GPU: NVIDIA GeForce RTX 5090
vLLM import: OK
editable target: /workspace/vllm
```

- Added baseline script:

```text
/workspace/vllm/scripts/mpr_baseline_qwen3_8b.py
```

- Added Step 1.0 baseline notes:

```text
/workspace/milestone_1_step_1_0_baseline.md
```

- User ran the baseline and got a valid generation:

```text
KV cache is a technique used in transformer models to store the keys and values
of previous attention computations, allowing the model to efficiently process
sequential data by reusing these ...
```

Step 1.0 is complete.

### Step 1.1 Sidecar Scaffold

Initial scaffold added:

```text
/workspace/vllm/vllm/v1/mixed_precision_recovery/__init__.py
/workspace/vllm/vllm/v1/mixed_precision_recovery/config.py
/workspace/vllm/vllm/v1/mixed_precision_recovery/sidecar.py
/workspace/vllm/vllm/v1/mixed_precision_recovery/debug.py
```

Also registered the initial MPR environment variables in `vllm/envs.py` so
vLLM environment validation recognizes them:

```text
VLLM_MPR_ENABLE
VLLM_MPR_DEBUG_DIR
VLLM_MPR_TOPK
VLLM_MPR_MAX_LAYERS
VLLM_MPR_MAX_STEPS
VLLM_MPR_DUMP_EVERY
VLLM_MPR_WINDOW_SIZE
```

Current behavior:

```text
VLLM_MPR_ENABLE unset or 0 -> sidecar is disabled and observer methods return early
VLLM_MPR_ENABLE=1 -> sidecar initializes counters and optional JSONL debug writer
```

Validation completed in the Windows workspace:

```text
python -m py_compile vllm/v1/mixed_precision_recovery/*.py vllm/envs.py
```

Validation still pending in the target `/workspace/vllm` conda environment:

```text
import smoke test with torch 2.11.0+cu130
baseline generation with VLLM_MPR_ENABLE unset
baseline generation with VLLM_MPR_ENABLE=1
```

The Windows Python import smoke test could not be used because its local torch
version does not provide `torch.library.infer_schema`, which this vLLM tree
expects.

Target server validation completed by user:

```text
MPR off import smoke: passed
MPR on import smoke: passed
VLLM_MPR_ENABLE=1 produced True {'init': 1}
debug JSONL init event: passed
baseline generation with MPR off: passed
baseline generation with MPR on: passed
```

Step 1.1 is complete.

### Step 1.2 KV Write Observation Hook

Discussion decisions:

```text
scope = metadata-only KV write observation
store original key/value in sidecar = no
failure policy = fail-fast
slot id behavior = PAD_SLOT_ID(-1) is padding; slot ids < -1 fail fast
block_size source = attn_layer.impl.block_size
missing block_size behavior = fail-fast
smoke debug recommendation = VLLM_MPR_MAX_LAYERS=10
```

First target-server Step 1.2 validation failed during engine initialization:

```text
ValueError: MPR requires attention impl FlashAttentionImpl to expose block_size.
```

Root cause:

```text
FlashAttentionImpl does not expose block_size as an instance attribute.
FlashAttention KV cache layout carries it as [2, num_blocks, block_size, ...].
```

Fix:

```text
_maybe_observe_mpr_kv_write first checks attn_layer.impl.block_size.
If absent, the fallback is explicitly FlashAttention-specific.
For FlashAttentionImpl and FlashAttention-shaped KV cache, it infers:
  block_size = kv_cache.shape[2]
Otherwise it fails fast with the backend name and KV cache shape in the error
message.
```

This is intentionally backend-specific for Milestone 1 because the target scope
is vLLM v1 + FlashAttention. Other attention backends must not silently reuse
the `kv_cache.shape[2]` interpretation unless their KV cache layout has been
checked separately.

Second target-server Step 1.2 validation failed with:

```text
AssertionError: MPR observed negative slot id ... min_slot=-1.
```

Root cause:

```text
vLLM defines PAD_SLOT_ID = -1.
The slot mapping kernel pads unused CUDA graph slots with PAD_SLOT_ID.
These padded slots are not KV writes.
```

Fix:

```text
observe_kv_write now filters PAD_SLOT_ID before block/offset summaries.
slot ids < PAD_SLOT_ID still fail fast.
debug JSONL records num_pad_slots in addition to num_slots/num_valid_slots.
```

Implementation added:

```text
vllm/model_executor/layers/attention/attention.py
  unified_kv_cache_update(...)
  _maybe_observe_mpr_kv_write(...)

vllm/v1/mixed_precision_recovery/sidecar.py
  RecoverySidecar.observe_kv_write(...)
```

The hook runs only when `VLLM_MPR_ENABLE=1`. With MPR disabled, the new
attention-path code returns before importing the sidecar.

The sidecar now records JSONL summaries for KV writes:

```text
event
layer_name
layer_event_idx
key_shape
value_shape
slot_mapping_shape
num_slots
num_valid_slots
block_size
unique_block_ids
min_block_offset
max_block_offset
num_pad_slots
```

The slot mapping contract assumed for Milestone 1 remains:

```text
slot_id = physical_block_id * block_size + block_offset
physical_block_id = slot_id // block_size
block_offset = slot_id % block_size
```

This is valid for the current single-GPU/no-CP/no-DCP scope. `PAD_SLOT_ID=-1`
is filtered out as CUDA graph padding; any slot id less than `PAD_SLOT_ID`
still raises an assertion.

Debug limiting behavior added:

```text
VLLM_MPR_MAX_LAYERS limits the first N unique layer names recorded
VLLM_MPR_MAX_STEPS limits per-layer KV write records
VLLM_MPR_DUMP_EVERY records every Nth per-layer KV write
```

Validation completed in the Windows workspace:

```text
python -m py_compile vllm/model_executor/layers/attention/attention.py \
  vllm/v1/mixed_precision_recovery/*.py
```

Target GPU validation completed by user:

```text
generation completed
block_size inference error resolved
PAD_SLOT_ID=-1 padding handled
observe_kv_write JSONL records created
10 observed layers each recorded 10 KV write events
```

Step 1.2 is complete for the current smoke scope.

### Step 1.3 Digest Cache Design Notes

ArkVale reference checked:

```text
ArkVale/source/arkvale/infer_state.py
  InferState._summarize_keys(...)
  InferState.prefill_save_digests(...)
  InferState.decode_save_1_digest(...)
```

For vLLM Milestone 1, we should reference ArkVale's digest formula and
full-page/key-only policy, not its whole KV pool/cache lifecycle.

Initial vLLM sidecar storage candidate:

```text
(layer_name, physical_block_id) -> digest_max
(layer_name, physical_block_id) -> digest_min
(layer_name, physical_block_id) -> valid_token_count
```

The simple first implementation can store these in layer-local Python dicts,
for example:

```text
dict[layer_name, dict[physical_block_id, BlockDigest]]
```

Potential direction:

```text
If digest count or Python dict overhead becomes large for long sequences,
multi-request batches, or many layers, move digest storage to a paged tensor
layout similar in spirit to ArkVale's digest cache. This would make digest
packing/scoring and future ArkVale-kernel adapter work more natural, but it is
not required for the first Step 1.3 prototype.
```

### Step 1.3 Digest Cache v0 Implementation

Discussion decisions before implementation:

```text
digest generation logic = separate helper function, not inline in sidecar
raw KV backup in sidecar = no
digest source = vLLM FlashAttention KV cache key plane, kv_cache[0]
full block detection = track observed block offsets per layer/block
digest storage = layer-local Python dict keyed by physical_block_id
debug JSONL = metadata only, not full digest tensor values
```

Implementation added:

```text
vllm/v1/mixed_precision_recovery/digest.py
  KeyBlockDigest
  summarize_key_block(...)

vllm/v1/mixed_precision_recovery/sidecar.py
  BlockDigest
  _block_offsets
  _digest_cache
  _observe_block_offsets(...)

vllm/model_executor/layers/attention/attention.py
  passes kv_cache into observe_kv_write(...)

tests/v1/mixed_precision_recovery/test_digest.py
  digest formula test
  sidecar full-block digest creation test
```

Digest helper contract:

```text
input:
  key_block: [block_size, num_kv_heads, head_dim]

output:
  digest_min: [num_kv_heads, head_dim]
  digest_max: [num_kv_heads, head_dim]
  valid_token_count = block_size
```

The helper uses the ArkVale-style formula:

```text
raw_max = key_block.max(dim=0)
raw_min = key_block.min(dim=0)
center = (raw_max + raw_min) / 2
dist = mean(abs(center - key_block), dim=block_tokens)
digest_min = center - dist
digest_max = center + dist
```

Sidecar behavior:

```text
slot_mapping PAD_SLOT_ID(-1) entries are still filtered out
slot ids < PAD_SLOT_ID still fail fast
physical_block_id = slot_id // block_size
block_offset = slot_id % block_size
```

For every valid write, the sidecar records the observed offsets for:

```text
(layer_name, physical_block_id)
```

Once the observed offsets cover the full range:

```text
0 ... block_size - 1
```

the sidecar reads:

```text
key_block = kv_cache[0, physical_block_id]
```

and stores:

```text
_digest_cache[layer_name][physical_block_id] = BlockDigest(...)
```

Current v0 limitation:

```text
physical KV block lifecycle/reuse cleanup is not implemented yet
the first full transition for a layer/block creates one digest entry
future multi-request/prefix-cache/block-reuse support must invalidate or
refresh digest state when vLLM reuses a physical block for different tokens
```

Serving/lifecycle risk:

```text
The current sidecar keys digest state only by:
  (layer_name, physical_block_id)

This is sufficient for a single-request smoke run, but not sufficient for a
long-lived serving engine. After a request finishes, vLLM can return its
physical KV blocks to a free list and later assign the same physical_block_id
to a different request. If the sidecar keeps the old entry:
  _block_offsets[layer_name][physical_block_id]
  _digest_cache[layer_name][physical_block_id]

then the digest can describe the previous request's tokens while the KV cache
block now contains another request's tokens. The current `block_id not in
layer_digests` guard can also prevent the new request's digest from being
recomputed.

Before serving-style validation or long-lived engine experiments, revisit this
design and add one of:
  request/block ownership tracking with invalidation on block free/reuse
  a physical-block generation/epoch in the sidecar key
  integration with vLLM KV cache manager free/reuse events
```

Debug JSONL additions:

```text
kv_cache_shape
digest_created_block_ids
num_digest_blocks_for_layer
total_digest_blocks
```

When a block digest is created, the debug writer also emits:

```text
event = digest_created
layer_name
physical_block_id
digest_min_shape
digest_max_shape
valid_token_count
block_size
num_digest_blocks_for_layer
total_digest_blocks
```

The `VLLM_MPR_MAX_LAYERS`, `VLLM_MPR_MAX_STEPS`, and
`VLLM_MPR_DUMP_EVERY` settings are treated as JSONL/debug limiting controls.
They do not stop sidecar block-offset tracking or digest-cache updates for
layers that are actually observed by the hook.

Validation completed in the Windows workspace:

```text
python -m py_compile vllm/v1/mixed_precision_recovery/__init__.py \
  vllm/v1/mixed_precision_recovery/config.py \
  vllm/v1/mixed_precision_recovery/debug.py \
  vllm/v1/mixed_precision_recovery/digest.py \
  vllm/v1/mixed_precision_recovery/sidecar.py \
  vllm/model_executor/layers/attention/attention.py \
  tests/v1/mixed_precision_recovery/test_digest.py
```

Local pytest validation was not possible in the Windows workspace because this
Python environment does not have `pytest` installed. A direct import smoke also
cannot be used here because the local torch version lacks
`torch.library.infer_schema`, which this vLLM tree expects. Run the pytest and
GPU smoke in the target `/workspace/vllm` environment.

Documentation follow-up:

```text
Added docstrings and explicit shape comments to the Step 1.3 digest path:
  summarize_key_block(...)
  RecoverySidecar.observe_kv_write(...)
  RecoverySidecar._observe_block_offsets(...)
  BlockDigest / KeyBlockDigest

The comments now state the expected FlashAttention KV cache shape:
  kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
  key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
  key_block: [block_size, num_kv_heads, head_dim]
  digest_min/max: [num_kv_heads, head_dim]
```

Validation tooling follow-up:

```text
Added scripts/mpr_validate_debug_jsonl.py to validate Step 1.3 smoke JSONL
without manually reading raw cat/jq output.

The validator checks:
  observe_kv_write block/offset invariants
  digest_created metadata invariants
  digest_min_shape == digest_max_shape
  valid_token_count == block_size for full-block digest v0
  each digest_created event matches an observe_kv_write event with the same
    (layer_name, layer_event_idx)
  physical_block_id appears in the observe event's digest_created_block_ids

Added tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py for
matching and unmatched digest event cases.

Token-count follow-up:

```text
scripts/mpr_baseline_qwen3_8b.py now prints:
  prompt_token_count
  generated_token_count
  total_token_count

scripts/mpr_validate_debug_jsonl.py now prints per-layer observed KV write
slot totals:
  valid_slots
  block_size
  lower_bound_full_blocks = valid_slots // block_size
  digests

The baseline total is the user request's prompt+generated token count. The
validator total is what the MPR hook actually observed in KV writes and is the
more direct signal for digest-count debugging. It can include profile/warmup
events if vLLM emits KV writes before the user request.
```

Profile-run isolation follow-up:

```text
The first script-level --defer-mpr-enable implementation tried to set
VLLM_MPR_ENABLE=0 during LLM(...), then restore it before llm.generate(...).
That failed for the target SyncMP/EngineCore path because EngineCore runs in a
separate process. The child process inherited VLLM_MPR_ENABLE=0 during startup
and did not see the parent process restore it, so no JSONL file was created.

Fix:
  vllm.forward_context.ForwardContext now carries is_dummy_run
  GPUModelRunner.execute_model passes dummy_run into set_forward_context(...)
  _maybe_observe_mpr_kv_write skips observation when forward_context.is_dummy_run

This keeps VLLM_MPR_ENABLE=1 inside the EngineCore process while filtering
dummy/profile model-runner forwards in the MPR hook itself.

scripts/mpr_baseline_qwen3_8b.py keeps --defer-mpr-enable as a compatibility
flag, but it no longer toggles VLLM_MPR_ENABLE. The script still prints:
  mpr_defer_enable
  mpr_enable_during_init
  mpr_enable_during_generate
```

## Next Step

Validate Step 1.3 on the target GPU server.

Recommended validation:

```bash
cd /workspace/vllm
rm -rf /tmp/vllm_mpr_debug

python -m pytest tests/v1/mixed_precision_recovery/test_digest.py -q

VLLM_MPR_ENABLE=1 \
VLLM_MPR_DEBUG_DIR=/tmp/vllm_mpr_debug \
VLLM_MPR_MAX_LAYERS=10 \
VLLM_MPR_MAX_STEPS=20 \
VLLM_MPR_DUMP_EVERY=1 \
python scripts/mpr_baseline_qwen3_8b.py \
  --defer-mpr-enable \
  --prompt "Say hello." \
  --max-tokens 8
```

Expected debug check:

```bash
cat /tmp/vllm_mpr_debug/*.jsonl | grep observe_kv_write | head
cat /tmp/vllm_mpr_debug/*.jsonl | grep digest_created | head
python scripts/mpr_validate_debug_jsonl.py /tmp/vllm_mpr_debug/*.jsonl
```

Expected outcome:

```text
generation completes
observe_kv_write events exist
digest_created events appear after at least one full physical block is observed
digest_min_shape and digest_max_shape are [num_kv_heads, head_dim]
num_valid_slots > 0 for decode writes
num_pad_slots may be > 0 because CUDA graph padding uses PAD_SLOT_ID=-1
unique_block_ids is non-empty for decode writes
0 <= min_block_offset <= max_block_offset < block_size
no invalid negative-slot assertion
```

CUDA graph capture isolation follow-up:

```text
Target smoke validation showed this layer-0 num_valid_slots histogram:
  [(1, 31), (34, 1), (256, 1), (512, 1)]

Interpretation:
  34      likely the actual prompt prefill KV write
  1 x 31  likely actual decode KV writes
  256/512 likely CUDA graph capture or warmup-style forwards

The previous dummy/profile isolation only covered GPUModelRunner.execute_model
paths where dummy_run=True is passed into set_forward_context(...). CUDA graph
capture uses prepare_dummy_inputs(...) and prepare_inputs_to_capture(...), but
its capture-time forward context did not mark is_dummy_run=True. As a result,
MPR could observe capture-time dummy KV writes as if they belonged to the real
request.

Fix:
  vllm/v1/worker/gpu/cudagraph_utils.py now passes is_dummy_run=True to
  set_forward_context(...) inside CudagraphModelRunner.capture().

Expected revalidation signal:
  The 256/512 entries should disappear from the per-layer num_valid_slots
  histogram. For the same smoke run, layer 0 should be close to:
    [(1, 31), (34, 1)]

Notes:
  valid_slots in the validator summary is still an observed KV-write volume
  counter, not a unique-slot count and not a final request token count. It is
  useful for detecting unexpected extra forwards, as above.
```

CUDA graph capture isolation revalidation note:

```text
After the is_dummy_run=True capture-context fix, the target server still showed
large num_valid_slots entries in the layer-0 histogram. Therefore, the 256/512
events are not fully explained by the capture-time set_forward_context path
patched above, or there is another warmup/capture/replay path that still lacks
an MPR skip signal.

Current decision:
  Do not block Step 1.3 on this. The originally intended smoke checks are still
  satisfied:
    generation completed
    no negative-slot assertion
    no block_size/backend assertion
    digest_created events exist
    validator matching checks pass

Open follow-up:
  Before serving-style validation, revisit MPR observation boundaries and split
  the debug counters into:
    actual request KV writes
    warmup/profile/capture/replay writes
    unique physical slots observed per layer

This matters for interpreting validator summary counters, but the current
digest creation path still appears correct for the blocks it observes.
```

### Step 1.4 Query-Time Score Hook Discussion

Initial decisions before implementation:

```text
Hook location:
  Use vllm/model_executor/layers/attention/attention.py:
    unified_attention_with_output(...)
  Place the MPR score-only observation after get_attention_context(...) and
  before self.impl.forward(...), so query, attn_metadata, kv_cache, attn_layer,
  and layer_name are all available while the attention output remains unchanged.

Rolling query buffer identity:
  Step 1.4 v0 will use layer_name as the rolling-buffer key.
  This is intentionally single-request smoke scope only. Multi-batch and
  serving scenarios need request identity / sequence ownership in the key;
  otherwise decode queries from different requests can be mixed in one layer
  buffer.

ArkVale reference:
  ArkVale's decode score path is q_len == 1:
    adapter/modeling.py calls estimate_select_recall(cur_id, query_states)
    only in the decode branch.
  ArkVale's prefill eviction path uses query_states[:, -1:, ...], not the
  whole prefill query sequence.
  Therefore Step 1.4 should not treat full prefill query tensors as decode
  scoring input. The v0 filter should score only max_query_len == 1 decode
  batches and skip prefill/mixed/chunked cases.

Top-k:
  VLLM_MPR_TOPK is debug-output policy for compact inspection, not a fundamental
  scoring requirement. The score vector is the primary artifact; top-k block ids
  are only a convenient reduced view for smoke/debug.

Query window and score aggregation:
  If the number of generated decode tokens is smaller than VLLM_MPR_WINDOW_SIZE,
  compute window_query from the queries observed so far:
    window_query = mean(last min(num_seen, window_size) decode queries)
  Record window_query_len in debug JSONL so early-step scores can be interpreted.

  For GQA/MQA, Step 1.4 should follow ArkVale's order:
    1. compute scores per query head against the corresponding KV-head digest
    2. aggregate query-head scores within each group

  The aggregation policy should be configurable:
    max   preserve the strongest query-head signal; default for accuracy-first
    mean  average query-head scores; useful for ArkVale-style comparison

  Proposed implementation config:
    VLLM_MPR_SCORE_AGG=max|mean, default max
```

### Step 1.4 Query-Time Score Hook Implementation

Implemented Step 1.4 score-only path:

```text
Config:
  Added VLLM_MPR_SCORE_AGG=max|mean, default max.
  max preserves the strongest query-head score for accuracy-first inspection.
  mean is available for ArkVale-style all-head average comparison.

Hook:
  unified_attention_with_output(...) now calls _maybe_observe_mpr_query(...)
  after get_attention_context(...) and before self.impl.forward(...).
  The hook skips MPR disabled and ForwardContext.is_dummy_run paths.

Decode-only scope:
  sidecar.observe_query(...) scores only max_query_len == 1.
  max_query_len != 1 records score_skipped(reason=non_decode_query).
  In max_query_len == 1, num_actual_tokens is treated as active request count.
  num_actual_tokens != 1 records
  score_skipped(reason=non_single_request_decode). This avoids failing engine
  startup when vLLM emits synthetic warmup/capture batches such as
  max_query_len=1, num_actual_tokens=256.
  Step 1.4 v0 is still single-request scoring only; multi-request/serving needs
  request-scoped query windows and block ownership before scoring non-single
  rows.

Query window:
  _query_windows uses dict[layer_name] -> deque[Tensor] as v0 bookkeeping.
  The stored query shape is [num_q_heads, head_dim].
  window_query is mean(last min(num_seen, VLLM_MPR_WINDOW_SIZE) decode queries).
  This dict/deque structure is not part of the future kernel-facing API.

Scoring core:
  Added vllm/v1/mixed_precision_recovery/scoring.py.
  estimate_digest_scores(...) is tensor-only and does not know about sidecar
  dicts, layer names, request ids, or block ownership.
  Inputs:
    query_window: [num_q_heads, head_dim]
    digest_min:   [num_blocks, num_kv_heads, head_dim]
    digest_max:   [num_blocks, num_kv_heads, head_dim]
  It maps query heads to KV heads for GQA/MQA, computes per-query-head cuboid
  scores, then aggregates query-head scores with VLLM_MPR_SCORE_AGG.
  The Step 1.4 output is layer-local physical KV block scoring:
    layer_name -> physical_block_id -> score
  It is not head-level scoring. Head/group scores are intermediate values only.
  Future recall policies may need head/group-aware decisions, but this v0 debug
  artifact intentionally records a single score per block per layer.

Debug:
  score_estimated JSONL records:
    query_shape
    window_query_shape
    window_query_len
    num_digest_blocks
    score_count
    score_agg
    topk
    topk_block_ids
    topk_scores
  score_skipped JSONL records skip reason and relevant shapes/counts.
  scripts/mpr_validate_debug_jsonl.py now supports --min-score-events and
  validates score_estimated invariants.

Tests:
  Added tests/v1/mixed_precision_recovery/test_scoring.py for manual formula
  matching, max/mean aggregation, invalid head grouping, rolling query window,
  and non-single decode skip behavior.
```

Step 1.4 startup warmup fix:

```text
Target smoke with longer decode failed during EngineCore initialization:
  max_query_len=1, num_actual_tokens=256

This is decode-shaped but not the real single-request generation path. It is
likely a vLLM startup warmup/capture/profile path that is not fully covered by
ForwardContext.is_dummy_run. The original fail-fast assertion for
num_actual_tokens != 1 was too broad because it also killed these synthetic
startup batches.

Fix:
  Keep Step 1.4 scoring single-request only, but skip non-single decode-shaped
  rows with score_skipped(reason=non_single_request_decode) instead of raising.

Serving implication:
  This does not implement multi-request scoring. It only keeps smoke validation
  alive in the presence of startup batches. Real serving support still needs
  request-scoped query windows and request/block ownership tracking.
```

Step 1.4 validation result:

```text
Unit tests:
  tests/v1/mixed_precision_recovery/test_scoring.py
  tests/v1/mixed_precision_recovery/test_digest.py
  tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py
  passed on the target server. Only dependency deprecation warnings were seen.

Long decode smoke:
  A 512-token decode smoke was run with:
    VLLM_MPR_ENABLE=1
    VLLM_MPR_WINDOW_SIZE=64
    VLLM_MPR_TOPK=8
    VLLM_MPR_SCORE_AGG=max
  The JSONL validator passed with --min-score-events and
  --allow-unmatched-digest-events for the long run.

Manual long-run checks:
  window_query_len reached and capped at 64.
  score_count matched num_digest_blocks.
  digest/score count progression looked normal.
  top-k block ids could be inspected over later decode events.

Strict digest/observe alignment:
  A shorter smoke with VLLM_MPR_DUMP_EVERY=1 passed the JSONL validator without
  --allow-unmatched-digest-events.

Conclusion:
  Step 1.4 is complete for the current single-request score-only smoke scope.
```

### Step 1.5 Debug Output and Inspection Implementation

Implemented Step 1.5 debug metadata for score inspection:

```text
score_estimated now records request/block context in addition to score values:
  num_reqs
  max_query_len
  num_actual_tokens
  seq_lens
  block_size
  block_table_shape
  block_table_row
  valid_block_ids
  finalized_block_ids
  observed_digest_block_ids
  missing_digest_blocks
  extra_digest_blocks

Definitions:
  block_table_row is the first request row from FlashAttention metadata.
  valid_block_ids are block_table entries covering ceil(seq_len / block_size).
  finalized_block_ids are block_table entries covering floor(seq_len / block_size).
  observed_digest_block_ids are the cached physical block IDs packed and scored
  for the current layer.
  missing_digest_blocks are finalized request blocks without a cached digest.
  extra_digest_blocks are cached/scored digest blocks outside the current
  request's valid block-table range.

Important scope note:
  Step 1.4/1.5 still scores every cached digest for the layer. The new
  missing/extra fields are debug diagnostics that expose stale/warmup/reuse
  effects; they do not yet filter scoring to request-owned blocks. Serving
  support still needs request/block ownership tracking before this should be
  treated as a production recall policy.
```

Validator updates:

```text
scripts/mpr_validate_debug_jsonl.py now validates:
  score_count == num_digest_blocks
  observed_digest_block_ids length == score_count
  topk_block_ids subset of observed_digest_block_ids
  block/debug list fields contain non-negative ints

The validator summary also prints score events with missing finalized digest
blocks or extra cached digest blocks when those diagnostics are non-empty.
```

Tests added:

```text
tests/v1/mixed_precision_recovery/test_scoring.py:
  verifies seq_lens/block_table -> valid/finalized/missing/extra debug fields.

tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py:
  verifies the expanded score event schema and rejects top-k blocks that were
  not part of the scored digest block list.

Local verification:
  python -m py_compile passed for sidecar, validator, and updated tests.
  Local Windows Python did not have pytest installed, so pytest should be run
  in the target vLLM server environment.
```

Step 1.5 smoke validation result:

```text
The target server smoke run completed successfully.

Confirmed by user:
  generation completed
  JSONL validation passed
  no missing finalized digest block summary section was printed
  no extra cached digest block summary section was printed

Interpretation:
  For the validated single-request run, scored digest blocks matched the
  current request's valid/finalized block-table view. The Step 1.5 debug fields
  are therefore usable for Step 1.6 strict metadata validation.
```

### Step 1.6 Validation Implementation

Added stricter validation helpers for Milestone 1 completion checks:

```text
scripts/mpr_validate_debug_jsonl.py:
  --strict-current-request-scores

This mode is intended for the current single-request smoke scope. It requires:
  observed_digest_block_ids to be present
  finalized_block_ids to be present
  missing_digest_blocks == []
  extra_digest_blocks == []
  set(observed_digest_block_ids) == set(finalized_block_ids)
  topk_block_ids subset of finalized_block_ids
  topk_scores are finite numbers

This is intentionally not a serving/multi-request validation mode. Multi-request
serving still needs request-scoped query windows and request/block ownership
tracking before strict request-local scoring can be required.
```

Added deterministic generation comparison helper:

```text
scripts/mpr_compare_generation_outputs.py

It reads two mpr_baseline_qwen3_8b.py logs, extracts generated_token_ids, and
fails if the sidecar-on run differs from the sidecar-off baseline.
```

Recommended Step 1.6 commands on the target server:

```bash
python scripts/mpr_validate_debug_jsonl.py \
  --min-score-events 50 \
  --allow-unmatched-digest-events \
  --strict-current-request-scores \
  /tmp/vllm_mpr_debug/*.jsonl

python scripts/mpr_compare_generation_outputs.py \
  /tmp/mpr_baseline_off.log \
  /tmp/mpr_baseline_on.log
```

Step 1.6 validation result:

```text
The target server validation completed successfully.

Confirmed by user:
  Baseline non-regression passed:
    sidecar-off and sidecar-on generated_token_ids matched.
  Metadata alignment passed:
    strict JSONL validation passed with --strict-current-request-scores.
  Digest sanity passed:
    digest/score invariants checked by scripts/mpr_validate_debug_jsonl.py
    completed without assertion failures.

Interpretation:
  For the current single-GPU, single-request, score-only smoke scope,
  Milestone 1 Step 1.6 validation is complete.

Remaining scope caveat:
  This does not validate multi-request serving, request/block ownership under
  block reuse, or production recall behavior. Those remain future design and
  implementation items.
```

### Step 1.7 ArkVale Kernel Analysis and First-Pass Compatibility

Created report:

```text
/home/han/KV_cache_quant/proposed_method_develop/vllm/MPR_implementation_plan/milestone_1_step_1_7_arkvale_kernel_compatibility_report.md
```

Summary:

```text
direct_reuse = no
single_request_reuse_after_packing = likely yes
multi_request_reuse_after_packing = not yet
```

Key findings:

```text
ArkVale estimate_scores mathematically matches the current cuboid digest score.
The kernel expects FlashInfer-style paged digest-cache inputs, not the current
sidecar's Python dict keyed by vLLM physical block id.
ArkVale dg_indices are digest-cache page ids, not vLLM physical block ids.
The current MPR default aggregation is max, while ArkVale's exposed wrapper
mean-reduces query heads within groups.
The kernel is safe for the current single-request smoke scope, but should not
be treated as general variable-length multi-request serving support yet.
```

Open Step 1.7 decisions before adapter design:

```text
1. Should ArkVale-kernel mode initially force score_agg=mean, or should the
   wrapper/kernel expose max aggregation?
2. Should the first packing prototype allocate temporary compact dg_data per
   score call, or introduce persistent sidecar paged digest storage?
```

### Reference Scoring Reports: Quest And DiffKV

Created Quest scoring reference report:

```text
/home/han/KV_cache_quant/proposed_method_develop/vllm/MPR_implementation_plan/reference_quest_scoring_report.md
```

Created DiffKV scoring reference report:

```text
/home/han/KV_cache_quant/proposed_method_develop/vllm/MPR_implementation_plan/reference_diffkv_scoring_report.md
```

Current interpretation:

```text
Quest is the closest mainstream reference for query-aware page scoring with
page-level max/min metadata.

DiffKV is not a digest scorer. It uses accumulated post-softmax attention mass
as a token importance signal, max-aggregated over GQA query heads, then applies
thresholds to choose high precision, low precision, or prune.

For MPR, Quest/ArkVale are better references for runtime digest scoring.
DiffKV is more useful as a precision-allocation policy reference and as a
possible attention-mass oracle/calibration signal.
```

### Step 1.7 Modular Scoring Implementation

Created implementation report:

```text
/home/han/KV_cache_quant/proposed_method_develop/vllm/MPR_implementation_plan/milestone_1_step_1_7_modular_scoring_implementation_report.md
```

Implemented the first modular scoring pass:

```text
vllm/v1/mixed_precision_recovery/config.py
  added scoring_backend and digest_kind

vllm/v1/mixed_precision_recovery/digest.py
  summarize_key_block(..., digest_kind=...)
  digest_kind=arkvale keeps existing tightened digest behavior
  digest_kind=raw_minmax enables Quest-style raw extrema metadata

vllm/v1/mixed_precision_recovery/scoring.py
  added DigestScoringBackend protocol
  added TorchQuestScorer
  added DigestScoreResult
  exposed per_query_head_scores and per_kv_head_scores
  kept estimate_digest_scores(...) as a legacy block-score wrapper

vllm/v1/mixed_precision_recovery/sidecar.py
  resolves configured scorer once in RecoverySidecar.__post_init__
  creates configured digest kind in observe_kv_write
  calls backend.estimate(...) in observe_query

vllm/envs.py
  registered VLLM_MPR_SCORING_BACKEND
  registered VLLM_MPR_DIGEST_KIND
  registered VLLM_MPR_SCORE_GRANULARITY
```

Default behavior:

```text
VLLM_MPR_SCORING_BACKEND=torch_quest
VLLM_MPR_DIGEST_KIND=raw_minmax
VLLM_MPR_SCORE_AGG=max
VLLM_MPR_SCORE_GRANULARITY=kv_head
```

GQA policy implemented for the current Quest-style path:

```text
q_head -> kv_head mapping:
  group_size = num_q_heads // num_kv_heads
  kv_head = q_head // group_size

score_agg=max:
  per_kv_head_score = max over query heads in that KV group
  block_score = max over KV groups

This is the agreed conservative union policy for GQA in the current block-level
top-k debug path.
```

Head-level score selection follow-up:

```text
Added VLLM_MPR_SCORE_GRANULARITY=block|kv_head|query_head.

block:
  Existing behavior. topk_block_ids/topk_scores are computed from block_scores:
    [num_blocks]

kv_head:
  Keeps the existing block aggregate top-k fields for backward compatibility.
  Adds topk_block_ids_by_head/topk_scores_by_head from per_kv_head_scores:
    [num_blocks, num_kv_heads]

query_head:
  Keeps the existing block aggregate top-k fields for backward compatibility.
  Adds topk_block_ids_by_head/topk_scores_by_head from per_query_head_scores:
    [num_blocks, num_q_heads]

The default Quest-style head-level smoke is now:
  VLLM_MPR_DIGEST_KIND=raw_minmax
  VLLM_MPR_SCORE_GRANULARITY=kv_head
  VLLM_MPR_SCORE_AGG=max
```

Default-policy update:

```text
The project default moved from ArkVale tightened digest + block score output to:
  Quest-style raw_minmax digest
  head-level output at KV-head granularity

For MHA, num_q_heads == num_kv_heads, so kv_head granularity is equivalent to
attention-head granularity. For GQA/MQA, kv_head granularity uses the
conservative max/union over query heads in each KV group.

ArkVale tightened digest and block-level scoring remain available via:
  VLLM_MPR_DIGEST_KIND=arkvale
  VLLM_MPR_SCORE_GRANULARITY=block
```

Debug JSONL additions:

```text
digest_kind
scoring_backend
num_q_heads
num_kv_heads
gqa_group_size
score_granularity
num_score_heads
head_score_count
topk_block_ids_by_head
topk_scores_by_head
```

Validator/test updates:

```text
scripts/mpr_validate_debug_jsonl.py accepts and validates optional modular
scoring fields while preserving older JSONL compatibility.

tests/v1/mixed_precision_recovery/test_digest.py covers raw_minmax digest.
tests/v1/mixed_precision_recovery/test_scoring.py covers structured scorer
metadata and conservative GQA max aggregation.
tests/v1/mixed_precision_recovery/test_debug_jsonl_validator.py covers the
expanded score event schema.
```

Local validation:

```text
python -m py_compile passed for selected MPR source/test/validator files.
git diff --check passed.
direct importlib smoke for digest.py and scoring.py passed.
```

Local pytest status:

```text
Full pytest is blocked in the local base environment before MPR tests run:
  tests/conftest.py imports transformers/sklearn/scipy
  local NumPy is 2.3.5
  installed SciPy/sklearn extension was built against NumPy 1.x

Running with parent conftest disabled avoids that issue, but this base
environment is missing cbor2, which vLLM imports through vllm.config.

The target vLLM environment should run the focused MPR pytest suite.
```

### Step 1.7 Quest Estimate Kernel Detailed Analysis

Created detailed Quest estimate kernel report:

```text
/home/han/KV_cache_quant/proposed_method_develop/vllm/MPR_implementation_plan/milestone_1_step_1_7_quest_estimate_kernel_analysis_report.md
```

Key conclusion:

```text
Quest's exposed Python/C++ wrapper is not directly reusable for MPR because it
is batch-size-1, assumes q_heads == metadata_heads, and depends on Quest's own
InferenceController/KvCache metadata cache.

Quest's underlying estimate kernel design is a strong fit for the current MPR
default because it computes raw_minmax page scores per query head and its lower
level dispatch is GQA-shaped.
```

Recommended adapter direction:

```text
Implement estimate-only CUDA backend first.
Pack MPR raw_minmax digests into Quest-style metadata pages:
  [metadata_pages, 2, page_size, num_kv_heads, head_dim]

Expose or modify a binding that keeps num_q_heads and num_kv_heads separate.
Return per-query-head scores from CUDA.
Reuse current Python aggregation for:
  per_query_head_scores -> per_kv_head_scores -> topk_block_ids_by_head

Defer kernel-side KV-head max aggregation and Quest topk_filtering reuse until
after PyTorch-vs-CUDA score parity is established.
```

### Step 1.7 Quest-Style Candidate Packing Implementation

Implemented the pre-CUDA adapter layer for Quest-style estimate integration.

Config addition:

```text
VLLM_MPR_RECENT_TOKENS
  default: 64
  meaning: recent token window excluded from score/top-k candidates and kept
  protected by policy
```

Sidecar candidate selection now uses current-request logical block order when
`block_table` and `seq_lens` are available:

```text
valid_block_ids:
  block_table row up to ceil(seq_len / block_size)

finalized_block_ids:
  block_table row up to floor(seq_len / block_size)

protected_tail_entries:
  ceil(recent_tokens / block_size)

protected_block_ids:
  tail of valid_block_ids

score_candidate_block_ids:
  finalized_block_ids excluding protected_block_ids
```

This means scoring is no longer based on sorted physical block ids when current
request block metadata is available. It preserves the request's logical block
order and makes the recent-token protection policy explicit.

Added Quest-compatible metadata packer:

```text
vllm/v1/mixed_precision_recovery/quest_packing.py

pack_quest_metadata_cache(...)
  input:
    digest_min/max [num_candidates, num_kv_heads, head_dim]
    metadata_page_size
    entry_block_ids

  output:
    metadata_data [num_metadata_pages, 2, page_size, num_kv_heads, head_dim]
    metadata_indices int32 [num_metadata_pages]
    metadata_indptr int32 [2]
    metadata_last_page_len
    metadata_last_page_idx
    entry_block_ids
```

The packer appends one guard metadata entry by default. This matches the current
Quest estimate kernel behavior, which subtracts/excludes the last logical
metadata entry. With the guard entry, all real MPR score candidates remain
visible to the unmodified Quest estimate kernel.

Important caveat:

```text
The guard entry is an explicit Quest-kernel compatibility shim, not a general
MPR policy decision.

MPR recent-token protection is handled before packing through:
  score_candidate_block_ids = finalized blocks excluding protected tail blocks

The extra guard exists only because the current Quest estimate kernel always
drops the final logical metadata entry. If the MPR estimate binding later
accepts an explicit score-entry count or exclusion count, this guard should be
removed and the exact candidate length should be passed to the kernel instead.
```

Debug JSONL additions:

```text
recent_tokens
protected_tail_entries
protected_block_ids
score_candidate_block_ids
```

The validator now treats `score_candidate_block_ids` as the strict comparison
target when present. This allows strict validation to pass when recent protected
blocks are intentionally excluded from scoring.

Validation:

```text
python -m py_compile passed for updated MPR source/test/validator files.
git diff --check passed.
quest_packing importlib smoke passed.

Focused pytest is still blocked in the local base environment before tests run:
  tests/conftest.py imports transformers/sklearn/scipy
  local NumPy is 2.3.5
  installed SciPy/sklearn extension was built against NumPy 1.x
```

### Step 1.7 Quest CUDA Backend Binding Surface

Implemented the first binding surface for the Quest CUDA scoring backend.

Python backend:

```text
QuestCudaScorer
  backend name: quest_cuda

Flow:
  validate query/digest shapes
  require GQA group_size in {1, 4, 8}
  require CUDA query tensors
  pack digest_min/max with pack_quest_metadata_cache(add_guard_entry=True)
  allocate output [num_q_heads, num_score_entries]
  call vllm._custom_ops.mpr_estimate_attn_score(...)
  transpose output to [num_score_entries, num_q_heads]
  reuse Python aggregate_query_head_scores(...)
```

Configuration:

```text
VLLM_MPR_SCORING_BACKEND=quest_cuda
```

is now accepted by `MPRConfig`.

vLLM custom op surface:

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

The current C++ registration is intentionally a stub:

```text
csrc/mpr/quest_estimate_stub.cpp
```

It fixes the op schema and build integration point but throws at runtime. The
next implementation step is to replace this stub with the actual Quest estimate
wrapper that separates:

```text
num_q_heads  = q.size(1)
num_kv_heads = metadata_data.size(3)  # NHD
```

and calls the existing Quest lower-level estimate kernel with:

```text
paged_kv.num_heads = num_kv_heads
num_qo_heads       = num_q_heads
```

Validation:

```text
python -m py_compile passed for updated Python files.
git diff --check passed.
```

Open performance note:

```text
QuestCudaScorer.estimate currently calls pack_quest_metadata_cache(...) on every
score event.

This is correctness-first and keeps the backend stateless, but it may add
noticeable overhead because each decode scoring call repacks compact
metadata_data/indices/indptr from digest_min/digest_max.

Do not optimize this before profiling. Candidate follow-ups after profiling:
  cache packed metadata per layer/request candidate set
  update packed metadata incrementally when new digest blocks arrive
  keep a persistent Quest-style paged digest cache instead of per-call packing
```

### Step 1.7 Quest Kernel Dependency Slicing Analysis

Created Quest kernel dependency slicing report:

```text
/home/han/KV_cache_quant/proposed_method_develop/vllm/MPR_implementation_plan/milestone_1_step_1_7_quest_kernel_dependency_slicing_report.md
```

Key conclusion:

```text
Use an estimate-only vendored subset rather than copying Quest's full
decode_attn.cuh/decode_page.cuh stack directly.

The required score path is:
  compute_max_possible
  MaxPossibleSampleWithPagedKVCacheKernel
  MaxPossibleSampleWithPagedKVCache launcher
  minimal PageStorage/paged_kv_t accessors
  small FlashInfer compatibility subset
  MPR-facing torch binding

The full Quest decode file pulls in unrelated full-attention dependencies such
as cascade/state/RoPE/partition decode code, so it is a larger and riskier
compile surface than needed for Step 1.7.
```

Recommended local file shape:

```text
csrc/mpr/quest_estimate.cu
csrc/mpr/quest_estimate_kernel.cuh
csrc/mpr/quest_paged_kv.cuh
csrc/mpr/flashinfer_compat/layout.cuh
csrc/mpr/flashinfer_compat/utils.cuh
csrc/mpr/flashinfer_compat/math.cuh
csrc/mpr/flashinfer_compat/cp_async.cuh
csrc/mpr/flashinfer_compat/vec_dtypes.cuh
```

Binding direction:

```text
Keep the existing mpr_estimate_attn_score op schema for the first PoC.
Replace csrc/mpr/quest_estimate_stub.cpp with csrc/mpr/quest_estimate.cu.
Derive num_q_heads from q.size(1).
Derive num_kv_heads from metadata_data.size(3) for NHD.
Call the lower-level Quest estimate launcher with:
  paged_kv.num_heads = num_kv_heads
  num_qo_heads = num_q_heads
Return per-query-head scores and keep Python-side KV-head aggregation/top-k.
```

Important pre-kernel fix found during analysis:

```text
Quest/FlashInfer TensorLayout.NHD == 0
Quest/FlashInfer TensorLayout.HND == 1

Current MPR Python code still has:
  QUEST_NHD_LAYOUT = 1

The current stub does not use layout, so tests did not catch this. It must be
changed to 0 before the real CUDA kernel is connected.
```

First CUDA PoC limits:

```text
single-request compact metadata_indptr [0, num_metadata_pages]
NHD metadata layout only
PageStorage::kIndices only
rotary_mode = none
partition_kv = false
fp16 first
GQA group_size in {1, 4, 8}
head_dim in {64, 128, 256}
```

Validation plan:

```text
Build _C with quest_estimate.cu.
Add a CUDA parity test comparing QuestCudaScorer output to TorchQuestScorer
per_query_head_scores.
Cover group_size 1/4/8 and output-length/shape/layout errors.
Then run focused MPR pytest and one short generation smoke with:
  VLLM_MPR_ENABLE=1
  VLLM_MPR_SCORING_BACKEND=quest_cuda
```

### Step 1.7 Quest CUDA Estimate Kernel Implementation

Implemented the first estimate-only Quest CUDA backend instead of the previous
runtime stub.

Third-party FlashInfer header snapshot:

```text
csrc/third_party/flashinfer/README.md
csrc/third_party/flashinfer/include/flashinfer/layout.cuh
csrc/third_party/flashinfer/include/flashinfer/utils.cuh
csrc/third_party/flashinfer/include/flashinfer/math.cuh
csrc/third_party/flashinfer/include/flashinfer/cp_async.cuh
csrc/third_party/flashinfer/include/flashinfer/vec_dtypes.cuh
```

These headers are copied from Quest's FlashInfer third-party snapshot and keep
their original Apache-2.0 license headers.

MPR-local derived Quest estimate files:

```text
csrc/mpr/quest_paged_kv.cuh
csrc/mpr/quest_estimate_kernel.cuh
csrc/mpr/quest_estimate.cu
```

Implementation shape:

```text
quest_paged_kv.cuh
  minimal PageStorage::kIndices paged digest accessor
  treats metadata_data plane 0 as digest_max
  treats metadata_data plane 1 as digest_min

quest_estimate_kernel.cuh
  estimate-only compute_max_possible path
  MaxPossibleSampleWithPagedKVCacheKernel
  MaxPossibleSampleWithPagedKVCache launcher
  keeps Quest's final-entry exclusion behavior

quest_estimate.cu
  validates the existing mpr_estimate_attn_score op inputs
  supports NHD metadata layout only for the first PoC
  supports batch size 1
  supports fp16 only
  supports GQA group_size in {1, 4, 8}
  supports head_dim in {64, 128, 256}
```

Build integration:

```text
CMake now builds csrc/mpr/quest_estimate.cu instead of:
  csrc/mpr/quest_estimate_stub.cpp

The _C target include path now includes:
  csrc/third_party/flashinfer/include
```

Python-side layout fix:

```text
QUEST_NHD_LAYOUT changed from 1 to 0.

Quest/FlashInfer layout enum:
  NHD = 0
  HND = 1
```

Tests added:

```text
test_quest_cuda_uses_flashinfer_nhd_layout_value

test_quest_cuda_backend_matches_torch_reference_when_op_is_built
  CUDA-gated
  skips if the local _C op has not been rebuilt yet
  compares quest_cuda per_query_head_scores to TorchQuestScorer
```

Validation completed:

```text
git diff --check
  passed

python -m py_compile vllm/v1/mixed_precision_recovery/scoring.py \
  tests/v1/mixed_precision_recovery/test_scoring.py
  passed

/home/han/anaconda3/envs/20260528_vllm/bin/python -m pytest \
  tests/v1/mixed_precision_recovery/test_scoring.py
  19 passed, 1 skipped

/usr/local/cuda-12.8/bin/nvcc -std=c++20 --expt-relaxed-constexpr \
  -gencode arch=compute_80,code=sm_80 \
  -Icsrc -Icsrc/third_party/flashinfer/include \
  -I...torch includes... \
  -c csrc/mpr/quest_estimate.cu -o /tmp/mpr_quest_estimate.o
  passed
```

Validation limitations:

```text
The local execution environment reports torch.cuda.is_available() == False, so
the CUDA runtime parity test was skipped.

Full CMake configure/build was not completed because this checkout attempts to
FetchContent external dependencies from GitHub. Local Cutlass and Triton
sources avoided the first two downloads, but DeepGEMM still required network
access and blocked configure in the restricted environment.
```
