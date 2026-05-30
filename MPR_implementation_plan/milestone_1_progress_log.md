# Milestone 1 Progress Log

Last updated: 2026-05-28

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
