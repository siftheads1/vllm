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
