# Milestone 1 Progress Log

Last updated: 2026-05-27

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
negative slot id behavior = assertion failure
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
If absent and the KV cache is FlashAttention-shaped, it infers:
  block_size = kv_cache.shape[2]
Otherwise it still fails fast with the KV cache shape in the error message.
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
```

The slot mapping contract assumed for Milestone 1 remains:

```text
slot_id = physical_block_id * block_size + block_offset
physical_block_id = slot_id // block_size
block_offset = slot_id % block_size
```

This is valid for the current single-GPU/no-CP/no-DCP scope. If any negative
slot id is observed, Step 1.2 intentionally raises an assertion instead of
filtering it out.

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

Full runtime validation is still pending in the target GPU environment.

## Next Step

Run Step 1.2 validation in the target vLLM environment.

Recommended first validation:

```bash
cd /workspace/vllm
rm -rf /tmp/vllm_mpr_debug

VLLM_MPR_ENABLE=1 \
VLLM_MPR_DEBUG_DIR=/tmp/vllm_mpr_debug \
VLLM_MPR_MAX_LAYERS=10 \
VLLM_MPR_MAX_STEPS=20 \
VLLM_MPR_DUMP_EVERY=1 \
python scripts/mpr_baseline_qwen3_8b.py \
  --prompt "Say hello." \
  --max-tokens 8
```

Expected debug check:

```bash
cat /tmp/vllm_mpr_debug/*.jsonl | grep observe_kv_write | head
```

Expected outcome:

```text
generation completes
observe_kv_write events exist
num_valid_slots > 0 for decode writes
unique_block_ids is non-empty for decode writes
0 <= min_block_offset <= max_block_offset < block_size
no negative-slot assertion
```
