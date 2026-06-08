# Milestone 4.5 Progress Log

## 2026-06-08: Milestone Created

Created Milestone 4.5 as a separate INT4 integration milestone between
Milestone 4 and Milestone 5.

Rationale:

```text
INT4 support is not just a Post-M5 optimization item
INT4 changes the intended precision tier set to fp16/int8/int4/skip
M5 optimization should run after the full intended tier set is integrated
PyTorch does not provide a general-purpose torch.int4 tensor dtype for this path
therefore M4.5 should implement packed uint8 payloads with explicit pack/unpack
```

Initial scope:

```text
packed INT4 backup codec
top-ratio policy extension
CPU backup/provider integration
recovery materialization into normal GPU KV cache dtype
debug JSONL accounting and validator support
ratio-sweep smoke with generated text visibility
simulated degraded-residency skip validation with INT4 present
```

Out of scope:

```text
direct mixed-dtype attention
GPU low-precision staging buffer
real scheduler-owned offload/eviction
multi-request/preemption correctness
FP8 or vLLM native FP8 KV cache support
```

## 2026-06-08: Step 4.5.0 INT4 Design Review

Completed the pre-implementation INT4 design review.

Added:

```text
MPR_implementation_plan/milestones/milestone_4_5/reports/step_4_5_0_int4_design_review.md
```

Status:

```text
investigation complete
user decisions pending
no implementation changes made
```

Key finding:

```text
The local PyTorch 2.11 environment exposes torch.int4 and torch.uint4 names,
but basic eager operations such as float_tensor.to(torch.int4), fill, copy,
and add are not implemented. M4.5 should not use torch.int4/torch.uint4 as
the backup payload dtype.
```

Recommended M4.5 direction:

```text
represent INT4 payloads as packed torch.uint8 bytes plus explicit scale and
shape metadata
use signed symmetric quantization with emitted values in [-7, 7]
use two's-complement nibble encoding
pack along head_dim with packed shape:
  [2, block_size, num_kv_heads, ceil(head_dim / 2)]
keep per-token-per-kv-head fp32 scales for the reference codec
extend top_ratio first and defer threshold INT4 semantics
add eager_fp16_int8_int4 while keeping eager_fp16_int8 as the default
```
