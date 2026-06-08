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
