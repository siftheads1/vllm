# Step 4.5.0 INT4 Design Review

Date: 2026-06-08

Status:

```text
investigation complete
user decisions pending
no implementation changes made
```

## Question

Before implementing INT4 recovery, decide whether the planned representation is
reasonable and how the first M4.5 implementation should encode, store, and
materialize INT4 payloads.

## Findings

PyTorch has `torch.int4` and `torch.uint4` names in the current local
environment, but they should not be used as the MPR backup payload dtype.

Local check:

```text
torch: 2.11.0+cu130
has torch.int4: true
has torch.uint4: true
torch.empty(..., dtype=torch.int4): creates a tensor object
float_tensor.to(torch.int4): NotImplementedError
torch.ones(..., dtype=torch.int4): NotImplementedError
int4_tensor + int4_tensor: NotImplementedError
```

PyTorch/TorchAO documentation matches this interpretation:

```text
torch.int1..torch.int7 and torch.uint1..torch.uint7 exist as low-precision
dtype names/placeholders, but the integer sub-byte dtypes do not have general
eager implementations.

TorchAO describes practical int4 tensors as derived/quantized tensors with a
packing format and metadata, not as plain eager int4 tensors.
```

vLLM also points in the same direction:

```text
vllm.scalar_type can describe sub-byte scalar types because torch.dtype does
not generally cover them as regular tensor dtypes.

vLLM weight quantization paths use packed sub-byte formats such as uint4b8,
packed uint8/int32 buffers, and explicit conversion/repacking utilities.

The current local vLLM tree also has an nvfp4 KV cache path. That path stores
packed fp4 data plus fp8 block scales in uint8-backed KV cache buffers and uses
FlashInfer/TRTLLM attention integration. It is useful as a packing/layout
reference, but it is not the same as signed symmetric INT4 recovery payloads.
```

Conclusion:

```text
M4.5 should implement INT4 as packed torch.uint8 payload bytes plus explicit
scale/original-shape metadata and explicit unpack/dequant materialization.

Do not use torch.int4/torch.uint4 tensors as the logical backup payload.
```

## Recommended Decisions

### 1. Quantized Range

Recommendation:

```text
signed symmetric INT4
encoder emits values in [-7, 7]
scale = max(abs(vector)) / 7
zero-vector scale = 1.0 and quantized values remain 0
```

Reasoning:

```text
matches the existing M4 INT8 reference style, which uses [-127, 127]
keeps zero exactly representable
avoids asymmetric zero-point handling in the first implementation
leaves -8 as an unused representable value in the packed signed nibble format
```

### 2. Nibble Encoding

Recommendation:

```text
use signed two's-complement nibble encoding
q_nibble = q_int4 & 0xF
unpack sign-extends values >= 8 back to negative int8 values
encoder never emits -8
```

Reasoning:

```text
zero is encoded as nibble 0
zero padding remains semantically zero
the codec is self-contained and does not need GPTQ-style bias-8 semantics
future kernel-specific encodings can be hidden behind the codec boundary
```

Alternative:

```text
bias-8 / uint4b8 encoding stores q + 8
```

This aligns with some vLLM weight quantization formats, but it encodes zero as
8 and makes padding/zero-vector handling less direct for this MPR payload.

### 3. Pack Order and Shape

Recommendation:

```text
pack along the head_dim dimension
packed shape = [2, block_size, num_kv_heads, ceil(head_dim / 2)]
even head_dim element -> low nibble
odd head_dim element -> high nibble
if head_dim is odd, pad only the final high nibble of each vector with zero
track original_shape for exact materialization
```

Reasoning:

```text
preserves per-token-per-kv-head vector boundaries
keeps scale broadcasting simple
matches common least-significant-nibble-first packing loops
avoids cross-vector packing complexity in the correctness-first implementation
```

This refines the initial action-plan wording, which said `ceil(numel / 2)`.
Flat packing is slightly denser only for odd vector lengths, but preserving
vector boundaries is cleaner and safer. Most KV head dimensions are even.

### 4. Scale Granularity and Dtype

Recommendation:

```text
scale granularity = per-token-per-kv-head
scale shape = [2, block_size, num_kv_heads]
scale dtype = torch.float32 for the reference codec
```

Reasoning:

```text
matches the M4 INT8 codec
matches the local vLLM concept of per-token-head KV quantization modes
keeps INT4 error bounded per K/V token/head vector
avoids introducing a new granularity variable before optimization work
```

Future optimization can evaluate fp16/bf16/fp8 scale storage, grouped scales,
or GPU-side quantization.

### 5. Threshold Policy

Recommendation:

```text
extend top_ratio to fp16 -> int8 -> int4 -> skip in M4.5
do not extend threshold policy yet
if precision_policy == threshold and tier_int4_ratio > 0, reject config
```

Reasoning:

```text
existing threshold policy has only high/low thresholds
adding INT4 needs high/mid/low or a different rule
silently mapping INT4 into the existing threshold policy would be ambiguous
top_ratio is the current smoke-stable M4 policy
```

### 6. Storage Mode

Recommendation:

```text
add backup_storage_mode = eager_fp16_int8_int4
keep eager_fp16_int8 as the default
M4.5 INT4 smoke should explicitly use eager_fp16_int8_int4
```

Reasoning:

```text
preserves existing M4 default behavior
lets INT4 be enabled deliberately
keeps fp16_only available for future on-the-fly provider experiments
avoids making every M4 run pay INT4 quantization/packing cost by default
```

## Decision Items for User

```text
1. Accept signed symmetric [-7, 7] INT4?
2. Accept two's-complement nibble encoding instead of bias-8 uint4b8?
3. Accept packing along head_dim with shape [2, block_size, num_kv_heads, ceil(head_dim / 2)]?
4. Keep scale granularity per-token-per-kv-head with fp32 scale for reference?
5. Defer threshold INT4 support and reject threshold + tier_int4_ratio > 0?
6. Add eager_fp16_int8_int4 but keep eager_fp16_int8 as default?
```

## Sources Checked

Primary/reference sources:

```text
PyTorch dtype docs:
  https://docs.pytorch.org/docs/main/tensor_attributes.html

TorchAO quantization overview:
  https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html

TorchAO quantized inference docs:
  https://docs.pytorch.org/ao/stable/workflows/inference.html

vLLM scalar type docs:
  https://docs.vllm.ai/en/stable/api/vllm/scalar_type/

vLLM quantization docs:
  https://docs.vllm.ai/en/stable/features/quantization/index.html

vLLM quantized KV cache docs:
  https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/
```

Local code checked:

```text
vllm/scalar_type.py
vllm/config/cache.py
vllm/utils/torch_utils.py
vllm/v1/kv_cache_interface.py
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/flashinfer.py
vllm/model_executor/layers/quantization/utils/quant_utils.py
vllm/v1/mixed_precision_recovery/backup_codec.py
vllm/v1/mixed_precision_recovery/precision_policy.py
```
