# Milestone 1 Step 1.0 Baseline

Step 1.0 fixes the baseline environment and command before adding any MPR
sidecar hooks.

## Environment

```text
conda executable: /opt/miniforge3/bin/conda
conda env: vllm
target vLLM tree: /workspace/vllm
reference ArkVale tree: /workspace/ArkVale/source/arkvale
torch: 2.11.0+cu130
CUDA available: true
GPU: NVIDIA GeForce RTX 5090
vLLM version: 0.21.1rc1.dev278+gd4004455d
vLLM commit: d4004455d2357985830af10e432709b42c820455
HF_HOME: /workspace/.hf_home
```

Note: `/workspace/ArkVale/vllm` and `/workspace/vllm` are currently the same
commit, but the editable install points to `/workspace/vllm`. Code changes for
Milestone 1 should target `/workspace/vllm`.

## Baseline Model

```text
model: Qwen/Qwen3-8B
```

The model is not currently cached locally. The first vLLM run will download it
through Hugging Face and store it under `HF_HOME`, expected to be:

```text
/workspace/.hf_home/hub
```

## Baseline Script

```text
/workspace/vllm/scripts/mpr_baseline_qwen3_8b.py
```

The script sets:

```text
VLLM_USE_V1=1
enforce_eager=True
enable_chunked_prefill=False
temperature=0.0
seed=0
tensor_parallel_size=1
max_model_len=2048
max_tokens=32
gpu_memory_utilization=0.75
```

It does not enable speculative decoding, DCP/CP, CPU offload, or MPR hooks.

## Run Command

From `/workspace/vllm`:

```bash
/opt/miniforge3/bin/conda run -n vllm \
  python scripts/mpr_baseline_qwen3_8b.py
```

Optional smaller smoke prompt:

```bash
/opt/miniforge3/bin/conda run -n vllm \
  python scripts/mpr_baseline_qwen3_8b.py \
  --prompt "Say hello." \
  --max-tokens 8
```

## Validation Criteria

The run is considered a valid Step 1.0 baseline if:

1. Qwen3-8B downloads or loads successfully.
2. vLLM uses the `/workspace/vllm` editable tree.
3. Generation completes for a single prompt.
4. Output includes generated text and token ids.
5. No MPR sidecar code is imported or enabled.

## Status

Completed:

- Conda env import check
- PyTorch/CUDA check
- vLLM import check
- Editable target check
- Baseline script creation
- Baseline script syntax check
- Baseline script CLI help check
- Full Qwen3-8B baseline generation

Observed baseline output:

```text
KV cache is a technique used in transformer models to store the keys and values
of previous attention computations, allowing the model to efficiently process
sequential data by reusing these ...
```

This satisfies Step 1.0: Qwen3-8B loaded, vLLM generated output for a single
prompt, and no MPR sidecar code was enabled.
