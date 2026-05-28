#!/usr/bin/env python3
"""Deterministic vLLM baseline for MPR Milestone 1 Step 1.0."""

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small deterministic vLLM baseline generation."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--prompt",
        default="Explain KV cache in one sentence.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Allow CUDA graphs/compile paths instead of eager mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set before importing vLLM so environment-based config is visible early.
    os.environ.setdefault("VLLM_USE_V1", "1")

    from vllm import LLM, SamplingParams

    print("baseline_model:", args.model)
    print("hf_home:", os.environ.get("HF_HOME"))
    print("vllm_use_v1:", os.environ.get("VLLM_USE_V1"))

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.no_enforce_eager,
        enable_chunked_prefill=False,
        trust_remote_code=args.trust_remote_code,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    outputs = llm.generate([args.prompt], sampling_params)
    for output in outputs:
        prompt_token_count = (
            len(output.prompt_token_ids)
            if output.prompt_token_ids is not None
            else 0
        )
        generated_token_count = len(output.outputs[0].token_ids)
        total_token_count = prompt_token_count + generated_token_count

        print("prompt:", output.prompt)
        print("prompt_token_count:", prompt_token_count)
        print("generated_text:", output.outputs[0].text)
        print("generated_token_ids:", output.outputs[0].token_ids)
        print("generated_token_count:", generated_token_count)
        print("total_token_count:", total_token_count)


if __name__ == "__main__":
    main()
