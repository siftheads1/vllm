#!/usr/bin/env python3
"""Benchmark vLLM offline engine step latency for MPR experiments."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM generation and report per-engine-step latency.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--prompt",
        default="Explain KV cache in one sentence.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--print-steps",
        action="store_true",
        help="Print every measured engine step latency.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Keep generating until max_tokens for latency comparisons.",
    )
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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    sorted_values = sorted(values)
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def predicted_boundary_steps(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    block_size: int,
) -> list[int]:
    """Return approximate step indices where a new full block appears.

    Step 0 is the prefill step. For single-request, non-chunked prefill runs,
    generated token index ``t`` roughly appears at engine step ``t``.
    """
    if block_size <= 0:
        return []
    boundaries = []
    for generated_idx in range(1, generated_tokens + 1):
        if (prompt_tokens + generated_idx) % block_size == 0:
            boundaries.append(generated_idx)
    return boundaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step_idx",
                "phase",
                "latency_ms",
                "num_step_outputs",
                "has_finished_output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    os.environ.setdefault("VLLM_USE_V1", "1")

    from vllm import LLM, SamplingParams

    print("benchmark: mpr_step_latency")
    print(f"model: {args.model}")
    print(f"dtype: {args.dtype}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"block_size: {args.block_size}")
    print(f"mpr_enable: {os.environ.get('VLLM_MPR_ENABLE')}")
    print(f"mpr_cpu_backup: {os.environ.get('VLLM_MPR_CPU_BACKUP')}")
    print(f"mpr_scoring_enable: {os.environ.get('VLLM_MPR_SCORING_ENABLE')}")
    print(f"mpr_scoring_backend: {os.environ.get('VLLM_MPR_SCORING_BACKEND')}")

    load_start = time.perf_counter()
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
    load_elapsed_sec = time.perf_counter() - load_start

    step_rows: list[dict[str, Any]] = []
    original_step = llm.llm_engine.step

    def timed_step():
        step_idx = len(step_rows)
        start = time.perf_counter()
        outputs = original_step()
        latency_ms = (time.perf_counter() - start) * 1000.0
        row = {
            "step_idx": step_idx,
            "phase": "prefill" if step_idx == 0 else "decode",
            "latency_ms": latency_ms,
            "num_step_outputs": len(outputs),
            "has_finished_output": any(
                bool(getattr(output, "finished", False))
                for output in outputs
            ),
        }
        step_rows.append(row)
        return outputs

    llm.llm_engine.step = timed_step  # type: ignore[method-assign]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
    )

    generate_start = time.perf_counter()
    request_ids = llm.enqueue(
        [args.prompt],
        sampling_params,
        use_tqdm=False,
    )
    outputs = llm.wait_for_completion(use_tqdm=False)
    generate_elapsed_sec = time.perf_counter() - generate_start

    latencies = [float(row["latency_ms"]) for row in step_rows]
    prefill_latencies = [
        float(row["latency_ms"])
        for row in step_rows
        if row["phase"] == "prefill"
    ]
    decode_latencies = [
        float(row["latency_ms"])
        for row in step_rows
        if row["phase"] == "decode"
    ]
    prompt_tokens = sum(
        len(output.prompt_token_ids)
        if output.prompt_token_ids is not None
        else 0
        for output in outputs
    )
    generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    boundary_steps = predicted_boundary_steps(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        block_size=args.block_size,
    )
    boundary_set = set(boundary_steps)
    boundary_latencies = [
        float(row["latency_ms"])
        for row in step_rows
        if row["phase"] == "decode" and int(row["step_idx"]) in boundary_set
    ]
    non_boundary_latencies = [
        float(row["latency_ms"])
        for row in step_rows
        if row["phase"] == "decode" and int(row["step_idx"]) not in boundary_set
    ]

    print(f"load_elapsed_sec: {load_elapsed_sec:.6f}")
    print(f"generate_elapsed_sec: {generate_elapsed_sec:.6f}")
    print(f"request_ids: {request_ids}")
    print(f"prompt_token_count: {prompt_tokens}")
    print(f"generated_token_count: {generated_tokens}")
    print(f"engine_step_count: {len(step_rows)}")
    print(f"predicted_boundary_steps: {boundary_steps}")
    if generated_tokens and generate_elapsed_sec > 0:
        print(
            "generated_tokens_per_sec: "
            f"{generated_tokens / generate_elapsed_sec:.6f}"
        )
    if latencies:
        print("step_latency_ms:")
        print(f"  mean: {statistics.mean(latencies):.6f}")
        print(f"  median: {statistics.median(latencies):.6f}")
        print(f"  p90: {percentile(latencies, 90):.6f}")
        print(f"  p95: {percentile(latencies, 95):.6f}")
        print(f"  p99: {percentile(latencies, 99):.6f}")
        print(f"  max: {max(latencies):.6f}")
    if prefill_latencies:
        print("prefill_step_latency_ms:")
        print(f"  mean: {statistics.mean(prefill_latencies):.6f}")
        print(f"  max: {max(prefill_latencies):.6f}")
    if decode_latencies:
        print("decode_step_latency_ms:")
        print(f"  mean: {statistics.mean(decode_latencies):.6f}")
        print(f"  median: {statistics.median(decode_latencies):.6f}")
        print(f"  p90: {percentile(decode_latencies, 90):.6f}")
        print(f"  p95: {percentile(decode_latencies, 95):.6f}")
        print(f"  p99: {percentile(decode_latencies, 99):.6f}")
        print(f"  max: {max(decode_latencies):.6f}")
    if boundary_latencies:
        print("predicted_boundary_decode_step_latency_ms:")
        print(f"  mean: {statistics.mean(boundary_latencies):.6f}")
        print(f"  max: {max(boundary_latencies):.6f}")
    if non_boundary_latencies:
        print("non_boundary_decode_step_latency_ms:")
        print(f"  mean: {statistics.mean(non_boundary_latencies):.6f}")
        print(f"  max: {max(non_boundary_latencies):.6f}")

    if args.print_steps:
        print(
            "columns: step_idx, phase, latency_ms, num_step_outputs, "
            "has_finished_output"
        )
        for row in step_rows:
            print(
                f"{row['step_idx']}, {row['phase']}, "
                f"{row['latency_ms']:.6f}, "
                f"{row['num_step_outputs']}, {row['has_finished_output']}"
            )
    if args.csv is not None:
        write_csv(args.csv, step_rows)
        print(f"csv_path: {args.csv}")


if __name__ == "__main__":
    main()
