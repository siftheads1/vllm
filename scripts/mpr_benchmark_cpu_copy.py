#!/usr/bin/env python3
"""Benchmark pure GPU-to-CPU KV block copy variants for MPR."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pure D2H copy latency for one semantic KV block.")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Source GPU KV cache dtype.",
    )
    parser.add_argument(
        "--dst-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Destination CPU backup dtype.",
    )
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


def time_sync_latency_ms(
    fn: Callable[[int], object],
    *,
    warmup: int,
    iters: int,
) -> float:
    for i in range(warmup):
        fn(i)
    synchronize()

    start = time.perf_counter()
    for i in range(iters):
        fn(i)
        synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if args.num_blocks <= 0:
        raise SystemExit("--num-blocks must be positive.")

    src_dtype = dtype_from_name(args.dtype)
    dst_dtype = dtype_from_name(args.dst_dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    kv_cache = torch.randn(
        (
            2,
            args.num_blocks,
            args.block_size,
            args.num_kv_heads,
            args.head_dim,
        ),
        dtype=src_dtype,
        device="cuda",
    )
    block_shape = (
        2,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
    )
    pageable_dst = torch.empty(block_shape, dtype=dst_dtype, device="cpu")
    try:
        pinned_dst = torch.empty(
            block_shape,
            dtype=dst_dtype,
            device="cpu",
            pin_memory=True,
        )
        pinned_available = True
    except RuntimeError:
        pinned_dst = pageable_dst
        pinned_available = False

    def block_for_iter(i: int) -> torch.Tensor:
        return kv_cache[:, i % args.num_blocks]

    def alloc_to_cpu(i: int) -> torch.Tensor:
        return block_for_iter(i).detach().to(
            device="cpu",
            dtype=dst_dtype,
            copy=True,
        )

    def pageable_copy(i: int) -> torch.Tensor:
        pageable_dst.copy_(block_for_iter(i), non_blocking=False)
        return pageable_dst

    def pinned_copy_sync(i: int) -> torch.Tensor:
        pinned_dst.copy_(block_for_iter(i), non_blocking=False)
        return pinned_dst

    def pinned_copy_nonblocking(i: int) -> torch.Tensor:
        pinned_dst.copy_(block_for_iter(i), non_blocking=True)
        return pinned_dst

    bytes_per_copy = (
        2 * args.block_size * args.num_kv_heads * args.head_dim
        * torch.tensor([], dtype=dst_dtype).element_size()
    )

    alloc_ms = time_sync_latency_ms(
        alloc_to_cpu,
        warmup=args.warmup,
        iters=args.iters,
    )
    pageable_ms = time_sync_latency_ms(
        pageable_copy,
        warmup=args.warmup,
        iters=args.iters,
    )
    if pinned_available:
        pinned_sync_ms = time_sync_latency_ms(
            pinned_copy_sync,
            warmup=args.warmup,
            iters=args.iters,
        )
        pinned_nonblocking_ms = time_sync_latency_ms(
            pinned_copy_nonblocking,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        pinned_sync_ms = float("nan")
        pinned_nonblocking_ms = float("nan")

    print("benchmark: mpr_cpu_copy")
    print(f"src_dtype: {args.dtype}")
    print(f"dst_dtype: {args.dst_dtype}")
    print(f"block_size: {args.block_size}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"head_dim: {args.head_dim}")
    print(f"num_blocks: {args.num_blocks}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"bytes_per_copy: {bytes_per_copy}")
    print(f"pinned_available: {pinned_available}")
    print(
        "columns: alloc_to_cpu_wall_ms, pageable_copy_wall_ms, "
        "pinned_copy_sync_wall_ms, pinned_copy_nonblocking_wall_ms"
    )
    print(
        f"{alloc_ms:.6f}, "
        f"{pageable_ms:.6f}, "
        f"{pinned_sync_ms:.6f}, "
        f"{pinned_nonblocking_ms:.6f}"
    )


if __name__ == "__main__":
    main()
