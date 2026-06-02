#!/usr/bin/env python3
"""Benchmark MPR semantic CPU backup overhead on synthetic KV blocks."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from vllm.v1.mixed_precision_recovery.cpu_backup import SemanticCPUBackupStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark synchronous MPR CPU backup latency.")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
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


def make_kv_cache(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(
        (2, num_blocks, block_size, num_kv_heads, head_dim),
        dtype=dtype,
        device="cuda",
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if args.num_blocks <= 0:
        raise SystemExit("--num-blocks must be positive.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    kv_cache = make_kv_cache(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
    )
    bytes_per_backup = (
        2 * args.block_size * args.num_kv_heads * args.head_dim
        * torch.tensor([], dtype=torch.float16).element_size()
    )

    def block_for_iter(i: int) -> torch.Tensor:
        return kv_cache[:, i % args.num_blocks]

    def raw_d2h_copy(i: int) -> torch.Tensor:
        return block_for_iter(i).detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        )

    put_store = SemanticCPUBackupStore()
    put_counter = 0

    def store_put(i: int) -> object:
        nonlocal put_counter
        physical_block_id = put_counter
        put_counter += 1
        return put_store.put(
            layer_name="benchmark.layer",
            physical_block_id=physical_block_id,
            kv_block=block_for_iter(i),
        )

    release_store = SemanticCPUBackupStore()
    for i in range(args.iters):
        release_store.put(
            layer_name="benchmark.layer",
            physical_block_id=i,
            kv_block=block_for_iter(i),
        )
    synchronize()

    def release_all(_: int) -> object:
        return release_store.release_blocks(set(range(args.iters)))

    raw_wall_ms = time_sync_latency_ms(
        raw_d2h_copy,
        warmup=args.warmup,
        iters=args.iters,
    )
    store_put_wall_ms = time_sync_latency_ms(
        store_put,
        warmup=args.warmup,
        iters=args.iters,
    )
    store_stats = put_store.stats()

    release_start = time.perf_counter()
    release_result = release_all(0)
    release_wall_ms = (time.perf_counter() - release_start) * 1000.0

    print("benchmark: mpr_cpu_backup")
    print(f"dtype: {args.dtype}")
    print(f"block_size: {args.block_size}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"head_dim: {args.head_dim}")
    print(f"num_blocks: {args.num_blocks}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"bytes_per_backup_fp16: {bytes_per_backup}")
    print(
        "columns: raw_d2h_copy_wall_ms, store_put_wall_ms, "
        "store_put_reported_copy_wall_ms, store_put_overhead_wall_ms, "
        "release_all_wall_ms, release_entries, release_bytes"
    )
    reported_copy_ms = (
        store_stats.total_copy_wall_seconds * 1000.0 / store_stats.put_count
        if store_stats.put_count else 0.0
    )
    print(
        f"{raw_wall_ms:.6f}, "
        f"{store_put_wall_ms:.6f}, "
        f"{reported_copy_ms:.6f}, "
        f"{store_put_wall_ms - raw_wall_ms:.6f}, "
        f"{release_wall_ms:.6f}, "
        f"{release_result.released_entries}, "
        f"{release_result.released_bytes}"
    )


if __name__ == "__main__":
    main()
