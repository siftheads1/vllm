#!/usr/bin/env python3
"""Benchmark MPR digest scoring backends on synthetic tensors."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch

from vllm.v1.mixed_precision_recovery.quest_packing import (
    pack_quest_metadata_cache,
)
from vllm.v1.mixed_precision_recovery.scoring import (
    QUEST_NHD_LAYOUT,
    QuestCudaScorer,
    TorchQuestScorer,
    aggregate_query_head_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MPR Quest-style scoring latency.")
    parser.add_argument(
        "--candidates",
        default="8,16,32,64,128",
        help="Comma-separated candidate digest counts.",
    )
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--metadata-page-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-agg", choices=("max", "mean"), default="max")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"),
                        default="float16")
    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_sync_latency_ms(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
        synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def time_cuda_event_ms(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def make_inputs(
    *,
    num_candidates: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.randn(
        (num_q_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    center = torch.randn(
        (num_candidates, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    radius = torch.rand(
        (num_candidates, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    digest_min = center - radius
    digest_max = center + radius
    return query, digest_min, digest_max


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    candidate_counts = [
        int(value) for value in args.candidates.split(",") if value.strip()
    ]
    if not candidate_counts:
        raise SystemExit("At least one candidate count is required.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch_scorer = TorchQuestScorer()
    quest_scorer = QuestCudaScorer()

    try:
        from vllm import _custom_ops as ops
    except Exception as exc:
        raise SystemExit(f"Could not import vLLM custom ops: {exc}") from exc
    if not hasattr(ops, "mpr_estimate_attn_score"):
        raise SystemExit("mpr_estimate_attn_score custom op is unavailable.")

    print("benchmark: mpr_scoring")
    print(f"dtype: {args.dtype}")
    print(f"num_q_heads: {args.num_q_heads}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"head_dim: {args.head_dim}")
    print(f"metadata_page_size: {args.metadata_page_size}")
    print(f"topk: {args.topk}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(
        "columns: candidates, torch_total_wall_ms, torch_total_cuda_ms, "
        "quest_cuda_total_wall_ms, quest_cuda_total_cuda_ms, "
        "quest_cuda_persistent_wall_ms, quest_cuda_persistent_cuda_ms, "
        "quest_pack_wall_ms, quest_pack_cuda_ms, "
        "quest_cuda_kernel_wall_ms, quest_cuda_kernel_cuda_ms, "
        "quest_postprocess_wall_ms, quest_postprocess_cuda_ms, "
        "topk_block_wall_ms, topk_kv_head_wall_ms, "
        "topk_query_head_wall_ms")

    for num_candidates in candidate_counts:
        query, digest_min, digest_max = make_inputs(
            num_candidates=num_candidates,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
        )

        def torch_total() -> object:
            return torch_scorer.estimate(
                query_window=query,
                digest_min=digest_min,
                digest_max=digest_max,
                score_agg=args.score_agg,
                metadata_page_size=args.metadata_page_size,
            )

        def quest_total() -> object:
            return quest_scorer.estimate(
                query_window=query,
                digest_min=digest_min,
                digest_max=digest_max,
                score_agg=args.score_agg,
                metadata_page_size=args.metadata_page_size,
            )

        def quest_persistent_total() -> object:
            return quest_scorer.estimate_packed(
                query_window=query,
                packed=packed,
                num_kv_heads=args.num_kv_heads,
                score_agg=args.score_agg,
            )

        def quest_pack_only() -> object:
            return pack_quest_metadata_cache(
                digest_min=digest_min,
                digest_max=digest_max,
                metadata_page_size=args.metadata_page_size,
                add_guard_entry=True,
            )

        packed = pack_quest_metadata_cache(
            digest_min=digest_min,
            digest_max=digest_max,
            metadata_page_size=args.metadata_page_size,
            add_guard_entry=True,
        )
        query_batch = query.unsqueeze(0).contiguous()
        kernel_output = torch.empty(
            (args.num_q_heads, packed.num_score_entries),
            dtype=dtype,
            device=device,
        )

        def quest_kernel_only() -> object:
            ops.mpr_estimate_attn_score(
                query_batch,
                kernel_output,
                packed.metadata_data,
                packed.metadata_indices,
                packed.metadata_indptr,
                packed.metadata_last_page_len,
                packed.metadata_last_page_idx,
                QUEST_NHD_LAYOUT,
            )
            return kernel_output

        quest_kernel_only()
        synchronize()

        def quest_postprocess_only() -> object:
            per_query_head_scores = kernel_output.transpose(0, 1).contiguous()
            block_scores, per_kv_head_scores = aggregate_query_head_scores(
                per_query_head_scores,
                num_kv_heads=args.num_kv_heads,
                score_agg=args.score_agg,
            )
            return block_scores, per_kv_head_scores

        per_query_head_scores = kernel_output.transpose(0, 1).contiguous()
        block_scores, _ = aggregate_query_head_scores(
            per_query_head_scores,
            num_kv_heads=args.num_kv_heads,
            score_agg=args.score_agg,
        )
        _, per_kv_head_scores = aggregate_query_head_scores(
            per_query_head_scores,
            num_kv_heads=args.num_kv_heads,
            score_agg=args.score_agg,
        )
        physical_block_ids = list(range(num_candidates))
        topk = min(args.topk, num_candidates)

        def topk_block_only() -> object:
            if topk <= 0:
                return [], []
            topk_scores, topk_indices = torch.topk(block_scores, k=topk)
            topk_index_values = topk_indices.detach().cpu().tolist()
            topk_block_ids = [
                physical_block_ids[int(index)] for index in topk_index_values
            ]
            topk_score_values = [
                float(score)
                for score in topk_scores.detach().to(torch.float32).cpu().tolist()
            ]
            return topk_block_ids, topk_score_values

        def topk_by_head_only(scores_by_block_head: torch.Tensor) -> object:
            if topk <= 0:
                return [], []
            topk_block_ids_by_head: list[list[int]] = []
            topk_scores_by_head: list[list[float]] = []
            num_heads = int(scores_by_block_head.shape[1])
            for head_idx in range(num_heads):
                head_scores = scores_by_block_head[:, head_idx]
                topk_scores, topk_indices = torch.topk(head_scores, k=topk)
                topk_index_values = topk_indices.detach().cpu().tolist()
                topk_block_ids_by_head.append([
                    physical_block_ids[int(index)] for index in topk_index_values
                ])
                topk_scores_by_head.append([
                    float(score)
                    for score in topk_scores.detach().to(torch.float32).cpu().tolist()
                ])
            return topk_block_ids_by_head, topk_scores_by_head

        def topk_kv_head_only() -> object:
            return topk_by_head_only(per_kv_head_scores)

        def topk_query_head_only() -> object:
            return topk_by_head_only(per_query_head_scores)

        torch_wall = time_sync_latency_ms(
            torch_total,
            warmup=args.warmup,
            iters=args.iters,
        )
        torch_cuda = time_cuda_event_ms(
            torch_total,
            warmup=args.warmup,
            iters=args.iters,
        )
        quest_wall = time_sync_latency_ms(
            quest_total,
            warmup=args.warmup,
            iters=args.iters,
        )
        quest_cuda = time_cuda_event_ms(
            quest_total,
            warmup=args.warmup,
            iters=args.iters,
        )
        persistent_wall = time_sync_latency_ms(
            quest_persistent_total,
            warmup=args.warmup,
            iters=args.iters,
        )
        persistent_cuda = time_cuda_event_ms(
            quest_persistent_total,
            warmup=args.warmup,
            iters=args.iters,
        )
        pack_wall = time_sync_latency_ms(
            quest_pack_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        pack_cuda = time_cuda_event_ms(
            quest_pack_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        kernel_wall = time_sync_latency_ms(
            quest_kernel_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        kernel_cuda = time_cuda_event_ms(
            quest_kernel_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        postprocess_wall = time_sync_latency_ms(
            quest_postprocess_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        postprocess_cuda = time_cuda_event_ms(
            quest_postprocess_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        topk_block_wall = time_sync_latency_ms(
            topk_block_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        topk_kv_head_wall = time_sync_latency_ms(
            topk_kv_head_only,
            warmup=args.warmup,
            iters=args.iters,
        )
        topk_query_head_wall = time_sync_latency_ms(
            topk_query_head_only,
            warmup=args.warmup,
            iters=args.iters,
        )

        print(
            f"{num_candidates}, "
            f"{torch_wall:.6f}, {torch_cuda:.6f}, "
            f"{quest_wall:.6f}, {quest_cuda:.6f}, "
            f"{persistent_wall:.6f}, {persistent_cuda:.6f}, "
            f"{pack_wall:.6f}, {pack_cuda:.6f}, "
            f"{kernel_wall:.6f}, {kernel_cuda:.6f}, "
            f"{postprocess_wall:.6f}, {postprocess_cuda:.6f}, "
            f"{topk_block_wall:.6f}, {topk_kv_head_wall:.6f}, "
            f"{topk_query_head_wall:.6f}")


if __name__ == "__main__":
    main()
