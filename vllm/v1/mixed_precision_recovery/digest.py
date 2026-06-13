# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Key digest helpers for MPR."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


ARKVALE_DIGEST_KIND = "arkvale"
RAW_MINMAX_DIGEST_KIND = "raw_minmax"
SUPPORTED_DIGEST_KINDS = frozenset({ARKVALE_DIGEST_KIND, RAW_MINMAX_DIGEST_KIND})


@dataclass(frozen=True)
class KeyBlockDigest:
    """Summary for one full key-cache block.

    Attributes:
        digest_min: Lower digest bound, shaped ``[num_kv_heads, head_dim]``.
        digest_max: Upper digest bound, shaped ``[num_kv_heads, head_dim]``.
        valid_token_count: Number of token slots summarized. For Step 1.3 this
            is always equal to ``block_size`` because partial-block digests are
            not created.
        block_size: Number of token slots in the source block.
        digest_kind: Digest construction policy.
    """

    digest_min: torch.Tensor
    digest_max: torch.Tensor
    valid_token_count: int
    block_size: int
    digest_kind: str = ARKVALE_DIGEST_KIND


def _validate_key_block(key_block: torch.Tensor) -> tuple[torch.Tensor, int]:
    if key_block.ndim != 3:
        raise ValueError(
            "MPR key block digest expects a 3D tensor shaped "
            f"[block_size, num_kv_heads, head_dim], got {tuple(key_block.shape)}."
        )

    key_block = key_block.detach()
    block_size = int(key_block.shape[0])
    if block_size <= 0:
        raise ValueError("MPR key block digest requires a non-empty block.")
    return key_block, block_size


def summarize_key_block(
    key_block: torch.Tensor,
    digest_kind: str = ARKVALE_DIGEST_KIND,
    profile_callback: Callable[[str, float], None] | None = None,
) -> KeyBlockDigest:
    """Summarize one full KV-cache key block.

    Args:
        key_block: FlashAttention key-cache block with shape
            ``[block_size, num_kv_heads, head_dim]``. This is expected to be
            ``kv_cache[0, physical_block_id]`` where the leading ``0`` selects
            keys rather than values from FlashAttention's KV cache.
        digest_kind: ``"arkvale"`` uses ArkVale's tightened center +/- mean
            distance bounds. ``"raw_minmax"`` uses Quest-style raw extrema.

    Returns:
        A bounding digest with min/max tensors shaped
        ``[num_kv_heads, head_dim]``.
    """
    if digest_kind not in SUPPORTED_DIGEST_KINDS:
        raise ValueError(
            "MPR digest kind must be 'arkvale' or 'raw_minmax', "
            f"got {digest_kind!r}."
        )

    validate_start = time.perf_counter() if profile_callback is not None else 0.0
    key_block, block_size = _validate_key_block(key_block)
    if profile_callback is not None:
        profile_callback("counter_digest_validate", validate_start)

    amax_start = time.perf_counter() if profile_callback is not None else 0.0
    raw_max = key_block.amax(dim=0)
    if profile_callback is not None:
        profile_callback("counter_digest_amax", amax_start)

    amin_start = time.perf_counter() if profile_callback is not None else 0.0
    raw_min = key_block.amin(dim=0)
    if profile_callback is not None:
        profile_callback("counter_digest_amin", amin_start)

    if digest_kind == RAW_MINMAX_DIGEST_KIND:
        result_start = time.perf_counter() if profile_callback is not None else 0.0
        result = KeyBlockDigest(
            digest_min=raw_min,
            digest_max=raw_max,
            valid_token_count=block_size,
            block_size=block_size,
            digest_kind=digest_kind,
        )
        if profile_callback is not None:
            profile_callback("counter_digest_raw_minmax_result", result_start)
        return result

    centers_start = time.perf_counter() if profile_callback is not None else 0.0
    # centers: midpoint of the raw bounding box.
    # Shape: [num_kv_heads, head_dim].
    centers = (raw_max + raw_min) / 2
    if profile_callback is not None:
        profile_callback("counter_digest_arkvale_centers", centers_start)

    dists_start = time.perf_counter() if profile_callback is not None else 0.0
    # dists: mean absolute distance from center over the block token axis.
    # centers.unsqueeze(0) broadcasts to [block_size, num_kv_heads, head_dim].
    # Shape: [num_kv_heads, head_dim].
    dists = (centers.unsqueeze(0) - key_block).abs().mean(dim=0)
    if profile_callback is not None:
        profile_callback("counter_digest_arkvale_dists", dists_start)

    result_start = time.perf_counter() if profile_callback is not None else 0.0
    result = KeyBlockDigest(
        # digest_min/digest_max: ArkVale-style tightened bounds.
        # Shape: [num_kv_heads, head_dim].
        digest_min=centers - dists,
        digest_max=centers + dists,
        valid_token_count=block_size,
        block_size=block_size,
        digest_kind=digest_kind,
    )
    if profile_callback is not None:
        profile_callback("counter_digest_arkvale_result", result_start)
    return result
