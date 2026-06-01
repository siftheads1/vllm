# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Disabled-by-default MPR sidecar scaffold."""

from __future__ import annotations

import math
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.debug import MPRDebugWriter
from vllm.v1.mixed_precision_recovery.digest import (
    KeyBlockDigest,
    summarize_key_block,
)
from vllm.v1.mixed_precision_recovery.scoring import (
    DigestScoringBackend,
    get_digest_scoring_backend,
)

logger = init_logger(__name__)


@dataclass
class BlockDigest:
    """Sidecar-owned digest cache entry for one physical KV block.

    Attributes:
        digest_min: Lower digest bound, shaped ``[num_kv_heads, head_dim]``.
        digest_max: Upper digest bound, shaped ``[num_kv_heads, head_dim]``.
        valid_token_count: Number of token slots summarized. For Step 1.3 this
            equals ``block_size`` because only full-block digests are created.
        block_size: Number of token slots in the physical KV block.
        digest_kind: Digest construction policy used for this entry.
        layer_event_idx: Per-layer KV-write event index that first completed
            the block. ``-1`` means the digest was created while the event was
            outside the debug dump window.
    """

    digest_min: Any
    digest_max: Any
    valid_token_count: int
    block_size: int
    layer_event_idx: int
    digest_kind: str = "arkvale"


@dataclass(frozen=True)
class RequestBlockContext:
    """Current-request block context used for score candidate selection."""

    num_reqs: int | None
    max_query_len: int | None
    num_actual_tokens: int | None
    seq_lens: list[int] | None
    block_size: int | None
    block_table_shape: list[int] | None
    block_table_row: list[int]
    valid_block_ids: list[int]
    finalized_block_ids: list[int]
    recent_tokens: int
    protected_tail_entries: int
    protected_block_ids: list[int]
    score_candidate_block_ids: list[int]
    has_block_table_context: bool


@dataclass
class RecoverySidecar:
    """Score-only sidecar state for Milestone 1.

    The Step 1.3 prototype tracks written block offsets and summarizes full
    FlashAttention key-cache blocks into ArkVale-style digests. Debug dump
    limits affect JSONL records only; they do not gate sidecar state updates.
    """

    config: MPRConfig = field(default_factory=MPRConfig.from_env)
    counters: Counter[str] = field(default_factory=Counter)
    # Debug-only layer bookkeeping for VLLM_MPR_MAX_LAYERS dump limiting.
    _layer_indices: dict[str, int] = field(default_factory=dict)
    _kv_write_counts: Counter[str] = field(default_factory=Counter)
    _score_counts: Counter[str] = field(default_factory=Counter)
    _block_offsets: dict[str, dict[int, set[int]]] = field(default_factory=dict)
    _digest_cache: dict[str, dict[int, BlockDigest]] = field(default_factory=dict)
    # Step 1.4 v0 uses layer_name-only query windows for single-request smoke.
    # Serving/multi-request support needs request-scoped keys to avoid mixing
    # decode queries from different requests in the same layer buffer.
    _query_windows: dict[str, deque[torch.Tensor]] = field(default_factory=dict)
    _debug_writer: MPRDebugWriter = field(init=False)
    _scoring_backend: DigestScoringBackend = field(init=False)

    def __post_init__(self) -> None:
        """Initialize optional debug output and emit the init event."""
        self._debug_writer = MPRDebugWriter(self.config)
        self._scoring_backend = get_digest_scoring_backend(
            self.config.scoring_backend,
        )
        if self.config.enabled:
            logger.info(
                "MPR sidecar enabled: topk=%d, window_size=%d, "
                "recent_tokens=%d, score_agg=%s, scoring_backend=%s, "
                "digest_kind=%s, score_granularity=%s, debug_dir=%s",
                self.config.topk,
                self.config.window_size,
                self.config.recent_tokens,
                self.config.score_agg,
                self.config.scoring_backend,
                self.config.digest_kind,
                self.config.score_granularity,
                self.config.debug_dir,
            )
            self._record("init")

    def enabled(self) -> bool:
        """Return whether MPR sidecar observation is enabled."""
        return self.config.enabled

    def observe_kv_write(
        self,
        layer_name: str,
        key: Any = None,
        value: Any = None,
        kv_cache: Any = None,
        slot_mapping: Any = None,
        block_size: int | None = None,
    ) -> None:
        """Observe a KV-cache write and create digests for newly full blocks.

        Args:
            layer_name: vLLM attention layer name.
            key: Incoming key tensor for the current KV update. It is used only
                for debug shape reporting here. Expected shape is the
                attention-backend update shape for this forward pass, commonly
                ``[num_tokens, num_kv_heads, head_dim]`` for FlashAttention.
            value: Incoming value tensor for the current KV update. It is used
                only for debug shape reporting. Expected shape matches ``key``.
            kv_cache: FlashAttention KV cache after the write, shaped
                ``[2, num_blocks, block_size, num_kv_heads, head_dim]``. The
                leading dimension selects key/value; ``kv_cache[0]`` is the key
                cache used for digest generation.
            slot_mapping: Flat physical slot ids for the current update. Shape
                is typically ``[num_slots]`` after flattening. ``PAD_SLOT_ID``
                entries are CUDA graph padding and are ignored.
            block_size: Number of token slots per physical KV block.
        """
        if not self.config.enabled:
            return
        should_record, layer_event_idx = self._should_record_layer_event(
            layer_name,
            self._kv_write_counts,
        )
        if block_size is None or block_size <= 0:
            raise ValueError(
                f"MPR requires a positive block_size for {layer_name}, "
                f"got {block_size}."
            )
        if slot_mapping is None:
            raise ValueError(f"MPR requires slot_mapping for {layer_name}.")

        # flat_slots: flattened physical slot ids for this KV write.
        # Shape: [num_slots].
        flat_slots = slot_mapping.detach().reshape(-1)
        num_slots = int(flat_slots.numel())
        if num_slots == 0:
            unique_block_ids: list[int] = []
            min_block_offset = None
            max_block_offset = None
            num_pad_slots = 0
            digest_created_block_ids: list[int] = []
        else:
            has_invalid_negative_slot = bool((flat_slots < PAD_SLOT_ID).any().item())
            if has_invalid_negative_slot:
                min_slot = int(flat_slots.min().item())
                raise AssertionError(
                    f"MPR observed invalid negative slot id for {layer_name}: "
                    f"min_slot={min_slot}."
                )

            # vLLM pads unused CUDA graph slots with PAD_SLOT_ID. These are not
            # KV writes and should not participate in block/offset summaries.
            # valid_slots: physical slot ids that correspond to real KV writes.
            # Shape: [num_valid_slots].
            valid_slots = flat_slots[flat_slots != PAD_SLOT_ID]
            num_pad_slots = num_slots - int(valid_slots.numel())
            if valid_slots.numel() == 0:
                unique_block_ids = []
                min_block_offset = None
                max_block_offset = None
                digest_created_block_ids = []
            else:
                # block_ids: physical KV block id for each valid slot.
                # Shape: [num_valid_slots].
                block_ids = valid_slots // block_size

                # block_offsets: token offset within the physical block.
                # Shape: [num_valid_slots].
                block_offsets = valid_slots % block_size
                unique_block_ids = [
                    int(block_id)
                    for block_id in sorted(block_ids.unique().detach().cpu().tolist())
                ]
                min_block_offset = int(block_offsets.min().item())
                max_block_offset = int(block_offsets.max().item())
                digest_created_block_ids = self._observe_block_offsets(
                    layer_name=layer_name,
                    block_ids=block_ids,
                    block_offsets=block_offsets,
                    block_size=block_size,
                    kv_cache=kv_cache,
                    layer_event_idx=layer_event_idx,
                )

        if should_record:
            self._record(
                "observe_kv_write",
                layer_name=layer_name,
                layer_event_idx=layer_event_idx,
                key_shape=self._shape_of(key),
                value_shape=self._shape_of(value),
                kv_cache_shape=self._shape_of(kv_cache),
                slot_mapping_shape=self._shape_of(slot_mapping),
                num_slots=num_slots,
                num_valid_slots=num_slots - num_pad_slots,
                num_pad_slots=num_pad_slots,
                block_size=block_size,
                unique_block_ids=unique_block_ids,
                min_block_offset=min_block_offset,
                max_block_offset=max_block_offset,
                digest_created_block_ids=digest_created_block_ids,
                num_digest_blocks_for_layer=len(
                    self._digest_cache.get(layer_name, {})
                ),
                total_digest_blocks=self._num_digest_blocks(),
                digest_kind=self.config.digest_kind,
            )

    def observe_query(
        self,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
    ) -> None:
        """Observe a decode query and emit score-only digest estimates.

        Step 1.4 v0 is intentionally limited to single-request decode batches.
        Query windows are keyed only by ``layer_name``; multi-request serving
        would mix requests and requires request-scoped query windows plus block
        ownership tracking.
        """
        if not self.config.enabled:
            return
        should_record, layer_event_idx = self._should_record_layer_event(
            layer_name,
            self._score_counts,
        )
        if query is None:
            self._record_score_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "missing_query",
                query=query,
                attn_metadata=attn_metadata,
            )
            return
        if attn_metadata is None:
            self._record_score_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "missing_attn_metadata",
                query=query,
                attn_metadata=attn_metadata,
            )
            return

        max_query_len = getattr(attn_metadata, "max_query_len", None)
        if max_query_len != 1:
            self._record_score_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "non_decode_query",
                query=query,
                attn_metadata=attn_metadata,
            )
            return

        num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
        # In a max_query_len == 1 decode-shaped batch, num_actual_tokens equals
        # the number of active rows. Step 1.4 v0 uses a layer_name-only rolling
        # query window, so multi-request batches would mix request A/B queries.
        # vLLM startup can also emit synthetic warmup/capture batches with
        # shapes like max_query_len=1, num_actual_tokens=256, so this must be a
        # skip for smoke validation rather than an engine-failing assertion.
        # Serving support still needs request-scoped query windows and
        # request/block ownership tracking before scoring non-single rows.
        if num_actual_tokens != 1:
            self._record_score_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "non_single_request_decode",
                query=query,
                attn_metadata=attn_metadata,
            )
            return
        if query.ndim != 3:
            raise ValueError(
                "MPR Step 1.4 expects query shaped "
                f"[num_tokens, num_q_heads, head_dim], got {tuple(query.shape)} "
                f"for {layer_name}."
            )
        if int(query.shape[0]) < 1:
            self._record_score_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "empty_query",
                query=query,
                attn_metadata=attn_metadata,
            )
            return

        # query_for_window: current decode query for the single active request.
        # Shape: [num_q_heads, head_dim].
        query_for_window = query[0].detach().clone()
        if not torch.is_floating_point(query_for_window):
            self._record_score_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "non_floating_query",
                query=query,
                attn_metadata=attn_metadata,
            )
            return
        window = self._query_windows.setdefault(
            layer_name,
            deque(maxlen=self.config.window_size),
        )
        window.append(query_for_window)
        # window_query: average of the observed decode queries so far, up to
        # VLLM_MPR_WINDOW_SIZE. Shape: [num_q_heads, head_dim].
        window_query = torch.stack(tuple(window), dim=0).mean(dim=0)

        block_size = self._block_size_for_layer(layer_name)
        request_context = self._request_block_context(
            attn_metadata=attn_metadata,
            block_size=block_size,
        )
        candidate_block_ids = (
            request_context.score_candidate_block_ids
            if request_context.has_block_table_context
            else None
        )
        physical_block_ids, digest_min, digest_max = self._pack_layer_digests(
            layer_name,
            device=window_query.device,
            dtype=window_query.dtype,
            candidate_block_ids=candidate_block_ids,
        )
        if not physical_block_ids:
            skip_reason = (
                "no_score_candidate_blocks"
                if request_context.has_block_table_context
                else "no_digest_blocks"
            )
            self._record_score_skip(
                should_record,
                layer_name,
                layer_event_idx,
                skip_reason,
                query=query,
                attn_metadata=attn_metadata,
                window_query=window_query,
                window_query_len=len(window),
            )
            return

        score_result = self._scoring_backend.estimate(
            query_window=window_query,
            digest_min=digest_min,
            digest_max=digest_max,
            score_agg=self.config.score_agg,
            metadata_page_size=block_size,
        )
        scores = score_result.block_scores
        topk = min(self.config.topk, int(scores.numel()))
        topk_block_ids, topk_score_values = self._topk_block_scores(
            scores=scores,
            physical_block_ids=physical_block_ids,
            topk=topk,
        )
        head_debug = self._head_score_debug_fields(
            score_result=score_result,
            physical_block_ids=physical_block_ids,
            topk=topk,
        )

        score_block_debug = self._score_block_debug_fields(
            request_context=request_context,
            observed_digest_block_ids=physical_block_ids,
        )

        if should_record:
            self._record(
                "score_estimated",
                layer_name=layer_name,
                layer_event_idx=layer_event_idx,
                query_shape=self._shape_of(query),
                window_query_shape=self._shape_of(window_query),
                window_query_len=len(window),
                num_digest_blocks=len(physical_block_ids),
                score_count=int(scores.numel()),
                score_agg=self.config.score_agg,
                scoring_backend=score_result.scoring_backend,
                digest_kind=self.config.digest_kind,
                num_q_heads=score_result.num_q_heads,
                num_kv_heads=score_result.num_kv_heads,
                gqa_group_size=score_result.group_size,
                score_granularity=self.config.score_granularity,
                topk=topk,
                topk_block_ids=topk_block_ids,
                topk_scores=topk_score_values,
                **head_debug,
                **score_block_debug,
            )

    def _record_score_skip(
        self,
        should_record: bool,
        layer_name: str,
        layer_event_idx: int | None,
        reason: str,
        *,
        query: Any = None,
        attn_metadata: Any = None,
        window_query: Any = None,
        window_query_len: int | None = None,
    ) -> None:
        """Record a skipped score event when debug limits allow it."""
        if not should_record:
            return
        self._record(
            "score_skipped",
            layer_name=layer_name,
            layer_event_idx=layer_event_idx,
            skipped_reason=reason,
            query_shape=self._shape_of(query),
            max_query_len=getattr(attn_metadata, "max_query_len", None),
            num_actual_tokens=getattr(attn_metadata, "num_actual_tokens", None),
            window_query_shape=self._shape_of(window_query),
            window_query_len=window_query_len,
            num_digest_blocks=len(self._digest_cache.get(layer_name, {})),
            score_agg=self.config.score_agg,
            scoring_backend=self.config.scoring_backend,
            digest_kind=self.config.digest_kind,
            score_granularity=self.config.score_granularity,
        )

    def _head_score_debug_fields(
        self,
        *,
        score_result: Any,
        physical_block_ids: list[int],
        topk: int,
    ) -> dict[str, Any]:
        """Build optional head-level top-k fields for score debug records."""
        score_granularity = self.config.score_granularity
        if score_granularity == "block":
            return {}
        if score_granularity == "kv_head":
            head_scores = score_result.per_kv_head_scores
        elif score_granularity == "query_head":
            head_scores = score_result.per_query_head_scores
        else:
            raise ValueError(
                "MPR score_granularity must be 'block', 'kv_head', or "
                f"'query_head', got {score_granularity!r}."
            )

        topk_block_ids_by_head, topk_scores_by_head = self._topk_head_scores(
            scores_by_block_head=head_scores,
            physical_block_ids=physical_block_ids,
            topk=topk,
        )
        return {
            "num_score_heads": int(head_scores.shape[1]),
            "head_score_count": int(head_scores.numel()),
            "topk_block_ids_by_head": topk_block_ids_by_head,
            "topk_scores_by_head": topk_scores_by_head,
        }

    def _score_block_debug_fields(
        self,
        *,
        request_context: RequestBlockContext,
        observed_digest_block_ids: list[int],
    ) -> dict[str, Any]:
        """Build request/block metadata for score debug records.

        Args:
            request_context: Parsed FlashAttention request/block context.
            observed_digest_block_ids: Physical block IDs that were packed and
                scored for this layer. Shape-equivalent: ``[num_digest_blocks]``.

        Returns:
            JSON-serializable fields describing the single request's block table
            row, the finalized blocks that should have digests, and any missing
            or extra digest block IDs. Missing/extra fields are diagnostics for
            the current score-candidate set after recent-tail protection.
        """
        observed_set = set(observed_digest_block_ids)
        score_candidate_set = set(request_context.score_candidate_block_ids)
        missing_digest_blocks = (
            sorted(score_candidate_set - observed_set)
            if request_context.has_block_table_context
            else []
        )
        extra_digest_blocks = (
            sorted(observed_set - score_candidate_set)
            if request_context.has_block_table_context
            else []
        )

        return {
            "num_reqs": request_context.num_reqs,
            "max_query_len": request_context.max_query_len,
            "num_actual_tokens": request_context.num_actual_tokens,
            "seq_lens": request_context.seq_lens,
            "block_size": request_context.block_size,
            "block_table_shape": request_context.block_table_shape,
            "block_table_row": request_context.block_table_row,
            "valid_block_ids": request_context.valid_block_ids,
            "finalized_block_ids": request_context.finalized_block_ids,
            "recent_tokens": request_context.recent_tokens,
            "protected_tail_entries": request_context.protected_tail_entries,
            "protected_block_ids": request_context.protected_block_ids,
            "score_candidate_block_ids": request_context.score_candidate_block_ids,
            "observed_digest_block_ids": list(observed_digest_block_ids),
            "missing_digest_blocks": missing_digest_blocks,
            "extra_digest_blocks": extra_digest_blocks,
        }

    def _request_block_context(
        self,
        *,
        attn_metadata: Any,
        block_size: int | None,
    ) -> RequestBlockContext:
        """Parse the single-request block table and select score candidates."""
        seq_lens = self._tensor_to_int_list(getattr(attn_metadata, "seq_lens", None))
        block_table = getattr(attn_metadata, "block_table", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_table_tensor", None)
        block_table_shape = self._shape_of(block_table)
        block_table_row: list[int] = []
        if block_table is not None and getattr(block_table, "ndim", 0) >= 2:
            row_values = self._tensor_to_int_list(block_table[0])
            block_table_row = [] if row_values is None else row_values
        elif block_table is not None:
            row_values = self._tensor_to_int_list(block_table)
            block_table_row = [] if row_values is None else row_values

        inferred_num_reqs = len(seq_lens) if seq_lens is not None else None
        num_reqs = getattr(attn_metadata, "num_reqs", None)
        if num_reqs is None:
            num_reqs = inferred_num_reqs

        seq_len = seq_lens[0] if seq_lens else None
        valid_block_ids: list[int] = []
        finalized_block_ids: list[int] = []
        protected_tail_entries = 0
        protected_block_ids: list[int] = []
        score_candidate_block_ids: list[int] = []
        has_block_table_context = (
            block_size is not None
            and seq_len is not None
            and block_size > 0
            and bool(block_table_row)
        )
        if has_block_table_context:
            valid_block_count = (seq_len + block_size - 1) // block_size
            finalized_block_count = seq_len // block_size
            valid_block_ids = block_table_row[:valid_block_count]
            finalized_block_ids = block_table_row[:finalized_block_count]
            if self.config.recent_tokens > 0:
                protected_tail_entries = min(
                    len(valid_block_ids),
                    math.ceil(self.config.recent_tokens / block_size),
                )
                protected_block_ids = valid_block_ids[-protected_tail_entries:]
            protected_set = set(protected_block_ids)
            score_candidate_block_ids = [
                block_id
                for block_id in finalized_block_ids
                if block_id not in protected_set
            ]

        return RequestBlockContext(
            num_reqs=num_reqs,
            max_query_len=getattr(attn_metadata, "max_query_len", None),
            num_actual_tokens=getattr(attn_metadata, "num_actual_tokens", None),
            seq_lens=seq_lens,
            block_size=block_size,
            block_table_shape=block_table_shape,
            block_table_row=block_table_row,
            valid_block_ids=valid_block_ids,
            finalized_block_ids=finalized_block_ids,
            recent_tokens=self.config.recent_tokens,
            protected_tail_entries=protected_tail_entries,
            protected_block_ids=protected_block_ids,
            score_candidate_block_ids=score_candidate_block_ids,
            has_block_table_context=has_block_table_context,
        )

    def _block_size_for_layer(self, layer_name: str) -> int | None:
        """Return the cached block size for a layer, if any digest exists."""
        layer_digests = self._digest_cache.get(layer_name, {})
        for digest in layer_digests.values():
            return int(digest.block_size)
        return None

    def _block_size_for_layer_blocks(
        self,
        layer_name: str,
        physical_block_ids: list[int],
    ) -> int | None:
        """Return the cached block size for scored layer blocks, if present."""
        layer_digests = self._digest_cache.get(layer_name, {})
        for block_id in physical_block_ids:
            digest = layer_digests.get(block_id)
            if digest is not None:
                return int(digest.block_size)
        return None

    @staticmethod
    def _topk_block_scores(
        *,
        scores: torch.Tensor,
        physical_block_ids: list[int],
        topk: int,
    ) -> tuple[list[int], list[float]]:
        """Map a block score vector's top-k indices to physical block IDs."""
        if topk <= 0:
            return [], []
        topk_scores, topk_indices = torch.topk(scores, k=topk)
        topk_index_values = topk_indices.detach().cpu().tolist()
        topk_block_ids = [physical_block_ids[int(index)] for index in topk_index_values]
        topk_score_values = [
            float(score)
            for score in topk_scores.detach().to(torch.float32).cpu().tolist()
        ]
        return topk_block_ids, topk_score_values

    @classmethod
    def _topk_head_scores(
        cls,
        *,
        scores_by_block_head: torch.Tensor,
        physical_block_ids: list[int],
        topk: int,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Map per-head score columns to head-local top-k physical block IDs."""
        if scores_by_block_head.ndim != 2:
            raise ValueError(
                "MPR head top-k expects scores_by_block_head shaped "
                f"[num_blocks, num_heads], got {tuple(scores_by_block_head.shape)}."
            )
        num_heads = int(scores_by_block_head.shape[1])
        if topk <= 0:
            return [[] for _ in range(num_heads)], [[] for _ in range(num_heads)]

        topk_block_ids_by_head: list[list[int]] = []
        topk_scores_by_head: list[list[float]] = []
        for head_idx in range(num_heads):
            head_block_ids, head_scores = cls._topk_block_scores(
                scores=scores_by_block_head[:, head_idx],
                physical_block_ids=physical_block_ids,
                topk=topk,
            )
            topk_block_ids_by_head.append(head_block_ids)
            topk_scores_by_head.append(head_scores)
        return topk_block_ids_by_head, topk_scores_by_head

    def snapshot_stats(self) -> dict[str, int]:
        """Return a copy of sidecar event counters for smoke tests."""
        return dict(self.counters)

    def close(self) -> None:
        """Close any debug writer resources owned by this sidecar."""
        self._debug_writer.close()

    def _should_record_layer_event(
        self,
        layer_name: str,
        event_counts: Counter[str],
    ) -> tuple[bool, int | None]:
        """Update per-layer debug counters and decide whether to dump JSONL.

        Args:
            layer_name: vLLM attention layer name.
            event_counts: Counter keyed by layer name for the event family being
                limited.

        Returns:
            A tuple ``(should_record, layer_event_idx)``. ``should_record``
            controls only JSONL/debug output; sidecar state updates should still
            proceed when it is false. ``layer_event_idx`` is the 1-based count
            for this layer, or ``None`` if the layer is outside
            ``VLLM_MPR_MAX_LAYERS``.
        """
        if layer_name not in self._layer_indices:
            if (
                self.config.max_layers is not None
                and len(self._layer_indices) >= self.config.max_layers
            ):
                self.counters["skipped_layer_limit"] += 1
                return False, None
            self._layer_indices[layer_name] = len(self._layer_indices)

        event_counts[layer_name] += 1
        layer_event_idx = event_counts[layer_name]
        if (
            self.config.max_steps is not None
            and layer_event_idx > self.config.max_steps
        ):
            self.counters["skipped_step_limit"] += 1
            return False, layer_event_idx
        if (layer_event_idx - 1) % self.config.dump_every != 0:
            self.counters["skipped_dump_every"] += 1
            return False, layer_event_idx
        return True, layer_event_idx

    @staticmethod
    def _shape_of(value: Any) -> list[int] | None:
        """Return ``value.shape`` as JSON-serializable ints when present."""
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    def _observe_block_offsets(
        self,
        layer_name: str,
        block_ids: Any,
        block_offsets: Any,
        block_size: int,
        kv_cache: Any,
        layer_event_idx: int | None,
    ) -> list[int]:
        """Track written block offsets and digest blocks that become full.

        Args:
            layer_name: vLLM attention layer name.
            block_ids: Physical KV block id for each valid slot, shaped
                ``[num_valid_slots]``.
            block_offsets: Token offset within each physical block, shaped
                ``[num_valid_slots]``.
            block_size: Number of token slots per physical KV block.
            kv_cache: FlashAttention KV cache after the write, shaped
                ``[2, num_blocks, block_size, num_kv_heads, head_dim]``.
            layer_event_idx: Per-layer KV-write event index for debug output,
                or ``None`` when the layer is outside the debug dump window.

        Returns:
            Physical block ids whose digest was created by this observation.
        """
        if kv_cache is None:
            raise ValueError(f"MPR requires kv_cache for digesting {layer_name}.")
        if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
            raise ValueError(
                "MPR digest cache currently expects FlashAttention KV cache "
                "layout [2, num_blocks, block_size, num_kv_heads, head_dim], "
                f"got {tuple(kv_cache.shape)} for {layer_name}."
            )
        if int(kv_cache.shape[2]) != block_size:
            raise AssertionError(
                f"MPR block_size mismatch for {layer_name}: "
                f"block_size={block_size}, kv_cache.shape[2]={int(kv_cache.shape[2])}."
            )

        # key_cache: FlashAttention key plane.
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim].
        key_cache = kv_cache[0]

        # layer_offsets: observed token offsets by physical block id.
        # Structure: dict[physical_block_id, set[block_offset]].
        layer_offsets = self._block_offsets.setdefault(layer_name, {})

        # layer_digests: cached digest entries by physical block id.
        # Each digest_min/digest_max has shape [num_kv_heads, head_dim].
        layer_digests = self._digest_cache.setdefault(layer_name, {})
        created_block_ids: list[int] = []

        # block_id_values/block_offset_values are CPU scalar lists used only for
        # Python-side bookkeeping. They preserve the [num_valid_slots] pairing.
        block_id_values = block_ids.detach().cpu().tolist()
        block_offset_values = block_offsets.detach().cpu().tolist()
        for block_id_raw, block_offset_raw in zip(block_id_values, block_offset_values):
            # block_id: physical KV block index into key_cache dim 0.
            block_id = int(block_id_raw)

            # block_offset: token slot index within key_cache[block_id] dim 0.
            block_offset = int(block_offset_raw)
            if block_id < 0 or block_id >= int(key_cache.shape[0]):
                raise AssertionError(
                    f"MPR observed block id outside KV cache for {layer_name}: "
                    f"block_id={block_id}, num_blocks={int(key_cache.shape[0])}."
                )
            if block_offset < 0 or block_offset >= block_size:
                raise AssertionError(
                    f"MPR observed block offset outside block for {layer_name}: "
                    f"block_offset={block_offset}, block_size={block_size}."
                )

            offsets = layer_offsets.setdefault(block_id, set())
            offsets.add(block_offset)
            if len(offsets) == block_size and block_id not in layer_digests:
                # key_cache[block_id]: one full key block.
                # Shape: [block_size, num_kv_heads, head_dim].
                digest = summarize_key_block(
                    key_cache[block_id],
                    digest_kind=self.config.digest_kind,
                )
                layer_digests[block_id] = self._to_block_digest(
                    digest,
                    layer_event_idx,
                )
                created_block_ids.append(block_id)
                if layer_name in self._layer_indices:
                    self._record(
                        "digest_created",
                        layer_name=layer_name,
                        layer_event_idx=layer_event_idx,
                        physical_block_id=block_id,
                        digest_min_shape=self._shape_of(digest.digest_min),
                        digest_max_shape=self._shape_of(digest.digest_max),
                        valid_token_count=digest.valid_token_count,
                        block_size=digest.block_size,
                        digest_kind=digest.digest_kind,
                        num_digest_blocks_for_layer=len(layer_digests),
                        total_digest_blocks=self._num_digest_blocks(),
                    )

        return created_block_ids

    def _pack_layer_digests(
        self,
        layer_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        candidate_block_ids: list[int] | None = None,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Pack cached layer digests into tensor inputs for scoring.

        Args:
            layer_name: vLLM attention layer name.
            device: Target device for the packed digest tensors.
            dtype: Target dtype for the packed digest tensors.
            candidate_block_ids: Optional current-request block ids to score, in
                logical order. Missing digests are skipped and reported through
                score debug metadata.

        Returns:
            A tuple ``(physical_block_ids, digest_min, digest_max)`` where the
            digest tensors are shaped ``[num_blocks, num_kv_heads, head_dim]``.
        """
        layer_digests = self._digest_cache.get(layer_name, {})
        if not layer_digests:
            empty = torch.empty(0, device=device, dtype=dtype)
            return [], empty, empty

        if candidate_block_ids is None:
            physical_block_ids = sorted(layer_digests)
        else:
            physical_block_ids = [
                block_id
                for block_id in candidate_block_ids
                if block_id in layer_digests
            ]
            if not physical_block_ids:
                empty = torch.empty(0, device=device, dtype=dtype)
                return [], empty, empty

        digest_min = torch.stack(
            [
                layer_digests[block_id].digest_min.to(device=device, dtype=dtype)
                for block_id in physical_block_ids
            ],
            dim=0,
        )
        digest_max = torch.stack(
            [
                layer_digests[block_id].digest_max.to(device=device, dtype=dtype)
                for block_id in physical_block_ids
            ],
            dim=0,
        )
        return physical_block_ids, digest_min, digest_max

    @staticmethod
    def _to_block_digest(
        digest: KeyBlockDigest,
        layer_event_idx: int | None,
    ) -> BlockDigest:
        """Convert a helper digest into the sidecar cache entry type.

        Args:
            digest: Digest returned by ``summarize_key_block``. Its min/max
                tensors are shaped ``[num_kv_heads, head_dim]``.
            layer_event_idx: Per-layer KV-write event index, or ``None`` when
                the digest was created outside the debug dump window.

        Returns:
            A ``BlockDigest`` stored in ``_digest_cache``.
        """
        return BlockDigest(
            digest_min=digest.digest_min,
            digest_max=digest.digest_max,
            valid_token_count=digest.valid_token_count,
            block_size=digest.block_size,
            layer_event_idx=-1 if layer_event_idx is None else layer_event_idx,
            digest_kind=digest.digest_kind,
        )

    def _num_digest_blocks(self) -> int:
        """Return the total number of cached block digests across layers."""
        return sum(len(layer_digests) for layer_digests in self._digest_cache.values())

    @staticmethod
    def _tensor_to_int_list(value: Any) -> list[int] | None:
        """Convert a tensor-like value to a flat Python int list for JSONL."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "reshape") and hasattr(value, "tolist"):
            return [int(item) for item in value.reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        return None

    def _record(self, event: str, **fields: Any) -> None:
        """Increment an event counter and append one JSONL debug record."""
        self.counters[event] += 1
        self._debug_writer.write(
            {
                "event": event,
                "counters": dict(self.counters),
                **fields,
            }
        )


_GLOBAL_SIDECAR: RecoverySidecar | None = None
_GLOBAL_LOCK = threading.Lock()


def get_mpr_sidecar() -> RecoverySidecar:
    global _GLOBAL_SIDECAR
    if _GLOBAL_SIDECAR is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_SIDECAR is None:
                _GLOBAL_SIDECAR = RecoverySidecar()
    return _GLOBAL_SIDECAR


def reset_mpr_sidecar() -> None:
    global _GLOBAL_SIDECAR
    with _GLOBAL_LOCK:
        if _GLOBAL_SIDECAR is not None:
            _GLOBAL_SIDECAR.close()
        _GLOBAL_SIDECAR = None
