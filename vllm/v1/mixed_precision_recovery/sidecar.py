# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Disabled-by-default MPR sidecar scaffold."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.cpu_backup import (
    CPUBackupStore,
    CPUBackupStats,
    SemanticCPUBackupStore,
)
from vllm.v1.mixed_precision_recovery.debug import MPRDebugWriter
from vllm.v1.mixed_precision_recovery.digest import (
    KeyBlockDigest,
    summarize_key_block,
)
from vllm.v1.mixed_precision_recovery.quest_packing import (
    PackedQuestDigestCache,
    QuestMetadataStore,
)
from vllm.v1.mixed_precision_recovery.precision_policy import (
    PrecisionPolicy,
    ThresholdPrecisionPolicy,
    TierAssignment,
    TopRatioPrecisionPolicy,
)
from vllm.v1.mixed_precision_recovery.recovery import (
    BlockRecoveryManager,
    select_recovery_block_ids,
)
from vllm.v1.mixed_precision_recovery.recovery_payload import (
    EagerRecoveryPayloadProvider,
    RecoveryPayloadProvider,
    TieredRecoveryPayloads,
)
from vllm.v1.mixed_precision_recovery.scoring import (
    DigestScoreResult,
    DigestScoringBackend,
    PACKED_ESTIMATE_PROFILE_TIMING_NAMES,
    get_digest_scoring_backend,
    get_packed_estimate_profile_timing,
    reset_packed_estimate_profile_timing,
)

logger = init_logger(__name__)

_BOUNDARY_PROFILE_TIMING_NAMES = (
    "counter_prepare",
    "counter_observe",
    "counter_key_cache",
    "counter_block_lookup",
    "counter_create_digest",
    "counter_summarize_key_block",
    "counter_to_block_digest",
    "counter_append_quest_metadata",
    "counter_backup",
)

_SCORING_PROFILE_TIMING_NAMES = (
    "scoring_estimate_query_scores",
    "scoring_record_estimated",
    "scoring_should_record",
    "scoring_query_clone",
    "scoring_window_stack_mean",
    "scoring_block_size",
    "scoring_request_block_context",
    "scoring_request_ctx_seq_lens",
    "scoring_request_ctx_block_table_lookup",
    "scoring_request_ctx_block_table_row",
    "scoring_request_ctx_candidates",
    "scoring_select_digest_blocks",
    "scoring_quest_packed_prefix",
    "scoring_quest_packed_estimate",
    "scoring_pack_layer_digests",
    "scoring_backend_estimate",
    "scoring_score_packing_debug",
    "scoring_block_topk",
    "scoring_head_topk_debug",
    "scoring_block_debug_fields",
    "scoring_context_build",
    *PACKED_ESTIMATE_PROFILE_TIMING_NAMES,
)


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
class QueryWindowState:
    """Rolling query-window state for one layer."""

    entries: deque[torch.Tensor]
    running_sum: torch.Tensor | None = None


@dataclass(frozen=True)
class QueryScoreContext:
    """Internal score result plus debug metadata for query-time MPR paths."""

    layer_name: str
    should_record: bool
    layer_event_idx: int | None
    query: Any
    window_query: torch.Tensor
    window_query_len: int
    request_context: RequestBlockContext
    physical_block_ids: list[int]
    score_result: DigestScoreResult
    topk: int
    topk_block_ids: list[int]
    topk_scores: list[float]
    score_packing_debug: dict[str, Any]
    head_debug: dict[str, Any]
    score_block_debug: dict[str, Any]


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
    _quest_metadata_stores: dict[str, QuestMetadataStore] = field(
        default_factory=dict
    )
    # Step 1.4 v0 uses layer_name-only query windows for single-request smoke.
    # Serving/multi-request support needs request-scoped keys to avoid mixing
    # decode queries from different requests in the same layer buffer.
    _query_windows: dict[str, QueryWindowState] = field(default_factory=dict)
    _cpu_backup_store: CPUBackupStore = field(
        default_factory=SemanticCPUBackupStore
    )
    _debug_writer: MPRDebugWriter = field(init=False)
    _scoring_backend: DigestScoringBackend = field(init=False)
    _recovery_manager: BlockRecoveryManager = field(
        default_factory=BlockRecoveryManager
    )
    _recovery_payload_provider: RecoveryPayloadProvider = field(
        default_factory=EagerRecoveryPayloadProvider
    )
    _observe_kv_write_timing: dict[str, float] = field(
        default_factory=lambda: {
            "count": 0.0,
            "pre_observe_block_offsets_total_ms": 0.0,
            "pre_observe_block_offsets_max_ms": 0.0,
            "observe_block_offsets_count": 0.0,
            "observe_block_offsets_total_ms": 0.0,
            "observe_block_offsets_max_ms": 0.0,
            **{
                f"{name}_{suffix}": 0.0
                for name in _BOUNDARY_PROFILE_TIMING_NAMES
                for suffix in ("count", "total_ms", "max_ms")
            },
        }
    )
    _scoring_profile_timing: dict[str, float] = field(
        default_factory=lambda: {
            f"{name}_{suffix}": 0.0
            for name in _SCORING_PROFILE_TIMING_NAMES
            for suffix in ("count", "total_ms", "max_ms")
        }
    )
    _counter_leader_layer_name: str | None = None
    _counter_seq_lens_by_row: list[int] = field(default_factory=list)
    _counter_seq_lens_initialized: bool = False
    _request_block_context_cache_key: tuple[Any, ...] | None = None
    _request_block_context_cache_value: RequestBlockContext | None = None

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
                "digest_kind=%s, score_granularity=%s, cpu_backup=%s, "
                "scoring_enabled=%s, enable_logging=%s, debug_dir=%s",
                self.config.topk,
                self.config.window_size,
                self.config.recent_tokens,
                self.config.score_agg,
                self.config.scoring_backend,
                self.config.digest_kind,
                self.config.score_granularity,
                self.config.cpu_backup_enabled,
                self.config.scoring_enabled,
                self.config.enable_logging,
                self.config.debug_dir,
            )
            if self.config.enable_logging:
                self._record("init")

    def enabled(self) -> bool:
        """Return whether MPR sidecar observation is enabled."""
        return self.config.enabled

    def cpu_backup_stats(self) -> CPUBackupStats:
        """Return point-in-time CPU backup stats."""
        return self._cpu_backup_store.stats()

    def reset_observe_kv_write_timing(self) -> None:
        timing = self._observe_kv_write_timing
        for key in timing:
            timing[key] = 0.0

    def observe_kv_write_timing(self) -> dict[str, float | int]:
        timing = self._observe_kv_write_timing
        count = int(timing["count"])
        pre_total_ms = timing["pre_observe_block_offsets_total_ms"]
        offsets_count = int(timing["observe_block_offsets_count"])
        offsets_total_ms = timing["observe_block_offsets_total_ms"]
        return {
            "count": count,
            "pre_observe_block_offsets_total_ms": pre_total_ms,
            "pre_observe_block_offsets_mean_ms": (
                pre_total_ms / count if count > 0 else 0.0
            ),
            "pre_observe_block_offsets_max_ms": timing[
                "pre_observe_block_offsets_max_ms"
            ],
            "observe_block_offsets_count": offsets_count,
            "observe_block_offsets_total_ms": offsets_total_ms,
            "observe_block_offsets_mean_ms": (
                offsets_total_ms / offsets_count if offsets_count > 0 else 0.0
            ),
            "observe_block_offsets_max_ms": timing[
                "observe_block_offsets_max_ms"
            ],
            **self._boundary_profile_timing_snapshot(),
        }

    def _boundary_profile_timing_snapshot(self) -> dict[str, float | int]:
        timing = self._observe_kv_write_timing
        result: dict[str, float | int] = {}
        for name in _BOUNDARY_PROFILE_TIMING_NAMES:
            count = int(timing.get(f"{name}_count", 0.0))
            total_ms = timing.get(f"{name}_total_ms", 0.0)
            result[f"{name}_count"] = count
            result[f"{name}_total_ms"] = total_ms
            result[f"{name}_mean_ms"] = total_ms / count if count > 0 else 0.0
            result[f"{name}_max_ms"] = timing.get(f"{name}_max_ms", 0.0)
        return result

    def _record_boundary_profile_timing(
        self,
        name: str,
        start: float,
    ) -> None:
        if not self.config.boundary_profile_enabled:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timing = self._observe_kv_write_timing
        count_key = f"{name}_count"
        total_key = f"{name}_total_ms"
        max_key = f"{name}_max_ms"
        timing[count_key] = timing.get(count_key, 0.0) + 1.0
        timing[total_key] = timing.get(total_key, 0.0) + elapsed_ms
        timing[max_key] = max(timing.get(max_key, 0.0), elapsed_ms)

    def reset_scoring_profile_timing(self) -> None:
        timing = self._scoring_profile_timing
        for key in timing:
            timing[key] = 0.0
        reset_packed_estimate_profile_timing()

    def scoring_profile_timing(self) -> dict[str, float | int]:
        timing = self._scoring_profile_timing
        result: dict[str, float | int] = {}
        for name in _SCORING_PROFILE_TIMING_NAMES:
            count = int(timing.get(f"{name}_count", 0.0))
            total_ms = timing.get(f"{name}_total_ms", 0.0)
            result[f"{name}_count"] = count
            result[f"{name}_total_ms"] = total_ms
            result[f"{name}_mean_ms"] = total_ms / count if count > 0 else 0.0
            result[f"{name}_max_ms"] = timing.get(f"{name}_max_ms", 0.0)
        result.update(get_packed_estimate_profile_timing())
        return result

    def _record_scoring_profile_timing(
        self,
        name: str,
        start: float,
    ) -> None:
        if not self.config.scoring_profile_enabled:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timing = self._scoring_profile_timing
        count_key = f"{name}_count"
        total_key = f"{name}_total_ms"
        max_key = f"{name}_max_ms"
        timing[count_key] = timing.get(count_key, 0.0) + 1.0
        timing[total_key] = timing.get(total_key, 0.0) + elapsed_ms
        timing[max_key] = max(timing.get(max_key, 0.0), elapsed_ms)

    def release_blocks(
        self,
        block_ids: Any,
        *,
        reason: str = "free",
        request_id: str | None = None,
    ) -> None:
        """Release MPR state keyed by GPU KV physical block id.

        This is a Milestone 2 lifecycle cleanup entry point. The caller passes
        vLLM GPU KV physical block ids, not CPU backup block ids. The CPU
        backup store maps those external ids to any internal storage handle.
        This method does not free vLLM GPU KV blocks; vLLM's BlockPool still
        owns that lifecycle. It only drops MPR metadata and CPU backups that
        would become stale once those GPU block ids are reusable.
        """
        if not self.config.enabled:
            return
        physical_block_ids = self._normalize_physical_block_ids(block_ids)
        if not physical_block_ids:
            # Defensive no-op path for callers that report a release event
            # without any concrete GPU physical block ids.
            self._record(
                "blocks_released",
                request_id=request_id,
                release_reason=reason,
                released_block_ids=[],
                released_offset_entries=0,
                released_digest_entries=0,
                released_cpu_backup_entries=0,
                released_cpu_backup_bytes=0,
                invalidated_quest_metadata_stores=0,
            )
            return

        block_id_set = set(physical_block_ids)

        # First drop write-completion bookkeeping. These offset sets are used
        # only to decide when a physical block has become full enough to digest.
        released_offset_entries = 0
        for layer_name in list(self._block_offsets):
            layer_offsets = self._block_offsets[layer_name]
            for block_id in block_id_set:
                if layer_offsets.pop(block_id, None) is not None:
                    released_offset_entries += 1
            if not layer_offsets:
                del self._block_offsets[layer_name]

        # Then drop digest entries keyed by the same GPU physical block ids.
        # QuestMetadataStore is append-only in the current fast path, so when
        # a digest is removed we invalidate the whole layer store rather than
        # risk scoring with stale packed metadata.
        released_digest_entries = 0
        invalidated_quest_metadata_stores = 0
        for layer_name in list(self._digest_cache):
            layer_digests = self._digest_cache[layer_name]
            removed_for_layer = 0
            for block_id in block_id_set:
                if layer_digests.pop(block_id, None) is not None:
                    released_digest_entries += 1
                    removed_for_layer += 1
            if removed_for_layer and layer_name in self._quest_metadata_stores:
                # QuestMetadataStore is append-only in M1. Drop the persistent
                # packed store on release to avoid stale fast-path metadata.
                # A cleaner production path should use generation-aware keys
                # or a removable metadata store.
                del self._quest_metadata_stores[layer_name]
                invalidated_quest_metadata_stores += 1
            if not layer_digests:
                del self._digest_cache[layer_name]

        # Finally release sidecar-owned CPU backup entries. The backup store
        # maps these external GPU block ids to whatever internal CPU storage
        # representation it owns.
        release_result = self._cpu_backup_store.release_blocks(block_id_set)
        backup_stats = self._cpu_backup_store.stats()
        self._record(
            "blocks_released",
            request_id=request_id,
            release_reason=reason,
            released_block_ids=physical_block_ids,
            released_offset_entries=released_offset_entries,
            released_digest_entries=released_digest_entries,
            released_cpu_backup_entries=release_result.released_entries,
            released_cpu_backup_bytes=release_result.released_bytes,
            released_cpu_backup_fp16_payload_bytes=(
                release_result.fp16_payload_bytes
            ),
            released_cpu_backup_int8_payload_bytes=(
                release_result.int8_payload_bytes
            ),
            released_cpu_backup_int8_scale_bytes=(
                release_result.int8_scale_bytes
            ),
            released_cpu_backup_int4_payload_bytes=(
                release_result.int4_payload_bytes
            ),
            released_cpu_backup_int4_scale_bytes=(
                release_result.int4_scale_bytes
            ),
            released_cpu_backup_total_actual_bytes=(
                release_result.total_actual_backup_bytes
            ),
            cpu_backup_block_count=backup_stats.block_count,
            cpu_backup_bytes=backup_stats.total_bytes,
            cpu_backup_fp16_payload_bytes=backup_stats.fp16_payload_bytes,
            cpu_backup_int8_payload_bytes=backup_stats.int8_payload_bytes,
            cpu_backup_int8_scale_bytes=backup_stats.int8_scale_bytes,
            cpu_backup_int4_payload_bytes=backup_stats.int4_payload_bytes,
            cpu_backup_int4_scale_bytes=backup_stats.int4_scale_bytes,
            cpu_backup_total_actual_bytes=(
                backup_stats.total_actual_backup_bytes
            ),
            invalidated_quest_metadata_stores=invalidated_quest_metadata_stores,
        )

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
        observe_start = time.perf_counter()
        pre_timing_recorded = False

        def record_pre_observe_block_offsets_time() -> None:
            nonlocal pre_timing_recorded
            if pre_timing_recorded:
                return
            pre_timing_recorded = True
            elapsed_ms = (time.perf_counter() - observe_start) * 1000.0
            timing = self._observe_kv_write_timing
            timing["count"] += 1.0
            timing["pre_observe_block_offsets_total_ms"] += elapsed_ms
            timing["pre_observe_block_offsets_max_ms"] = max(
                timing["pre_observe_block_offsets_max_ms"],
                elapsed_ms,
            )

        should_record = False
        layer_event_idx = None
        if self.config.enable_logging:
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
            num_pad_slots = 0
            digest_created_block_ids: list[int] = []
            if should_record:
                unique_block_ids: list[int] = []
                min_block_offset = None
                max_block_offset = None
            record_pre_observe_block_offsets_time()
        else:
            # has_invalid_negative_slot = bool((flat_slots < PAD_SLOT_ID).any().item())
            # if has_invalid_negative_slot:
            #     min_slot = int(flat_slots.min().item())
            #     raise AssertionError(
            #         f"MPR observed invalid negative slot id for {layer_name}: "
            #         f"min_slot={min_slot}."
            #     )

            # vLLM pads unused CUDA graph slots with PAD_SLOT_ID. These are not
            # KV writes and should not participate in block/offset summaries.
            # valid_slots: physical slot ids that correspond to real KV writes.
            # Shape: [num_valid_slots].
            valid_slots = flat_slots[flat_slots != PAD_SLOT_ID]
            num_pad_slots = num_slots - int(valid_slots.numel())
            if valid_slots.numel() == 0:
                digest_created_block_ids = []
                if should_record:
                    unique_block_ids = []
                    min_block_offset = None
                    max_block_offset = None
                record_pre_observe_block_offsets_time()
            else:
                # block_ids: physical KV block id for each valid slot.
                # Shape: [num_valid_slots].
                block_ids = valid_slots // block_size

                # block_offsets: token offset within the physical block.
                # Shape: [num_valid_slots].
                block_offsets = valid_slots % block_size
                if should_record:
                    if not self.config.enable_logging:
                        raise Exception("Trying to record even though logging is disabled")
                    unique_block_ids = [
                        int(block_id)
                        for block_id in sorted(
                            block_ids.unique().detach().cpu().tolist()
                        )
                    ]
                    min_block_offset = int(block_offsets.min().item())
                    max_block_offset = int(block_offsets.max().item())
                record_pre_observe_block_offsets_time()
                observe_offsets_start = time.perf_counter()
                try:
                    digest_created_block_ids = self._observe_block_offsets(
                        layer_name=layer_name,
                        block_ids=block_ids,
                        block_offsets=block_offsets,
                        block_size=block_size,
                        kv_cache=kv_cache,
                        layer_event_idx=layer_event_idx,
                    )
                finally:
                    elapsed_ms = (
                        time.perf_counter() - observe_offsets_start
                    ) * 1000.0
                    timing = self._observe_kv_write_timing
                    timing["observe_block_offsets_count"] += 1.0
                    timing["observe_block_offsets_total_ms"] += elapsed_ms
                    timing["observe_block_offsets_max_ms"] = max(
                        timing["observe_block_offsets_max_ms"],
                        elapsed_ms,
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

    def prepare_counter_kv_write(
        self,
        *,
        layer_name: str,
        attn_metadata: Any,
    ) -> tuple[bool, list[int] | None]:
        """Advance shared sequence counters and decide counter observation.

        Only the first observed layer advances sequence lengths. Other layers
        consume the leader's latest snapshot so boundary detection is shared
        while digest creation remains layer-local.
        """
        profile_enabled = self.config.boundary_profile_enabled
        profile_start = time.perf_counter() if profile_enabled else 0.0
        try:
            self.counters["counter_observe_candidate_count"] += 1
            if attn_metadata is None:
                self.counters["counter_observe_missing_attn_metadata"] += 1
                return False, None
            block_table_start = time.perf_counter() if profile_enabled else 0.0
            has_block_table = self._counter_block_table(attn_metadata) is not None
            if profile_enabled:
                self._record_boundary_profile_timing(
                    "counter_block_lookup",
                    block_table_start,
                )
            if not has_block_table:
                self.counters["counter_observe_missing_block_table"] += 1
                return False, None
            query_len = self._counter_single_request_query_len(attn_metadata)
            if query_len is None:
                self.counters["counter_observe_unsupported_num_reqs"] += 1
                self._counter_seq_lens_initialized = False
                self._counter_seq_lens_by_row = []
                return False, None
            try:
                if query_len <= 0:
                    self.counters["counter_observe_bad_query_len"] += 1
                    return False, None
                if self._counter_leader_layer_name is None:
                    self._counter_leader_layer_name = layer_name
                    self.counters["counter_observe_leader_selected"] += 1
                if layer_name == self._counter_leader_layer_name:
                    if query_len == 1:
                        if not self._counter_seq_lens_initialized:
                            self.counters["counter_observe_uninitialized"] += 1
                            return False, None
                        self._counter_seq_lens_by_row[0] += 1
                        self.counters["counter_observe_decode_advanced"] += 1
                    else:
                        self._counter_seq_lens_by_row = [query_len]
                        self._counter_seq_lens_initialized = True
                        self.counters["counter_observe_prefill_initialized"] += 1
                if not self._counter_seq_lens_initialized:
                    self.counters["counter_observe_uninitialized"] += 1
                    return False, None
                seq_lens = list(self._counter_seq_lens_by_row)
                if query_len != 1:
                    self.counters["counter_observe_not_pure_decode"] += 1
                    return False, seq_lens
                self.counters["counter_observe_accepted"] += 1
                return True, seq_lens
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self.counters["counter_observe_bad_query_len"] += 1
                return False, None
        finally:
            if profile_enabled:
                self._record_boundary_profile_timing(
                    "counter_prepare",
                    profile_start,
                )

    def observe_kv_write_by_counter(
        self,
        layer_name: str,
        key: Any = None,
        value: Any = None,
        kv_cache: Any = None,
        attn_metadata: Any = None,
        block_size: int | None = None,
        seq_lens: list[int] | None = None,
    ) -> list[int]:
        """Observe pure-decode KV writes using sequence counters.

        The shared counter is advanced once per step by a leader layer. A block
        becomes full when the post-update sequence length is block-aligned.
        """
        if not self.config.enabled:
            return []
        profile_enabled = self.config.boundary_profile_enabled
        profile_start = time.perf_counter() if profile_enabled else 0.0
        self.counters["counter_observe_used"] += 1
        should_record = False
        layer_event_idx = None
        if self.config.enable_logging:
            should_record, layer_event_idx = self._should_record_layer_event(
                layer_name,
                self._kv_write_counts,
            )
        if block_size is None or block_size <= 0:
            raise ValueError(
                f"MPR requires a positive block_size for {layer_name}, "
                f"got {block_size}."
            )
        if kv_cache is None:
            raise ValueError(f"MPR requires kv_cache for digesting {layer_name}.")

        key_cache_start = time.perf_counter() if profile_enabled else 0.0
        key_cache = self._key_cache_for_digest(
            layer_name=layer_name,
            kv_cache=kv_cache,
            block_size=block_size,
        )
        if profile_enabled:
            self._record_boundary_profile_timing(
                "counter_key_cache",
                key_cache_start,
            )
        block_table_start = time.perf_counter() if profile_enabled else 0.0
        block_table = self._counter_block_table(attn_metadata)
        if profile_enabled:
            self._record_boundary_profile_timing(
                "counter_block_lookup",
                block_table_start,
            )
        if seq_lens is None or block_table is None:
            self.counters["counter_observe_fallback_required"] += 1
            if profile_enabled:
                self._record_boundary_profile_timing(
                    "counter_observe",
                    profile_start,
                )
            return []

        created_block_ids: list[int] = []
        boundary_request_indices: list[int] = []
        for req_idx, seq_len in enumerate(seq_lens):
            if seq_len <= 0 or seq_len % block_size != 0:
                continue
            logical_block_idx = seq_len // block_size - 1
            block_lookup_start = time.perf_counter() if profile_enabled else 0.0
            block_id = self._counter_block_id(
                block_table=block_table,
                req_idx=req_idx,
                logical_block_idx=logical_block_idx,
            )
            if profile_enabled:
                self._record_boundary_profile_timing(
                    "counter_block_lookup",
                    block_lookup_start,
                )
            if block_id is None:
                continue
            boundary_request_indices.append(req_idx)
            self.counters["counter_observe_boundary_requests"] += 1
            digest_start = time.perf_counter() if profile_enabled else 0.0
            if self._create_digest_for_full_block(
                layer_name=layer_name,
                block_id=block_id,
                block_size=block_size,
                key_cache=key_cache,
                kv_cache=kv_cache,
                layer_event_idx=layer_event_idx,
            ):
                created_block_ids.append(block_id)
                self.counters["counter_observe_digest_created"] += 1
            if profile_enabled:
                self._record_boundary_profile_timing(
                    "counter_create_digest",
                    digest_start,
                )
        if not boundary_request_indices:
            self.counters["counter_observe_no_boundary"] += 1

        if should_record:
            self._record(
                "observe_kv_write",
                layer_name=layer_name,
                layer_event_idx=layer_event_idx,
                observe_backend="counter",
                key_shape=self._shape_of(key),
                value_shape=self._shape_of(value),
                kv_cache_shape=self._shape_of(kv_cache),
                block_size=block_size,
                seq_lens=seq_lens,
                boundary_request_indices=boundary_request_indices,
                digest_created_block_ids=created_block_ids,
                num_digest_blocks_for_layer=len(
                    self._digest_cache.get(layer_name, {})
                ),
                total_digest_blocks=self._num_digest_blocks(),
                digest_kind=self.config.digest_kind,
            )
        if profile_enabled:
            self._record_boundary_profile_timing(
                "counter_observe",
                profile_start,
            )
        return created_block_ids

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
        if not self.config.scoring_enabled:
            self.counters["score_skipped_disabled"] += 1
            return
        profile_enabled = self.config.scoring_profile_enabled
        estimate_start = time.perf_counter() if profile_enabled else 0.0
        score_context = self._estimate_query_scores_optimized(
            layer_name=layer_name,
            query=query,
            attn_metadata=attn_metadata,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_estimate_query_scores",
                estimate_start,
            )
        if score_context is None:
            return
        record_start = time.perf_counter() if profile_enabled else 0.0
        self._record_score_estimated(score_context)
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_record_estimated",
                record_start,
            )

    def recover_before_attention(
        self,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
        kv_cache: Any = None,
        block_size: int | None = None,
    ) -> None:
        """Materialize selected CPU-backed KV blocks before attention.

        M3 keeps this as a sidecar-level API first. The attention hook should
        call this instead of ``observe_query`` when recovery is enabled, so the
        decode query is scored once and appended to the query window once.
        """
        self._recover_before_attention_impl(
            layer_name=layer_name,
            query=query,
            attn_metadata=attn_metadata,
            kv_cache=kv_cache,
            block_size=block_size,
            apply_test_mutation=False,
        )

    def recover_before_attention_with_test_mutation(
        self,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
        kv_cache: Any = None,
        block_size: int | None = None,
    ) -> None:
        """M3 validation-only recovery path with configured fault injection."""
        self._recover_before_attention_impl(
            layer_name=layer_name,
            query=query,
            attn_metadata=attn_metadata,
            kv_cache=kv_cache,
            block_size=block_size,
            apply_test_mutation=True,
        )

    def _recover_before_attention_impl(
        self,
        *,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
        kv_cache: Any = None,
        block_size: int | None = None,
        apply_test_mutation: bool,
    ) -> None:
        """Shared recovery implementation for production and test paths."""
        if not self.config.enabled:
            return
        if not self.config.recovery_enabled:
            return
        should_record, layer_event_idx = self._should_record_layer_event(
            layer_name,
            self._score_counts,
        )
        if not self.config.cpu_backup_enabled:
            self._record_recovery_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "cpu_backup_disabled",
                query=query,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
            )
            return
        if not self.config.scoring_enabled:
            self._record_recovery_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "scoring_disabled",
                query=query,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
            )
            return
        if kv_cache is None:
            self._record_recovery_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "missing_kv_cache",
                query=query,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
            )
            return

        profile_enabled = self.config.scoring_profile_enabled
        estimate_start = time.perf_counter() if profile_enabled else 0.0
        score_context = self._estimate_query_scores_optimized(
            layer_name=layer_name,
            query=query,
            attn_metadata=attn_metadata,
            should_record=should_record,
            layer_event_idx=layer_event_idx,
            block_size_override=block_size,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_estimate_query_scores",
                estimate_start,
            )
        if score_context is None:
            self._record_recovery_skip(
                should_record,
                layer_name,
                layer_event_idx,
                "no_score_context",
                query=query,
                attn_metadata=attn_metadata,
                kv_cache=kv_cache,
            )
            return

        tier_assignment: TierAssignment | None = None
        tiered_payloads: TieredRecoveryPayloads | None = None
        if self.config.precision_tiering_enabled:
            tier_assignment = self._assign_precision_tiers(score_context)
            selected_block_ids = tier_assignment.all_block_ids
        else:
            selected_block_ids = select_recovery_block_ids(
                score_result=score_context.score_result,
                physical_block_ids=score_context.physical_block_ids,
                policy=self.config.recovery_policy,
                topk=self.config.recovery_topk,
                threshold=self.config.recovery_threshold,
            )
        test_mutated_block_ids = []
        if apply_test_mutation:
            test_mutated_block_ids = self._maybe_mutate_recovery_targets(
                kv_cache=kv_cache,
                selected_block_ids=selected_block_ids,
            )
            if self.config.recovery_test_mode == "mutate_only":
                self._record_recovery_test_mutated(
                    score_context=score_context,
                    kv_cache=kv_cache,
                    selected_block_ids=selected_block_ids,
                    mutated_block_ids=test_mutated_block_ids,
                )
                return
            if self.config.recovery_test_mode != "recover":
                raise ValueError(
                    "MPR recovery_test_mode must be 'recover' or "
                    f"'mutate_only', got {self.config.recovery_test_mode!r}."
                )
        if self.config.precision_tiering_enabled:
            assert tier_assignment is not None
            tiered_payloads = self._recovery_payload_provider.fetch(
                assignment=tier_assignment,
                cpu_backup_store=self._cpu_backup_store,
                layer_name=layer_name,
            )
            self._raise_on_missing_tier_payloads(tiered_payloads)
            recovery_result = (
                self._recovery_manager.materialize_tiered_payloads(
                    tiered_payloads=tiered_payloads,
                    kv_cache=kv_cache,
                )
            )
        else:
            recovery_result = self._recovery_manager.materialize_blocks(
                selected_block_ids=selected_block_ids,
                kv_cache=kv_cache,
                cpu_backup_store=self._cpu_backup_store,
                layer_name=layer_name,
            )
        backup_stats = self._cpu_backup_store.stats()
        if score_context.should_record:
            self._record(
                "recovery_materialized",
                layer_name=layer_name,
                layer_event_idx=score_context.layer_event_idx,
                query_shape=self._shape_of(query),
                window_query_shape=self._shape_of(score_context.window_query),
                window_query_len=score_context.window_query_len,
                kv_cache_shape=self._shape_of(kv_cache),
                recovery_policy=self.config.recovery_policy,
                recovery_topk=self.config.recovery_topk,
                recovery_threshold=self.config.recovery_threshold,
                recovery_test_mutate=(
                    self.config.recovery_test_mutate
                    if apply_test_mutation
                    else "off"
                ),
                recovery_test_mutated_scope=(
                    self._recovery_test_mutated_scope()
                    if apply_test_mutation
                    else "selected"
                ),
                recovery_test_mode=(
                    self.config.recovery_test_mode
                    if apply_test_mutation
                    else "recover"
                ),
                recovery_test_mutated_block_ids=test_mutated_block_ids,
                recovery_selected_block_ids=recovery_result.selected_block_ids,
                recovered_block_ids=recovery_result.recovered_block_ids,
                missing_backup_block_ids=(
                    recovery_result.missing_backup_block_ids
                ),
                skipped_block_ids=recovery_result.skipped_block_ids,
                recovered_bytes=recovery_result.recovered_bytes,
                recovery_copy_wall_ms=(
                    recovery_result.copy_wall_seconds * 1000.0
                ),
                **self._tiered_recovery_debug_fields(
                    tier_assignment=tier_assignment,
                    recovery_result=recovery_result,
                ),
                cpu_backup_block_count=backup_stats.block_count,
                cpu_backup_bytes=backup_stats.total_bytes,
                cpu_backup_fp16_payload_bytes=backup_stats.fp16_payload_bytes,
                cpu_backup_int8_payload_bytes=backup_stats.int8_payload_bytes,
                cpu_backup_int8_scale_bytes=backup_stats.int8_scale_bytes,
                cpu_backup_int4_payload_bytes=backup_stats.int4_payload_bytes,
                cpu_backup_int4_scale_bytes=backup_stats.int4_scale_bytes,
                cpu_backup_total_actual_bytes=(
                    backup_stats.total_actual_backup_bytes
                ),
                **score_context.score_block_debug,
            )

    def _build_precision_policy(self) -> PrecisionPolicy:
        """Build the configured M4 score-to-tier policy."""
        if self.config.precision_policy == "top_ratio":
            return TopRatioPrecisionPolicy(
                fp16_ratio=self.config.tier_fp16_ratio,
                int8_ratio=self.config.tier_int8_ratio,
                int4_ratio=self.config.tier_int4_ratio,
            )
        if self.config.precision_policy == "threshold":
            return ThresholdPrecisionPolicy(
                high_threshold=self.config.tier_high_threshold,
                mid_threshold=self.config.tier_mid_threshold,
                low_threshold=self.config.tier_low_threshold,
            )
        raise ValueError(
            "MPR precision_policy must be 'top_ratio' or 'threshold', got "
            f"{self.config.precision_policy!r}."
        )

    def _assign_precision_tiers(
        self,
        score_context: QueryScoreContext,
    ) -> TierAssignment:
        """Assign score candidates to precision tiers without rescoring."""
        policy = self._build_precision_policy()
        return policy.assign_tiers(
            block_scores=score_context.score_result.block_scores,
            physical_block_ids=score_context.physical_block_ids,
        )

    def _raise_on_missing_tier_payloads(
        self,
        tiered_payloads: TieredRecoveryPayloads,
    ) -> None:
        """Fail fast when eager tiered recovery cannot fetch a tier payload."""
        if (
            not tiered_payloads.missing_fp16_block_ids
            and not tiered_payloads.missing_int8_block_ids
            and not tiered_payloads.missing_int4_block_ids
        ):
            return
        raise ValueError(
            "MPR tiered recovery expected all selected tier payloads to "
            "exist for the eager provider, got "
            f"missing_fp16_block_ids={tiered_payloads.missing_fp16_block_ids}, "
            f"missing_int8_block_ids={tiered_payloads.missing_int8_block_ids}, "
            f"missing_int4_block_ids={tiered_payloads.missing_int4_block_ids}."
        )

    def _tiered_recovery_debug_fields(
        self,
        *,
        tier_assignment: TierAssignment | None,
        recovery_result: Any,
    ) -> dict[str, Any]:
        """Return M4 tier debug fields only when tiering is active."""
        if tier_assignment is None:
            return {}
        return {
            "precision_tiering_enabled": True,
            "precision_policy": self.config.precision_policy,
            "tier_fp16_block_ids": tier_assignment.fp16_block_ids,
            "tier_int8_block_ids": tier_assignment.int8_block_ids,
            "tier_int4_block_ids": tier_assignment.int4_block_ids,
            "tier_skip_block_ids": tier_assignment.skipped_block_ids,
            "recovered_fp16_block_ids": (
                recovery_result.recovered_fp16_block_ids
            ),
            "recovered_int8_block_ids": (
                recovery_result.recovered_int8_block_ids
            ),
            "recovered_int4_block_ids": (
                recovery_result.recovered_int4_block_ids
            ),
            "missing_fp16_block_ids": recovery_result.missing_fp16_block_ids,
            "missing_int8_block_ids": recovery_result.missing_int8_block_ids,
            "missing_int4_block_ids": recovery_result.missing_int4_block_ids,
            "fp16_payload_bytes": recovery_result.fp16_payload_bytes,
            "int8_payload_bytes": recovery_result.int8_payload_bytes,
            "int4_payload_bytes": recovery_result.int4_payload_bytes,
            "int4_scale_bytes": recovery_result.int4_scale_bytes,
            "effective_recovery_transfer_bytes": (
                recovery_result.effective_recovery_transfer_bytes
            ),
        }

    def _estimate_query_scores(
        self,
        *,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
        should_record: bool | None = None,
        layer_event_idx: int | None = None,
        block_size_override: int | None = None,
    ) -> QueryScoreContext | None:
        """Reference query scoring path retained for parity/debug comparison.

        This original implementation keeps the conservative behavior that was
        used during MPR bring-up: it clones the decode query and calls the
        full-head scoring backend path so ``DigestScoreResult`` always carries
        per-query-head and per-KV-head scores. The production hot path is wired
        to ``_estimate_query_scores_optimized`` below.
        """
        profile_enabled = self.config.scoring_profile_enabled
        if should_record is None:
            should_record_start = time.perf_counter() if profile_enabled else 0.0
            should_record, layer_event_idx = self._should_record_layer_event(
                layer_name,
                self._score_counts,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_should_record",
                    should_record_start,
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
        query_clone_start = time.perf_counter() if profile_enabled else 0.0
        query_for_window = query[0].detach().clone()
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_query_clone",
                query_clone_start,
            )
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
        # window_query: average of the observed decode queries so far, up to
        # VLLM_MPR_WINDOW_SIZE. Shape: [num_q_heads, head_dim].
        # window_start = time.perf_counter() if profile_enabled else 0.0
        # window_query, window_query_len = self._update_query_window(
        #     layer_name,
        #     query_for_window,
        # )
        # if profile_enabled:
        #     self._record_scoring_profile_timing(
        #         "scoring_window_stack_mean",
        #         window_start,
        #     )

        # Experimental current-query scoring path: bypass the rolling query
        # window and score only the active decode step's query.
        window_query = query_for_window
        window_query_len = 1

        block_size_start = time.perf_counter() if profile_enabled else 0.0
        block_size = (
            block_size_override
            if block_size_override is not None
            else self._block_size_for_layer(layer_name)
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_block_size",
                block_size_start,
            )
        request_context_start = time.perf_counter() if profile_enabled else 0.0
        request_context = self._request_block_context(
            attn_metadata=attn_metadata,
            block_size=block_size,
            cache_step=layer_event_idx,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_request_block_context",
                request_context_start,
            )
        candidate_block_ids = (
            request_context.score_candidate_block_ids
            if request_context.has_block_table_context
            else None
        )
        select_digest_start = time.perf_counter() if profile_enabled else 0.0
        physical_block_ids = self._select_layer_digest_block_ids(
            layer_name,
            candidate_block_ids=candidate_block_ids,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_select_digest_blocks",
                select_digest_start,
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
                window_query_len=window_query_len,
            )
            return

        score_result = None
        quest_packed_fast_path = False
        quest_packed_fallback_reason = "backend_not_quest_cuda"
        if self.config.scoring_backend == "quest_cuda" and hasattr(
            self._scoring_backend,
            "estimate_packed",
        ):
            packed_prefix_start = time.perf_counter() if profile_enabled else 0.0
            packed, quest_packed_fallback_reason = (
                self._try_get_quest_packed_prefix(
                    layer_name=layer_name,
                    physical_block_ids=physical_block_ids,
                    block_size=block_size,
                    device=window_query.device,
                    dtype=window_query.dtype,
                )
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_quest_packed_prefix",
                    packed_prefix_start,
                )
            if packed is not None:
                packed_estimate_start = (
                    time.perf_counter() if profile_enabled else 0.0
                )
                score_result = self._scoring_backend.estimate_packed(
                    query_window=window_query,
                    packed=packed,
                    num_kv_heads=int(packed.metadata_data.shape[3]),
                    score_agg=self.config.score_agg,
                    profile_enabled=profile_enabled,
                )
                if profile_enabled:
                    self._record_scoring_profile_timing(
                        "scoring_quest_packed_estimate",
                        packed_estimate_start,
                    )
                quest_packed_fast_path = True
                quest_packed_fallback_reason = None

        if score_result is None:
            pack_start = time.perf_counter() if profile_enabled else 0.0
            physical_block_ids, digest_min, digest_max = self._pack_layer_digests(
                layer_name,
                device=window_query.device,
                dtype=window_query.dtype,
                candidate_block_ids=candidate_block_ids,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_pack_layer_digests",
                    pack_start,
                )
            estimate_start = time.perf_counter() if profile_enabled else 0.0
            score_result = self._scoring_backend.estimate(
                query_window=window_query,
                digest_min=digest_min,
                digest_max=digest_max,
                score_agg=self.config.score_agg,
                metadata_page_size=block_size,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_backend_estimate",
                    estimate_start,
                )
            if self.config.scoring_backend != "quest_cuda":
                quest_packed_fallback_reason = "backend_not_quest_cuda"
            elif quest_packed_fallback_reason is None:
                quest_packed_fallback_reason = "fallback_after_fast_path_miss"

        scores = score_result.block_scores
        topk = min(self.config.topk, int(scores.numel()))
        debug_record_enabled = self.config.enable_logging and bool(should_record)
        score_packing_debug: dict[str, Any] = {}
        topk_block_ids: list[int] = []
        topk_score_values: list[float] = []
        head_debug: dict[str, Any] = {}
        score_block_debug: dict[str, Any] = {}
        if debug_record_enabled:
            score_packing_debug_start = (
                time.perf_counter() if profile_enabled else 0.0
            )
            score_packing_debug = self._score_packing_debug_fields(
                quest_packed_fast_path=quest_packed_fast_path,
                quest_packed_fallback_reason=quest_packed_fallback_reason,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_score_packing_debug",
                    score_packing_debug_start,
                )
            block_topk_start = time.perf_counter() if profile_enabled else 0.0
            topk_block_ids, topk_score_values = self._topk_block_scores(
                scores=scores,
                physical_block_ids=physical_block_ids,
                topk=topk,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_block_topk",
                    block_topk_start,
                )
            head_topk_start = time.perf_counter() if profile_enabled else 0.0
            head_debug = self._head_score_debug_fields(
                score_result=score_result,
                physical_block_ids=physical_block_ids,
                topk=topk,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_head_topk_debug",
                    head_topk_start,
                )
            block_debug_start = time.perf_counter() if profile_enabled else 0.0
            score_block_debug = self._score_block_debug_fields(
                request_context=request_context,
                observed_digest_block_ids=physical_block_ids,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_block_debug_fields",
                    block_debug_start,
                )

        context_start = time.perf_counter() if profile_enabled else 0.0
        score_context = QueryScoreContext(
            layer_name=layer_name,
            should_record=should_record,
            layer_event_idx=layer_event_idx,
            query=query,
            window_query=window_query,
            window_query_len=window_query_len,
            request_context=request_context,
            physical_block_ids=physical_block_ids,
            score_result=score_result,
            topk=topk,
            topk_block_ids=topk_block_ids,
            topk_scores=topk_score_values,
            score_packing_debug=score_packing_debug,
            head_debug=head_debug,
            score_block_debug=score_block_debug,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_context_build",
                context_start,
            )
        return score_context

    def _estimate_query_scores_optimized(
        self,
        *,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
        should_record: bool | None = None,
        layer_event_idx: int | None = None,
        block_size_override: int | None = None,
    ) -> QueryScoreContext | None:
        """Optimized query scoring path for the current MPR hot path.

        This method intentionally sits next to ``_estimate_query_scores`` instead
        of replacing it. It preserves the same skip semantics and context shape,
        but avoids two hot-path costs when debug head-score materialization is
        not needed:

        * use a detached query view instead of cloning the single decode row;
        * call ``QuestCudaScorer.estimate_packed_optimized`` so block scores are
          reduced from the Quest CUDA output layout without transposing the full
          query-head score matrix.

        If logging is enabled and ``score_granularity`` requests head-level
        debug fields, the packed scorer is asked to materialize full head scores
        and falls back to the reference packed path.
        """
        profile_enabled = self.config.scoring_profile_enabled
        if should_record is None:
            should_record_start = time.perf_counter() if profile_enabled else 0.0
            should_record, layer_event_idx = self._should_record_layer_event(
                layer_name,
                self._score_counts,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_should_record",
                    should_record_start,
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

        query_clone_start = time.perf_counter() if profile_enabled else 0.0
        query_for_window = query[0].detach()
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_query_clone",
                query_clone_start,
            )
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

        window_query = query_for_window
        window_query_len = 1

        block_size_start = time.perf_counter() if profile_enabled else 0.0
        block_size = (
            block_size_override
            if block_size_override is not None
            else self._block_size_for_layer(layer_name)
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_block_size",
                block_size_start,
            )
        request_context_start = time.perf_counter() if profile_enabled else 0.0
        request_context = self._request_block_context(
            attn_metadata=attn_metadata,
            block_size=block_size,
            cache_step=layer_event_idx,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_request_block_context",
                request_context_start,
            )
        candidate_block_ids = (
            request_context.score_candidate_block_ids
            if request_context.has_block_table_context
            else None
        )
        select_digest_start = time.perf_counter() if profile_enabled else 0.0
        physical_block_ids = self._select_layer_digest_block_ids(
            layer_name,
            candidate_block_ids=candidate_block_ids,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_select_digest_blocks",
                select_digest_start,
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
                window_query_len=window_query_len,
            )
            return

        debug_record_enabled = self.config.enable_logging and bool(should_record)
        materialize_head_scores = debug_record_enabled and (
            self.config.score_granularity != "block"
        )
        score_result = None
        quest_packed_fast_path = False
        quest_packed_fallback_reason = "backend_not_quest_cuda"
        estimate_packed_optimized = getattr(
            self._scoring_backend,
            "estimate_packed_optimized",
            None,
        )
        if self.config.scoring_backend == "quest_cuda" and callable(
            estimate_packed_optimized
        ):
            packed_prefix_start = time.perf_counter() if profile_enabled else 0.0
            packed, quest_packed_fallback_reason = (
                self._try_get_quest_packed_prefix(
                    layer_name=layer_name,
                    physical_block_ids=physical_block_ids,
                    block_size=block_size,
                    device=window_query.device,
                    dtype=window_query.dtype,
                )
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_quest_packed_prefix",
                    packed_prefix_start,
                )
            if packed is not None:
                packed_estimate_start = (
                    time.perf_counter() if profile_enabled else 0.0
                )
                score_result = estimate_packed_optimized(
                    query_window=window_query,
                    packed=packed,
                    num_kv_heads=int(packed.metadata_data.shape[3]),
                    score_agg=self.config.score_agg,
                    profile_enabled=profile_enabled,
                    materialize_head_scores=materialize_head_scores,
                )
                if profile_enabled:
                    self._record_scoring_profile_timing(
                        "scoring_quest_packed_estimate",
                        packed_estimate_start,
                    )
                quest_packed_fast_path = True
                quest_packed_fallback_reason = None

        if score_result is None:
            pack_start = time.perf_counter() if profile_enabled else 0.0
            physical_block_ids, digest_min, digest_max = self._pack_layer_digests(
                layer_name,
                device=window_query.device,
                dtype=window_query.dtype,
                candidate_block_ids=candidate_block_ids,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_pack_layer_digests",
                    pack_start,
                )
            estimate_start = time.perf_counter() if profile_enabled else 0.0
            score_result = self._scoring_backend.estimate(
                query_window=window_query,
                digest_min=digest_min,
                digest_max=digest_max,
                score_agg=self.config.score_agg,
                metadata_page_size=block_size,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_backend_estimate",
                    estimate_start,
                )
            if self.config.scoring_backend != "quest_cuda":
                quest_packed_fallback_reason = "backend_not_quest_cuda"
            elif quest_packed_fallback_reason is None:
                quest_packed_fallback_reason = "fallback_after_fast_path_miss"

        scores = score_result.block_scores
        topk = min(self.config.topk, int(scores.numel()))
        score_packing_debug: dict[str, Any] = {}
        topk_block_ids: list[int] = []
        topk_score_values: list[float] = []
        head_debug: dict[str, Any] = {}
        score_block_debug: dict[str, Any] = {}
        if debug_record_enabled:
            score_packing_debug_start = (
                time.perf_counter() if profile_enabled else 0.0
            )
            score_packing_debug = self._score_packing_debug_fields(
                quest_packed_fast_path=quest_packed_fast_path,
                quest_packed_fallback_reason=quest_packed_fallback_reason,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_score_packing_debug",
                    score_packing_debug_start,
                )
            block_topk_start = time.perf_counter() if profile_enabled else 0.0
            topk_block_ids, topk_score_values = self._topk_block_scores(
                scores=scores,
                physical_block_ids=physical_block_ids,
                topk=topk,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_block_topk",
                    block_topk_start,
                )
            head_topk_start = time.perf_counter() if profile_enabled else 0.0
            head_debug = self._head_score_debug_fields(
                score_result=score_result,
                physical_block_ids=physical_block_ids,
                topk=topk,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_head_topk_debug",
                    head_topk_start,
                )
            block_debug_start = time.perf_counter() if profile_enabled else 0.0
            score_block_debug = self._score_block_debug_fields(
                request_context=request_context,
                observed_digest_block_ids=physical_block_ids,
            )
            if profile_enabled:
                self._record_scoring_profile_timing(
                    "scoring_block_debug_fields",
                    block_debug_start,
                )

        context_start = time.perf_counter() if profile_enabled else 0.0
        score_context = QueryScoreContext(
            layer_name=layer_name,
            should_record=should_record,
            layer_event_idx=layer_event_idx,
            query=query,
            window_query=window_query,
            window_query_len=window_query_len,
            request_context=request_context,
            physical_block_ids=physical_block_ids,
            score_result=score_result,
            topk=topk,
            topk_block_ids=topk_block_ids,
            topk_scores=topk_score_values,
            score_packing_debug=score_packing_debug,
            head_debug=head_debug,
            score_block_debug=score_block_debug,
        )
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_context_build",
                context_start,
            )
        return score_context

    def _update_query_window(
        self,
        layer_name: str,
        query_for_window: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Append one query and return the rolling query average."""
        state = self._query_windows.setdefault(
            layer_name,
            QueryWindowState(deque(maxlen=self.config.window_size)),
        )
        running_sum = state.running_sum
        if running_sum is not None and (
            running_sum.shape != query_for_window.shape
            or running_sum.dtype != query_for_window.dtype
            or running_sum.device != query_for_window.device
        ):
            state.entries.clear()
            state.running_sum = None
            running_sum = None

        if running_sum is None:
            state.entries.append(query_for_window)
            state.running_sum = query_for_window.clone()
        else:
            if len(state.entries) == self.config.window_size:
                state.running_sum.sub_(state.entries[0])
            state.entries.append(query_for_window)
            state.running_sum.add_(query_for_window)

        window_query_len = len(state.entries)
        return state.running_sum / window_query_len, window_query_len

    def _record_score_estimated(self, score_context: QueryScoreContext) -> None:
        """Record a score_estimated event from reusable score context."""
        if not score_context.should_record:
            return
        if not self.config.enable_logging:
            self.counters["score_estimated"] += 1
            return
        score_result = score_context.score_result
        self._record(
            "score_estimated",
            layer_name=score_context.layer_name,
            layer_event_idx=score_context.layer_event_idx,
            query_shape=self._shape_of(score_context.query),
            window_query_shape=self._shape_of(score_context.window_query),
            window_query_len=score_context.window_query_len,
            num_digest_blocks=len(score_context.physical_block_ids),
            score_count=int(score_result.block_scores.numel()),
            score_agg=self.config.score_agg,
            scoring_backend=score_result.scoring_backend,
            digest_kind=self.config.digest_kind,
            num_q_heads=score_result.num_q_heads,
            num_kv_heads=score_result.num_kv_heads,
            gqa_group_size=score_result.group_size,
            score_granularity=self.config.score_granularity,
            topk=score_context.topk,
            topk_block_ids=score_context.topk_block_ids,
            topk_scores=score_context.topk_scores,
            **score_context.score_packing_debug,
            **score_context.head_debug,
            **score_context.score_block_debug,
        )

    def _record_recovery_test_mutated(
        self,
        *,
        score_context: QueryScoreContext,
        kv_cache: Any,
        selected_block_ids: list[int],
        mutated_block_ids: list[int],
    ) -> None:
        """Record validation-only mutation that intentionally skips recovery."""
        if not score_context.should_record:
            return
        backup_stats = self._cpu_backup_store.stats()
        self._record(
            "recovery_test_mutated",
            layer_name=score_context.layer_name,
            layer_event_idx=score_context.layer_event_idx,
            query_shape=self._shape_of(score_context.query),
            window_query_shape=self._shape_of(score_context.window_query),
            window_query_len=score_context.window_query_len,
            kv_cache_shape=self._shape_of(kv_cache),
            recovery_policy=self.config.recovery_policy,
            recovery_topk=self.config.recovery_topk,
            recovery_threshold=self.config.recovery_threshold,
            recovery_test_mutate=self.config.recovery_test_mutate,
            recovery_test_mutated_scope=self._recovery_test_mutated_scope(),
            recovery_test_mode=self.config.recovery_test_mode,
            recovery_selected_block_ids=selected_block_ids,
            recovery_test_mutated_block_ids=mutated_block_ids,
            cpu_backup_block_count=backup_stats.block_count,
            cpu_backup_bytes=backup_stats.total_bytes,
            cpu_backup_fp16_payload_bytes=backup_stats.fp16_payload_bytes,
            cpu_backup_int8_payload_bytes=backup_stats.int8_payload_bytes,
            cpu_backup_int8_scale_bytes=backup_stats.int8_scale_bytes,
            cpu_backup_int4_payload_bytes=backup_stats.int4_payload_bytes,
            cpu_backup_int4_scale_bytes=backup_stats.int4_scale_bytes,
            cpu_backup_total_actual_bytes=(
                backup_stats.total_actual_backup_bytes
            ),
            **score_context.score_block_debug,
        )

    def _maybe_mutate_recovery_targets(
        self,
        *,
        kv_cache: Any,
        selected_block_ids: list[int],
    ) -> list[int]:
        """Apply optional M3 fault-injection mutation before recovery."""
        if self.config.recovery_test_mutate == "off":
            return []
        if self.config.recovery_test_mutate not in ("zero_selected", "zero_all"):
            raise ValueError(
                "MPR recovery_test_mutate must be 'off', 'zero_selected', "
                "or 'zero_all', "
                f"got {self.config.recovery_test_mutate!r}."
            )
        if not isinstance(kv_cache, torch.Tensor):
            return []
        if kv_cache.ndim != 5 or int(kv_cache.shape[0]) != 2:
            return []

        if self.config.recovery_test_mutate == "zero_all":
            kv_cache.zero_()
            return list(range(int(kv_cache.shape[1])))

        mutated_block_ids: list[int] = []
        num_blocks = int(kv_cache.shape[1])
        for block_id in selected_block_ids:
            block_id = int(block_id)
            if block_id < 0 or block_id >= num_blocks:
                continue
            kv_cache[:, block_id].zero_()
            mutated_block_ids.append(block_id)
        return mutated_block_ids

    def _recovery_test_mutated_scope(self) -> str:
        """Return debug scope for validation-only recovery mutation."""
        if self.config.recovery_test_mutate == "zero_all":
            return "all_kv_cache"
        return "selected"

    def _record_recovery_skip(
        self,
        should_record: bool,
        layer_name: str,
        layer_event_idx: int | None,
        reason: str,
        *,
        query: Any = None,
        attn_metadata: Any = None,
        kv_cache: Any = None,
    ) -> None:
        """Record a skipped recovery event when debug limits allow it."""
        if not should_record:
            return
        self._record(
            "recovery_skipped",
            layer_name=layer_name,
            layer_event_idx=layer_event_idx,
            skipped_reason=reason,
            query_shape=self._shape_of(query),
            max_query_len=getattr(attn_metadata, "max_query_len", None),
            num_actual_tokens=getattr(attn_metadata, "num_actual_tokens", None),
            kv_cache_shape=self._shape_of(kv_cache),
            recovery_policy=self.config.recovery_policy,
            recovery_topk=self.config.recovery_topk,
            recovery_threshold=self.config.recovery_threshold,
            cpu_backup_enabled=self.config.cpu_backup_enabled,
            scoring_enabled=self.config.scoring_enabled,
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
        cache_step: int | None = None,
    ) -> RequestBlockContext:
        """Parse the single-request block table and select score candidates."""
        cache_key = self._request_block_context_cache_key_for(
            attn_metadata=attn_metadata,
            block_size=block_size,
            cache_step=cache_step,
        )
        if (
            cache_key is not None
            and cache_key == self._request_block_context_cache_key
            and self._request_block_context_cache_value is not None
        ):
            self.counters["request_block_context_cache_hit"] += 1
            return self._request_block_context_cache_value
        if cache_key is not None:
            self.counters["request_block_context_cache_miss"] += 1

        profile_enabled = self.config.scoring_profile_enabled
        seq_lens_start = time.perf_counter() if profile_enabled else 0.0
        seq_lens = self._tensor_to_int_list(getattr(attn_metadata, "seq_lens", None))
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_request_ctx_seq_lens",
                seq_lens_start,
            )

        block_table_lookup_start = time.perf_counter() if profile_enabled else 0.0
        block_table = getattr(attn_metadata, "block_table", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_table_tensor", None)
        block_table_shape = self._shape_of(block_table)
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_request_ctx_block_table_lookup",
                block_table_lookup_start,
            )

        block_table_row_start = time.perf_counter() if profile_enabled else 0.0
        block_table_row: list[int] = []
        if block_table is not None and getattr(block_table, "ndim", 0) >= 2:
            row_values = self._tensor_to_int_list(block_table[0])
            block_table_row = [] if row_values is None else row_values
        elif block_table is not None:
            row_values = self._tensor_to_int_list(block_table)
            block_table_row = [] if row_values is None else row_values
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_request_ctx_block_table_row",
                block_table_row_start,
            )

        candidates_start = time.perf_counter() if profile_enabled else 0.0
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
        if profile_enabled:
            self._record_scoring_profile_timing(
                "scoring_request_ctx_candidates",
                candidates_start,
            )

        request_context = RequestBlockContext(
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
        if cache_key is not None:
            self._request_block_context_cache_key = cache_key
            self._request_block_context_cache_value = request_context
        return request_context

    def _request_block_context_cache_key_for(
        self,
        *,
        attn_metadata: Any,
        block_size: int | None,
        cache_step: int | None,
    ) -> tuple[Any, ...] | None:
        """Return a per-step cache key for request/block metadata parsing."""
        if cache_step is None:
            return None
        block_table = getattr(attn_metadata, "block_table", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_table_tensor", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if block_table is None and seq_lens is None:
            return None
        return (
            int(cache_step),
            int(block_size) if block_size is not None else None,
            int(self.config.recent_tokens),
            self._cache_scalar_identity(
                getattr(attn_metadata, "max_query_len", None)
            ),
            self._cache_scalar_identity(
                getattr(attn_metadata, "num_actual_tokens", None)
            ),
            self._cache_scalar_identity(getattr(attn_metadata, "num_reqs", None)),
            self._tensor_cache_identity(seq_lens),
            self._tensor_cache_identity(block_table),
        )

    @staticmethod
    def _cache_scalar_identity(value: Any) -> Any:
        """Return a tuple-safe scalar identity for cache keys."""
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        item_fn = getattr(value, "item", None)
        if callable(item_fn):
            try:
                item_value = item_fn()
                if isinstance(item_value, (bool, int, float, str)):
                    return item_value
            except (RuntimeError, TypeError, ValueError):
                pass
        return repr(value)

    @staticmethod
    def _tensor_cache_identity(value: Any) -> tuple[Any, ...] | None:
        """Return tensor identity metadata without materializing device data."""
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        shape_tuple = (
            tuple(int(dim) for dim in shape)
            if shape is not None
            else None
        )
        stride_value = None
        stride_fn = getattr(value, "stride", None)
        if callable(stride_fn):
            try:
                stride_value = tuple(int(dim) for dim in stride_fn())
            except (RuntimeError, TypeError, ValueError):
                stride_value = None
        storage_offset = None
        storage_offset_fn = getattr(value, "storage_offset", None)
        if callable(storage_offset_fn):
            try:
                storage_offset = int(storage_offset_fn())
            except (RuntimeError, TypeError, ValueError):
                storage_offset = None
        data_ptr = None
        data_ptr_fn = getattr(value, "data_ptr", None)
        if callable(data_ptr_fn):
            try:
                data_ptr = int(data_ptr_fn())
            except (RuntimeError, TypeError, ValueError):
                data_ptr = None
        identity = data_ptr if data_ptr is not None else id(value)
        return (
            type(value).__name__,
            identity,
            shape_tuple,
            stride_value,
            storage_offset,
            str(getattr(value, "device", None)),
            str(getattr(value, "dtype", None)),
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

    @staticmethod
    def _counter_block_table(attn_metadata: Any) -> Any | None:
        """Return the request block table used by the counter backend."""
        if attn_metadata is None:
            return None
        block_table = getattr(attn_metadata, "block_table_tensor", None)
        if block_table is None:
            block_table = getattr(attn_metadata, "block_table", None)
        return block_table

    @staticmethod
    def _counter_single_request_query_len(attn_metadata: Any) -> int | None:
        """Return query length only when metadata represents one request."""
        num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
        max_query_len = getattr(attn_metadata, "max_query_len", None)
        if num_actual_tokens is None or max_query_len is None:
            return None
        num_actual_tokens = int(num_actual_tokens)
        max_query_len = int(max_query_len)
        if num_actual_tokens != max_query_len:
            return None
        return num_actual_tokens

    @staticmethod
    def _counter_block_id(
        *,
        block_table: Any,
        req_idx: int,
        logical_block_idx: int,
    ) -> int | None:
        """Read one physical block id from a request block table."""
        try:
            if getattr(block_table, "ndim", 0) >= 2:
                value = block_table[req_idx, logical_block_idx]
            else:
                value = block_table[logical_block_idx]
            block_id = int(value.item() if hasattr(value, "item") else value)
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None
        return block_id if block_id >= 0 else None

    def _key_cache_for_digest(
        self,
        *,
        layer_name: str,
        kv_cache: Any,
        block_size: int,
    ) -> Any:
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
        return kv_cache[0]

    def _create_digest_for_full_block(
        self,
        *,
        layer_name: str,
        block_id: int,
        block_size: int,
        key_cache: Any,
        kv_cache: Any,
        layer_event_idx: int | None,
    ) -> bool:
        """Create all sidecar state for one full physical KV block."""
        if block_id < 0 or block_id >= int(key_cache.shape[0]):
            raise AssertionError(
                f"MPR observed block id outside KV cache for {layer_name}: "
                f"block_id={block_id}, num_blocks={int(key_cache.shape[0])}."
            )

        layer_digests = self._digest_cache.setdefault(layer_name, {})
        if block_id in layer_digests:
            return False

        profile_enabled = self.config.boundary_profile_enabled
        summarize_start = time.perf_counter() if profile_enabled else 0.0
        digest = summarize_key_block(
            key_cache[block_id],
            digest_kind=self.config.digest_kind,
        )
        if profile_enabled:
            self._record_boundary_profile_timing(
                "counter_summarize_key_block",
                summarize_start,
            )
        to_digest_start = time.perf_counter() if profile_enabled else 0.0
        layer_digests[block_id] = self._to_block_digest(
            digest,
            layer_event_idx,
        )
        if profile_enabled:
            self._record_boundary_profile_timing(
                "counter_to_block_digest",
                to_digest_start,
            )
        quest_metadata_start = time.perf_counter() if profile_enabled else 0.0
        self._append_quest_metadata_digest(
            layer_name=layer_name,
            block_id=block_id,
            digest=digest,
        )
        if profile_enabled:
            self._record_boundary_profile_timing(
                "counter_append_quest_metadata",
                quest_metadata_start,
            )
        backup_start = time.perf_counter() if profile_enabled else 0.0
        self._maybe_backup_kv_block(
            layer_name=layer_name,
            block_id=block_id,
            kv_block=kv_cache[:, block_id],
            layer_event_idx=layer_event_idx,
        )
        if profile_enabled:
            self._record_boundary_profile_timing(
                "counter_backup",
                backup_start,
            )
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
        return True

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
        key_cache = self._key_cache_for_digest(
            layer_name=layer_name,
            kv_cache=kv_cache,
            block_size=block_size,
        )

        # layer_offsets: observed token offsets by physical block id.
        # Structure: dict[physical_block_id, set[block_offset]].
        layer_offsets = self._block_offsets.setdefault(layer_name, {})
        self._digest_cache.setdefault(layer_name, {})

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
            created = False
            if len(offsets) == block_size:
                created = self._create_digest_for_full_block(
                    layer_name=layer_name,
                    block_id=block_id,
                    block_size=block_size,
                    key_cache=key_cache,
                    kv_cache=kv_cache,
                    layer_event_idx=layer_event_idx,
                )
            if created:
                created_block_ids.append(block_id)

        return created_block_ids

    def _maybe_backup_kv_block(
        self,
        *,
        layer_name: str,
        block_id: int,
        kv_block: torch.Tensor,
        layer_event_idx: int | None,
    ) -> None:
        """Create the M2 semantic CPU fp16 backup for one full K/V block."""
        if not self.config.cpu_backup_enabled:
            return
        result = self._cpu_backup_store.put(
            layer_name=layer_name,
            physical_block_id=block_id,
            kv_block=kv_block,
            backup_storage_mode=self.config.backup_storage_mode,
        )
        stats = self._cpu_backup_store.stats()
        if layer_name in self._layer_indices:
            self._record(
                "cpu_backup_created",
                layer_name=layer_name,
                layer_event_idx=layer_event_idx,
                physical_block_id=block_id,
                cpu_backup_shape=list(result.shape),
                cpu_backup_dtype=str(result.dtype),
                cpu_backup_bytes=result.num_bytes,
                cpu_backup_fp16_payload_bytes=result.fp16_payload_bytes,
                cpu_backup_int8_payload_bytes=result.int8_payload_bytes,
                cpu_backup_int8_scale_bytes=result.int8_scale_bytes,
                cpu_backup_int4_payload_bytes=result.int4_payload_bytes,
                cpu_backup_int4_scale_bytes=result.int4_scale_bytes,
                cpu_backup_total_actual_bytes=(
                    result.total_actual_backup_bytes
                ),
                cpu_backup_copy_wall_ms=result.copy_wall_seconds * 1000.0,
                cpu_backup_block_count=stats.block_count,
                cpu_backup_total_bytes=stats.total_bytes,
                cpu_backup_total_fp16_payload_bytes=(
                    stats.fp16_payload_bytes
                ),
                cpu_backup_total_int8_payload_bytes=(
                    stats.int8_payload_bytes
                ),
                cpu_backup_total_int8_scale_bytes=stats.int8_scale_bytes,
                cpu_backup_total_int4_payload_bytes=(
                    stats.int4_payload_bytes
                ),
                cpu_backup_total_int4_scale_bytes=stats.int4_scale_bytes,
                cpu_backup_store_total_actual_bytes=(
                    stats.total_actual_backup_bytes
                ),
                cpu_backup_total_copy_wall_ms=(
                    stats.total_copy_wall_seconds * 1000.0
                ),
            )
        else:
            self.counters["cpu_backup_created"] += 1

    def _append_quest_metadata_digest(
        self,
        *,
        layer_name: str,
        block_id: int,
        digest: KeyBlockDigest,
    ) -> None:
        """Append a newly created digest to the layer's Quest metadata store."""
        if self.config.scoring_backend != "quest_cuda":
            return
        store = self._quest_metadata_stores.get(layer_name)
        num_kv_heads = int(digest.digest_min.shape[0])
        head_dim = int(digest.digest_min.shape[1])
        if store is None:
            store = QuestMetadataStore(
                metadata_page_size=int(digest.block_size),
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=digest.digest_max.dtype,
                device=digest.digest_max.device,
            )
            self._quest_metadata_stores[layer_name] = store
        elif (
            store.metadata_page_size != int(digest.block_size)
            or store.num_kv_heads != num_kv_heads
            or store.head_dim != head_dim
            or store.dtype != digest.digest_max.dtype
            or store.device != digest.digest_max.device
        ):
            raise AssertionError(
                "MPR Quest metadata store shape/dtype/device mismatch for "
                f"{layer_name}."
            )

        store.append_digest(
            block_id=block_id,
            digest_min=digest.digest_min,
            digest_max=digest.digest_max,
        )

    def _select_layer_digest_block_ids(
        self,
        layer_name: str,
        *,
        candidate_block_ids: list[int] | None = None,
    ) -> list[int]:
        """Select existing layer digest block ids without stacking tensors."""
        layer_digests = self._digest_cache.get(layer_name, {})
        if not layer_digests:
            return []
        if candidate_block_ids is None:
            return sorted(layer_digests)
        return [block_id for block_id in candidate_block_ids if block_id in layer_digests]

    def _try_get_quest_packed_prefix(
        self,
        *,
        layer_name: str,
        physical_block_ids: list[int],
        block_size: int | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[PackedQuestDigestCache | None, str | None]:
        """Return a persistent Quest metadata prefix view when safe."""
        if not physical_block_ids:
            return None, "no_digest_blocks"
        if block_size is None or block_size <= 0:
            return None, "missing_block_size"
        store = self._quest_metadata_stores.get(layer_name)
        if store is None:
            return None, "missing_quest_metadata_store"
        if store.metadata_page_size != block_size:
            return None, "metadata_page_size_mismatch"
        if store.device != device:
            return None, "metadata_device_mismatch"
        if store.dtype != dtype:
            return None, "metadata_dtype_mismatch"
        if not store.can_view_prefix(physical_block_ids):
            return None, "non_prefix_score_candidates"
        return store.view_prefix(len(physical_block_ids)), None

    @staticmethod
    def _score_packing_debug_fields(
        *,
        quest_packed_fast_path: bool,
        quest_packed_fallback_reason: str | None,
    ) -> dict[str, Any]:
        """Build debug metadata for Quest packed-cache fast path usage."""
        fields: dict[str, Any] = {
            "quest_packed_fast_path": quest_packed_fast_path,
        }
        if quest_packed_fallback_reason is not None:
            fields["quest_packed_fallback_reason"] = (
                quest_packed_fallback_reason
            )
        return fields

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
    def _normalize_physical_block_ids(block_ids: Any) -> list[int]:
        """Flatten a one-level block id container into sorted unique ids."""
        if block_ids is None:
            return []
        # Tensor-like whole input, e.g. torch.tensor([1, 2, 3]) or a shaped
        # block-id tensor. This is mostly defensive for direct sidecar callers;
        # the current KVCacheManager hook passes a Python list.
        if hasattr(block_ids, "detach"):
            return sorted(
                {
                    int(block_id)
                    for block_id in block_ids.detach().cpu().reshape(-1).tolist()
                }
            )
        normalized: set[int] = set()
        # General Python container path, e.g. [1, 2], ([1], [2]), or a mixed
        # list containing tensors. This is the path used by the current
        # KVCacheManager hook after flattening group-wise block ids.
        for item in block_ids:
            if item is None:
                continue
            if hasattr(item, "detach"):
                normalized.update(
                    int(block_id)
                    for block_id in item.detach().cpu().reshape(-1).tolist()
                )
            elif isinstance(item, (list, tuple, set)):
                normalized.update(int(block_id) for block_id in item)
            else:
                normalized.add(int(item))
        return sorted(normalized)

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
        if not self.config.enable_logging:
            return
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


def get_mpr_observe_kv_write_timing() -> dict[str, float | int]:
    if _GLOBAL_SIDECAR is None:
        result: dict[str, float | int] = {
            "count": 0,
            "pre_observe_block_offsets_total_ms": 0.0,
            "pre_observe_block_offsets_mean_ms": 0.0,
            "pre_observe_block_offsets_max_ms": 0.0,
            "observe_block_offsets_count": 0,
            "observe_block_offsets_total_ms": 0.0,
            "observe_block_offsets_mean_ms": 0.0,
            "observe_block_offsets_max_ms": 0.0,
        }
        for name in _BOUNDARY_PROFILE_TIMING_NAMES:
            result[f"{name}_count"] = 0
            result[f"{name}_total_ms"] = 0.0
            result[f"{name}_mean_ms"] = 0.0
            result[f"{name}_max_ms"] = 0.0
        return result
    return _GLOBAL_SIDECAR.observe_kv_write_timing()


def reset_mpr_observe_kv_write_timing() -> None:
    if _GLOBAL_SIDECAR is not None:
        _GLOBAL_SIDECAR.reset_observe_kv_write_timing()


def get_mpr_scoring_profile_timing() -> dict[str, float | int]:
    if _GLOBAL_SIDECAR is None:
        result: dict[str, float | int] = {}
        for name in _SCORING_PROFILE_TIMING_NAMES:
            result[f"{name}_count"] = 0
            result[f"{name}_total_ms"] = 0.0
            result[f"{name}_mean_ms"] = 0.0
            result[f"{name}_max_ms"] = 0.0
        return result
    return _GLOBAL_SIDECAR.scoring_profile_timing()


def reset_mpr_scoring_profile_timing() -> None:
    reset_packed_estimate_profile_timing()
    if _GLOBAL_SIDECAR is not None:
        _GLOBAL_SIDECAR.reset_scoring_profile_timing()


def reset_mpr_sidecar() -> None:
    global _GLOBAL_SIDECAR
    with _GLOBAL_LOCK:
        if _GLOBAL_SIDECAR is not None:
            _GLOBAL_SIDECAR.close()
        _GLOBAL_SIDECAR = None
