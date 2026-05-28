# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Disabled-by-default MPR sidecar scaffold."""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.mixed_precision_recovery.config import MPRConfig
from vllm.v1.mixed_precision_recovery.debug import MPRDebugWriter
from vllm.v1.mixed_precision_recovery.digest import (
    KeyBlockDigest,
    summarize_key_block,
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
        layer_event_idx: Per-layer KV-write event index that first completed
            the block. ``-1`` means the digest was created while the event was
            outside the debug dump window.
    """

    digest_min: Any
    digest_max: Any
    valid_token_count: int
    block_size: int
    layer_event_idx: int


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
    _block_offsets: dict[str, dict[int, set[int]]] = field(default_factory=dict)
    _digest_cache: dict[str, dict[int, BlockDigest]] = field(default_factory=dict)
    _debug_writer: MPRDebugWriter = field(init=False)

    def __post_init__(self) -> None:
        """Initialize optional debug output and emit the init event."""
        self._debug_writer = MPRDebugWriter(self.config)
        if self.config.enabled:
            logger.info(
                "MPR sidecar enabled: topk=%d, window_size=%d, "
                "debug_dir=%s",
                self.config.topk,
                self.config.window_size,
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
            )

    def observe_query(
        self,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
    ) -> None:
        """Observe an attention query tensor.

        Step 1.3 does not score queries yet, so this method only records a
        debug event when enabled.
        """
        if not self.config.enabled:
            return
        self._record("observe_query", layer_name=layer_name)

    def estimate_scores(
        self,
        layer_name: str,
        window_query: Any = None,
        attn_metadata: Any = None,
        block_size: int | None = None,
    ) -> None:
        """Placeholder for future digest/query scoring.

        Args:
            layer_name: vLLM attention layer name.
            window_query: Future query window summary. The planned shape is a
                query tensor or query summary over up to
                ``VLLM_MPR_WINDOW_SIZE`` decode tokens.
            attn_metadata: vLLM attention metadata for the current layer.
            block_size: Number of token slots per physical KV block.

        Returns:
            ``None`` until Step 1.4 scoring is implemented.
        """
        if not self.config.enabled:
            return None
        self._record(
            "estimate_scores",
            layer_name=layer_name,
            block_size=block_size,
        )
        return None

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
                digest = summarize_key_block(key_cache[block_id])
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
                        num_digest_blocks_for_layer=len(layer_digests),
                        total_digest_blocks=self._num_digest_blocks(),
                    )

        return created_block_ids

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
        )

    def _num_digest_blocks(self) -> int:
        """Return the total number of cached block digests across layers."""
        return sum(len(layer_digests) for layer_digests in self._digest_cache.values())

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
