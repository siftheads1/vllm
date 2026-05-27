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

logger = init_logger(__name__)


@dataclass
class RecoverySidecar:
    """Score-only sidecar state for Milestone 1.

    Step 1.1 intentionally does not observe tensors yet. The methods are no-op
    placeholders that let later attention hooks increment counters behind one
    fast ``enabled`` guard.
    """

    config: MPRConfig = field(default_factory=MPRConfig.from_env)
    counters: Counter[str] = field(default_factory=Counter)
    # Debug-only layer bookkeeping for VLLM_MPR_MAX_LAYERS dump limiting.
    _layer_indices: dict[str, int] = field(default_factory=dict)
    _kv_write_counts: Counter[str] = field(default_factory=Counter)
    _debug_writer: MPRDebugWriter = field(init=False)

    def __post_init__(self) -> None:
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
        return self.config.enabled

    def observe_kv_write(
        self,
        layer_name: str,
        key: Any = None,
        value: Any = None,
        slot_mapping: Any = None,
        block_size: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        should_record, layer_event_idx = self._should_record_layer_event(
            layer_name,
            self._kv_write_counts,
        )
        if not should_record:
            return
        if block_size is None or block_size <= 0:
            raise ValueError(
                f"MPR requires a positive block_size for {layer_name}, "
                f"got {block_size}."
            )
        if slot_mapping is None:
            raise ValueError(f"MPR requires slot_mapping for {layer_name}.")

        flat_slots = slot_mapping.detach().reshape(-1)
        num_slots = int(flat_slots.numel())
        if num_slots == 0:
            unique_block_ids: list[int] = []
            min_block_offset = None
            max_block_offset = None
            num_pad_slots = 0
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
            valid_slots = flat_slots[flat_slots != PAD_SLOT_ID]
            num_pad_slots = num_slots - int(valid_slots.numel())
            if valid_slots.numel() == 0:
                unique_block_ids = []
                min_block_offset = None
                max_block_offset = None
            else:
                block_ids = valid_slots // block_size
                block_offsets = valid_slots % block_size
                unique_block_ids = [
                    int(block_id)
                    for block_id in sorted(block_ids.unique().detach().cpu().tolist())
                ]
                min_block_offset = int(block_offsets.min().item())
                max_block_offset = int(block_offsets.max().item())

        self._record(
            "observe_kv_write",
            layer_name=layer_name,
            layer_event_idx=layer_event_idx,
            key_shape=self._shape_of(key),
            value_shape=self._shape_of(value),
            slot_mapping_shape=self._shape_of(slot_mapping),
            num_slots=num_slots,
            num_valid_slots=num_slots - num_pad_slots,
            num_pad_slots=num_pad_slots,
            block_size=block_size,
            unique_block_ids=unique_block_ids,
            min_block_offset=min_block_offset,
            max_block_offset=max_block_offset,
        )

    def observe_query(
        self,
        layer_name: str,
        query: Any = None,
        attn_metadata: Any = None,
    ) -> None:
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
        if not self.config.enabled:
            return None
        self._record(
            "estimate_scores",
            layer_name=layer_name,
            block_size=block_size,
        )
        return None

    def snapshot_stats(self) -> dict[str, int]:
        return dict(self.counters)

    def close(self) -> None:
        self._debug_writer.close()

    def _should_record_layer_event(
        self,
        layer_name: str,
        event_counts: Counter[str],
    ) -> tuple[bool, int | None]:
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
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        return [int(dim) for dim in shape]

    def _record(self, event: str, **fields: Any) -> None:
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
