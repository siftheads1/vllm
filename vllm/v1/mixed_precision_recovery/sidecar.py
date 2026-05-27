# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Disabled-by-default MPR sidecar scaffold."""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
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
        self._record(
            "observe_kv_write",
            layer_name=layer_name,
            block_size=block_size,
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
