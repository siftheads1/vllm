# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Environment-backed configuration for Mixed-Precision Recovery."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}.")


def _parse_int(name: str, default: int, min_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")
    return value


def _parse_optional_limit(name: str) -> int | None:
    value = _parse_int(name, -1, -1)
    return None if value < 0 else value


@dataclass(frozen=True)
class MPRConfig:
    """Configuration for the score-only MPR sidecar.

    Defaults keep the feature completely disabled. The values here are small
    Milestone 1 controls only; they do not define recovery or offload policy.
    """

    enabled: bool = False
    debug_dir: str | None = None
    topk: int = 8
    max_layers: int | None = None
    max_steps: int | None = None
    dump_every: int = 1
    window_size: int = 64

    @classmethod
    def from_env(cls) -> "MPRConfig":
        return cls(
            enabled=_parse_bool("VLLM_MPR_ENABLE", False),
            debug_dir=os.getenv("VLLM_MPR_DEBUG_DIR") or None,
            topk=_parse_int("VLLM_MPR_TOPK", 8, 1),
            max_layers=_parse_optional_limit("VLLM_MPR_MAX_LAYERS"),
            max_steps=_parse_optional_limit("VLLM_MPR_MAX_STEPS"),
            dump_every=_parse_int("VLLM_MPR_DUMP_EVERY", 1, 1),
            window_size=_parse_int("VLLM_MPR_WINDOW_SIZE", 64, 1),
        )
