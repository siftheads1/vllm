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


def _parse_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.getenv(name)
    value = default if raw is None else raw.strip().lower()
    if value not in choices:
        choices_text = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {choices_text}, got {raw!r}.")
    return value


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
    recent_tokens: int = 64
    score_agg: str = "max"
    scoring_backend: str = "torch_quest"
    digest_kind: str = "raw_minmax"
    score_granularity: str = "kv_head"

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
            recent_tokens=_parse_int("VLLM_MPR_RECENT_TOKENS", 64, 0),
            score_agg=_parse_choice(
                "VLLM_MPR_SCORE_AGG",
                "max",
                {"max", "mean"},
            ),
            scoring_backend=_parse_choice(
                "VLLM_MPR_SCORING_BACKEND",
                "torch_quest",
                {"torch_quest", "quest_cuda"},
            ),
            digest_kind=_parse_choice(
                "VLLM_MPR_DIGEST_KIND",
                "raw_minmax",
                {"arkvale", "raw_minmax"},
            ),
            score_granularity=_parse_choice(
                "VLLM_MPR_SCORE_GRANULARITY",
                "kv_head",
                {"block", "kv_head", "query_head"},
            ),
        )
