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


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _validate_ratio(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


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
    cpu_backup_enabled: bool = False
    scoring_enabled: bool = True
    recovery_enabled: bool = False
    recovery_topk: int = 8
    recovery_policy: str = "topk_block"
    recovery_threshold: float = 0.0
    recovery_test_mutate: str = "off"
    recovery_test_mode: str = "recover"
    precision_tiering_enabled: bool = False
    precision_policy: str = "top_ratio"
    tier_fp16_ratio: float = 0.25
    tier_int8_ratio: float = 0.50
    tier_high_threshold: float = 0.0
    tier_low_threshold: float = 0.0
    backup_storage_mode: str = "eager_fp16_int8"

    def __post_init__(self) -> None:
        """Validate MPR config values that depend on multiple fields."""
        if self.precision_policy not in {"top_ratio", "threshold"}:
            raise ValueError(
                "precision_policy must be 'top_ratio' or 'threshold', got "
                f"{self.precision_policy!r}."
            )
        if self.backup_storage_mode not in {"eager_fp16_int8", "fp16_only"}:
            raise ValueError(
                "backup_storage_mode must be 'eager_fp16_int8' or "
                f"'fp16_only', got {self.backup_storage_mode!r}."
            )
        if (
            self.cpu_backup_enabled
            and not self.precision_tiering_enabled
            and self.backup_storage_mode == "eager_fp16_int8"
        ):
            raise ValueError(
                "backup_storage_mode='eager_fp16_int8' requires "
                "precision_tiering_enabled=True when cpu_backup_enabled=True; "
                "use backup_storage_mode='fp16_only' for M3 fp16 recovery."
            )
        _validate_ratio("tier_fp16_ratio", self.tier_fp16_ratio)
        _validate_ratio("tier_int8_ratio", self.tier_int8_ratio)
        ratio_sum = self.tier_fp16_ratio + self.tier_int8_ratio
        if ratio_sum > 1.0:
            raise ValueError(
                "tier_fp16_ratio + tier_int8_ratio must be <= 1, got "
                f"{ratio_sum}."
            )
        if self.tier_high_threshold < self.tier_low_threshold:
            raise ValueError(
                "tier_high_threshold must be >= tier_low_threshold, got "
                f"{self.tier_high_threshold} < {self.tier_low_threshold}."
            )

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
            cpu_backup_enabled=_parse_bool("VLLM_MPR_CPU_BACKUP", False),
            scoring_enabled=_parse_bool("VLLM_MPR_SCORING_ENABLE", True),
            recovery_enabled=_parse_bool("VLLM_MPR_RECOVERY_ENABLE", False),
            recovery_topk=_parse_int("VLLM_MPR_RECOVERY_TOPK", 8, 1),
            recovery_policy=_parse_choice(
                "VLLM_MPR_RECOVERY_POLICY",
                "topk_block",
                {"topk_block", "threshold_block"},
            ),
            recovery_threshold=_parse_float(
                "VLLM_MPR_RECOVERY_THRESHOLD",
                0.0,
            ),
            recovery_test_mutate=_parse_choice(
                "VLLM_MPR_RECOVERY_TEST_MUTATE",
                "off",
                {"off", "zero_selected", "zero_all"},
            ),
            recovery_test_mode=_parse_choice(
                "VLLM_MPR_RECOVERY_TEST_MODE",
                "recover",
                {"recover", "mutate_only"},
            ),
            precision_tiering_enabled=_parse_bool(
                "VLLM_MPR_PRECISION_TIERING_ENABLE",
                False,
            ),
            precision_policy=_parse_choice(
                "VLLM_MPR_PRECISION_POLICY",
                "top_ratio",
                {"top_ratio", "threshold"},
            ),
            tier_fp16_ratio=_parse_float(
                "VLLM_MPR_TIER_FP16_RATIO",
                0.25,
            ),
            tier_int8_ratio=_parse_float(
                "VLLM_MPR_TIER_INT8_RATIO",
                0.50,
            ),
            tier_high_threshold=_parse_float(
                "VLLM_MPR_TIER_HIGH_THRESHOLD",
                0.0,
            ),
            tier_low_threshold=_parse_float(
                "VLLM_MPR_TIER_LOW_THRESHOLD",
                0.0,
            ),
            backup_storage_mode=_parse_choice(
                "VLLM_MPR_BACKUP_STORAGE_MODE",
                "eager_fp16_int8",
                {"eager_fp16_int8", "fp16_only"},
            ),
        )
