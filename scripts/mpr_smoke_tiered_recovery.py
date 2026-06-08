#!/usr/bin/env python3
"""Run an M4/M4.5 tiered recovery ratio sweep smoke.

This smoke runs one MPR-off baseline, then runs tiered MPR recovery across
FP16/INT8 and opt-in FP16/INT8/INT4 top-ratio compositions. It is intended to
validate tiered recovery path coverage, debug JSONL shape, and byte accounting
while printing generated outputs for inspection.

Skip-tier correctness under degraded/missing residency is intentionally out of
scope for this script and remains a Step 4.9 responsibility.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mpr_compare_generation_outputs import extract_generated_token_ids
from scripts.mpr_smoke_recovery_quality import (
    clean_mpr_env,
    common_text_prefix_len,
    count_events,
    extract_generated_text,
    first_mismatch_index,
    load_jsonl_events,
    run_generation,
)


DEFAULT_TIER_RATIOS = (
    "1.0:0.0",
    "0.75:0.25",
    "0.5:0.5",
    "0.25:0.75",
    "0.0:1.0",
)
SKIP_ACCOUNTING_RATIO = "0.25:0.50"


@dataclass(frozen=True)
class TierRatio:
    fp16: float
    int8: float
    int4: float = 0.0

    @property
    def skip(self) -> float:
        return 1.0 - self.fp16 - self.int8 - self.int4

    @property
    def has_int4(self) -> bool:
        return self.int4 > 0.0

    @property
    def label(self) -> str:
        fp16 = f"{self.fp16:.2f}".replace(".", "p")
        int8 = f"{self.int8:.2f}".replace(".", "p")
        if self.has_int4:
            int4 = f"{self.int4:.2f}".replace(".", "p")
            return f"fp16_{fp16}_int8_{int8}_int4_{int4}"
        return f"fp16_{fp16}_int8_{int8}"

    @property
    def display(self) -> str:
        if self.has_int4:
            return f"{self.fp16:.2f}:{self.int8:.2f}:{self.int4:.2f}"
        return f"{self.fp16:.2f}:{self.int8:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MPR M4/M4.5 tiered recovery ratio sweep smoke."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text. Defaults depend on --prompt-preset.",
    )
    parser.add_argument(
        "--prompt-preset",
        choices=("kv_cache", "name_recall"),
        default="kv_cache",
        help="Built-in prompt for the smoke.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for logs and debug JSONL. Defaults to /tmp/mpr_m4_tiered_*.",
    )
    parser.add_argument(
        "--recent-tokens",
        type=int,
        default=256,
        help="Recent-token protection for score candidate selection.",
    )
    parser.add_argument("--debug-max-layers", type=int, default=2)
    parser.add_argument("--debug-max-steps", type=int, default=2048)
    parser.add_argument(
        "--tier-ratios",
        default=",".join(DEFAULT_TIER_RATIOS),
        help=(
            "Comma-separated FP16:INT8 or FP16:INT8:INT4 ratios. Existing "
            "two-part ratios imply INT4=0. Defaults to a no-skip M4 sweep: "
            + ",".join(DEFAULT_TIER_RATIOS)
            + "."
        ),
    )
    parser.add_argument(
        "--include-skip-ratio",
        action="store_true",
        help=(
            f"Also run {SKIP_ACCOUNTING_RATIO}; this checks skip assignment "
            "and accounting only, not degraded skip correctness. The added "
            "ratio uses INT4=0."
        ),
    )
    parser.add_argument(
        "--text-preview-chars",
        type=int,
        default=360,
        help="Generated text preview length printed per run.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary with generated outputs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to the baseline generation script.",
    )
    parser.add_argument(
        "--show-full-text",
        action="store_true",
        help="Print full generated_text for baseline and each tiered run.",
    )
    return parser.parse_args()


def parse_tier_ratios(ratio_text: str, *, include_skip_ratio: bool) -> list[TierRatio]:
    ratio_items = [item.strip() for item in ratio_text.split(",") if item.strip()]
    if include_skip_ratio and SKIP_ACCOUNTING_RATIO not in ratio_items:
        ratio_items.append(SKIP_ACCOUNTING_RATIO)
    if not ratio_items:
        raise ValueError("At least one --tier-ratios entry is required.")

    ratios: list[TierRatio] = []
    for item in ratio_items:
        try:
            parts = item.split(":")
            if len(parts) not in (2, 3):
                raise ValueError
            fp16_text, int8_text = parts[:2]
            fp16 = float(fp16_text)
            int8 = float(int8_text)
            int4 = float(parts[2]) if len(parts) == 3 else 0.0
        except ValueError as exc:
            raise ValueError(
                "Tier ratios must be comma-separated FP16:INT8 or "
                f"FP16:INT8:INT4 entries, got {item!r}."
            ) from exc
        if fp16 < 0.0 or int8 < 0.0 or int4 < 0.0:
            raise ValueError(f"Tier ratios must be non-negative, got {item!r}.")
        if fp16 > 1.0 or int8 > 1.0 or int4 > 1.0:
            raise ValueError(f"Tier ratios must be <= 1.0, got {item!r}.")
        ratio_sum = fp16 + int8 + int4
        if ratio_sum > 1.0 + 1e-9:
            raise ValueError(
                f"Tier ratio sum must be <= 1.0, got {item!r} "
                f"(sum={ratio_sum:.6f})."
            )
        ratios.append(TierRatio(fp16=fp16, int8=int8, int4=int4))
    return ratios


def preview_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def ratio_env(
    *,
    args: argparse.Namespace,
    ratio: TierRatio,
    debug_dir: Path,
) -> dict[str, str]:
    env = clean_mpr_env()
    env.update(
        {
            "VLLM_MPR_ENABLE": "1",
            "VLLM_MPR_CPU_BACKUP": "1",
            "VLLM_MPR_BACKUP_STORAGE_MODE": (
                "eager_fp16_int8_int4" if ratio.has_int4 else "eager_fp16_int8"
            ),
            "VLLM_MPR_SCORING_ENABLE": "1",
            "VLLM_MPR_RECOVERY_ENABLE": "1",
            "VLLM_MPR_RECOVERY_POLICY": "threshold_block",
            "VLLM_MPR_RECOVERY_THRESHOLD": "-1e30",
            "VLLM_MPR_RECOVERY_TOPK": "1",
            "VLLM_MPR_PRECISION_TIERING_ENABLE": "1",
            "VLLM_MPR_PRECISION_POLICY": "top_ratio",
            "VLLM_MPR_TIER_FP16_RATIO": str(ratio.fp16),
            "VLLM_MPR_TIER_INT8_RATIO": str(ratio.int8),
            "VLLM_MPR_TIER_INT4_RATIO": str(ratio.int4),
            "VLLM_MPR_RECOVERY_TEST_MUTATE": "zero_selected",
            "VLLM_MPR_RECOVERY_TEST_MODE": "recover",
            "VLLM_MPR_RECENT_TOKENS": str(args.recent_tokens),
            "VLLM_MPR_DEBUG_DIR": str(debug_dir),
            "VLLM_MPR_MAX_LAYERS": str(args.debug_max_layers),
            "VLLM_MPR_MAX_STEPS": str(args.debug_max_steps),
        }
    )
    return env


def jsonl_paths(debug_dir: Path) -> list[Path]:
    return sorted(debug_dir.glob("*.jsonl"))


def prepare_debug_dir(debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = jsonl_paths(debug_dir)
    if existing_paths:
        raise AssertionError(
            f"Debug directory already contains JSONL files: {debug_dir}. "
            "Use a fresh --work-dir for this smoke run."
        )


def run_jsonl_validator(
    *,
    args: argparse.Namespace,
    ratio: TierRatio,
    debug_dir: Path,
    log_path: Path,
) -> None:
    paths = jsonl_paths(debug_dir)
    if not paths:
        raise AssertionError(f"No debug JSONL files found in {debug_dir}.")

    cmd = [
        args.python,
        "scripts/mpr_validate_debug_jsonl.py",
        *[str(path) for path in paths],
        "--min-digest-events",
        "1",
        "--min-score-events",
        "0",
        "--min-recovery-events",
        "1",
        "--allow-unmatched-digest-events",
        "--show",
        "5",
    ]
    if ratio.has_int4:
        cmd.append("--require-tiered-int4-recovery")
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(
                f"debug JSONL validator failed; see log: {log_path}",
                file=sys.stderr,
            )
            lines = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            print(f"last {min(80, len(lines))} validator lines:", file=sys.stderr)
            for line in lines[-80:]:
                print(line, file=sys.stderr)
            raise


def int_list(event: dict[str, Any], field: str) -> list[int]:
    value = event.get(field, [])
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise AssertionError(f"{field} must be a list of ints in {event}.")
    return value


def int_value(event: dict[str, Any], field: str) -> int:
    value = event.get(field, 0)
    if not isinstance(value, int):
        raise AssertionError(f"{field} must be an int in {event}.")
    return value


def sum_list_lengths(events: list[dict[str, Any]], field: str) -> int:
    return sum(len(int_list(event, field)) for event in events)


def sum_ints(events: list[dict[str, Any]], field: str) -> int:
    return sum(int_value(event, field) for event in events)


def summarize_tiered_events(
    *,
    ratio: TierRatio,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    tiered_events = [
        event
        for event in events
        if event.get("event") == "recovery_materialized"
        and event.get("precision_tiering_enabled") is True
    ]
    if not tiered_events:
        raise AssertionError(f"{ratio.display}: no tiered recovery events found.")

    missing_fp16_count = sum_list_lengths(tiered_events, "missing_fp16_block_ids")
    missing_int8_count = sum_list_lengths(tiered_events, "missing_int8_block_ids")
    missing_int4_count = sum_list_lengths(tiered_events, "missing_int4_block_ids")
    if missing_fp16_count or missing_int8_count or missing_int4_count:
        raise AssertionError(
            f"{ratio.display}: missing tier payloads observed "
            f"(fp16={missing_fp16_count}, int8={missing_int8_count}, "
            f"int4={missing_int4_count})."
        )

    effective_bytes = sum_ints(tiered_events, "effective_recovery_transfer_bytes")
    if effective_bytes <= 0:
        raise AssertionError(
            f"{ratio.display}: effective_recovery_transfer_bytes did not "
            "become positive."
        )

    tier_fp16_count = sum_list_lengths(tiered_events, "tier_fp16_block_ids")
    tier_int8_count = sum_list_lengths(tiered_events, "tier_int8_block_ids")
    tier_int4_count = sum_list_lengths(tiered_events, "tier_int4_block_ids")
    tier_skip_count = sum_list_lengths(tiered_events, "tier_skip_block_ids")
    recovered_fp16_count = sum_list_lengths(tiered_events, "recovered_fp16_block_ids")
    recovered_int8_count = sum_list_lengths(tiered_events, "recovered_int8_block_ids")
    recovered_int4_count = sum_list_lengths(tiered_events, "recovered_int4_block_ids")
    int4_payload_bytes = sum_ints(tiered_events, "int4_payload_bytes")
    int4_scale_bytes = sum_ints(tiered_events, "int4_scale_bytes")
    if ratio.skip <= 1e-9 and tier_skip_count != 0:
        raise AssertionError(
            f"{ratio.display}: no-skip ratio unexpectedly produced "
            f"{tier_skip_count} skip assignments."
        )
    if ratio.skip > 1e-9 and tier_skip_count <= 0:
        raise AssertionError(
            f"{ratio.display}: skip ratio > 0 but no skip assignments "
            "appeared."
        )
    if ratio.fp16 > 0.0 and (tier_fp16_count <= 0 or recovered_fp16_count <= 0):
        raise AssertionError(
            f"{ratio.display}: fp16 ratio > 0 but no fp16 tier/recovery "
            "ids appeared."
        )
    if ratio.int8 > 0.0 and (tier_int8_count <= 0 or recovered_int8_count <= 0):
        raise AssertionError(
            f"{ratio.display}: int8 ratio > 0 but no int8 tier/recovery "
            "ids appeared."
        )
    if ratio.int4 > 0.0 and (tier_int4_count <= 0 or recovered_int4_count <= 0):
        raise AssertionError(
            f"{ratio.display}: int4 ratio > 0 but no int4 tier/recovery "
            "ids appeared."
        )
    if ratio.int4 > 0.0 and (int4_payload_bytes <= 0 or int4_scale_bytes <= 0):
        raise AssertionError(
            f"{ratio.display}: int4 ratio > 0 but INT4 byte accounting was "
            f"not positive (payload={int4_payload_bytes}, "
            f"scale={int4_scale_bytes})."
        )

    return {
        "tiered_recovery_event_count": len(tiered_events),
        "recovery_materialized_event_count": count_events(
            events, "recovery_materialized"
        ),
        "tier_fp16_count": tier_fp16_count,
        "tier_int8_count": tier_int8_count,
        "tier_int4_count": tier_int4_count,
        "tier_skip_count": tier_skip_count,
        "recovered_fp16_count": recovered_fp16_count,
        "recovered_int8_count": recovered_int8_count,
        "recovered_int4_count": recovered_int4_count,
        "missing_fp16_count": missing_fp16_count,
        "missing_int8_count": missing_int8_count,
        "missing_int4_count": missing_int4_count,
        "fp16_payload_bytes": sum_ints(tiered_events, "fp16_payload_bytes"),
        "int8_payload_bytes": sum_ints(tiered_events, "int8_payload_bytes"),
        "int4_payload_bytes": int4_payload_bytes,
        "int4_scale_bytes": int4_scale_bytes,
        "effective_recovery_transfer_bytes": effective_bytes,
        "recovered_bytes": sum_ints(tiered_events, "recovered_bytes"),
    }


def print_run_summary(
    *,
    ratio: TierRatio,
    summary: dict[str, Any],
    generated_text: str,
    show_full_text: bool,
    preview_chars: int,
) -> None:
    print("")
    print(f"ratio {ratio.display} passed")
    print(f"  generated_token_count: {summary['generated_token_count']}")
    mismatch = summary["first_token_mismatch_vs_baseline"]
    print(
        "  first_token_mismatch_vs_baseline: "
        f"{'none' if mismatch is None else mismatch}"
    )
    print(
        "  text_common_prefix_chars_vs_baseline: "
        f"{summary['text_common_prefix_chars_vs_baseline']}"
    )
    print(
        "  tier_counts: "
        f"fp16={summary['tier_fp16_count']}, "
        f"int8={summary['tier_int8_count']}, "
        f"int4={summary['tier_int4_count']}, "
        f"skip={summary['tier_skip_count']}"
    )
    print(
        "  recovered_counts: "
        f"fp16={summary['recovered_fp16_count']}, "
        f"int8={summary['recovered_int8_count']}, "
        f"int4={summary['recovered_int4_count']}"
    )
    print(
        "  bytes: "
        f"fp16_payload={summary['fp16_payload_bytes']}, "
        f"int8_payload={summary['int8_payload_bytes']}, "
        f"int4_payload={summary['int4_payload_bytes']}, "
        f"int4_scale={summary['int4_scale_bytes']}, "
        f"effective_transfer={summary['effective_recovery_transfer_bytes']}"
    )
    print(f"  log_path: {summary['log_path']}")
    print(f"  debug_dir: {summary['debug_dir']}")
    print(f"  validator_log_path: {summary['validator_log_path']}")
    print("  generated_text:")
    print(generated_text if show_full_text else preview_text(generated_text, preview_chars))


def main() -> None:
    args = parse_args()
    ratios = parse_tier_ratios(
        args.tier_ratios,
        include_skip_ratio=args.include_skip_ratio,
    )

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mpr_m4_tiered_", dir="/tmp"))
    work_dir.mkdir(parents=True, exist_ok=True)

    baseline_log = work_dir / "baseline_off.log"
    run_generation(
        name="baseline_off",
        args=args,
        env=clean_mpr_env(),
        log_path=baseline_log,
    )
    baseline_ids = extract_generated_token_ids(baseline_log)
    if not baseline_ids:
        raise AssertionError("baseline generated_token_ids was empty.")
    baseline_text = extract_generated_text(baseline_log)

    print("")
    print("MPR-off baseline completed")
    print(f"work_dir: {work_dir}")
    print(f"baseline_log_path: {baseline_log}")
    print(f"baseline_generated_token_count: {len(baseline_ids)}")
    print("baseline_generated_text:")
    print(baseline_text if args.show_full_text else preview_text(
        baseline_text, args.text_preview_chars))

    ratio_summaries: list[dict[str, Any]] = []
    for ratio in ratios:
        debug_dir = work_dir / f"debug_{ratio.label}"
        prepare_debug_dir(debug_dir)
        log_path = work_dir / f"tiered_{ratio.label}.log"
        validator_log = work_dir / f"validator_{ratio.label}.log"

        run_generation(
            name=f"tiered_{ratio.display}",
            args=args,
            env=ratio_env(args=args, ratio=ratio, debug_dir=debug_dir),
            log_path=log_path,
        )
        generated_ids = extract_generated_token_ids(log_path)
        if not generated_ids:
            raise AssertionError(f"{ratio.display}: generated_token_ids was empty.")
        generated_text = extract_generated_text(log_path)

        events = load_jsonl_events(debug_dir)
        run_jsonl_validator(
            args=args,
            ratio=ratio,
            debug_dir=debug_dir,
            log_path=validator_log,
        )
        event_summary = summarize_tiered_events(ratio=ratio, events=events)
        mismatch = first_mismatch_index(baseline_ids, generated_ids)
        text_prefix = common_text_prefix_len(baseline_text, generated_text)

        summary = {
            "ratio": ratio.display,
            "fp16_ratio": ratio.fp16,
            "int8_ratio": ratio.int8,
            "int4_ratio": ratio.int4,
            "skip_ratio": ratio.skip,
            "generated_token_count": len(generated_ids),
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
            "generated_text_preview": preview_text(
                generated_text,
                args.text_preview_chars,
            ),
            "first_token_mismatch_vs_baseline": mismatch,
            "text_common_prefix_chars_vs_baseline": text_prefix,
            "log_path": str(log_path),
            "debug_dir": str(debug_dir),
            "validator_log_path": str(validator_log),
            **event_summary,
        }
        ratio_summaries.append(summary)
        print_run_summary(
            ratio=ratio,
            summary=summary,
            generated_text=generated_text,
            show_full_text=args.show_full_text,
            preview_chars=args.text_preview_chars,
        )

    output_summary = {
        "work_dir": str(work_dir),
        "baseline_log_path": str(baseline_log),
        "baseline_generated_token_count": len(baseline_ids),
        "baseline_generated_token_ids": baseline_ids,
        "baseline_generated_text": baseline_text,
        "baseline_generated_text_preview": preview_text(
            baseline_text,
            args.text_preview_chars,
        ),
        "ratio_summaries": ratio_summaries,
        "skip_correctness_validated": False,
        "skip_correctness_note": (
            "Step 4.8 only observes optional skip assignment/accounting. "
            "Step 4.9 validates degraded-residency skip behavior."
        ),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(output_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("")
        print(f"summary_json: {args.summary_json}")

    print("")
    print("MPR M4 tiered recovery ratio sweep smoke passed")


if __name__ == "__main__":
    main()
