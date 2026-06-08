#!/usr/bin/env python3
"""Run an M4 degraded-residency skip-tier ratio sweep smoke.

This smoke uses validation-only ``zero_selected`` mutation to simulate old KV
blocks being degraded before tiered recovery. FP16/INT8 tiers are materialized
from CPU backup payloads, while skip-tier blocks must remain degraded and
unrecovered. Output differences are reported for inspection only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mpr_compare_generation_outputs import extract_generated_token_ids
from scripts.mpr_smoke_recovery_quality import (
    clean_mpr_env,
    common_text_prefix_len,
    extract_generated_text,
    first_mismatch_index,
    load_jsonl_events,
    run_generation,
)
from scripts.mpr_smoke_tiered_recovery import (
    TierRatio,
    int_list,
    parse_tier_ratios,
    prepare_debug_dir,
    preview_text,
    sum_ints,
    sum_list_lengths,
)


DEFAULT_DEGRADED_TIER_RATIOS = (
    "0.25:0.25",
    "0.25:0.50",
    "0.50:0.25",
    "0.10:0.25",
    "0.25:0.10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MPR M4 degraded-residency skip-tier smoke."
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
        help=(
            "Directory for logs and debug JSONL. Defaults to "
            "/tmp/mpr_m4_degraded_*."
        ),
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
        default=",".join(DEFAULT_DEGRADED_TIER_RATIOS),
        help=(
            "Comma-separated FP16:INT8 ratio pairs. Every ratio must leave "
            "room for skip tier. Defaults to "
            + ",".join(DEFAULT_DEGRADED_TIER_RATIOS)
            + "."
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
        help="Print full generated_text for baseline and each degraded run.",
    )
    return parser.parse_args()


def parse_degraded_ratios(ratio_text: str) -> list[TierRatio]:
    ratios = parse_tier_ratios(ratio_text, include_skip_ratio=False)
    for ratio in ratios:
        if ratio.skip <= 1e-9:
            raise ValueError(
                "Step 4.9 degraded-residency ratios must leave skip tier "
                f"capacity, got {ratio.display}."
            )
    return ratios


def degraded_env(
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
            "VLLM_MPR_BACKUP_STORAGE_MODE": "eager_fp16_int8",
            "VLLM_MPR_SCORING_ENABLE": "1",
            "VLLM_MPR_RECOVERY_ENABLE": "1",
            "VLLM_MPR_RECOVERY_POLICY": "threshold_block",
            "VLLM_MPR_RECOVERY_THRESHOLD": "-1e30",
            "VLLM_MPR_RECOVERY_TOPK": "1",
            "VLLM_MPR_PRECISION_TIERING_ENABLE": "1",
            "VLLM_MPR_PRECISION_POLICY": "top_ratio",
            "VLLM_MPR_TIER_FP16_RATIO": str(ratio.fp16),
            "VLLM_MPR_TIER_INT8_RATIO": str(ratio.int8),
            "VLLM_MPR_RECOVERY_TEST_MUTATE": "zero_selected",
            "VLLM_MPR_RECOVERY_TEST_MODE": "recover",
            "VLLM_MPR_RECENT_TOKENS": str(args.recent_tokens),
            "VLLM_MPR_DEBUG_DIR": str(debug_dir),
            "VLLM_MPR_MAX_LAYERS": str(args.debug_max_layers),
            "VLLM_MPR_MAX_STEPS": str(args.debug_max_steps),
        }
    )
    return env


def run_jsonl_validator(
    *,
    args: argparse.Namespace,
    debug_dir: Path,
    log_path: Path,
) -> None:
    paths = sorted(debug_dir.glob("*.jsonl"))
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
        "--require-tiered-skip-unrecovered",
        "--show",
        "5",
    ]
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


def summarize_degraded_events(
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
    if missing_fp16_count or missing_int8_count:
        raise AssertionError(
            f"{ratio.display}: missing tier payloads observed "
            f"(fp16={missing_fp16_count}, int8={missing_int8_count})."
        )

    effective_bytes = sum_ints(tiered_events, "effective_recovery_transfer_bytes")
    if effective_bytes <= 0:
        raise AssertionError(
            f"{ratio.display}: effective_recovery_transfer_bytes did not "
            "become positive."
        )

    tier_fp16_count = sum_list_lengths(tiered_events, "tier_fp16_block_ids")
    tier_int8_count = sum_list_lengths(tiered_events, "tier_int8_block_ids")
    tier_skip_count = sum_list_lengths(tiered_events, "tier_skip_block_ids")
    recovered_fp16_count = sum_list_lengths(tiered_events, "recovered_fp16_block_ids")
    recovered_int8_count = sum_list_lengths(tiered_events, "recovered_int8_block_ids")
    mutated_count = sum_list_lengths(tiered_events, "recovery_test_mutated_block_ids")
    if mutated_count <= 0:
        raise AssertionError(f"{ratio.display}: no degraded block ids observed.")
    if tier_skip_count <= 0:
        raise AssertionError(
            f"{ratio.display}: no skip tier assignments observed. Increase "
            "generated context or reduce VLLM_MPR_RECENT_TOKENS."
        )
    if ratio.fp16 > 0.0 and recovered_fp16_count <= 0:
        raise AssertionError(
            f"{ratio.display}: fp16 ratio > 0 but no recovered fp16 ids appeared."
        )
    if ratio.int8 > 0.0 and recovered_int8_count <= 0:
        raise AssertionError(
            f"{ratio.display}: int8 ratio > 0 but no recovered int8 ids appeared."
        )

    skip_validating_event_count = 0
    for event in tiered_events:
        skip_ids = set(int_list(event, "tier_skip_block_ids"))
        if not skip_ids:
            continue
        mutated_ids = set(int_list(event, "recovery_test_mutated_block_ids"))
        skipped_ids = set(int_list(event, "skipped_block_ids"))
        recovered_ids = set(int_list(event, "recovered_block_ids"))
        if not skip_ids.issubset(mutated_ids):
            raise AssertionError(
                f"{ratio.display}: skip ids were not all degraded: "
                f"skip={sorted(skip_ids)}, mutated={sorted(mutated_ids)}."
            )
        if not skip_ids.issubset(skipped_ids):
            raise AssertionError(
                f"{ratio.display}: skip ids were not reported as skipped: "
                f"skip={sorted(skip_ids)}, skipped={sorted(skipped_ids)}."
            )
        recovered_skip_ids = skip_ids & recovered_ids
        if recovered_skip_ids:
            raise AssertionError(
                f"{ratio.display}: skip ids were unexpectedly recovered: "
                f"{sorted(recovered_skip_ids)}."
            )
        skip_validating_event_count += 1

    return {
        "tiered_recovery_event_count": len(tiered_events),
        "skip_validating_event_count": skip_validating_event_count,
        "mutated_block_count": mutated_count,
        "tier_fp16_count": tier_fp16_count,
        "tier_int8_count": tier_int8_count,
        "tier_skip_count": tier_skip_count,
        "recovered_fp16_count": recovered_fp16_count,
        "recovered_int8_count": recovered_int8_count,
        "unrecovered_skip_count": tier_skip_count,
        "missing_fp16_count": missing_fp16_count,
        "missing_int8_count": missing_int8_count,
        "fp16_payload_bytes": sum_ints(tiered_events, "fp16_payload_bytes"),
        "int8_payload_bytes": sum_ints(tiered_events, "int8_payload_bytes"),
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
    print(f"degraded ratio {ratio.display} passed")
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
    print(f"  mutated_block_count: {summary['mutated_block_count']}")
    print(
        "  tier_counts: "
        f"fp16={summary['tier_fp16_count']}, "
        f"int8={summary['tier_int8_count']}, "
        f"skip={summary['tier_skip_count']}"
    )
    print(
        "  recovered_counts: "
        f"fp16={summary['recovered_fp16_count']}, "
        f"int8={summary['recovered_int8_count']}, "
        f"unrecovered_skip={summary['unrecovered_skip_count']}"
    )
    print(
        "  bytes: "
        f"fp16_payload={summary['fp16_payload_bytes']}, "
        f"int8_payload={summary['int8_payload_bytes']}, "
        f"effective_transfer={summary['effective_recovery_transfer_bytes']}"
    )
    print(f"  log_path: {summary['log_path']}")
    print(f"  debug_dir: {summary['debug_dir']}")
    print(f"  validator_log_path: {summary['validator_log_path']}")
    print("  generated_text:")
    print(generated_text if show_full_text else preview_text(generated_text, preview_chars))


def main() -> None:
    args = parse_args()
    ratios = parse_degraded_ratios(args.tier_ratios)

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mpr_m4_degraded_", dir="/tmp"))
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
        log_path = work_dir / f"degraded_{ratio.label}.log"
        validator_log = work_dir / f"validator_{ratio.label}.log"

        run_generation(
            name=f"degraded_{ratio.display}",
            args=args,
            env=degraded_env(args=args, ratio=ratio, debug_dir=debug_dir),
            log_path=log_path,
        )
        generated_ids = extract_generated_token_ids(log_path)
        if not generated_ids:
            raise AssertionError(f"{ratio.display}: generated_token_ids was empty.")
        generated_text = extract_generated_text(log_path)

        events = load_jsonl_events(debug_dir)
        run_jsonl_validator(args=args, debug_dir=debug_dir, log_path=validator_log)
        event_summary = summarize_degraded_events(ratio=ratio, events=events)
        mismatch = first_mismatch_index(baseline_ids, generated_ids)
        text_prefix = common_text_prefix_len(baseline_text, generated_text)

        summary = {
            "ratio": ratio.display,
            "fp16_ratio": ratio.fp16,
            "int8_ratio": ratio.int8,
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
        "skip_correctness_validated": True,
        "skip_correctness_note": (
            "Step 4.9 validates simulated degraded-residency skip behavior "
            "with validation-only zero_selected mutation, not real offload "
            "eviction."
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
    print("MPR M4 degraded-residency skip-tier smoke passed")


if __name__ == "__main__":
    main()
