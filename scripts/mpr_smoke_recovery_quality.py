#!/usr/bin/env python3
"""Run a three-way M3 recovery semantic smoke.

The smoke checks:
1. Baseline generation without MPR.
2. Threshold-selected validation mutation without recovery changes output.
3. The same mutation followed by CPU-backup recovery restores baseline output.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mpr_compare_generation_outputs import extract_generated_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MPR M3 mutate-only vs mutate+recover smoke."
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
        help="Directory for logs and debug JSONL. Defaults to /tmp/mpr_m3_quality_*.",
    )
    parser.add_argument(
        "--recent-tokens",
        type=int,
        default=256,
        help=(
            "Recent-token protection for threshold selection. With 512 generated "
            "tokens this targets roughly the older half of the KV blocks."
        ),
    )
    parser.add_argument("--debug-max-layers", type=int, default=2)
    parser.add_argument("--debug-max-steps", type=int, default=2048)
    parser.add_argument(
        "--test-mutate",
        choices=("zero_selected", "zero_all"),
        default="zero_selected",
        help="Default validation-only KV mutation mode for both MPR runs.",
    )
    parser.add_argument(
        "--mutate-only-test-mutate",
        choices=("zero_selected", "zero_all"),
        default=None,
        help="Override validation mutation mode for the mutate-only run.",
    )
    parser.add_argument(
        "--recover-test-mutate",
        choices=("zero_selected", "zero_all"),
        default=None,
        help="Override validation mutation mode for the mutate+recover run.",
    )
    parser.add_argument(
        "--min-mutated-finalized-ratio",
        type=float,
        default=0.35,
        help=(
            "Minimum observed mutated/finalized block ratio required in both "
            "mutate-only and mutate+recover runs."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to the baseline generation script.",
    )
    parser.add_argument(
        "--show-full-text",
        action="store_true",
        help="Print full generated_text for baseline, mutate-only, and recover.",
    )
    parser.add_argument(
        "--expected-answer",
        default=None,
        help=(
            "Optional substring that baseline and mutate+recover outputs must "
            "contain. Defaults to ailikehuman for --prompt-preset name_recall."
        ),
    )
    return parser.parse_args()


def clean_mpr_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("VLLM_MPR_"):
            env.pop(key)
    env["VLLM_USE_V1"] = "1"
    return env


def build_name_recall_prompt() -> str:
    """Build a long prompt with the target name fact near the middle."""
    intro = [
        "Complete the final lookup by copying the exact USER_NAME value from "
        "the notes. The notes contain filler, but one line defines USER_NAME.",
        "The final answer must start with the exact lowercase USER_NAME value.",
        "",
    ]
    before_fact = [
        f"Background note {idx}: This smoke test studies KV cache recovery, "
        "attention context, and deterministic generation. This filler places "
        "the identity fact away from the final question."
        for idx in range(1, 13)
    ]
    fact = [
        "",
        "Critical identity note:",
        "USER_NAME = ailikehuman",
        "This exact lowercase value is the answer requested at the end.",
        "",
    ]
    after_fact = [
        f"Follow-up note {idx}: Continue tracking threshold selection, CPU "
        "backup materialization, protected recent tokens, and mutate-only "
        "failure observation. This filler follows the identity fact."
        for idx in range(1, 13)
    ]
    question = [
        "",
        "Final lookup: copy the USER_NAME value from the earlier critical "
        "identity note.",
        "USER_NAME =",
    ]
    return "\n".join(intro + before_fact + fact + after_fact + question)


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_preset == "name_recall":
        return build_name_recall_prompt()
    return "Explain KV cache in one sentence."


def resolve_expected_answer(args: argparse.Namespace) -> str | None:
    if args.expected_answer is not None:
        return args.expected_answer
    if args.prompt_preset == "name_recall":
        return "ailikehuman"
    return None


def build_generation_cmd(args: argparse.Namespace) -> list[str]:
    prompt = resolve_prompt(args)
    cmd = [
        args.python,
        "scripts/mpr_baseline_qwen3_8b.py",
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--seed",
        str(args.seed),
        "--prompt",
        prompt,
        "--ignore-eos",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def run_generation(
    *,
    name: str,
    args: argparse.Namespace,
    env: dict[str, str],
    log_path: Path,
) -> None:
    cmd = build_generation_cmd(args)
    print(f"running {name}: {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(
                f"{name} failed; see log: {log_path}",
                file=sys.stderr,
            )
            lines = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            print(f"last {min(80, len(lines))} log lines:", file=sys.stderr)
            for line in lines[-80:]:
                print(line, file=sys.stderr)
            raise


def mpr_env(
    *,
    args: argparse.Namespace,
    debug_dir: Path,
    test_mode: str,
    test_mutate: str,
) -> dict[str, str]:
    env = clean_mpr_env()
    env.update(
        {
            "VLLM_MPR_ENABLE": "1",
            "VLLM_MPR_CPU_BACKUP": "1",
            "VLLM_MPR_SCORING_ENABLE": "1",
            "VLLM_MPR_RECOVERY_ENABLE": "1",
            "VLLM_MPR_RECOVERY_POLICY": "threshold_block",
            "VLLM_MPR_RECOVERY_THRESHOLD": "-1e30",
            "VLLM_MPR_RECOVERY_TOPK": "1",
            "VLLM_MPR_RECOVERY_TEST_MUTATE": test_mutate,
            "VLLM_MPR_RECOVERY_TEST_MODE": test_mode,
            "VLLM_MPR_RECENT_TOKENS": str(args.recent_tokens),
            "VLLM_MPR_DEBUG_DIR": str(debug_dir),
            "VLLM_MPR_MAX_LAYERS": str(args.debug_max_layers),
            "VLLM_MPR_MAX_STEPS": str(args.debug_max_steps),
        }
    )
    return env


def load_jsonl_events(debug_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path_text in glob.glob(str(debug_dir / "*.jsonl")):
        path = Path(path_text)
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    return events


def count_events(events: list[dict[str, Any]], event_name: str) -> int:
    return sum(1 for event in events if event.get("event") == event_name)


def has_nonempty_mutation(events: list[dict[str, Any]], event_name: str) -> bool:
    for event in events:
        if event.get("event") != event_name:
            continue
        mutated = event.get("recovery_test_mutated_block_ids", [])
        if isinstance(mutated, list) and mutated:
            return True
    return False


def has_recovered_mutation(events: list[dict[str, Any]]) -> bool:
    for event in events:
        if event.get("event") != "recovery_materialized":
            continue
        mutated = set(event.get("recovery_test_mutated_block_ids", []))
        recovered = set(event.get("recovered_block_ids", []))
        if mutated and mutated.issubset(recovered):
            return True
    return False


def event_mutation_ratio(
    event: dict[str, Any],
) -> tuple[float, int, int, int, int]:
    finalized = event.get("finalized_block_ids", [])
    candidate = event.get("score_candidate_block_ids", [])
    protected = event.get("protected_block_ids", [])
    mutated = event.get("recovery_test_mutated_block_ids", [])
    if not isinstance(finalized, list) or not finalized:
        return 0.0, 0, 0, 0, 0
    if not isinstance(candidate, list):
        candidate = []
    if not isinstance(protected, list):
        protected = []
    if not isinstance(mutated, list):
        mutated = []
    mutated_set = set(int(block_id) for block_id in mutated)
    finalized_set = set(int(block_id) for block_id in finalized)
    mutated_count = len(mutated_set & finalized_set)
    finalized_count = len(finalized_set)
    candidate_count = len(set(int(block_id) for block_id in candidate))
    protected_count = len(set(int(block_id) for block_id in protected))
    ratio = mutated_count / finalized_count
    return ratio, mutated_count, finalized_count, candidate_count, protected_count


def best_mutation_coverage(
    events: list[dict[str, Any]],
    *,
    event_name: str,
) -> tuple[float, int, int, int, int]:
    best = (0.0, 0, 0, 0, 0)
    for event in events:
        if event.get("event") != event_name:
            continue
        coverage = event_mutation_ratio(event)
        if coverage[0] > best[0]:
            best = coverage
    return best


def require_mutation_coverage(
    events: list[dict[str, Any]],
    *,
    event_name: str,
    min_ratio: float,
) -> tuple[float, int, int, int, int]:
    coverage = best_mutation_coverage(events, event_name=event_name)
    ratio, mutated_count, finalized_count, _, _ = coverage
    if ratio < min_ratio:
        raise AssertionError(
            f"{event_name} mutated/finalized ratio too low: "
            f"{ratio:.3f} < {min_ratio:.3f} "
            f"({mutated_count}/{finalized_count} blocks)."
        )
    return coverage


def extract_text_preview(path: Path) -> str:
    text = extract_generated_text(path)
    return text[:240] if text else "<generated_text line not found>"


def extract_generated_text(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_idx, line in enumerate(lines):
        if not line.startswith("generated_text:"):
            continue
        text_lines = [line.split(":", maxsplit=1)[1].strip()]
        for continuation in lines[line_idx + 1:]:
            if continuation.startswith("generated_token_ids:"):
                break
            text_lines.append(continuation)
        return "\n".join(text_lines).strip()
    return ""


def first_mismatch_index(left: list[int], right: list[int]) -> int | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def common_text_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit


def text_window(text: str, center: int, *, radius: int = 180) -> str:
    start = max(0, center - radius)
    end = min(len(text), center + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", "\\n") + suffix


def recovery_copy_summary(events: list[dict[str, Any]]) -> dict[str, float | int]:
    recovery_events = [
        event for event in events if event.get("event") == "recovery_materialized"
    ]
    total_bytes = 0
    total_copy_ms = 0.0
    recovered_block_count = 0
    for event in recovery_events:
        recovered_bytes = event.get("recovered_bytes", 0)
        if isinstance(recovered_bytes, int):
            total_bytes += recovered_bytes
        copy_ms = event.get("recovery_copy_wall_ms", 0.0)
        if isinstance(copy_ms, (int, float)):
            total_copy_ms += float(copy_ms)
        recovered_ids = event.get("recovered_block_ids", [])
        if isinstance(recovered_ids, list):
            recovered_block_count += len(recovered_ids)

    total_gib = total_bytes / float(1024**3)
    copy_seconds = total_copy_ms / 1000.0
    bandwidth_gib_s = total_gib / copy_seconds if copy_seconds > 0 else 0.0
    return {
        "event_count": len(recovery_events),
        "recovered_block_count": recovered_block_count,
        "total_recovered_bytes": total_bytes,
        "total_recovered_gib": total_gib,
        "total_recovery_copy_ms": total_copy_ms,
        "estimated_recovery_copy_gib_s": bandwidth_gib_s,
    }


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="mpr_m3_quality_", dir="/tmp"))
    work_dir.mkdir(parents=True, exist_ok=True)

    baseline_log = work_dir / "baseline_off.log"
    mutate_only_log = work_dir / "mutate_only.log"
    recover_log = work_dir / "mutate_recover.log"
    mutate_debug_dir = work_dir / "debug_mutate_only"
    recover_debug_dir = work_dir / "debug_recover"
    mutate_debug_dir.mkdir(exist_ok=True)
    recover_debug_dir.mkdir(exist_ok=True)

    run_generation(
        name="baseline_off",
        args=args,
        env=clean_mpr_env(),
        log_path=baseline_log,
    )
    run_generation(
        name="mutate_only",
        args=args,
        env=mpr_env(
            args=args,
            debug_dir=mutate_debug_dir,
            test_mode="mutate_only",
            test_mutate=args.mutate_only_test_mutate or args.test_mutate,
        ),
        log_path=mutate_only_log,
    )
    run_generation(
        name="mutate_recover",
        args=args,
        env=mpr_env(
            args=args,
            debug_dir=recover_debug_dir,
            test_mode="recover",
            test_mutate=args.recover_test_mutate or args.test_mutate,
        ),
        log_path=recover_log,
    )

    baseline_ids = extract_generated_token_ids(baseline_log)
    mutate_only_ids = extract_generated_token_ids(mutate_only_log)
    recover_ids = extract_generated_token_ids(recover_log)
    mutate_mismatch = first_mismatch_index(baseline_ids, mutate_only_ids)
    recover_mismatch = first_mismatch_index(baseline_ids, recover_ids)

    if mutate_mismatch is None:
        raise AssertionError(
            "mutate-only output unexpectedly matched baseline token IDs."
        )
    if recover_mismatch is not None:
        raise AssertionError(
            "mutate+recover output did not match baseline token IDs."
        )

    mutate_events = load_jsonl_events(mutate_debug_dir)
    recover_events = load_jsonl_events(recover_debug_dir)
    if count_events(mutate_events, "recovery_test_mutated") < 1:
        raise AssertionError("mutate-only run had no recovery_test_mutated events.")
    if not has_nonempty_mutation(mutate_events, "recovery_test_mutated"):
        raise AssertionError("mutate-only run had no non-empty mutation events.")
    if count_events(recover_events, "recovery_materialized") < 1:
        raise AssertionError("mutate+recover run had no recovery_materialized events.")
    if not has_recovered_mutation(recover_events):
        raise AssertionError(
            "mutate+recover run had no event where mutated IDs were recovered."
        )
    mutate_coverage = require_mutation_coverage(
        mutate_events,
        event_name="recovery_test_mutated",
        min_ratio=args.min_mutated_finalized_ratio,
    )
    recover_coverage = require_mutation_coverage(
        recover_events,
        event_name="recovery_materialized",
        min_ratio=args.min_mutated_finalized_ratio,
    )
    recover_copy = recovery_copy_summary(recover_events)
    baseline_text = extract_generated_text(baseline_log)
    mutate_text = extract_generated_text(mutate_only_log)
    recover_text = extract_generated_text(recover_log)
    expected_answer = resolve_expected_answer(args)
    if expected_answer is not None:
        if expected_answer not in baseline_text:
            raise AssertionError(
                "baseline output did not contain expected answer "
                f"{expected_answer!r}."
            )
        if expected_answer not in recover_text:
            raise AssertionError(
                "mutate+recover output did not contain expected answer "
                f"{expected_answer!r}."
            )

    print("MPR M3 recovery quality smoke passed")
    print(f"work_dir: {work_dir}")
    print(f"baseline_generated_tokens: {len(baseline_ids)}")
    print(f"mutate_only_generated_tokens: {len(mutate_only_ids)}")
    print(f"recover_generated_tokens: {len(recover_ids)}")
    print(f"mutate_only_first_token_mismatch_index: {mutate_mismatch}")
    print("recover_first_token_mismatch_index: none")
    print(f"mutate_only_recovery_test_mutated: "
          f"{count_events(mutate_events, 'recovery_test_mutated')}")
    print(f"recover_recovery_materialized: "
          f"{count_events(recover_events, 'recovery_materialized')}")
    print(
        "mutate_only_best_mutated_finalized_ratio: "
        f"{mutate_coverage[0]:.3f} "
        f"({mutate_coverage[1]}/{mutate_coverage[2]} blocks, "
        f"candidate={mutate_coverage[3]}, protected={mutate_coverage[4]})"
    )
    print(
        "recover_best_mutated_finalized_ratio: "
        f"{recover_coverage[0]:.3f} "
        f"({recover_coverage[1]}/{recover_coverage[2]} blocks, "
        f"candidate={recover_coverage[3]}, protected={recover_coverage[4]})"
    )
    print(
        "recover_copy_summary: "
        f"events={recover_copy['event_count']}, "
        f"blocks={recover_copy['recovered_block_count']}, "
        f"bytes={recover_copy['total_recovered_bytes']}, "
        f"gib={recover_copy['total_recovered_gib']:.6f}, "
        f"copy_ms={recover_copy['total_recovery_copy_ms']:.3f}, "
        f"est_gib_s={recover_copy['estimated_recovery_copy_gib_s']:.3f}"
    )
    print(
        "recover_copy_per_generated_token_bytes: "
        f"{recover_copy['total_recovered_bytes'] / max(1, len(recover_ids)):.1f}"
    )
    if expected_answer is not None:
        print(f"expected_answer: {expected_answer}")
        print(f"baseline_contains_expected: {expected_answer in baseline_text}")
        print(f"mutate_only_contains_expected: {expected_answer in mutate_text}")
        print(f"recover_contains_expected: {expected_answer in recover_text}")
    print("")
    print("baseline preview:")
    print(extract_text_preview(baseline_log))
    print("")
    print("mutate-only preview:")
    print(extract_text_preview(mutate_only_log))
    print("")
    print("mutate+recover preview:")
    print(extract_text_preview(recover_log))

    mutate_text_prefix = common_text_prefix_len(baseline_text, mutate_text)
    recover_text_prefix = common_text_prefix_len(baseline_text, recover_text)
    print("")
    print(f"mutate_only_text_common_prefix_chars: {mutate_text_prefix}")
    print(f"recover_text_common_prefix_chars: {recover_text_prefix}")
    print("")
    print("baseline text around mutate-only divergence:")
    print(text_window(baseline_text, mutate_text_prefix))
    print("")
    print("mutate-only text around divergence:")
    print(text_window(mutate_text, mutate_text_prefix))

    if args.show_full_text:
        print("")
        print("==== baseline full text ====")
        print(baseline_text)
        print("")
        print("==== mutate-only full text ====")
        print(mutate_text)
        print("")
        print("==== mutate+recover full text ====")
        print(recover_text)


if __name__ == "__main__":
    main()
