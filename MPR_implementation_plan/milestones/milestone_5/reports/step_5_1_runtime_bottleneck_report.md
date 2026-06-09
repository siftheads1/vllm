# Step 5.1 Runtime Bottleneck Report

This report records the first runtime baseline received for Milestone 5 Step
5.1 and turns it into concrete profiling and optimization targets.

## Measurement Context

Runtime output directory recorded in the summary:

```text
/workspace/mpr_m5_step51_20260609_150531
```

Decode-only latency is the primary metric. The all-step metrics include prefill
and are not the main M5 optimization target.

## Runtime Summary

```text
mode             decode mean   median    p95      max       tok/s   mean vs baseline
baseline          25.00 ms    22.02    37.14     52.29    39.46       1.00x
mpr_enable_only   43.06 ms    40.86    54.66     66.24    23.03       1.72x
backup_only       42.77 ms    40.46    54.85     78.82    23.18       1.71x
scoring_only      90.53 ms    95.10   103.29    181.33    11.01       3.62x
fp16_recovery    126.45 ms   132.28   168.42    206.03     7.89       5.06x
mixed_int8       204.79 ms   153.45   865.31   1757.53     4.88       8.19x
mixed_int4       296.15 ms   202.45  1398.78   3348.17     3.38      11.85x
```

## Incremental Interpretation

```text
baseline -> mpr_enable_only:
  +18.07 ms decode mean
  MPR observe/digest bookkeeping is already a major hot-path cost.

mpr_enable_only -> backup_only:
  -0.29 ms decode mean
  fp16 backup-only cost is not visible above observe/digest overhead in this
  run. Do not treat CPU backup as free until synthetic copy/backup results and
  targeted backup timing confirm it.

backup_only -> scoring_only:
  +47.76 ms decode mean
  query scoring is the largest isolated cost after the base observe path.

scoring_only -> fp16_recovery:
  +35.92 ms decode mean
  fp16 recovery materialization is a clear second-stage cost.

fp16_recovery -> mixed_int8:
  +78.33 ms decode mean, but only +21.17 ms median
  the mixed INT8 path is dominated by tail stalls.

mixed_int8 -> mixed_int4:
  +91.37 ms decode mean and +49.00 ms median
  INT4 reference materialization/packing path adds both steady and tail cost.
```

Outlier status:

```text
baseline: outlier_rerun_recommended=true
mixed_int8: outlier_rerun_recommended=true
mixed_int4: outlier_rerun_recommended=true
```

The baseline outlier is small compared with mixed-tier tails. The important
follow-up is to inspect `mixed_int8_measured_*.csv` and
`mixed_int4_measured_*.csv` for spike localization, then rerun those modes with
`--warmup-runs 3 --measured-runs 5` only if the spikes look warmup-related.

## Profile And Optimization Targets

### Target 1: MPR Observe/Digest Base Overhead

Evidence:

```text
mpr_enable_only is 1.72x baseline and adds +18.07 ms decode mean even with
CPU backup, scoring, recovery, and debug JSONL disabled.
```

Profile next:

```text
split timings inside RecoverySidecar.observe_kv_write:
  slot_mapping detach/reshape
  invalid/PAD slot filtering
  block id and block offset derivation
  GPU->CPU sync points: any().item(), min().item(), max().item(),
    unique().detach().cpu().tolist()
  _observe_block_offsets bookkeeping
  summarize_key_block digest creation
  _record/_should_record gating when debug_dir is unset
```

Optimization candidates:

```text
avoid digest creation when scoring/recovery do not need digests
keep backup-only mode from paying digest/scoring metadata cost
reduce or batch GPU->CPU scalar/list synchronization from slot mapping
replace per-step Python set/dict block-offset tracking with cheaper
  boundary-aware state
cache or reuse per-layer block metadata where vLLM already has it
keep debug-field construction out of the hot path when JSONL is disabled
```

### Target 2: Scoring Path

Evidence:

```text
backup_only -> scoring_only adds +47.76 ms decode mean.
```

Profile next:

```text
split timings inside _estimate_query_scores:
  query-window append/build
  request/block context extraction
  candidate block id selection
  Quest/Torch scoring backend wall time
  score aggregation/topk
  persistent packed metadata use vs repacking
  score debug field construction and CPU conversions
```

Optimization candidates:

```text
avoid duplicate scoring and duplicate candidate metadata packing
make persistent packed Quest metadata the default fast path when valid
avoid item()/cpu().tolist() score/debug conversions unless JSONL requires them
reduce score granularity or candidate count for the measurement policy if
  correctness allows it
```

### Target 3: FP16 Recovery Materialization

Evidence:

```text
scoring_only -> fp16_recovery adds +35.92 ms decode mean.
```

Profile next:

```text
split BlockRecoveryManager.materialize_blocks:
  selected block id construction
  CPU backup lookup
  CPU->GPU copy wall time
  kv_cache write/indexing wall time
  recovered_bytes and block count per event
```

Optimization candidates:

```text
batch selected CPU blocks into fewer H2D transfers
evaluate pinned CPU memory and non_blocking copy
separate copy wall time from Python loop/indexing overhead
preserve fp16 recovery semantics while reducing per-block loop overhead
```

### Target 4: Mixed INT8/INT4 Tail Latency

Evidence:

```text
mixed_int8 p95/max are 865.31 ms / 1757.53 ms.
mixed_int4 p95/max are 1398.78 ms / 3348.17 ms.
boundary decode means are not worse than non-boundary means, so the tail is
not obviously isolated to predicted block-boundary steps.
```

Profile next:

```text
inspect top-latency rows in mixed_int8_measured_*.csv and
  mixed_int4_measured_*.csv
correlate spikes with recovered block counts, tier counts, and
  recovery_copy_wall_ms from debug smoke JSONL
separately time EagerRecoveryPayloadProvider.fetch and
  BlockRecoveryManager.materialize_tiered_payloads
split INT8/INT4 dequant/unpack/materialize costs
verify whether eager low-precision payload creation during backup is creating
  occasional long stalls
```

Optimization candidates:

```text
vectorize/batch tiered materialization instead of looping per payload
defer low-precision payload creation until recovery asks for the tier, or move
  it to a background path with readiness/fallback semantics
replace reference PyTorch INT4 unpack/dequant materialization with a fused or
  batched implementation if INT4 remains in scope
keep top-ratio policy stable while measuring; do not change threshold semantics
  as part of this cleanup
```

## Immediate Follow-Up

```text
1. Collect or inspect mixed_int8/mixed_int4 measured CSVs to localize tail
   spikes by step index and boundary status.
2. Run synthetic CPU copy/backup/scoring microbenchmarks to separate copy,
   backup-store, and scorer costs from end-to-end engine overhead.
3. Add targeted internal timing only after deciding the smallest set of probes
   needed for observe_kv_write, scoring, and materialization.
4. Do not start cleanup edits until the next profile target is selected and
   confirmed.
```
