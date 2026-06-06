# vLLM × ArkVale 기반 Mixed-Precision Recovery 큰 그림

## Collaboration and Execution Rules

These rules apply to all milestones in this MPR project.

1. Before starting each new milestone step, identify any design or policy
   decisions and ask the user to decide instead of choosing alone.
2. Before making implementation changes or creating a commit, brief the user on
   the concrete implementation plan.
3. Only proceed with implementation and commit after the user explicitly agrees
   with the plan, for example by saying to implement it as described.
4. Record important decisions and validation results in the relevant markdown
   progress documents so the context survives conversation compaction.
5. Treat scoring, recall policy, request/block ownership, serving behavior, and
   ArkVale kernel integration as design-sensitive areas that require explicit
   user confirmation before code changes.

## 1. 연구 Framing

이 작업의 목표는 기존 offloading 기반 LLM inference 연구를 다음 단계로 확장하는 것이다.

기존 계보는 대략 다음과 같이 볼 수 있다.

| System | 핵심 질문 | 한계 |
|---|---|---|
| FlexGen | 제한된 GPU memory에서 weight/KV/activation을 어디에 둘 것인가 | KV access가 대체로 bulk movement 중심 |
| InfiniGen | offloaded KV 중 어떤 token을 미리 가져올 것인가 | fetch decision이 binary에 가까움 |
| ArkVale | evicted KV를 page score 기반으로 recall할 수 있는가 | precision-aware recovery는 없음 |
| Ours | 어떤 page를, 언제, 어떤 precision으로 recover할 것인가 | 구현 대상 |

우리의 framing은 **selective KV recovery를 mixed-precision recovery로 일반화**하는 것이다.

즉, 기존 시스템이 `fetch / not fetch`, `evict / keep` 같은 binary decision에 가까웠다면, 우리는 page importance score를 이용해 다음과 같이 결정한다.

```text
high score   -> recover at high precision
medium score -> recover at lower precision
low score    -> skip or keep compressed/offloaded
```

이렇게 하면 GPU memory pressure와 CPU-GPU transfer cost를 줄이면서도, attention에 중요한 KV page는 더 높은 fidelity로 복구할 수 있다.

## 2. 왜 vLLM 기반인가

이번 구현의 1차 기반은 vLLM으로 둔다.

이유는 다음과 같다.

1. vLLM은 실제 serving stack에 가까운 paged KV cache, block table, scheduler, attention metadata를 이미 갖고 있다.
2. 최근 vLLM 코드에는 `vllm/v1/simple_kv_offload`와 `vllm/v1/kv_offload` 경로가 있어 CPU offload/reload 실험을 붙일 여지가 있다.
3. mixed-precision recovery는 본질적으로 token-level algorithm보다 block/page-level KV cache system 기능에 가깝다.
4. ArkVale도 page-level digest scoring과 CPU recall 구조를 제공하므로, vLLM의 block abstraction과 개념적으로 잘 맞는다.

단, vLLM은 구현 난이도가 높다. 그래서 초기 milestone은 작게 잡고, 너무 복잡하면 InfiniGen 기반 prototype으로 pivot할 수 있게 한다.

## 3. ArkVale에서 참고할 부분

ArkVale를 그대로 이식하지는 않는다. ArkVale의 HuggingFace adapter와 `InferState`는 자체 KV pool, digest pool, CPU backup, attention wrapper를 모두 들고 가는 구조라 vLLM과 직접 맞물리기 어렵다.

대신 다음 아이디어와 구현 조각을 reference로 사용한다.

| ArkVale component | 사용할 내용 |
|---|---|
| `source/arkvale/kv_cache.py` | page pool, page id mapping, recallable KV cache abstraction |
| `source/arkvale/infer_state.py` | digest 생성, score estimation, select, recall 흐름 |
| `source/arkvale/kernels.py` | `estimate_scores`, `select_topk`, paged KV kernel wrapper |
| `source/arkvale_cpp/src/estimate.cu` | page digest 기반 score estimation kernel |

특히 중요한 algorithmic pieces:

1. page 내 key vectors를 digest로 summarize
2. query와 digest 사이의 approximate page importance score 계산
3. score에 따라 recall target page 선택
4. CPU backup에서 GPU KV cache page로 recall

우리 쪽에서는 3번과 4번을 mixed precision-aware하게 바꾼다.

## 4. vLLM에서 붙일 후보 위치

현재 관찰한 vLLM 쪽 후보는 다음과 같다.

| Area | 역할 | 우리 작업과의 관계 |
|---|---|---|
| `vllm/v1/core/kv_cache_manager.py` | request별 block allocation/management | block lifecycle metadata를 붙일 후보 |
| `vllm/v1/worker/gpu/model_runner.py` | model execution entry | query/key 관찰 hook 후보 |
| `vllm/v1/attention/backend.py` | attention metadata/backend interface | score-only path 연결 후보 |
| `vllm/v1/attention/backends/*` | 실제 paged attention backend | 최종 attention integration 대상 |
| `vllm/v1/simple_kv_offload/*` | CPU offload block copy | CPU backup/recovery prototype 후보 |
| `vllm/v1/kv_offload/*` | 더 일반적인 offload abstraction | 장기적인 system integration 후보 |

처음부터 attention kernel을 수정하지 않는다. 먼저 sidecar metadata와 score-only path를 만든 뒤, recovery policy를 붙이는 순서로 간다.

## 5. Proposed Architecture

초기 구조는 vLLM의 기존 KV cache를 건드리는 범위를 최소화하고, ArkVale-style digest/recovery state를 sidecar로 둔다.

```text
vLLM KV cache
  ├─ GPU KV blocks
  ├─ block table
  └─ scheduler-managed block lifecycle

Mixed-Precision Recovery sidecar
  ├─ digest cache per layer
  ├─ CPU fp16 backup per block/page
  ├─ page importance scores
  ├─ precision policy
  └─ recovery metadata
```

초기에는 sidecar가 vLLM block id를 key로 삼는다.

```text
(layer_id, physical_block_id) -> digest
(layer_id, physical_block_id) -> CPU backup location
(request_id, logical_block_idx) -> score / precision tier
```

## 6. Milestones

### Milestone 0: Code Cartography

목표:
- vLLM decode path에서 query, key, block table, seq len이 어디서 만들어지고 전달되는지 확인
- simple KV offload가 실제로 어떤 metadata로 GPU/CPU block copy를 예약하는지 확인
- ArkVale kernel을 그대로 쓸 수 있는지, 아니면 PyTorch prototype이 먼저 필요한지 판단

산출물:
- `vllm_integration_notes.md`
- hook 후보 파일/함수 목록

### Milestone 1: Score-Only Prototype

목표:
- vLLM decode step에서 query와 current block table을 관찰
- block/page digest cache를 sidecar로 유지
- ArkVale-style page score를 계산
- 아직 recovery는 하지 않음

성공 기준:
- decode마다 `[batch/request, heads/groups, pages]` score tensor를 얻는다
- score가 sequence length와 block table에 맞게 정렬된다
- attention output은 기존 vLLM과 동일하게 유지된다

가능한 구현:
- 처음에는 ArkVale CUDA kernel 대신 PyTorch score 계산으로 시작
- correctness가 확인되면 `arkvale_cpp.estimate_scores` 또는 vLLM custom op로 이전

### Milestone 2: CPU fp16 Backup

목표:
- GPU KV block이 생성될 때 CPU fp16 backup을 sidecar에 저장
- vLLM block id와 CPU backup location을 매핑
- request 종료나 block free 시 backup도 release

성공 기준:
- block lifecycle과 backup lifecycle이 맞는다
- CPU memory usage가 예측 가능하다
- backup copy가 decode critical path를 크게 막지 않는다

### Milestone 3: Full-Precision Recovery

목표:
- score가 높은 page/block을 CPU backup에서 GPU KV cache로 recover
- 아직 mixed precision은 하지 않고 fp16 recovery만 구현
- ArkVale의 binary recall과 유사한 baseline을 vLLM 위에 만든다

성공 기준:
- selected block이 실제 attention 전에 GPU KV cache에 존재한다
- 기존 offload path와 충돌하지 않는다
- output sanity check를 통과한다

### Milestone 4: Mixed-Precision Recovery

목표:
- score를 precision tier로 매핑
- high score page는 fp16, medium score page는 lower precision, low score page는 skip
- transfer bytes와 GPU memory pressure를 줄인다

성공 기준:
- M3 fp16 recovery semantic smoke는 계속 통과한다
- score threshold가 precision tier 선택으로 이어진다
- lower-precision backup/materialization policy가 적어도 하나 구현된다
- baseline fp16 recovery 대비 transfer/storage bytes 감소를 측정할 수 있다

초기 precision policy:

```text
score >= tau_high -> fp16 recovery
score >= tau_low  -> int8/fp8 recovery
otherwise         -> skip
```

처음에는 attention kernel이 mixed precision page를 직접 읽도록 만들지 않는다. 대신 다음 중 하나로 시작한다.

1. CPU에서 lower precision으로 저장하고 GPU recovery 시 dequantize해서 기존 KV cache에 씀
2. GPU에 별도 low-precision staging buffer를 만들고, attention 전 fp16으로 materialize
3. vLLM의 existing FP8 KV cache support를 활용할 수 있는지 조사

진짜 mixed-precision attention kernel은 마지막 단계로 미룬다.

### Milestone 5: Recovery Cleanup and Optimization

목표:
- Milestone 4의 mixed-precision recovery skeleton을 유지하면서 hot-path overhead를 줄인다
- validation-only fault injection과 production recovery path를 더 명확히 분리한다
- 현재 환경에서 불필요한 debug/scoring/recovery 비용을 덜어낸다
- recovery/backup path의 비용 구조를 측정 가능하게 만든다

성공 기준:
- M4 mixed-precision semantic smoke는 계속 통과한다
- baseline 대비 decode overhead가 명확히 줄거나, 최소한 overhead breakdown이 분리된다
- recovered bytes, recovery copy wall time, per-token recovered bytes를 smoke/benchmark에서 확인할 수 있다
- Sidecar가 scoring, recovery, debug, fault-injection 책임을 지금보다 명확히 나눈다

우선순위:

```text
1. Recovery materialization 최적화
   - per-block Python loop 비용 측정
   - threshold selection이 많은 block을 고를 때 batched copy/index_copy_ 검토
   - pinned CPU memory / non_blocking copy / DMA-friendly layout 가능성 조사

2. Scoring/recovery hot-path 비용 덜어내기
   - debug용 tensor -> CPU/list 변환 최소화
   - recovery enabled일 때 중복 score/debug work 제거
   - smoke용 fault injection이 production entrypoint에 섞이지 않도록 정리

3. Sidecar 구조 정리
   - scoring context
   - CPU backup lifecycle
   - recovery materialization
   - debug event emission
   - validation-only mutation/fault injection
   를 분리 가능한 경계로 재배치

4. Measurement 정리
   - MPR debug event 기반 recovered_bytes/recovery_copy_wall_ms
   - latency benchmark에서 backup/scoring/recovery overhead 분리
   - 필요 시 nsys/dmon 기반 PCIe traffic 관측 절차 문서화
```

Milestone 5는 새 mixed precision 기능을 넓히는 단계가 아니라, M4 skeleton을
실제 다음 단계로 가져갈 수 있게 만드는 cleanup/optimization milestone이다.

## 7. Pivot Plan: InfiniGen 기반 Prototype

vLLM integration이 너무 무거우면 InfiniGen 기반으로 pivot한다.

InfiniGen에는 이미 다음이 있다.

1. selected KV index 생성
2. selected KV만 prefetch
3. FlexGen 기반 비교 실험 구조

따라서 InfiniGen pivot 시에는 token-level mixed precision recovery를 먼저 구현한다.

```text
InfiniGen prefetch_idx
  -> attach or recompute score
  -> split into high/medium tiers
  -> fetch selected KV with different precision policies
```

이 경우 장점은 빠른 prototype이고, 단점은 실제 vLLM-style paged serving system과 거리가 있다는 점이다.

## 8. 주요 Risk

### Risk 1: vLLM block lifecycle complexity

vLLM은 scheduler가 block allocation/free/preemption을 관리한다. sidecar digest/backup이 이 lifecycle과 어긋나면 stale metadata가 생긴다.

대응:
- 첫 구현은 single request, no preemption, fixed block size 조건에서 시작
- 그 다음 multi-request로 확장

### Risk 2: attention backend 다양성

vLLM은 FlashAttention, FlashInfer, Triton, MLA 등 backend가 다양하다.

대응:
- 한 backend만 target한다
- 가능하면 block table 기반 decode path가 명확한 backend를 고른다

### Risk 3: mixed precision page를 attention kernel이 바로 읽기 어려움

기존 attention kernel은 보통 uniform KV dtype/layout을 가정한다.

대응:
- 초기에는 recovery 결과를 기존 KV cache dtype으로 materialize
- mixed dtype direct attention은 후속 최적화로 둔다

### Risk 4: ArkVale kernel contract mismatch

ArkVale의 `estimate_scores`는 ArkVale의 own `KvCache` metadata shape을 전제로 한다.

대응:
- 먼저 PyTorch 구현으로 score correctness를 확인
- 이후 vLLM block table을 ArkVale kernel input format으로 변환

## 9. 현재 추천 Starting Point

가장 먼저 할 일은 구현이 아니라 code cartography다.

구체적으로는 다음 순서로 본다.

1. vLLM decode attention path에서 `query`, `kv_cache`, `block_table`, `seq_lens`가 어디서 만나는지 확인
2. `simple_kv_offload`가 scheduler metadata를 worker copy로 어떻게 넘기는지 확인
3. ArkVale `InferState`의 digest/recall 흐름을 vLLM block id 기준으로 다시 그리기
4. score-only sidecar의 최소 API 정의

초기 sidecar API sketch:

```python
class RecoverySidecar:
    def observe_kv_write(layer_id, block_ids, keys, values): ...
    def build_or_update_digest(layer_id, block_ids): ...
    def estimate_scores(layer_id, query, block_table, seq_lens): ...
    def choose_precision(scores): ...
    def request_recovery(layer_id, block_ids, precision_tiers): ...
    def release_blocks(layer_id, block_ids): ...
```

## 10. One-Sentence Contribution

We extend offloading-based LLM inference from binary KV fetch/evict decisions to page-level, score-guided mixed-precision recovery, using lightweight digest scoring to recover only the KV pages that matter and only at the precision they warrant.
