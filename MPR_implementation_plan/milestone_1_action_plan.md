# Milestone 1 Action Plan

Milestone 1의 목표는 vLLM decode 경로에 **score-only prototype**을 붙이는 것이다.

이 단계에서는 KV cache, attention output, scheduler, CPU offload/reload를 바꾸지 않는다. 목적은 vLLM의 실제 decode step에서 query와 block table을 이용해 ArkVale-style page importance score를 안정적으로 계산할 수 있는지 확인하는 것이다.

## Goal

vLLM v1 + FlashAttention decode 중 다음을 관측하고 기록한다.

```text
layer
request/batch position
query
seq_len
block_table
physical KV block ids
page/block scores
top-k candidate pages
```

성공 기준은 다음과 같다.

1. Decode step마다 layer별 page score를 계산할 수 있다.
2. Score index가 vLLM block table의 physical block id와 일관되게 대응된다.
3. Attention output은 기존 vLLM과 동일하게 유지된다.
4. CPU KV backup layout 없이도 동작한다.
5. 기능은 기본적으로 꺼져 있고, 명시적으로 켰을 때만 동작한다.

## Scope

초기 scope는 의도적으로 좁게 둔다.

```text
vLLM v1
FlashAttention backend
single GPU
single request first
decode path first
no DCP/CP
no speculative decoding
no sliding-window special handling
no recovery
no CPU backup
rolling window_query from recent 64 decode queries
PyTorch scoring first
```

Multi-request batching, ArkVale CUDA kernel 연결, CPU backup, recovery, mixed precision policy는 Milestone 1 이후로 둔다. Chunked prefill과 speculative decoding도 Milestone 1에서는 고려하지 않는다.

Milestone 1에서는 page/block 단위를 다음처럼 단순화해서 본다.

```text
vLLM logical block size
== vLLM physical KV block size
== digest scoring unit
== future recall unit
```

즉 vLLM physical KV block 하나를 score/recovery 후보 page 하나로 취급한다. 이 가정은 초기 metadata alignment를 쉽게 만들기 위한 것이며, 후속 milestone에서 digest page와 recall page를 더 coarse/fine하게 분리할 수 있다.

## Target Files

| File | Role |
|---|---|
| `vllm/model_executor/layers/attention/attention.py` | query/KV hook 위치 |
| `vllm/v1/attention/backends/flash_attn.py` | FlashAttention metadata/layout 확인 |
| `vllm/v1/worker/gpu/block_table.py` | slot mapping과 physical block id semantics |
| `vllm/v1/core/kv_cache_manager.py` | block lifecycle reference |
| `vllm/v1/simple_kv_offload/worker.py` | Milestone 2 CPU layout reference |

새 파일 후보:

```text
vllm/v1/mixed_precision_recovery/__init__.py
vllm/v1/mixed_precision_recovery/config.py
vllm/v1/mixed_precision_recovery/sidecar.py
vllm/v1/mixed_precision_recovery/scoring.py
vllm/v1/mixed_precision_recovery/debug.py
```

이름은 구현 전 다시 조정 가능하지만, sidecar를 attention layer 파일 안에 크게 넣지는 않는다.

## Step 1.0: Baseline 실행 경로 고정

먼저 score-only hook을 붙이기 전에 baseline command와 model/backend 조건을 고정한다.

확인할 것:

1. vLLM이 v1 engine path를 사용하는가
2. Attention backend가 FlashAttention인가
3. KV cache dtype과 block size가 무엇인가
4. CUDA graph/compile 설정이 hook 실험에 너무 불리하지 않은가
5. single request decode를 안정적으로 재현할 수 있는가

산출물:

```text
baseline_command = ...
model = ...
backend = FlashAttention
block_size = ...
kv_cache_dtype = ...
known_disabled_features = DCP/CP/spec/sliding-window-special
```

Completion:

- sidecar 없이 baseline generation이 정상 동작한다.
- 나중에 output non-regression을 비교할 prompt와 decode length를 정한다.

## Step 1.1: Disabled-by-Default Sidecar Scaffold

작은 sidecar module을 만든다. 기본값은 off이다.

초기 API sketch:

```python
class RecoverySidecar:
    def enabled(self) -> bool:
        ...

    def observe_kv_write(
        self,
        layer_name: str,
        key,
        value,
        slot_mapping,
        block_size: int,
    ) -> None:
        ...

    def observe_query(
        self,
        layer_name: str,
        query,
        attn_metadata,
    ) -> None:
        ...

    def estimate_scores(
        self,
        layer_name: str,
        window_query,
        attn_metadata,
        block_size: int,
    ):
        ...
```

초기 설정 방식 후보:

```text
VLLM_MPR_ENABLE=1
VLLM_MPR_DEBUG_DIR=/tmp/vllm_mpr_debug
VLLM_MPR_TOPK=...
VLLM_MPR_MAX_LAYERS=...
```

구현 원칙:

- Env flag가 꺼져 있으면 overhead가 거의 없어야 한다.
- Sidecar import 실패나 ArkVale kernel 부재가 baseline 실행을 깨면 안 된다.
- 첫 버전은 Python/PyTorch로 충분하다.

Completion:

- 기능 off 상태에서 vLLM behavior가 변하지 않는다.
- 기능 on 상태에서 sidecar 객체가 생성되고 debug counter가 증가한다.

## Step 1.2: KV Write Observation Hook

Hook 위치:

```text
vllm/model_executor/layers/attention/attention.py
  unified_kv_cache_update(...)
```

이 함수는 다음을 얻을 수 있다.

```text
layer_name
key
value
kv_cache
layer_slot_mapping
attn_layer
```

해야 할 일:

1. `layer_slot_mapping`에서 유효 slot만 추출한다.
2. `physical_block_id = slot_id // block_size`를 계산한다.
3. `block_offset = slot_id % block_size`를 계산한다.
4. 새로 write된 K/V가 어떤 physical block에 들어갔는지 sidecar에 알린다.
5. 이 단계에서는 KV cache write 자체를 바꾸지 않는다.

주의:

- `slot_mapping`에는 padding slot이 있을 수 있다.
- FlashAttention에서는 `reshape_and_cache_flash`가 `slot_mapping` shape을 기준으로 실제 token 수를 판단한다.
- Hook은 custom op 내부에 있으므로, side effect를 최소화하고 flag check를 가장 앞에 둔다.

Completion:

- Decode 중 layer별로 observed token count가 증가한다.
- Slot id에서 계산한 physical block id가 current block table 안의 block id와 일치한다.
- Padding slot은 sidecar state에 들어가지 않는다.

## Step 1.3: ArkVale-Style Digest Cache v0

Milestone 1의 digest는 ArkVale 구현과 같은 방식으로 생성한다. 임시 mean-key digest는 사용하지 않는다.

ArkVale의 digest 생성은 다음 파일에 있다.

```text
/workspace/ArkVale/source/arkvale/infer_state.py
  InferState._summarize_keys(...)
  InferState.prefill_save_digests(...)
  InferState.decode_save_1_digest(...)
```

ArkVale digest는 원본 KV page 하나마다 `maxs, mins` 한 쌍을 만든다. 여기서 `maxs/mins`는 단순 raw max/min이 아니라, page 안 key들의 axis-aligned cuboid summary이다.

```python
raw_max = filled_keys.max(dim=2).values
raw_min = filled_keys.min(dim=2).values
center = (raw_max + raw_min) / 2
dist = (center[:, :, None, :, :] - filled_keys).abs().mean(dim=2)
digest_max = center + dist
digest_min = center - dist
```

Input/output shape:

```text
filled_keys: [batch, n_filled_pages, page_size, n_kv_heads, head_dim]
digest_max: [batch, n_filled_pages, n_kv_heads, head_dim]
digest_min: [batch, n_filled_pages, n_kv_heads, head_dim]
```

vLLM sidecar의 최소 저장 형태:

```text
(layer_name, physical_block_id) -> digest_max
(layer_name, physical_block_id) -> digest_min
(layer_name, physical_block_id) -> valid_token_count
```

정확히는 digest를 page-shaped KV storage로 새로 만드는 것이 아니라, vLLM physical KV block 하나마다 ArkVale-style digest vector pair를 하나 둔다는 뜻이다. 나중에 ArkVale CUDA kernel을 직접 쓰려면 이 digest vector pair들을 ArkVale의 paged digest-cache layout으로 pack하면 된다.

ArkVale behavior를 따르는 초기 정책:

1. 완전히 채워진 vLLM physical block만 digest candidate로 삼는다.
2. 현재 쓰고 있는 partial last block은 digest score 대상에서 제외한다.
3. Sink/window block은 score로 뽑는 대상이라기보다 항상 보존/관찰되는 영역으로 따로 취급한다.
4. Decode 중 새 physical block이 시작되면 직전 block이 full block이므로 그 block digest를 finalize한다.

Completion:

- Sidecar가 layer별 ArkVale-style `digest_max/digest_min` entry를 만든다.
- Digest entry 하나는 vLLM physical KV block 하나를 대표한다.
- Partial last block은 기본 score candidate에 들어가지 않는다.
- Request가 길어질수록 finalized digest entry 수가 filled block 수와 같이 증가한다.

## Step 1.4: Query-Time Score Hook

Hook 위치:

```text
vllm/model_executor/layers/attention/attention.py
  unified_attention_with_output(...)
```

이 함수는 다음을 얻을 수 있다.

```text
layer_name
query
attn_metadata
kv_cache
attn_layer
```

Milestone 1에서는 현재 decode query 하나를 그대로 scoring query로 쓰지 않는다. 대신 layer/request별 최근 64개의 decode query를 rolling buffer로 유지하고, 그 평균을 `window_query`라고 부른다.

```text
window_query = mean(last up to 64 decode queries)
```

초기 token처럼 아직 64개가 쌓이지 않은 경우에는 현재까지 관측된 query만 평균낸다. Chunked prefill과 speculative decoding은 고려하지 않으므로, request별 decode step마다 query가 하나씩 들어온다고 가정한다.

해야 할 일:

1. 현재 decode `query`를 layer/request별 rolling query buffer에 append한다.
2. Buffer는 최근 64개 query만 유지한다.
3. `window_query`를 rolling buffer 평균으로 계산한다.
4. `attn_metadata.block_table`에서 현재 request의 physical block list를 읽는다.
5. `attn_metadata.seq_lens`로 유효 sequence length를 확인한다.
6. block size로 유효 block 수와 마지막 block length를 계산한다.
7. sidecar digest table에서 해당 physical block들의 ArkVale-style `digest_max/digest_min`을 조회한다.
8. `window_query`와 digest로 page score를 계산한다.
9. top-k candidate physical block ids를 기록한다.
10. attention backend의 기존 forward call은 그대로 실행한다.

초기 score shape:

```text
single request:
  scores: [num_layers_observed, num_pages] or per-layer [num_pages]

future multi-request:
  scores: [num_reqs, num_pages_per_req]
```

초기 PyTorch score는 ArkVale의 cuboid scoring과 맞춘다.

```text
score(block) = sum_i max(window_query_i * digest_max_i,
                         window_query_i * digest_min_i)
```

Head/group reduction은 ArkVale의 `estimate_scores` contract에 맞춰 잡는다. Milestone 1의 첫 구현에서는 layer별 score alignment가 우선이므로, 필요하면 head 평균 또는 group 평균을 debug option으로 둔다.

Completion:

- Decode step마다 `window_query`가 생성된다.
- `window_query`는 최근 최대 64개 decode query의 평균이다.
- Score vector 길이가 유효 finalized digest block 수와 맞다.
- Top-k 결과가 physical block id로 출력된다.
- Attention output tensor는 sidecar score 계산 전후로 바뀌지 않는다.

## Step 1.5: Debug Output and Inspection

Milestone 1은 기능 자체보다 metadata alignment가 중요하다. 따라서 debug artifact를 반드시 남긴다.

기록할 항목:

```text
step
layer_name
num_reqs
num_actual_tokens
seq_lens
block_table row
valid_block_ids
observed_digest_block_ids
score_shape
window_query_len
topk_block_ids
topk_scores
missing_digest_blocks
```

출력 방식 후보:

1. Rank-local JSONL file
2. Periodic logger summary
3. In-memory ring buffer plus final dump

추천 시작점:

```text
JSONL file under VLLM_MPR_DEBUG_DIR
```

주의:

- 매 decode step마다 모든 layer를 무제한 dump하면 느리고 파일이 커진다.
- `VLLM_MPR_MAX_LAYERS`, `VLLM_MPR_MAX_STEPS`, `VLLM_MPR_DUMP_EVERY` 같은 제한이 필요하다.

Completion:

- 한 prompt generation에 대해 debug JSONL을 읽고 block id/score 흐름을 추적할 수 있다.
- Missing digest가 있다면 어떤 layer/block인지 바로 보인다.

## Step 1.6: Validation

검증은 세 층으로 나눈다.

### 1. Baseline Non-Regression

Sidecar off:

```text
output_before == output_after
```

Sidecar on, score-only:

```text
generated token ids unchanged
attention output not intentionally modified
```

완전한 bitwise equality는 CUDA graph, sampling, runtime 설정에 따라 어려울 수 있으므로 deterministic decode 조건을 먼저 둔다.

### 2. Metadata Alignment

확인할 invariant:

```text
slot_id // block_size is in block_table
score length == number of finalized digest candidate blocks
window_query_len <= 64
last block valid length == seq_len % block_size or block_size
topk ids subset of valid finalized block ids
```

### 3. Digest Sanity

확인할 invariant:

```text
digest exists for blocks that have received KV writes
digest dtype/device is expected
no padding slot digest entry
no NaN/Inf score
```

Completion:

- Single request deterministic run에서 metadata invariants가 모두 통과한다.
- Sidecar on/off의 generated output이 동일하다.
- Debug dump로 score/top-k 흐름을 사람이 확인할 수 있다.

## Step 1.7: ArkVale Kernel Feasibility Check

Milestone 1의 기본 구현은 PyTorch scoring이다. 하지만 마지막에 ArkVale `estimate_scores`로 넘어갈 수 있는지 adapter shape를 점검한다.

확인할 mapping:

```text
vLLM block_table row
  -> packed dg_indices

seq_len + block_size
  -> dg_indptr
  -> dg_last_page_len
  -> dg_seq_len

sidecar digest tensor
  -> dg_data
```

주의:

- ArkVale kernel은 `dg_indptr`와 `dg_last_page_len`이 실제 유효 page/token만 나타내야 한다.
- Padded block table을 그대로 넘기면 안 된다.
- `dg_seq_len`은 kernel이 계산하는 `kv_chunk_len`과 맞아야 한다.

Completion:

- PyTorch scoring에서 사용하는 block id list를 ArkVale packed metadata로 바꾸는 adapter sketch가 작성된다.
- ArkVale CUDA kernel 연결이 Milestone 1.5 또는 Milestone 2 이전에 가능한지 판단한다.

## Step 1.8: Milestone 1 Completion Report

Milestone 1 완료 시 다음 문서를 업데이트한다.

```text
vllm_integration_notes.md
milestone_1_results.md
```

`milestone_1_results.md`에는 다음을 남긴다.

```text
implemented hooks
sidecar API
debug output format
known unsupported cases
validation command
sample debug excerpt
decision: proceed to M2 / improve M1 / pivot
```

## Proposed Implementation Order

1. Baseline run command 고정
2. Sidecar scaffold 추가
3. `unified_kv_cache_update`에 off-by-default observation hook 추가
4. Slot mapping -> physical block id 변환 검증
5. Digest table v0 구현
6. `unified_attention_with_output`에 score-only hook 추가
7. Debug JSONL dump 추가
8. Single request deterministic validation
9. ArkVale `estimate_scores` adapter feasibility 정리

## Out of Scope for Milestone 1

다음은 명시적으로 하지 않는다.

- CPU KV backup layout 확정
- CPU/GPU KV copy
- fp16 recovery
- lower precision recovery
- attention kernel 수정
- scheduler policy 수정
- block eviction policy 수정
- multi-GPU DCP/CP 지원
- speculative decoding 지원
- production performance optimization

## Main Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Custom op hook overhead | vLLM attention path는 CPU overhead에 민감함 | env flag, early return, debug limits |
| Torch compile/CUDA graph interaction | hook side effect가 graph capture와 충돌 가능 | eager/small config first, score-only guard |
| Padded slot/block metadata | 잘못된 block id에 digest/score가 붙을 수 있음 | padding filtering and invariant checks |
| Digest semantic mismatch | key가 RoPE 적용 전/후인지 불명확할 수 있음 | chosen model에서 hook point tensor semantics 확인 |
| ArkVale kernel shape mismatch | padded metadata가 score를 망칠 수 있음 | PyTorch first, packed adapter later |

## Decision Gate

Milestone 1 끝에서 다음 중 하나를 선택한다.

| Decision | Condition |
|---|---|
| Proceed to Milestone 2 | score-only sidecar가 안정적으로 동작하고 metadata invariant가 통과함 |
| Extend Milestone 1 | score는 가능하지만 digest quality나 hook overhead 검증이 부족함 |
| Add ArkVale kernel before M2 | PyTorch score는 맞지만 overhead가 너무 커서 kernel path가 필요함 |
| Pivot to InfiniGen prototype | vLLM hook이 compile/graph/runtime과 계속 충돌함 |

## Expected Outcome

Milestone 1이 끝나면 우리는 아직 recovery를 하지 않는다. 대신 다음 질문에 답할 수 있어야 한다.

```text
At each decode step, for each target layer, which vLLM physical KV blocks would
we recover if mixed-precision recovery were enabled?
```

이 답이 안정적으로 나오면 Milestone 2에서 CPU fp16 backup layout과 lifecycle을 설계할 수 있다.
