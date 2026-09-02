# main-dknn-eval-integration-202609 — main 평가 기반 통합 기록

## 1. 문서 성격

이 문서는 Dynamic-k NN 학습 및 평가 코드를 `main` 브랜치의 공용 실행
기반으로 통합한 작업을 기록한다. 특정 성능 주장을 위한 실험 보고서가
아니라, 통합의 배경과 범위, 기존 사용자에게 미치는 영향, 검증 결과를
남기는 engineering/provenance 기록이다.

- 작성일: 2026-09-02 (Asia/Seoul)
- 대상 브랜치: `main`
- 작업 시작 기준 커밋: `948a1117872e65bdb2adb8809faabb5e1f86b306`
- 통합 대상: canonical C2 평가, Dynamic-k population 기록 및 후처리,
  학습 설정 전달 경로

공용 실행 구현의 기준은 repository root의 `eval/`, `train_unified.py`,
`docker/`이다. `studies/` 아래의 기존 소스와 결과는 역사적 기록으로
취급하며 수정하거나 현재 실행 구현으로 사용하지 않는다.

## 2. 통합 배경

Dynamic-k NN 관련 평가 코드는 개발 worktree, sensitivity farm, test field,
과거 study 디렉터리에 서로 다른 형태로 존재했다. 각 코드는 당시 실험을
재현하는 데에는 유효했지만 다음 항목이 통일돼 있지 않았다.

- C2 수렴 조건과 경계의 엄격성
- state/action 시점 정렬과 수렴 비용 `J`의 구간
- checkpoint 선별과 최종 확인 절차
- seed별 trajectory 저장 형식과 provenance
- N=20/40 cutoff-radius 시각화 및 episode 평균 방식
- 재실행 시 기존 artifact를 재사용하거나 복구하는 규칙

이번 통합은 `main_c2_v1`을 단일 판정 기준으로 두고, 공용 평가와 분석을
root `eval/` 패키지에서 실행하도록 정리한 것이다. 과거 study 결과는
재분류하거나 새 protocol 결과로 바꾸지 않는다.

## 3. Canonical C2 protocol

`eval/protocol.py`의 `MAIN_C2_V1`이 C2 판정의 구현 기준이다.

- state scalar series는 초기 상태를 포함한 `T+1` 표본이다.
- 최근 50개 state에서 polarization의 최솟값이 `0.98`보다 커야 한다.
- 최근 300개 state에서 `distance < r0` proximity graph가 계속 연결돼야
  한다.
- 최근 300개 position-entropy의 peak-to-peak/mean 값이 `0.05`보다
  작아야 한다.
- canonical horizon은 6,000 step이며 수렴 판정은 rollout 후 offline으로
  수행한다.
- deterministic learned policy는 pointer logits의 argmax를 사용한다.
- 성공 episode의 비용은 `J = -sum(reward[1:t_fire+1])`로 정의한다.
- 실패 episode의 `J`는 정의하지 않고 결측값으로 보존한다.

기존 `eval.eval_c2`와 fixed-k reference judge도 같은 protocol 함수를
사용하도록 연결하여 판정 구현이 갈라지지 않게 했다.

## 4. 통합된 evaluation workflow

### 4.1 Unified CLI

공용 진입점은 `python -m eval`이며 다음 command를 제공한다.

- `checkpoints`: 학습 run의 평가 지표와 실제 checkpoint를 교차 확인하여
  top 후보와 final checkpoint manifest를 만든다.
- `c2 --lane dev`: 적은 seed로 하나 이상의 checkpoint를 screening한다.
- `c2 --lane confirm`: 명시적으로 고른 checkpoint 하나를 confirmation
  lane에서 평가한다.
- `population`: N과 policy arm별 full trajectory를 기록한다.
- `validate`: population bundle의 schema, summary 및 물리적 일관성을
  검증한다.
- `radii`: 한 deterministic episode의 cutoff circle과 contextual rank를
  정적 그림 또는 animation으로 만든다.
- `heatmaps`: deterministic population의 rank별 평균 cutoff-radius
  heatmap을 만든다.
- `control-effort`: main C2 기준 전후의 L1/L2 control effort를 분석한다.

### 4.2 C2 dev/confirmation lane

기본 dev lane은 N=20, L=250, seed 1000–1031의 32 episode를 사용한다.
confirmation의 공식 조건은 N=20, L=250, seed 1500–1999의 500 episode,
6,000 step, deterministic CPU 실행이다. seed, horizon, N 또는 L을
override한 실행은 artifact에 비공식 실행으로 기록한다.

confirmation은 후보 목록을 자동 승격하지 않으며 checkpoint 하나를
명시해야 한다. 선택된 package는 hash와 함께 confirmation bundle에
보관된다.

### 4.3 Population suite

기본 population suite는 다음 조합을 사용한다.

- population size: N=10, 20, 40
- seed: 0–49
- policy arm: deterministic learned, stochastic learned, PureACS
- horizon: 6,000 step

같은 N과 seed의 세 arm은 동일한 초기 swarm state를 사용한다. stochastic
arm의 action seed는 별도로 결정하고 metadata에 기록한다. 저장 artifact는
전체 state, pointer action, 실현된 binary mask, control input, 물리 scalar
series 및 C2 결과를 포함한다.

summary에는 episode별 결과, aggregate, PureACS와의 paired 차이, exact
McNemar와 co-success 비용 통계가 포함된다.

## 5. Artifact, provenance 및 validation

각 C2/population 실행은 독립된 versioned bundle을 사용한다.

- immutable 실행 specification의 fingerprint
- Git commit 및 dirty-tree hash
- checkpoint package/config/state hash
- Python, NumPy, Torch 및 platform 정보
- seed별 episode NPZ와 summary CSV
- bundle manifest와 validation report

파일은 임시 경로에 완전히 기록한 뒤 atomic replace한다. 기존 episode는
파일 존재만으로 재사용하지 않고 shape와 metadata를 검증한다. specification
fingerprint가 다르면 동일 bundle에 결과를 섞지 않는다. `--repair-invalid`
사용 시 잘못된 artifact는 `quarantine/`으로 옮긴 뒤 다시 생성한다.

deep validation은 다음 내용을 원본 trajectory에서 다시 계산한다.

- pointer action에서 cutoff mask 복원
- state에서 polarization과 position/velocity entropy 재계산
- state/control transition 및 raw reward 일치 여부
- 저장된 C2 scalar와 summary 일치 여부
- paired arm의 초기 상태 동일성
- bundle에 보관된 checkpoint package hash

source tree와 입력 checkpoint는 read-only로 취급하고 새 결과는 지정한
artifact root 아래에만 기록한다.

## 6. 통합된 분석 기능

### 6.1 Single-episode cutoff-radius animation

현재 `radii` animation은 한 deterministic episode에 대해 다음을 동시에
표시한다.

- swarm-centroid 기준 위치, heading, trail 및 pointer 연결선
- 각 ego agent가 선택한 cutoff radius 원
- swarm centroid에서 가까운 agent 순 cutoff-radius bar
- circular swarm mean heading과 가까운 agent 순 cutoff-radius bar
- radius, selected k, r0 및 contextual metric

이 레이아웃은 기존
`test_field/deterministic_swarm_radius_analysis`의
`*_dual_ranked_bars.mp4`와 기능적으로 대응한다. 기존 agent-ID layout과
centroid bar + heading-ranked time heatmap을 사용한 context-ranked layout은
이번 통합 범위에 포함하지 않았다.

### 6.2 Population-ranked heatmaps

deterministic episode마다 각 action time에서 agent를 독립적으로 정렬한 뒤,
동일한 physical-time/rank cell을 episode seed 전체에 걸쳐 평균한다.

- N=20/40 × centroid-distance rank
- N=20/40 × mean-heading-deviation rank
- full-horizon view
- physical time 30초 view
- 30초 view + velocity/position entropy panel

따라서 view별 네 장, 세 view에서 총 열두 장의 PNG를 만들 수 있다.
entropy panel은 heatmap과 physical-time axis를 공유하고 값의 양의 방향이
왼쪽을 향한다. 렌더링에는 population mean을 사용하며 derived NPZ에는
episode population standard deviation도 함께 저장한다.

기존 test-field의 flat output을 직접 덮어쓰지 않고 canonical population
bundle 아래의 다음 경로를 사용한다.

```text
population/analysis/ranked_heatmaps/full/
population/analysis/ranked_heatmaps/tmax30s/
population/analysis/ranked_heatmaps/tmax30s_with_entropies/
```

### 6.3 Control-effort

저장된 control input에서 main C2 기준 L1/L2 effort를 계산한다.

- full-horizon effort
- 성공 episode의 C2 firing 시점까지 누적 effort
- learned policy와 PureACS의 matched-seed 차이
- C2 event 기준 pre/post 및 event-aligned summary
- episode CSV, paired CSV와 정적 plot

후처리는 population bundle을 읽으며 policy rollout을 다시 수행하지 않는다.

## 7. 학습 재현성 보완

Docker runner에서 설정했지만 컨테이너 안의 `train_unified.py`까지 전달되지
않을 수 있던 주요 학습값을 명시적으로 연결했다.

| 환경변수 | 학습 인자 | 의미 |
|---|---|---|
| `FLOCK_STEPS` | `--steps` | 총 학습 environment step |
| `FLOCK_MINIBATCH` | `--minibatch` | SGD minibatch 크기 |
| `FLOCK_SGD_ITER` | `--sgd-iter` | train batch당 SGD 반복 횟수 |
| `FLOCK_LR_END` | `--lr-end` | learning-rate schedule의 마지막 값 |
| `FLOCK_EVAL_INTERVAL` | `--eval-interval` | 학습 중 evaluation 주기, `0`은 비활성화 |

환경변수를 지정하지 않으면 기존 profile 기본값을 그대로 사용한다. 따라서
이 변경은 기존 사용자의 기본 학습 설정을 바꾸지 않고, 명시적으로 요청한
값이 컨테이너 경계에서 사라지지 않게 한다.

오래된 cached training image가 entrypoint에서 SSH daemon을 시작하는 경우를
막기 위해 runner가 `START_SSHD=0`도 전달한다. 이 값은 학습 알고리즘과
hyperparameter에는 영향을 주지 않는다.

## 8. 기존 사용자와 historical study에 대한 영향

- W&B logging은 계속 opt-in이며 기본값은 비활성화다.
- 기존 training profile의 기본 hyperparameter는 변경하지 않았다.
- `eval.eval_c2`, `eval.run_knn_refs` 등 기존 진입점은 유지하되 canonical
  protocol을 공유한다.
- historical study artifact는 읽기 호환 대상으로만 사용하고 제자리에서
  변환하거나 재기록하지 않는다.
- legacy/noncanonical input을 분석할 때에는 별도의 output 경로를 요구한다.
- 평가 Docker image tag는 training image tag와 분리하여 평가 build가 다른
  사용자의 이후 training container에 영향을 주지 않게 했다.
- 기본 artifact 경로는 gitignored `test_results/evaluation/`이다.

## 9. 검증 기록

### 9.1 회귀 테스트

Docker의 학습/평가 dependency 환경에서 다음 일곱 test module을 실행했다.

- `test_c2_suite.py`
- `test_checkpoint_selection.py`
- `test_control_effort.py`
- `test_eval_analysis.py`
- `test_eval_artifacts.py`
- `test_eval_population.py`
- `test_eval_protocol.py`

결과는 총 48개 test 모두 통과였다.

### 9.2 실제 checkpoint smoke test

학습된 `checkpoint_000733`을 실제로 복원하여 CPU에서 다음 경로를 확인했다.

- C2 dev: N=20, seed 1000, 300 step
- C2 confirmation: N=20, seed 1500, 300 step
- population: N=20/40, seed 0,
  deterministic/stochastic/PureACS 총 6 episode
- deep validation: 6/6 episode `pass`, paired initial-state group 2개 확인
- radii: pointer-to-mask mismatch 0, preview 3장과 61-frame MP4 생성
- heatmaps: full/30초/30초+entropy에서 각 4장, 총 12장 생성
- control effort: 6 episode, paired record 4개, 정적 plot 3장 생성

두 C2 smoke episode는 300-step 제한 안에서 C2에 도달하지 않았다. 이 실행은
checkpoint 복원, 추론, 저장 및 분석 경로를 검사하기 위한 30초 smoke test로,
6,000-step 공식 성능 결과로 해석하지 않는다.

검증 당시 원본 policy state SHA-256은 다음과 같았고 실행 전후 동일했다.

```text
0406260796882e6bcf29911fa582dbdb38f2ce2d21e203967c3aa1975613fcb1
```

검증용 artifact는 71개 파일, 32MB였으며 검증 완료 후 삭제했다. 원본 학습
run과 `checkpoint_000733`은 보존했다.

## 10. 알려진 범위와 제약

1. 해당 학습 run은 학습 중 evaluation을 비활성화하여 `progress.csv`에
   checkpoint ranking용 C2 metric이 없다. 따라서 이 run에는 자동
   `checkpoints` funnel을 적용할 수 없지만, checkpoint를 명시한 dev 및
   confirmation 평가는 정상 작동한다.
2. single-episode MP4는 dual-ranked bar 레이아웃만 통합했다. 기존
   agent-ID MP4와 context-ranked heatmap MP4를 동일하게 재생성하는 layout
   selector는 없다.
3. heatmap의 계산 의미와 파일명은 기존 test-field 결과와 대응하지만,
   provenance와 재현성을 위해 canonical bundle 하위 디렉터리에 저장한다.
4. smoke-test artifact는 검증 후 삭제했으므로 이 문서에는 결과 수치와 hash만
   기록하며 대용량 생성물은 포함하지 않는다.

## 11. 변경 파일의 역할

주요 신규 구현은 다음과 같다.

- `eval/__main__.py`: unified CLI
- `eval/protocol.py`: canonical C2 judge
- `eval/checkpoint_selection.py`: checkpoint funnel
- `eval/c2_suite.py`: dev/confirmation lane
- `eval/population.py`: full-trace population suite와 validation
- `eval/artifacts.py`: versioned artifact와 atomic I/O
- `eval/provenance.py`: Git/runtime provenance
- `eval/analysis/`: radius, heatmap, control-effort 분석
- `docker/run_eval.sh`: one-shot evaluation container runner
- `docs/EVALUATION.md`: 사용자 실행 문서

기존 `eval/eval_c2.py`, `eval/run_knn_refs.py`, `eval/common.py`,
`eval/policies.py`는 공통 protocol, atomic artifact 및 Dynamic-k inference
adapter를 사용하도록 조정했다. `docker/run_train.sh`과 `train_unified.py`는
명시적인 학습 재현성 환경변수를 전달하도록 조정했다.
