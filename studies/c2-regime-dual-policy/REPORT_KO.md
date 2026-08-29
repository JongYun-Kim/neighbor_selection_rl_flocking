# B8–B9 최종 리포트 — 통일 C2 레짐에서의 두 정책(main × dynamic_k_nn) 비교

- 작성: 2026-08-23. 정본 계획 `/workspace/INTEGRATION_PLAN.md`(Q1–Q10, D1–D6)와 `/workspace/PHASE_B_HANDOFF.md` §5의 B8 설계를 그대로 실행한 결과. 세션 타임라인 원장은 `/workspace/B8_SESSION_NOTES.md`.
- 레포: `/workspace/neighbor_selection_rl_flocking-integration`, 브랜치 `integration/dual-policy`, remote 0, working tree clean, 커밋 6건(a7cfcec→0bbe175). 원본 2클론 무변경 유지 확인.

> ## ⚠ 정정 (2026-08-27) — 헤드라인 결론 일부 무효
>
> legacy 클론의 `distance_pointer` 체크포인트(`checkpoint_000848`, 8.2M 스텝, entropy 페널티
> 없음)를 **동일한 통일 평가 규약으로** 측정한 결과, 아래 §6-1의 주장이 반증되었다.
> 상세: **`/workspace/LEGACY848_REPORT.md`**.
>
> - **무효**: "cutoff-pointer 동적 k-NN은 stochastic으로만 기능하는 정책에 머문다"(§6-1, §6-2).
>   같은 아키텍처가 결정론 argmax로 **500/500 = 100%, J 152.6**을 낸다 — 본 리포트의 π_R′
>   (100% / J 180.9)와 k-NN 프론티어(k12 93.6% / J 160.6)를 모두 능가한다.
> - **수정**: §2-1의 "균등 함정"은 탈출 불가능한 함정이 아니라 **매우 느린 이행 구간**이다.
>   lr 2e-5·페널티 없음에서 포인터 entropy는 59.66 → 44.81(1.6M) → 25.91(3.3M) → 15.41(6.9M)로
>   감쇠한다. B8이 탐색한 0.5–2M 구간은 그 고원부 안에 통째로 들어간다.
> - **수정**: §3의 D1 6M 연장조항 미발동 근거. "상수 페널티 레시피는 연장해도 개선 없다"는
>   여전히 옳지만, 거기서 **방법의 한계**를 결론한 것이 오류였다. 필요했던 실험은 "페널티 없는
>   기본 레시피의 장기 학습"인데, 모든 프로브를 0.5M로 균일하게 자른 설계가 그 칸을 비웠다.
> - **여전히 유효**: §2-2의 `custom_loss` entropy 페널티 구현, §2-4의 KL 지표 정정,
>   §5의 π_R′ L-mix 강건성(legacy 체크포인트는 L500에서 전결합에도 지는 스케일 스페셜리스트다),
>   그리고 통계 방법론 일체.
> - **성립하지 않음**: "dknn > main". legacy 런과 π_R′는 예산(8.2M vs 1.92M)·보상·에피소드
>   규약·L 레짐이 동시에 다르므로 통제된 방법 비교가 아니다.

## 1. 결론 요약

**통일 C2 수렴 기준·argmax 평가 규약(Q8) 아래에서 main 신경 정책이 압도적으로 우세하다.**

| arm (n=500, 시드 1500–1999, argmax, 6000스텝) | 성공률 [Wilson 95%] | t_conv med | J med | CVaR10(J) |
|---|---|---|---|---|
| **π_R′** (main 대표, uni_policy_robust_s42 ck30) | **100.0%** [99.2, 100] | 554 | 180.9 | 269.6 |
| π_E′ (파인튜닝, c2C1ft_uni ck30) | 100.0% | 646 | 260.4 | 409.8 |
| **dknn-best** (p7_entpen_mb256 ck40) | **27.2%** [23.5, 31.3] | 792 | 295.2 | 607.7 |
| nearest k12 (레퍼런스) | 93.6% | 521 | 160.6 | 350.6 |
| nearest k13 / k14 | 92.6% / 93.6% | — | ~161 | — |

- **π_R′ vs k12**: McNemar b=0 c=32 (p=4.7e-10) — k12의 실패 32건 전부를 성공시키면서 자기 실패 0. 비용은 co-success dJ med +15.8 (~10% J 프리미엄, p=1.0e-6).
- **π_R′ vs dknn**: 364 vs 0 (p=5.3e-110), co-success에서도 dJ −117.9 (p=6.8e-21). 전 지표 우위.
- **dknn vs k12**: 343 vs 11 (p=1.3e-86) — dknn은 자신의 고정-k 조상에게도 완패.
- **π_R′ vs π_E′**: 성공 동률(양쪽 500/500), J는 π_R′가 med −76.8 우위(p=1.3e-73) — **이번 재현에서 파인튜닝 단계는 개선 실패**(§4 편차 참조).

## 2. B8-1 HP 프로브 — 핵심 서사 (예산 8/8런 × 0.5M, D2 내)

| # | config (dknn_c2 대비 변경) | argmax eval 최대 | 최종 entropy(최대 59.91) | 판정 |
|---|---|---|---|---|
| P1 | (B7) lr 2e-5 | 0 | 59.53 | 균등 정체 |
| P2 | lr 1e-4 flat | 0 | 59.63 | 균등 정체 |
| P3 | lr 5e-4→1e-4 | 0 | 59.45 | 균등 정체 |
| P4 | lr1e-4+mb256/sgd10 | 0 | 59.14 | 균등 정체 |
| P5 | lr1e-4+batch16k | 0 | 59.27 | 균등 정체 |
| P6′ | lr1e-4+ent페널티 1e-3 | 0 (train succ 0.22로 악화) | 58.45 | 단독 무효 |
| **P7** | **lr1e-4+페널티 1e-3+mb256/sgd10** | **0.25 (dev 6/32)** | 57.82 | **채택** |
| P8 | 페널티 2e-3+mb256/sgd10 | 0.062 | **4.33 (폭주 붕괴)** | 과잉 |

발견 사항:
1. **argmax 0의 기전은 lr 과소가 아니라 "균등 포인터의 자기 마스킹"**: 균등 랜덤 cutoff는 매 스텝 재샘플링(시간적 mixing)으로 그래프를 시간 평균 연결시켜 stochastic 성공 ~50–90%를 만들지만, 바로 그 mixing이 cutoff 선택 간 advantage 차이를 소거해 gradient가 균등점을 벗어나지 못한다. argmax는 mixing이 없어 초기 저-k 편향이 고착 → **단편화 실패**(B7 ckpt 포렌식: n_comp_end 2–11, deg_early 4–8).
2. RLlib은 `entropy_coeff<0`을 금지(validate_config) → **`DynamicKNNPPORLlib.custom_loss`에 opt-in 포인터 entropy 페널티 구현**(커밋 454bd55, 기본 0=무영향). 유효 성분은 **페널티(1e-3) × 강화 optimizer(mb 512→256, sgd 7→10)의 조합** — 각각 단독으로는 무효(P4/P6′).
3. 페널티는 양성 피드백 구조(균등=entropy 극대점 → 이탈 시 gradient 증가)라 **장기에서 폭주 붕괴** — P8은 0.5M 내, 최종 2M 런은 ~1.2M(it~150)에서 붕괴(entropy 52→3.5, train succ 0.91→0). 실용 해법은 10-iter 체크포인트 전수 스크리닝으로 피크 수확(실행함). 다음 후보: 페널티 어닐링(→0) 또는 entropy 하한 타깃.
4. 진단 주의: `kl_coeff=0`이면 RLlib learner_stats의 `kl`은 항상 정확히 0 — 정책 이동 지표로 쓸 수 없음(B7 진단문 정정). 참고: 원본 dknn 브랜치에는 포인터 정책의 argmax 평가가 애초 존재하지 않았다(evaluate_checkpoint.py는 binary 전용, 테스트는 stochastic) — argmax-0은 회귀가 아니라 Q8 규약이 새로 부과한 기준이다.

## 3. 최종 런과 대표 선정 (D5 절차)

| 런 | 설정 | 결과 | dev 스크리닝(시드 1000–1031, argmax) |
|---|---|---|---|
| π_R′ `uni_policy_robust_s42` | 확증 레시피 그대로(검증: config-diff 일치), 1.92M, cuda:1, 8.1h | eval 피크 it30–60(1.0), 후반 과학습 열화 | ck30 32/32 J182.8 **← 대표**, ck40 32/32 J221, ck50 32/32 J216, ck60 31/32, ck80 30/32 |
| π_E′ `c2C1ft_uni` | train_robust2, init=π_R′ it40, 80it flat 1e-4, cuda:3, 6.4h | 전 구간 eval ~1.0 | ck30 32/32 J253.6 (π_E′ 내 대표), 전 후보가 π_R′ ck30에 열위 |
| dknn-final `dknn_final_p7cfg` | P7 config, 2.0M, cuda:3, 5.5h | it~140까지 succ 0.91↑ 후 entropy 폭주 붕괴 | 피크 창 ck100–150만 생존, 최고 ck120 6/32 (t_conv 3598, J 1008) |
| (프로브 P7 런) | 동일 config 0.5M | — | **ck40 6/32, t_conv 828, J 269 ← dknn 대표**(성공 동률, J 타이브레이크) |

- **dknn 대표 = P7 런의 ck40**: 2M 최종 런 자체 최고(ckF120)와 dev 성공 동률(6/32)이나 t_conv·J에서 4배 우수. 확증에서 대표 136/500(27.2%) vs 감도 arm ckF120 50/500(10.0%)으로 선정 타당성 재확인. **최종 2M 런이 0.5M 시점보다 나은 argmax 정책을 만들지 못했다**는 점 자체가 상수 페널티의 한계 증거(§2-3).
- **D1 연장조항(6M) 미발동**: dknn의 병목은 스텝 부족(미수렴)이 아니라 페널티 폭주로 인한 과잉 수렴 — 같은 레시피로 6M을 돌려도 개선 근거 없음. 연장 대신 §6의 레시피 개선이 맞는 수순.

## 4. 계획과 달랐던 점 (게이트·중단 규칙 §5에 따른 명시)

1. **main 대표가 π_E 계열이 아님**: Q7은 π_E 계열 기준을 상정했으나, 이번 재현에서 π_E′(π_R′-it40 파인튜닝, A-it40 실물 부재로 인한 D5 fallback)는 성공 동률·J 열위(dev 253.6 vs 182.8, 확증 260.4 vs 180.9). D5의 dev 스크리닝 규칙대로 **main 대표 = π_R′ ck30**. (해석: flat 1e-4 80it 파인튜닝이 이 π_R′ 라인에서는 J를 되레 끌어올림 — A-라인의 π_R→π_E 개선이 재현되지 않음. init이 A-it40의 구조적 유사물일 뿐 실물이 아니라는 한계 포함.)
2. **dknn 대표가 프로브 런 출신**: §3 참조. 예산 관점에서는 dknn에 2.5M(프로브 0.5M+최종 2.0M)이 투입됐고 최선 지점이 0.5M에 있었던 것.
3. π_E′ 실행 GPU를 cuda:1→cuda:3으로 변경(π_R′와 병렬화, 2–3h 단축). 자원 정책 위반 없음(cuda:0 미사용, cuda:2 미사용, cuda:3 OOM 여유 유지, 스레드 ≤60).
4. 소요 실측: π_R′ 8.1h(추정 3–5h 초과, 242s/iter — 레시피 고유 비용), π_E′ 6.4h(추정 2–3.5h 초과). dknn 5.5h(추정 부합). 평가(CPU)는 전부 학습과 중첩 실행.
5. B7의 "KL≈0" 진단은 지표 자체가 무효였음을 정정(§2-4) — 결론(정책 미이동)은 entropy·eval 근거로 유지.

## 5. N/L 강건성 프로브 (Q8; N축 시드 1000–1499, L축 1500–1999, n=500/arm)

| 조건 | π_R′ ck30 | dknn ck40 | nearest 레퍼런스 | FC |
|---|---|---|---|---|
| 기본 N20@L250 | **100%** / J 181 | 27.2% / J 295 | k12: 93.6% / J 161 | — |
| N10@L177 | **100%** / J 184.7 | 12.0% / J 296 | k6: 95.8% / J 174.9 | k9: 100% / J 187.1 |
| N40@L354 | **100%** / J 194.5 | 39.0% / J 389 | k24: 96.4% / J 150.3 | k39: 100% / J 290.1 |
| L125 | **100%** / J 165.9 | 23.8% / J 390 | k12: 96.8% / J 141.5 | — |
| L500 | **99.8%** / J 220.5 | 29.2% / J 401 | k12: 92.8% / J 165.4 | — |

- π_R′는 학습 분포 밖 N(10/40)에서도 100% — 크기 일반화 완전. N10에서 FC와 성공 동률에 J 우위(184.7 vs 187.1), N40에서는 FC 대비 J 33% 절감(194.5 vs 290.1). k-스윕 대비로는 전 조건에서 실패를 보험하되 J 프리미엄(L125 +17%, N40 +29%)을 지불.
- dknn은 L축 안정(24–29%), N축은 **N↑ 방향 개선(N40 39.0%)** — per-ego cutoff가 후보 풀 확대에서 이득을 보는 유일한 우상향 신호. 그래도 전 조건 레퍼런스 대비 큰 열세.

## 6. 논문 방향 제안

1. **주 결론 축**: "통일 C2 기준·결정론 평가에서 학습된 선택 정책(main)은 고정 k-NN 스윕 전체를 지배(실패 0 + McNemar 유의)하고 크기·밀도 축에 완전 일반화한다. 반면 cutoff-pointer 동적 k-NN은 stochastic으로만 기능하는 정책에 머문다" — 비교 리포트의 헤드라인으로 충분히 강함.
2. **dknn의 학술적 기여는 실패 기전 분석에 있음**: (a) 시간적 mixing에 의한 advantage 자기 마스킹(균등 함정), (b) 결정론화 압력(entropy 페널티)의 조합 조건과 폭주 동역학, (c) N축 우상향. "왜 pointer식 동적 k가 argmax에서 무너지는가"는 방법 실패 사례 연구로 가치가 있고, 개선 방향(페널티 어닐링, entropy 하한 타깃(예: SAC식 자동 온도의 역방향), curriculum: stochastic→deterministic)을 후속 실험 제안으로.
3. π_E′ 미개선은 "파인튜닝 단계의 가치는 init 체크포인트 실물에 의존적"이라는 재현성 각주로 처리 권고(A-it40 실물 부재가 원인일 수 있음).
4. 통계 방법론은 acs-confirm 등록 분석(Wilson+exact McNemar+CVaR10+co-success dJ)을 그대로 계승 — `pair_judge.py`가 arm-쌍 일반화 구현.

## 7. 인수인계 노트

- **브랜치/커밋**: `integration/dual-policy` @ `0bbe175` (6커밋: 병합 a7cfcec → B4 9425235 → B5 88be7f1 → B6 2507b3d → 페널티 454bd55 → pair_judge 0bbe175).
- **원격**: `origin`(`git@github.com:JongYun-Kim/neighbor_selection_rl_flocking.git`)에 같은 이름 `integration/dual-policy`로 push 완료(upstream 설정).
- **대표 체크포인트** (전부 gitignored `test_results/` 안 — push 대상 아님, 별도 보존 필요 시 복사):
  - main 대표 π_R′: `test_results/uni_policy_robust_s42/GradLoggingPPO_*/checkpoint_000030`
  - π_E′: `test_results/c2C1ft_uni/manual/checkpoint_000030`
  - dknn 대표: `test_results/p7_entpen_mb256/GradLoggingPPO_*/checkpoint_000040` (감도: `dknn_final_p7cfg/.../checkpoint_000120`)
  - 각 checkpoint 디렉토리에 params.json 복사본 포함(eval 하네스 요건).
- **평가 산출**: `studies/acs-confirm/data/eval/*_summary.csv`(arm별 per-seed), `knnref/*.csv`(레퍼런스 9종), `b8_confirm_stats_{arms,pairs}.csv`(통계 매트릭스). 전부 gitignored.
- **재현 명령 요지**: dknn-best 재학습 `python train_unified.py --profile dknn_c2 --steps 500000 --lr 1e-4 --entropy-penalty 0.001 --minibatch 256 --sgd-iter 10 --gpu N`; 확증 `cd studies/acs-confirm/src && python eval_c2_r3.py --ckpt <ckpt_dir> --label X --seeds 1500-1999 --workers 24`; 통계 `python pair_judge.py --seeds 1500-1999 --arm ...`.
