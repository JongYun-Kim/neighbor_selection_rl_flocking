# c2-regime-dual-policy — the two policy families under one C2 regime

`REPORT_KO.md` is the head-to-head comparison that the C2 training regime was
built for: the binary edge-selection line (π_R′, π_E′) against Dynamic-k NN,
all trained under the same reward, episode convention and budget, all judged by
the same argmax C2 protocol on seeds 1500–1999. Phase-labelled **B8/B9** in the
integration plan, which is where the report's own section numbering and the
`B8-1` probe names come from; that plan is not in this repo.

**Read it together with `studies/legacy-ck848/`, not on its own.** The report
carries a correction banner dated 2026-08-27, added after the ck848
re-evaluation, and the two documents argue with each other on purpose:

| this report's claim | status after ck848 |
|---|---|
| "cutoff-pointer Dynamic-k only ever works as a stochastic policy" (§6-1, §6-2) | **refuted** — ck848 gives 100% / J 152.6 at argmax |
| the "uniform trap" is a trap (§2-1) | **corrected** — it is a very slow transition; pointer entropy 59.66 → 15.41 needs ~7M steps, and every probe here was cut at 0.5M |
| the D1 6M-extension clause was rightly not triggered (§3) | **corrected** — the un-penalised long run was the experiment the uniform-0.5M design left out |
| "dknn > main" | **does not hold either way** — budgets, rewards and L regimes all differ, so neither direction is a controlled comparison |
| `custom_loss` entropy penalty (§2-2), KL metric fix (§2-4), π_R′ L-mix robustness (§5), the statistics | **stand** |

Its π_R′ result is why `pi_r` is a profile of `train_unified.py`, and its §5
L-mix finding is what `studies/legacy-ck848/` later confirmed from the other
side: ck848 is a scale specialist that loses to fully-connected at L=500.

`REPORT_KO.md` is a record dated 2026-08-23, imported into the repo on
2026-08-29 from a path outside it. It is verbatim except for one omission at
import: §7's final bullet was working instructions for the next session, with
no bearing on any result, and was dropped. Nothing else is retro-edited, so
read it as of its date:

- No `src/` or `data/` of its own: it ran `train_unified.py` (then at commit
  `88be7f1`) and the acs-confirm harness, and its outputs went to
  `studies/acs-confirm/data/`, gitignored like every other eval artifact.
- §7's reproduction commands are the study-era invocation
  (`cd studies/acs-confirm/src && python eval_c2_r3.py …`); today's equivalent
  is `python -m eval.eval_c2 …` from the repo root.
- §7 is a session handoff note, not findings: where the representative
  checkpoints and eval outputs were left, and the commands that produced them.
- It cites `INTEGRATION_PLAN.md`, `PHASE_B_HANDOFF.md` and
  `B8_SESSION_NOTES.md`, which are still outside this repo.
