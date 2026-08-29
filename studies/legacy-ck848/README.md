# legacy-ck848 — re-evaluation of `checkpoint_000848`

`REPORT_KO.md` is the evidence behind the repo's headline claim: that the
pre-rename Dynamic-k NN checkpoint ck848 is the best arm measured under the C2
criterion (100% success, t_conv 532, J 152.6, CVaR10 201.0 at L=250/N=20, seeds
1500–1999), and therefore that Dynamic-k NN — not the C2-regime arm that the
earlier B8/B9 work concluded on — is the main line. `docs/DECISION_LOG.md` and
`checkpoints/PROVENANCE.md` both cite it.

This study directory is unusual in three ways, all of them facts about how the
work ran rather than gaps to be filled in later:

- **No `src/`.** It used the acs-confirm tooling as it stood
  (`studies/acs-confirm/src/{eval_c2_r3,run_knn_refs3,pair_judge}.py`) and its
  one code change is commit `500b8b4` there — the rename aliases that let a
  `distance_pointer` checkpoint load, plus the mis-dispatch bug that change
  exposed. That is now in the promoted harness as `eval/policies.py`.
- **No `data/`.** Outputs landed under `studies/acs-confirm/data/` (listed in
  the report's §10.5) and are gitignored like every other eval artifact.
- **No `PROBLEM` / `PLAN` / `RUNLOG`.** This was an evaluation of an existing
  checkpoint against an existing protocol, not a designed study.

`REPORT_KO.md` is a **verbatim record**, dated 2026-08-27, imported into the
repo on 2026-08-29 from a path outside it. Like every document under
`studies/`, it is not retro-edited, so read it as of its date:

- Its §10.4 reproduction commands are the study-era invocation
  (`cd studies/acs-confirm/src && python eval_c2_r3.py …`). The equivalent today
  is `python -m eval.eval_c2 …` from the repo root — same protocol, same
  numbers; see `README.md` § Evaluating.
- It cites `B8_B9_REPORT.md`, the arms its §9 argues against. That report is in
  the repo as `studies/c2-regime-dual-policy/REPORT_KO.md` and carries a
  2026-08-27 correction banner pointing back at this one — read the pair
  together. `LEGACY848_NOTES.md` (the session ledger) is still outside the repo.
- Its §10.1 "changed / did not change" table describes the working tree at the
  time of writing, several consolidation commits ago.
