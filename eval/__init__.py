"""Criterion-of-record evaluation harness (C2 protocol).

Run every tool as a module from the repo root, so that this package — not the
builtin ``eval()`` name, which lives in a different namespace and never
collides — is what ``eval.`` resolves to:

    python -m eval.eval_c2      --ckpt <ckpt> --label <label> [--seeds A-B]
    python -m eval.run_knn_refs --k 12 --L 250 [--seeds A-B]
    python -m eval.pair_judge   --arm pol=<label> --arm k12=knnref:12,250,20

The C2 protocol these tools implement: fixed-horizon rollouts capped at 6000
steps, deterministic (argmax) actions, L = 250 unless --bound says otherwise,
convergence judged offline (phi > 0.98 held over 50 steps, one r0-component
over 300 steps, spatial-entropy band < 5%), J = accumulated control cost up to
the firing step.

The copies under studies/*/src are the records of the studies that produced
them and are kept unmodified; this package is the version to run.
"""
