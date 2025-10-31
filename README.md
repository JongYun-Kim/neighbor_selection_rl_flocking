# neighbor_selection_rl_flocking

## TODOs
- [x] Re-configure/clone files
  - [x] Env: get it from lazy-message-listener-flocking
  - [x] Model: get it from lazy-message-listener-flocking
  - [x] Model: and update the names and clean it up
  - [x] Model: update it to be able to use address arbitrary numbers
  - [x] Fix: reward function
  - [x] Fix: done function
- [ ] Train neighbor selection policies
  - [ ] Reproduce 이전 레포지토리 성능
  - [ ] Tune the hyperparameters
- [ ] Neighbor Selection Comparison
  - [ ] 결정하기: baselines (비교 알고리즘)
    - [ ] 
  - [ ] 구현하기: baselines
  - [ ] Collect the results
  - [ ] Consider using different convergence criteria of ACS (e.g., num_neighbor-normalized spatial entropy––spatical density)
- [ ] Analyze the comparative study
  - [ ] Come up with metrics
  - [ ] Collect the data
  - [ ] Interpret them
- [ ] Extra work (less priority; not decided)
  - [ ] Ablation study
    - [ ] Compare network architectures: attention vs MLP (start with learning curves)
  - [ ] Look into the networks inside (e.g., attention scores)
  - [ ] Tests for non-leanring guys (e.g., robustness to noise)

