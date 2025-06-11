# neighbor_selection_rl_flocking

## TODOs
- [ ] Re-configure/clone files
  - [x] Env: get it from lazy-message-listener-flocking
  - [x] Model: get it from lazy-message-listener-flocking
  - [x] Model: and update the names and clean it up
  - [x] Model: update it to be able to use address arbitrary numbers
  - [ ] Fix: reward function
  - [ ] Fix: done function
  - [ ] CustomActors: create a template for heuristics
    - [ ] Distance-based (disc model)
    - [ ] Topology-based
    - [ ] Random
    - [ ] Any?
- [ ] Crate files for the experiments
  - [ ] Scalability
    - [ ] Add a training script for learning arbitrary number of agents
    - [ ] Test it with different ranges
  - [ ] Robustness (noise)
    - [ ] Noise tests
      - [ ] Observation noise (addictive noise)
      - [ ] Action noise
      - [ ] Communication noise (message reachability; _None_ noise)
  - [ ] Attention scores
    - [ ] Add a script to _view_ the attention scores; (any hooks? or callbacks?)
- [ ] Network type comparison
  - [ ] Add a train script for MLP

