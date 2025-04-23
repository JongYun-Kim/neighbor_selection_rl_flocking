# neighbor_selection_rl_flocking

## TODOs
- [ ] Re-configure/clone files
  - [ ] Env: get it from lazy-message-listener-flocking
  - [ ] Model: get it from lazy-message-listener-flocking
  - [ ] Model: and update the names and clean it up
  - [ ] Model: update it to be able to use address arbitrary numbers
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

