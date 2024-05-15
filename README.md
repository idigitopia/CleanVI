# CleanVI : MDP Solver Benchmark
## Clean Single File Implementation of Value Iteration


This repository contains code for single file implementations for benchmarking an MDP (Markov Decision Process) solver. The solvers are tested on the Frozen Lake environment, which is a dynamic grid-based environment with configurable map size and probability of holes.

### Features

- Implements a core MDP solver class MDP_CORE with support for GPU acceleration.
- Benchmarks the MDP solver on the Frozen Lake environment with configurable map size and hole probability.
- Optionally plots the optimal policy and value function for the solved MDP.


### Usage
To run the MDP solver benchmark for a particular mode run

``` python benchmark.py --gamma 0.9975 --epsilon 0.001 --map_size 100 100 --headless --mode torch_cpu```

The benchmark script will solve the MDP for the Frozen Lake environment using the specified parameters. It will display the time elapsed, number of backups performed, and the residual error upon convergence.
If not running in headless mode, the script will also plot the optimal policy and value function for the solved MDP


### Hardware Support 

The MDP solver has been implemented on different hardware types and frameworks. 

- CPU
  - [X] Pure Python
  - [X] Numpy
  - [X] Torch
- GPU
  - [X] Torch (NVIDIA)
  - [X] MLX (APPLE)
  - [ ] Jax (NVIDIA)
  - [ ] CUDA Kernel (NVIDIA)


## Benchmark Results

The following table shows the solve time for different modes of the MDP solver:

| Mode | Map Size | Hole Probability | Discount Factor | Epsilon | Solve Time (seconds) | Speedup | 
|------|----------|------------------|-----------------|---------|----------------------|---------|
| pure python | 1000x1000 | 0.05 | 0.9975 | 0.001 | >5000s | - |
| torch_cpu | 1000x1000 | 0.05 | 0.9975 | 0.001 | 26.91s | 1x |
| torch_gpu | 1000x1000 | 0.05 | 0.9975 | 0.001 | 1.59s | 17x |
