import jax.numpy as jnp
import time
from tqdm import tqdm
import argparse
import sys 
import os

class MDP_CORE():

    def bellman_backup_operator(Ti, Tp, R, Q, V, gamma):
        Q = jnp.sum(jnp.multiply(R, Tp), axis=2) + \
            gamma * jnp.sum(jnp.multiply(Tp, V[Ti]), axis=2)
        V_prime = jnp.max(Q, axis=1)
        Pi = jnp.argmax(Q, axis=1)
        return Q, Pi, V_prime

    def bellman_backup_operator_with_eps(Ti, Tp, R, Q, V, gamma):
        Q = jnp.sum(jnp.multiply(R, Tp), axis=2) + \
            gamma * jnp.sum(jnp.multiply(Tp, V[Ti]), axis=2)
        V_prime = jnp.max(Q, axis=1)
        Pi = jnp.argmax(Q, axis=1)
        epsilon = jnp.max(jnp.abs(V_prime - V))
        return epsilon, Q, Pi, V_prime

    def __init__(self, n_states, n_tran_types, n_tran_targets):
        nn, aa, tt = n_states, n_tran_types, n_tran_targets
        self.S = jnp.arange(nn).astype(jnp.int64)

        self.Ti = jnp.zeros((nn, aa, tt), dtype=jnp.int64)
        self.Tp = jnp.zeros((nn, aa, tt), dtype=jnp.float32)
        self.R = jnp.zeros((nn, aa, tt), dtype=jnp.float32)
        self.Q = jnp.zeros((nn, aa), dtype=jnp.float32)
        self.V = jnp.zeros(nn, dtype=jnp.float32)
        self.Pi = jnp.zeros(nn, dtype=jnp.int64)

        self.gamma = 0.99

    def single_bellman_backup_computation_with_eps(self):
        self.curr_error, self.Q, self.Pi, self.V = MDP_CORE.bellman_backup_operator_with_eps(
            self.Ti, self.Tp, self.R, self.Q, self.V, self.gamma)

    def single_bellman_backup_computation(self):
        self.Q, self.Pi, self.V = MDP_CORE.bellman_backup_operator(
            self.Ti, self.Tp, self.R, self.Q, self.V, self.gamma)

    def solve(self, max_n_backups=500, gamma=0.99, epsilon=0.001, 
              reset_values=False, verbose=False, bellman_backup_batch_size=25):
        st = time.time()

        if reset_values:
            self.reset_value_vectors()

        self.gamma = gamma
        v_iter = tqdm(range(max_n_backups), desc="Solving MDP", unit="backup") if verbose else range(max_n_backups)

        for i in v_iter:
            self.single_bellman_backup_computation()
            if verbose and i % bellman_backup_batch_size == 0:
                self.single_bellman_backup_computation_with_eps()
                if verbose:
                    v_iter.set_postfix({"Current Error": self.curr_error})
                if self.curr_error < epsilon:
                    break
        
        et = time.time()
        if verbose:
            GREEN = '\033[92m'
            RESET = '\033[0m'
            print(f"Solved MDP in {i} Backups, {GREEN}{et-st:.2f} Seconds{RESET}, Eps: {self.curr_error}")



    def reset_value_vectors(self):
        nn, aa, tt = self.Ti.shape
        self.Q = jnp.zeros((nn, aa), dtype=jnp.float32)
        self.V = jnp.zeros(nn, dtype=jnp.float32)
        self.Pi = jnp.zeros(nn, dtype=jnp.int64)

def main(args):
    sys.path.append(os.getcwd())
    
    print("#### Benchmarking MDP Solver for JAX GPU ####")
    
    from vi_core.env_frozen_lake import FrozenLakeEnvDynamic, plot_policy_image

    # Define Environment
    env = FrozenLakeEnvDynamic(map_size=tuple(args.map_size), h_prob=args.h_prob)
    Ti, Tp, Tr = env.calculate_transition_matrices()

    # Define MDP
    mdp = MDP_CORE(n_states=len(env.all_states),
                n_tran_types=len(env.all_actions), 
                n_tran_targets=4, # max number of states tran prob can be non-zero 
                )
    mdp.Ti[:], mdp.Tp[:], mdp.R[:] = jnp.array(Ti), jnp.array(Tp), jnp.array(Tr)
    mdp.solve(gamma=0.9975, verbose=True, max_n_backups=10000, 
            bellman_backup_batch_size=25)

    if not args.headless:
        plot_policy_image(mdp.V, mdp.Pi, env.map_grid, show_policy= env.map_size[0]<50)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve MDP for Frozen Lake environment.')
    parser.add_argument('--map_size', type=int, nargs=2, default=[25, 25], help='Size of the map')
    parser.add_argument('--h_prob', type=float, default=0.05, help='Probability of a hole')
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()
    main(args)