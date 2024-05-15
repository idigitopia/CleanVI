import time
from tqdm import tqdm
import argparse
import os 
import sys

class MDP_CORE():

    @staticmethod
    def bellman_backup_operator(Ti, Tp, R, Q, V, gamma):
        nn, aa, _ = len(Q), len(Q[0]), len(Tp[0][0])
        for s in range(nn):
            for a in range(aa):
                sum_term = sum(R[s][a][t] * Tp[s][a][t] for t in range(len(Tp[s][a])))
                sum_term += gamma * sum(Tp[s][a][t] * V[Ti[s][a][t]] for t in range(len(Ti[s][a])))
                Q[s][a] = sum_term

        V_prime = [max(Q[s]) for s in range(nn)]
        Pi = [Q[s].index(max(Q[s])) for s in range(nn)]
        return Q, Pi, V_prime

    @staticmethod
    def bellman_backup_operator_with_eps(Ti, Tp, R, Q, V, gamma):
        
        nn, aa, _ = len(Q), len(Q[0]), len(Tp[0][0])
        for s in range(nn):
            for a in range(aa):
                sum_term = sum(R[s][a][t] * Tp[s][a][t] for t in range(len(Tp[s][a])))
                sum_term += gamma * sum(Tp[s][a][t] * V[Ti[s][a][t]] for t in range(len(Ti[s][a])))
                Q[s][a] = sum_term

        V_prime = [max(Q[s]) for s in range(nn)]
        Pi = [Q[s].index(max(Q[s])) for s in range(nn)]
        epsilon = max(abs(V_prime[s] - V[s]) for s in range(nn))
        return epsilon, Q, Pi, V_prime

    def __init__(self, n_states, n_tran_types, n_tran_targets):
        self.Ti = [[[0]*n_tran_targets for _ in range(n_tran_types)] for _ in range(n_states)]
        self.Tp = [[[0.0]*n_tran_targets for _ in range(n_tran_types)] for _ in range(n_states)]
        self.R = [[[0.0]*n_tran_targets for _ in range(n_tran_types)] for _ in range(n_states)]
        self.Q = [[0.0]*n_tran_types for _ in range(n_states)]
        self.V = [0.0 for _ in range(n_states)]
        self.Pi = [0 for _ in range(n_states)]
        self.gamma = 0.99

    def single_bellman_backup_computation_with_eps(self):
        self.curr_error, self.Q, self.Pi, self.V = self.bellman_backup_operator_with_eps(
            self.Ti, self.Tp, self.R, self.Q, self.V, self.gamma)

    def single_bellman_backup_computation(self):
        self.Q, self.Pi, self.V = self.bellman_backup_operator(
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
            print(f"Solved MDP in {i} Backups, {et-st:.2f} Seconds, Eps: {self.curr_error}")

        return {"Time Elapsed": et-st, "Backups": i, "Error": self.curr_error}


    def reset_value_vectors(self):
        nn, aa = len(self.Ti), len(self.Ti[0])
        self.Q = [[0.0]*aa for _ in range(nn)]
        self.V = [0.0 for _ in range(nn)]
        self.Pi = [0 for _ in range(nn)]


def main(args):
    sys.path.append(os.getcwd())
    
    print("#### Benchmarking MDP Solver for Pure Python ####")
    
    from vi_core.env_frozen_lake import FrozenLakeEnvDynamic, plot_policy_image

    # Define Environment
    env = FrozenLakeEnvDynamic(map_size=tuple(args.map_size), h_prob=args.h_prob)
    Ti, Tp, Tr = env.calculate_transition_matrices()

    # Define MDP
    mdp = MDP_CORE(n_states=len(env.all_states),
               n_tran_types=len(env.all_actions), 
               n_tran_targets=4) 
    mdp.Ti, mdp.Tp, mdp.R = Ti, Tp, Tr
    result_dict = mdp.solve(gamma=args.gamma,
                            verbose=True, 
                            max_n_backups=10000, 
                            bellman_backup_batch_size=25)
    
    if not args.headless:
        plot_policy_image(mdp.V, mdp.Pi, env.map_grid, show_policy= env.map_size[0]<50)

    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve MDP for Frozen Lake environment.')
    parser.add_argument('--map_size', type=int, nargs=2, default=[25, 25], help='Size of the map')
    parser.add_argument('--h_prob', type=float, default=0.05, help='Probability of a hole')
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--gamma", type=float, default=0.99, help='Discount Factor')
    parser.add_argument("--epsilon", type=float, default=0.001, help='Residual error to end Value iteration')
    args = parser.parse_args()
    main(args)