from os import device_encoding
import os
import torch
from tqdm import tqdm
from functools import partial
from munch import Munch
import time
import argparse


class MDP_CORE():

    ######################## Nomenclature ########################################
    # nn, aa, tt = self.S.len, n_tran_types, n_tran_targets
    # self.Tp = torch.zeros((nn, aa, tt)).type(torch.float32).to(device=device)
    # self.Ti = torch.zeros((nn, aa, tt)).type(torch.LongTensor).to(device=device)
    # self.R = torch.zeros((nn, aa, tt)).type(torch.float32).to(device=device)
    # self.Tdist = torch.zeros((nn, aa, tt)).type(torch.float32).to(device=device) # Distances for s,a,s' approximations
    # self.Q = torch.zeros((nn, aa)).type(torch.float32).to(device=device)
    # self.V = torch.zeros((nn,)).type(torch.float32).to(device=device)
    # self.Pi = torch.zeros((nn)).type(torch.LongTensor).to(device=device)
    # self.V_safe = torch.zeros((nn,)).type(torch.float32).to(device=device)
    # self.Pi_safe = torch.zeros((nn)).type(torch.LongTensor).to(device=device)
    ##############################################################################

    ########################### Rabbit King ####################################
    # @torch.jit.script
    def bellman_backup_operator(Ti, Tp, R, Q, V, gamma):
        Q = torch.sum(torch.multiply(R, Tp), dim=2) + \
            gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
        max_obj = torch.max(Q, dim=1)
        V_prime, Pi = max_obj.values, max_obj.indices
        return Q, Pi, V_prime

    def bellman_backup_operator_with_eps(Ti, Tp, R, Q, V, gamma):
        Q = torch.sum(torch.multiply(R, Tp), dim=2) + \
            gamma[0]*torch.sum(torch.multiply(Tp, V[Ti]), dim=2)
        max_obj = torch.max(Q, dim=1)
        V_prime, Pi = max_obj.values, max_obj.indices
        epsilon = torch.max(torch.abs(V_prime-V))
        return epsilon, Q, Pi, V_prime
    ############################################################################

    def __init__(self, n_states, n_tran_types, n_tran_targets, device='cuda'):
        # ToDo Some sanity checkes for transitions

        super().__init__()

        self.device = device

        self.dac_constants = Munch()
        self.dac_constants.n_tran_types = n_tran_types
        self.dac_constants.n_tran_targets = n_tran_targets

        # Core States
        nn,aa,tt = n_states, n_tran_types, n_tran_targets
        self.S = torch.arange(nn).type(torch.LongTensor).to(self.device)

        # MDP Tensors
        self.Ti = torch.zeros(nn,aa,tt).type(torch.LongTensor).to(self.device) # (nn, aa, tt)  ||  Transiton Indexes
        self.Tp = torch.zeros(nn,aa,tt).type(torch.float32).to(self.device) # (nn, aa, tt)  ||  Transition Probabilities
        self.R = torch.zeros(nn,aa,tt).type(torch.float32).to(self.device) # (nn, aa, tt)  ||  Transition Rewards
        self.Q = torch.zeros(nn,aa).type(torch.float32).to(self.device) # (nn, aa)  ||  Q values for each state tran_type pair
        self.V = torch.zeros(nn).type(torch.float32).to(self.device) # (nn,)  ||  Value Vector
        self.Pi = torch.zeros(nn).type(torch.LongTensor).to(self.device) # (nn,)  ||  Policy Vector

        # MDP Solve helper variables
        self.gamma = torch.FloatTensor([0.99]).to(device)
            
    def single_bellman_backup_computation_with_eps(self):
        self.curr_error, self.Q, self.Pi, self.V = MDP_CORE.bellman_backup_operator_with_eps(self.Ti, self.Tp, self.R, self.Q, self.V, self.gamma)

    def single_bellman_backup_computation(self):
        self.Q, self.Pi, self.V = MDP_CORE.bellman_backup_operator(self.Ti, self.Tp, self.R, self.Q, self.V, self.gamma)

    def solve(self, max_n_backups=500, gamma=0.99, epsilon=0.001, reset_values = False, verbose=False, bellman_backup_batch_size=250) -> None:
        st = time.time()
        if reset_values:
            self.reset_value_vectors()

        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        v_iter = tqdm(range(max_n_backups), desc="Solving MDP", unit="backup") if verbose else range(max_n_backups)

        for i in v_iter:
            self.single_bellman_backup_computation()
            if verbose and i % bellman_backup_batch_size == 0:
                self.single_bellman_backup_computation_with_eps()
                if verbose:
                    v_iter.set_postfix({"Current Error": self.curr_error.cpu().item()})
                if self.curr_error.cpu() < epsilon:
                    break
        et = time.time()
        if verbose:
            print(f"Solved MDP in {i} Backups, {et-st:.2f} Seconds, Eps: {self.curr_error.cpu().item()}")

    # Extenstion Functions 
    def reset_value_vectors(self):
        nn,aa,tt = self.Ti.shape
        self.Q = torch.zeros((nn, aa)).type(torch.float32).to(device=self.device)
        self.V = torch.zeros((nn,)).type(torch.float32).to(device=self.device)
        self.C = torch.zeros((nn,)).type(torch.float32).to(device=self.device) # Cost of optimal policy for each state.
        self.Pi = torch.zeros((nn)).type(torch.LongTensor).to(device=self.device)


if __name__ == "__main__":
    from vi_core.env_frozen_lake import FrozenLakeEnvDynamic, plot_policy_image

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Solve MDP for Frozen Lake environment.')
    parser.add_argument('--map_size', type=int, nargs=2, default=[25, 25], help='Size of the map (default: [200, 200])')
    parser.add_argument('--h_prob', type=float, default=0.05, help='Probability of a hole (default: 0.05)')
    args = parser.parse_args()

    # Define Environment
    env = FrozenLakeEnvDynamic(map_size=tuple(args.map_size), h_prob=args.h_prob)
    Ti, Tp, Tr = env.calculate_transition_matrices()

    # Define MDP
    mdp = MDP_CORE(n_states=len(env.all_states),
                n_tran_types=len(env.all_actions), 
                n_tran_targets=4, # max number of states tran prob can be non-zero 
                device='cpu')
    mdp.Ti[:], mdp.Tp[:], mdp.R[:] = torch.Tensor(Ti), torch.Tensor(Tp), torch.Tensor(Tr)
    mdp.solve(gamma=0.9975, verbose=True, max_n_backups=10000, 
            bellman_backup_batch_size=25)

    plot_policy_image(mdp.V, mdp.Pi, env.map_grid, show_policy= env.map_size[0]<50)