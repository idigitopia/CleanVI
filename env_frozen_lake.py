import pickle
import sys
from statistics import mean
import matplotlib.pyplot as plt
import pickle as pk
import math
import numpy as np
from collections import namedtuple
import time
from tqdm import tqdm
import torch 

class FrozenLakeEnvDynamic():

    def __init__(self, map_size: tuple, h_prob=0.1):
        """
        Frozen Lake Environment
        :param map_size: Tuple of (Width, Height)
        Generates a random map with the given size, with holes and goal location. 
        The environment is a grid where each cell can be either safe (F), a hole (H), the start (S),
        or the goal (G). The agent starts at the start cell and aims 
        to reach the goal cell without falling into a hole.
        """
        self.map_grid = FrozenLakeEnvDynamic.generate_map(map_size, h_prob=h_prob)
        self.map_size = map_size
        self.nrow = map_size[0]
        self.ncol = map_size[1]
        self.state = 0

    @property
    def all_states(self):
        all_states = list(range(self.map_size[0] * self.map_size[1]))
        return all_states

    @property
    def all_actions(self):
        """
            0:UP
            1:Right
            2:Down
            3:Left
        """
        return list(range(4))

    def state_id_to_row_col(self, state):
        """ Takes in the state id and returns the row, col of the state"""
        return (math.floor(state / self.ncol), int(state % self.nrow))

    def row_col_to_state_id(self, row, col):
        """ Takes in the row, col and returns the state id """
        return row * self.ncol + col

    def sample_random_action(self):
        """ Returns a random action """
        return np.random.randint(4)

    def next_row_col(self, row, col, a):
        """ Takes in the current row, col and action and returns the next row, col
        Uses the action and environment boundaries to determine the next row, col

        Args:
            row (int): Row index
            col (int): Col index
            a (int): Action index

        Returns:
            int, int: Next row, col
        """
        if self.map_grid[row][col] in ["G", "H"]:
            return (row, col)

        action_map = {
            0: (max(row - 1, 0), col),  # Up
            1: (row, min(col + 1, self.ncol - 1)),  # Right
            2: (min(row + 1, self.nrow - 1), col),  # Down
            3: (row, max(col - 1, 0))  # Left
        }

        # print("NEXT:",row,col, action_map[a])
        return action_map[a]

    def get_successors(self, s, a):
        """ Returns the successors of the state s given the action a """
        successors = []
        row, col = self.state_id_to_row_col(s)

        succ_prob_map = {"F": 0.7, "H": 0.0001, "G": 0.7, "S": 0.7}
        succ_prob = succ_prob_map[self.map_grid[row][col]]
        fail_prob = (1 - succ_prob) / 3

        for r_a in self.all_actions:
            next_row, next_col = self.next_row_col(row, col, r_a)
            # print("next:", next_row, next_col)
            next_s = self.row_col_to_state_id(next_row, next_col)
            # print("tran", s, r_a,next_row, next_col,  next_s, r_a)
            successors.append([next_s, succ_prob] if r_a == a else [next_s,fail_prob])
        return successors

    def get_reward(self, s, a, s_prime):
        row, col = self.state_id_to_row_col(s)
        row_prime, col_prime = self.state_id_to_row_col(s_prime)
        # print("checking:" , row_prime, col_prime)
        s_cell = self.map_grid[row_prime][col_prime]
        # if s_cell in "GH": 
            # return 0
        
        t_cell = self.map_grid[row_prime][col_prime]
        reward_map = {
            "H": -1,  # Hole
            "G": 1,  # Goal
            "F": -1,  # Safe
            "S": -1,  # Start
        }

        return reward_map[t_cell]

    def reset(self):
        self.state = 0
        return np.array([self.state])

    def step(self, a):
        done = False

        row, col = self.state_id_to_row_col(self.state)
        if self.map_grid[row][col] == "G" or self.map_grid[row][col] == "H":
            done = True
            next_state = self.all_states[-1]
        else:
            successors = self.get_successors(self.state, a)
            states = [s[0] for s in successors]
            probs = [s[1] for s in successors]
            next_state = np.random.choice(states, 1, p=probs)[0]

        reward = self.get_reward(self.state, a)
        info = {}
        self.state = next_state
        return np.array([next_state]), reward, done, info


    @staticmethod
    def generate_map(shape, h_prob=0.1):
        """

        :param shape: Width x Height
        :return: List of text based map
        """
        # h_prob = 0.1
        grid_map = []

        for h in range(shape[1]):

            if h == 0:
                row = 'SF'
                row += generate_row(shape[0] - 2, h_prob)
            elif h == 1:
                row = 'FF'
                row += generate_row(shape[0] - 2, h_prob)

            elif h == shape[1] - 1:
                row = generate_row(shape[0] - 2, h_prob)
                row += 'FG'
            elif h == shape[1] - 2:
                row = generate_row(shape[0] - 2, h_prob)
                row += 'FF'
            else:
                row = generate_row(shape[0], h_prob)

            grid_map.append(row)
            del row

        return grid_map


    def calculate_transition_matrices(env, n_tran_targets=4):
        list_of_states = env.all_states
        list_of_actions = env.all_actions
        n_states = len(list_of_states)
        n_actions = len(list_of_actions)

        Ti = [[[0]*n_tran_targets for _ in range(n_actions)] for _ in range(n_states)]
        Tp = [[[0]*n_tran_targets for _ in range(n_actions)] for _ in range(n_states)]
        Tr = [[[0]*n_tran_targets for _ in range(n_actions)] for _ in range(n_states)]

        for si, s in tqdm(enumerate(list_of_states)):
            for ai, a in enumerate(list_of_actions):
                successors = env.get_successors(s, a)
                for ti, (s_prime, p) in enumerate(successors):
                    if ti < n_tran_targets:
                        Ti[si][ai][ti] = s_prime
                        Tp[si][ai][ti] = p
                        Tr[si][ai][ti] = env.get_reward(s, a, s_prime)

        return Ti, Tp, Tr
    
    # Transition Dictionary helper functions
    def get_tran_reward_dict(env):
        list_of_states = env.all_states
        list_of_actions = env.all_actions
        tran_reward_dict = {}

        # get transition dictionary

        for s in tqdm(list_of_states):
            tran_reward_dict[s] = {}
            for a in list_of_actions:
                tran_reward_dict[s][a] = {}
                successors = env.get_successors(s, a)
                for ns, p in successors:
                    if ns not in tran_reward_dict[s][a]:
                        tran_reward_dict[s][a][ns] = (p,env.get_reward(s, a, ns))
                    else:
                        _p, _r = tran_reward_dict[s][a][ns]
                        tran_reward_dict[s][a][ns] = (_p + p, _r)
        return tran_reward_dict



# Map Helper Functions

def generate_row(length, h_prob):
    row = np.random.choice(2, length, p=[1.0 - h_prob, h_prob])
    row = ''.join(list(map(lambda z: 'F' if z == 0 else 'H', row)))
    return row



# Print and Evaluate Helper Functions

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=2)

def evaluate_policy(env, policy, trials=10):
    total_reward = 0
    #     epoch = 10
    max_steps = 500

    for _ in range(trials):
        steps = 0
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while (not done) and (steps < max_steps):
            observation, reward, done, info = env.step(policy[observation[0]])
            total_reward += reward
            steps += 1
            # if(steps%100) == 0 :
            #     print(steps)
    return total_reward / trials


def evaluate_policy_discounted(env, policy, discount_factor, trials=10):
    epoch = 10
    reward_list = []
    max_steps = 500

    for _ in range(trials):
        steps = 0
        total_reward = 0
        trial_count = 0
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while (not done) and (steps < max_steps):
            observation, reward, done, info = env.step(policy[observation[0]])
            total_reward += (discount_factor ** trial_count) * reward
            trial_count += 1
            steps += 1
            # if(steps%100) == 0 :
                # print(steps)
        reward_list.append(total_reward)

    return mean(reward_list)


def print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np).reshape((map_size, map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np).reshape((map_size, map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor=beta)))
    print("State Value image view:\n")
    plt.imshow(np.array(v_np).reshape((map_size, map_size)))

    pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))


def save_and_print_results(v, pi, MAP, env, beta, name, show_val=False, show_pi=False,
                           results_dir="results/frozen_lake/"):
    map_size = len(MAP)
    v_np, pi_np = np.array(v), np.array(pi)
    if (show_val):
        print("\nState Value:\n")
        print(np.array(v_np).reshape((map_size, map_size)))
    if (show_pi):
        print("\nPolicy:\n")
        print(np.array(pi_np).reshape((map_size, map_size)))

    avg_reward = evaluate_policy(env, pi)
    avg_discounted_reward = evaluate_policy_discounted(env, pi, discount_factor=beta)
    print("\nAverage reward: {}\n".format(avg_reward))
    print("Avereage discounted reward: {}\n".format(avg_discounted_reward))
    print("State Value image view:\n")

    plt.imsave(results_dir + "value_" + str(map_size) + ".png", rescale_data(np.array(v_np).reshape((map_size, map_size))))
    pickle.dump(v, open(results_dir + name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(results_dir + name + "_" + str(map_size) + "_pi.pkl", "wb"))

    plot_policy_image(v,pi,MAP,results_dir, save=True)

    return avg_reward, avg_discounted_reward

def save_results(v, map_size):
    v_np = np.array(v)
    plt.imsave("latest_fig.png", np.array(v_np).reshape((map_size, map_size)), dpi=400)

def rescale_data(data):
    scale = int(1000/len(data))
    new_data = np.zeros(np.array(data.shape) * scale)
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            new_data[j * scale: (j+1) * scale, k * scale: (k+1) * scale] = data[j, k]
    return new_data


def plot_policy_image(value, pi, MAP, results_dir="results/frozen_lake/", show_policy =True, save = False):
    best_value = np.array(value).reshape(len(MAP), len(MAP))
    best_policy = np.array(pi).reshape(len(MAP), len(MAP))

    print("\n\nBest Q-value and Policy:\n")
    fig, ax = plt.subplots()
    im = ax.imshow(best_value)

    if show_policy:
        for i in range(best_value.shape[0]):
            for j in range(best_value.shape[1]):
                if MAP[i][j] in 'GH':
                    arrow = MAP[i][j]
                elif best_policy[i, j] == 0:
                    arrow = '^'
                elif best_policy[i, j] == 1:
                    arrow = '>'
                elif best_policy[i, j] == 2:
                    arrow = 'V'
                elif best_policy[i, j] == 3:
                    arrow = '<'
                if MAP[i][j] in 'S':
                    arrow = 'S ' + arrow
                text = ax.text(j, i, arrow,
                            ha="center", va="center", color="black")

    cbar = ax.figure.colorbar(im, ax=ax)

    fig.tight_layout()
    if save:
        plt.savefig(results_dir + "policy_" + str(len(MAP)) + ".png",)
        plt.show()  
    else:
        plt.show()


def get_performance_log(v, m, w, VI_time, matrix_time, results_dir="results/frozen_lake/", store = True):
    """
    :param v: VI Engine
    :param m: Map Size
    :param w: Workers Number
    :param VI_time: Time taken for VI to complete
    :param matrix_time: Time taken to calculate transition and reward Dictionaries
    :param results_dir:
    :param store: Save to disk Flag
    :return:
    """
    try:
        performance_log = pk.load(open(results_dir + "performance_log.pk", "rb"))
    except:
        performance_log = {}
    # print("loading",performance_log)
    performance_log[v] = {} if v not in performance_log else performance_log[v]
    performance_log[v][w] = {} if w not in performance_log[v] else performance_log[v][w]
    performance_log[v][w][m] = {} if m not in performance_log[v][w] else performance_log[v][w][m]
    performance_log[v][w][m]["matrix_time"] = [] if "matrix_time" not in performance_log[v][w][m] else performance_log[v][w][m]["matrix_time"]
    performance_log[v][w][m]["VI_time"] = [] if "VI_time" not in performance_log[v][w][m] else performance_log[v][w][m]["VI_time"]
    performance_log[v][w][m]["matrix_time"].append(matrix_time)
    performance_log[v][w][m]["VI_time"].append(VI_time)

    if(store):
        pk.dump(performance_log, open(results_dir + "performance_log.pk", "wb"))

    return performance_log

def process_log_data(perf_log):
    data = []
    for vi_engine in perf_log:
        for worker_num  in perf_log[vi_engine]:
            data_x = []
            data_y = []
            for map_size in perf_log[vi_engine][worker_num]:
                num_of_states = map_size[0]**2
                avg_runtime = mean(perf_log[vi_engine][worker_num][map_size]["VI_time"])
                data_x.append(num_of_states)
                data_y.append(avg_runtime)
            data.append((data_x, data_y, vi_engine + str(worker_num)))
    return data