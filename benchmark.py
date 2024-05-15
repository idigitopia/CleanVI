import argparse
from vi_core.env_frozen_lake import FrozenLakeEnvDynamic, plot_policy_image
from vi_core.hardware_log_utils import get_gpu_name
import time
import socket
import torch 
import subprocess


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Solve MDP for Frozen Lake environment.')
    parser.add_argument('--map_size', type=int, nargs=2, default=[25, 25], help='Size of the map')
    parser.add_argument('--h_prob', type=float, default=0.05, help='Probability of a hole')
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--gamma", type=float, default=0.99, help='Discount Factor')
    parser.add_argument("--epsilon", type=float, default=0.001, help='Residual error to end Value iteration')
    parser.add_argument("--mode", type=str, default="python", help="Head file to use")
    args = parser.parse_args()

    # Define MDP
    if args.mode == "python":
        from vi_core.vi_python import main
    elif args.mode == "numpy":
        from vi_core.vi_numpy import main
    elif args.mode == "torch_cpu":
        from vi_core.vi_torch_cpu import main
    elif args.mode == "torch_gpu":
        from vi_core.vi_torch_gpu import main
    elif args.mode == "jax":
        from vi_core.vi_jax import main
    elif args.mode == "mlx":
        from vi_core.vi_mlx import main
    else:
        raise ValueError("Invalid mode")
    
    result_dict = main(args)
    result_dict.update(args.__dict__)
    result_dict.update({"hostname": socket.gethostname()})
    result_dict.update({"Date": time.ctime()})
    result_dict.update({"exp_signature": f"{result_dict['hostname']}_{result_dict['mode']}_{time.time()}"})    
    result_dict.update({"gpu": get_gpu_name()})
    
    # append result in a results file
    # open file and write result dict to a file 
    with open(f"results/results{result_dict['exp_signature']}.txt", "a") as f:
        f.write(str(result_dict) + "\n")