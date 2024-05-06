import argparse
from vi_core.env_frozen_lake import FrozenLakeEnvDynamic, plot_policy_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve MDP for Frozen Lake environment.')
    parser.add_argument('--map_size', type=int, nargs=2, default=[25, 25], help='Size of the map')
    parser.add_argument('--h_prob', type=float, default=0.05, help='Probability of a hole')
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--mode", type=str, default="python", help="Head file to use")
    args = parser.parse_args()

    # Define MDP
    if args.mode == "python":
        from vi_core.vi_python import main
    elif args.mode == "numpy":
        from vi_core.vi_numpy import main
    elif args.mode == "torch_cpu":
        from vi_core.vi_torch_cpu import main
    elif args.mode == "mlx":
        from vi_core.vi_mlx import main
    else:
        raise ValueError("Invalid mode")
    
    main(args)