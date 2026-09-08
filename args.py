from argparse import ArgumentParser
import json

def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

def parse_args(argv=None):
    parser = ArgumentParser(description="DCVRP Solver Configuration")

    parser.add_argument("--config-file", "-f", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="Enable verbose output")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA")
    parser.add_argument("--rng-seed", type=int, default=1234,
                        help="Random number generator seed")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID to use, if available")
    # Data generation parameters
    data_group = parser.add_argument_group("Data generation parameters")
    data_group.add_argument("--problem-type", "-p", type=str,
                            default="DCVRP",
                            help="Type of VRP problem")
    data_group.add_argument("--customers-count", "-n", type=int, default=20,
                            help="Number of customers")
    data_group.add_argument("--vehicles-count", "-m", type=int, default=4,
                            help="Number of vehicles")
    data_group.add_argument("--veh-capa", type=int, default=150,
                            help="Vehicle capacity")
    data_group.add_argument("--veh-speed", type=int, default=1,
                            help="Vehicle speed")
    data_group.add_argument("--horizon", type=int, default=480,
                            help="Time horizon")
    data_group.add_argument("--min-cust-count", type=int, default=None,
                            help="Minimum customer count")
    data_group.add_argument("--loc-range", type=int, nargs=2, default=(0, 101),
                            help="Location range")
    data_group.add_argument("--dem-range", type=int, nargs=2, default=(5, 41),
                            help="Demand range")
    data_group.add_argument("--dur-range", type=int, nargs=2, default=(10, 31),
                            help="Duration range")
    # DVRP Environment parameters
    env_group = parser.add_argument_group("DVRP Environment parameters")
    env_group.add_argument("--pending-cost", type=float, default=5,
                           help="Pending cost")
    env_group.add_argument("--pend-cost-growth", type=float, default=None,
                           help="Pending cost growth rate")

    # Model parameters
    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument("--model-size", "-s", type=int, default=128,
                             help="Model size")
    model_group.add_argument("--layer-count", type=int, default=3,
                             help="Number of layers")
    model_group.add_argument("--head-count", type=int, default=8,
                             help="Number of attention heads")
    model_group.add_argument("--ff-size", type=int, default=512,
                             help="Feed-forward layer size")
    model_group.add_argument("--tanh-xplor", type=float, default=10,
                             help="Tanh exploration parameter")
    model_group.add_argument("--aggregator-type", type=str, default="argmax",
                             choices=["argmax", "mlp", "attention"],
                             help="Vehicle selection aggregation method (argmax/mlp/attention)")

    # Training parameters
    train_group = parser.add_argument_group("Training parameters")
    train_group.add_argument("--epoch-count", "-e", type=int, default=100,
                             help="Number of epochs")
    train_group.add_argument("--iter-count", "-i", type=int, default=1000,
                             help="Number of iterations")
    train_group.add_argument("--batch-size", "-b", type=int, default=100,
                             help="Batch size")
    train_group.add_argument("--learning-rate", "-r", type=float, default=0.0001,
                             help="Learning rate")
    train_group.add_argument("--rate-decay", "-d", type=float, default=None,
                             help="Learning rate decay")
    train_group.add_argument("--max-grad-norm", type=float, default=2,
                             help="Maximum gradient norm")
    train_group.add_argument("--grad-norm-decay", type=float, default=None,
                             help="Gradient norm decay")
    train_group.add_argument("--loss-use-cumul", action="store_true", default=False,
                             help="Use cumulative loss")

    # Baselines parameters
    baseline_group = parser.add_argument_group("Baselines parameters")
    baseline_group.add_argument("--baseline-type", type=str,
                                choices=["none", "nearnb", "rollout", "critic"],
                                default="rollout",
                                help="Type of baseline")
    baseline_group.add_argument("--rollout-count", type=int, default=3,
                                help="Number of rollouts")
    baseline_group.add_argument("--rollout-threshold", type=float, default=0.05,
                                help="Rollout threshold")
    baseline_group.add_argument("--critic-use-qval", action="store_true", default=False,
                                help="Use Q-value for critic")
    baseline_group.add_argument("--critic-rate", type=float, default=0.001,
                                help="Critic learning rate")
    baseline_group.add_argument("--critic-decay", type=float, default=None,
                                help="Critic decay rate")

    # Testing parameters
    test_group = parser.add_argument_group("Testing parameters")
    test_group.add_argument("--test-batch-size", type=int, default=128,
                            help="Test batch size")

    # Checkpointing
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument("--output-dir", "-o", type=str, default=None,
                                  help="Output directory")
    checkpoint_group.add_argument("--checkpoint-period", "-c", type=int, default=5,
                                  help="Checkpoint period")
    checkpoint_group.add_argument("--resume-state", type=str, default=None,
                                  help="Resume from state file")

    args = parser.parse_args(argv)

    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))
        args = parser.parse_args(argv)

    return args