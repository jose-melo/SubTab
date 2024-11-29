"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: - Collects arguments from command line, and loads configuration from the yaml files.
             - Prints a summary of all options and arguments.
"""

from argparse import ArgumentParser
import sys
import torch as th
from utils.utils import get_runtime_and_model_config, print_config


class ArgParser(ArgumentParser):
    """Inherits from ArgumentParser, and used to print helpful message if an error occurs"""

    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def get_arguments():
    """Gets command line arguments"""

    # Initialize parser
    parser = ArgParser()

    # Dataset can be provided via command line
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        help="Name of the dataset to use. It should have a config file with the same name.",
    )

    # Whether to use GPU.
    parser.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        action="store_true",
        help="Used to assign GPU as the device, assuming that GPU is available",
    )

    parser.add_argument(
        "-ng", "--no_gpu", dest="gpu", action="store_false", help="Used to assign CPU as the device"
    )

    parser.set_defaults(gpu=True)

    # GPU device number as in "cuda:0". Defaul is 0.
    parser.add_argument(
        "-dn",
        "--device_number",
        type=str,
        default="0",
        help="Defines which GPU to use. It is 0 by default",
    )

    # Experiment number if MLFlow is on
    parser.add_argument(
        "-ex",
        "--experiment",
        type=int,
        default=1,
        help="Used as a suffix to the name of MLFlow experiments if MLFlow is being used",
    )

    parser.add_argument("--n_dims", type=int, default=3, help="Number of dimensions in the dataset")

    parser.add_argument(
        "--hidden_dim_0",
        type=int,
        default=128,
        help="Number of hidden units in the first hidden layer",
    )
    parser.add_argument(
        "--hidden_dim_1",
        type=int,
        default=64,
        help="Number of hidden units in the second hidden layer",
    )
    parser.add_argument(
        "--hidden_dim_2",
        type=int,
        default=32,
        help="Number of hidden units in the third hidden layer",
    )

    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--aggregation", type=str, default="mean", help="Aggregation method")
    parser.add_argument("--noise_type", type=str, default="swap_noise", help="Noise type")
    parser.add_argument("--masking_ratio", type=float, default=0.2, help="Masking ratio")
    parser.add_argument("--n_subsets", type=int, default=4, help="Number of subsets")
    parser.add_argument("--random", type=bool, default=False, help="Random seed")
    parser.add_argument("--model_path", type=str, default=None, help="Model path")

    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Number of units in hidden layers."
    )
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the MLP.")
    parser.add_argument(
        "--learning_rate_mlp", type=float, default=1e-3, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--num_epochs_mlp", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "--weight_decay_mlp", type=float, default=1e-4, help="Weight decay for the optimizer."
    )

    # Return parser arguments
    return parser.parse_args()


def get_config(args):
    """Loads options using yaml files under /config folder and adds command line arguments to it"""
    # Load runtime config from config folder: ./config/ and flatten the runtime config
    config = get_runtime_and_model_config(args)
    # Define which device to use: GPU or CPU
    config["device"] = th.device(
        "cuda:" + args.device_number if th.cuda.is_available() and args.gpu else "cpu"
    )
    # Return
    return config


def print_config_summary(config, args=None):
    """Prints out summary of options and arguments used"""
    # Summarize config on the screen as a sanity check
    print(100 * "=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100 * "=")
    if args is not None:
        print(f"Arguments being used:\n")
        print_config(args)
        print(100 * "=")
