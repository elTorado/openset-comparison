""" Training script for Open-set Classification on Imagenet"""
import argparse
import openset_imagenet
import pathlib
import os


def get_args(command_line_options = None):
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Training Parameters",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "configuration",
        type = pathlib.Path,
        help = "The configuration file that defines the experiment"
    )
    
    parser.add_argument("--include_unknown", "-iu", action='store_false', dest="include_unknown", help="Exclude unknowns")

    parser.add_argument(
        "protocol",
        type=int,
        choices = (1,2,3),
        help="Open set protocol: 1, 2 or 3"
    )
    parser.add_argument(
        "--output-directory", "-o",
        type=pathlib.Path,
        default="./experiments",
        help="Directory to store the trained models into"
    )
    parser.add_argument(
        "--gpu", "-g",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
    )

    parser.add_argument(
        "--nice",
        type=int,
        default = 20,
        help = "Select Priority Level"
    )
    
    parser.add_argument(
        "--include_counterfactuals", "-inc_c",
        type=bool, default=False, 
        dest="include_counterfactuals",
        help="Include counterfactual images in the dataset")
    
    parser.add_argument("--include_arpl", "-inc_a",
                        type=bool, default=False,
                        dest="include_arpl", 
                        help="Include ARPL samples in the dataset")
    
    parser.add_argument("--mixed_unknowns", "-mu",
                        type=bool, default=False,
                        dest="mixed_unknowns",
                        help="Mix unknown samples in the dataset")
    

    args = parser.parse_args(command_line_options)

    os.nice(args.nice)
    return args


def main(command_line_options = None):

    args = get_args(command_line_options)

    config = openset_imagenet.util.load_yaml(args.configuration)
    if args.gpu:
        config.gpu = args.gpu
    config.protocol = args.protocol
    config.output_directory = args.output_directory
    config.include_counterfactuals = args.include_counterfactuals
    config.include_arpl = args.include_arpl
    config.mixed_unknowns = args.mixed_unknowns
    config.include_unknown = args.include_unknown

    openset_imagenet.train.worker(config)


if __name__ == "__main__":
    main()