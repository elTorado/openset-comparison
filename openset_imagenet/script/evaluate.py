""" Independent code for inference in testing dataset. The functions are included and executed
in the train.py script."""
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from vast.tools import set_device_gpu, set_device_cpu, device
from torchvision import transforms as tf
from torch.utils.data import DataLoader
import pathlib
import openset_imagenet


def get_args():
    """Gets the evaluation parameters."""
    parser = argparse.ArgumentParser("Get parameters for evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # directory parameters
    
    parser.add_argument(
        "configuration",
        type = pathlib.Path,
        default= "openset-imagenet/config/train.yaml",
        help = "The configuration file that defines the experiment"
    )
    
    parser.add_argument(
        "-l", "--loss",
        choices=["entropic", "softmax", "garbage"],
        default= "entropic",
        help="Which loss function to evaluate"
    )
    parser.add_argument(
        "-p", "--protocol",
        type=int,
        choices=(1, 2, 3),
        help="Which protocol to evaluate"
    )
    parser.add_argument(
        "--use-best", "-b",
        action="store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
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
        "--imagenet-directory",
        type=Path,
        default=Path("/local/scratch/datasets/ImageNet/ILSVRC2012/"),
        help="Imagenet root directory"
    )
    parser.add_argument(
        "--protocol-directory",
        type=Path,
        default = "protocols",
        help = "Where are the protocol files stored"
    )
    parser.add_argument(
        "--output-directory",
        default = "experiments/Protocol_{}",
        help = "Where to find the results of the experiments"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Select the batch size for the test set batches")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Data loaders number of workers, default:4")
    
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
    
    parser.add_argument("--include_unknown", "-iu", action='store_false', dest="include_unknown", help="Exclude unknowns")

    args = parser.parse_args()
    try:
        args.output_directory = args.output_directory.format(args.protocol)
    except:
        pass
    args.output_directory = Path(args.output_directory)
    return args

'''Creates a string suffix that can be used when writing files'''
def get_experiment_suffix(cfg):
    
    
    
    suffix = ""
    letters = True
    print(cfg)
    if cfg.include_counterfactuals:
        suffix += "_counterfactuals"
        letters = False
    if cfg.include_arpl:
        suffix += "_arpl"
        letters = False
    if cfg.mixed_unknowns:
        suffix += "_mixed"
        letters = False
    if not cfg.include_unknown:
        suffix += "_no_negatives"
        letters = False
    if letters:
        suffix += "_letters"

    return suffix

def main():
    args = get_args()
    
    config = openset_imagenet.util.load_yaml(args.configuration)
    
    if args.gpu:
        config.gpu = args.gpu
    config.protocol = args.protocol
    config.output_directory = args.output_directory
    config.include_counterfactuals = args.include_counterfactuals
    config.include_arpl = args.include_arpl
    config.mixed_unknowns = args.mixed_unknowns
    config.include_unknown = args.include_unknown
    
    
    cfg = config

    # Create transformations
    transform_val = tf.Compose(
        [tf.Resize(256),
         tf.CenterCrop(224),
         tf.ToTensor()])
    
    
    # Load dataset files and syntetic image files. Dataset file paths are hardcoded on config file    
    val_file = pathlib.Path(cfg.data.val_file.format(cfg.protocol))
    test_file = pathlib.Path(cfg.data.test_file.format(cfg.protocol))
    
    counterfactual_file = pathlib.Path(cfg.data.counterfactual_file) if cfg.include_counterfactuals else None
    arpl_file = pathlib.Path(cfg.data.arpl_file) if cfg.include_arpl else None

    # create datasets
    val_dataset = openset_imagenet.ImagenetDataset(
            csv_file=val_file,
            which_set="val",
            include_unknown=cfg.include_unknown,
            imagenet_path=cfg.data.imagenet_path,
            counterfactuals_path= counterfactual_file,
            mixed_unknowns=cfg.mixed_unknowns,
            arpl_path= arpl_file,
            transform=transform_val
        )
    test_dataset = openset_imagenet.ImagenetDataset(
        which_set="test",
        csv_file=test_file,
        include_unknown=cfg.include_unknown,
        imagenet_path=cfg.data.imagenet_path,
        counterfactuals_path= None,
        mixed_unknowns=None,
        arpl_path= None,
        transform=transform_val)

    # Info on console
    print("\n========== Data ==========")
    print(f"Val dataset len:{len(val_dataset)}, labels:{val_dataset.label_count}")
    print(f"Test dataset len:{len(test_dataset)}, labels:{test_dataset.label_count}")

    # create data loaders
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    # create device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
    else:
        print("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    if args.loss == "garbage":
        n_classes = val_dataset.label_count # we use one class for the negatives
    else:
        n_classes = val_dataset.label_count - 1  # number of classes - 1 when training with unknowns

    # create model
    suffix = get_experiment_suffix(cfg) + "_best" if args.use_best else "_curr"   
    
    model = openset_imagenet.ResNet50(fc_layer_dim=n_classes,
                                      out_features=n_classes, 
                                      logit_bias=False)
    
    start_epoch, best_score = openset_imagenet.train.load_checkpoint(model, args.output_directory /suffix/ (suffix+".pth"))
    print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    device(model)

    print("========== Evaluating ==========")
    print("Validation data:")
    # extracting arrays for validation
    gt, logits, features, scores = openset_imagenet.train.get_arrays(
        model=model,
        loader=val_loader
    )
    file_path = args.output_directory / f"{suffix}_val_arr.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")

    # extracting arrays for test
    print("Test data:")
    gt, logits, features, scores = openset_imagenet.train.get_arrays(
        model=model,
        loader=test_loader
    )
    file_path = args.output_directory / f"{suffix}_test_arr.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")