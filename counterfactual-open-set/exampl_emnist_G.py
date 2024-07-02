
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import os
import torch
from torchvision.datasets import EMNIST
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from vast import architectures, tools, losses
import pathlib
import json
from torch.nn.parallel import DistributedDataParallel
from loguru import logger
from vast.tools import set_device_gpu, set_device_cpu, device
from openset_imagenet.train import get_arrays, load_checkpoint
from openset_imagenet.losses import AverageMeter, EntropicOpensetLoss
from openset_imagenet.metrics import confidence
import numpy as np
from collections import OrderedDict, defaultdict
import random
import math
#Server
DATA_DIR = '/home/user/heizmann/data/'

#created by grid labeler
GENERATED_COUNTERFACTUALS_DIR = "/home/user/heizmann/openset-comparison/counterfactual-open-set/emnist_counterfactual.dataset"
GENERATED_ARPL_DIR = "/home/user/heizmann/openset-comparison/counterfactual-open-set/emnist_arpl.dataset"

#Local
_GENERATED_NEGATIVES_DIR = "/home/deanheizmann/masterthesis/openset-imagenet/counterfactual-open-set/generated_images_counterfactual.dataset"
_DATA_DIR = '/home/deanheizmann/data/'

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the main training script for all MNIST experiments. \
                    Where applicable roman letters are used as negatives. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approach", "-a", required=True, choices=['SoftMax', 'Garbage', 'EOS', 'Objectosphere'])
    parser.add_argument("--task", default='train', choices=['train', 'eval', "plot", "show"])
    parser.add_argument("--arch", default='LeNet', choices=['LeNet', 'LeNet_plus_plus'])
    parser.add_argument('--second_loss_weight', "-w", help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--Minimum_Knowns_Magnitude', "-m", help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument("--solver", dest="solver", default='sgd',choices=['sgd','adam'])
    parser.add_argument("--lr", "-l", dest="lr", default=0.01, type=float)
    parser.add_argument('--batch_size', "-b", help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
    parser.add_argument("--no_of_epochs", "-e", dest="no_of_epochs", type=int, default=70)
    parser.add_argument("--early_stopping", "-s", dest="early_stopping", type=int, default=3)
    parser.add_argument("--eval_directory", "-ed", dest= "eval_directory", default ="evaluation", help="Select the directory where evaluation details are.")
    parser.add_argument("--dataset_root", "-d", dest= "dataset_root", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--gpu", "-g", type=int, nargs="?",dest="gpu", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")
    parser.add_argument("--include_counterfactuals", "-inc_c", type=bool, default=False, dest="include_counterfactuals", help="Include counterfactual images in the dataset")
    parser.add_argument("--include_arpl", "-inc_a", type=bool, default=False, dest="include_arpl", help="Include ARPL samples in the dataset")
    parser.add_argument("--mixed_unknowns", "-mu", type=bool, default=False, dest="mixed_unknowns", help="Mix unknown samples in the dataset")
    parser.add_argument("--download", "-dwn", type=bool, default=False, dest="download", help="donwload emnist dataset")
    parser.add_argument("--include_unknown", "-iu", action='store_false', dest="include_unknown", help="Exclude unknowns")


    return parser.parse_args()


def transpose(x):
    """Used for correcting rotation of EMNIST Letters"""
    return x.transpose(2,1)


"""A split dataset for our experiments. It uses MNIST as known samples and EMNIST letters as unknowns.
    Particularly, the 11 letters will be used as negatives (for training and validation), and the 11 other letters will serve as unknowns (for testing only) -- 
    we removed letters `g`, `l`, `i` and `o` due to large overlap to the digits.
    The MNIST test set is used both in the validation and test split of this dataset.
    
    The dataset can be extended with synthetic negative samples from counterfactual image generation and 
    artificial reciprocal points learning. to activate this set the parameters 'include_arpl' and / or 
    'include_counterfactuals' to True.

    For the test set, you should consider to leave the parameters `include_unknown` and `has_garbage_class` at their respective defaults -- this might make things easier.

    Parameters:

    dataset_root: Where to find/download the data to.

    which_set: Which split of the dataset to use; can be 'train' , 'test' or 'validation' (anything besides 'train' and 'test' will be the validation set)

    include_unknown: Include unknown samples at all (might not be required in some cases, such as training with plain softmax)

    has_garbage_class: Set this to True when training softmax with background class. This way, unknown samples will get class label 10. If False (the default), unknown samples will get label -1.

    include_arpl: Set to true to load synthetic negative samples generated with artificial reciprocal point learing
    
    include_counterfactuals: Set to true to load synthetic negative samples generated with counterfactual imge generation

    mixed_unknowns: set to true to mix letters with synthetic samples 
"""


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, args, dataset_root, which_set="train", include_unknown = False, has_garbage_class=False, include_arpl = False, include_counterfactuals = False, mixed_unknowns = False):
        
        include_unknown = args.include_unknown
        self.which_letters = ""
        self.includes_synthetic_samples = include_arpl or include_counterfactuals
        assert not (which_set == "test" and self.includes_synthetic_samples), "TEST SET CANNOT INCLUDE SYNTHETIC SAMPLES!"

        
        # synthetic negative samples are stored in this list
        self.synthetic_samples = list()
        self.counterfactual_samples = list()
        self.arpl_samples = list()
        self.targets = list()
        self.letter_indexes = list()
        
        self.mnist = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=args.download,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=args.download,
            split='letters',
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.which_set = which_set
        self.has_garbage_class = has_garbage_class
        
        if include_unknown:
            self.classes = self.mnist.classes + [-1]
            
        else: 
            self.classes = self.mnist.classes
        
        
        print(" ++++++++++++++++++ " + which_set.upper() + " DATASET LOADING +++++++++++++++++++ ")
        print(" ========= INCLUDING NEGATIVES:" + str(include_unknown))
        print(" ========= INCLUDING COUNTERFACTUALS :" + str(include_counterfactuals))
        print(" ========= INCLUDING APRL:" + str(include_arpl))
        print(" ========= MIXING GENERATED SAMPLES WITH LETTERS: " + str(mixed_unknowns))
        print("- - - - - - - - - - - - - - - - - - - - ", end='\n')
        
        if include_unknown:
            # check if synthtic samples are included
            if self.includes_synthetic_samples: 
                        
                # fill synthetic samples list with samples, test set does not include synthetic samples
                if include_arpl:
                    self.arpl_samples = self.load_arpl()
                    if include_counterfactuals:
                        self.counterfactual_samples = self.load_counterfactuals()
                elif include_counterfactuals:
                    self.counterfactual_samples = self.load_counterfactuals()
                    print("after loading")
                    print(type(self.counterfactual_samples))
                
                random.shuffle(self.counterfactual_samples)
                random.shuffle(self.arpl_samples)
                
                if mixed_unknowns:
                    # letters are mixed with synthetic samples in train and validation set
                    self.targets, self.which_letters = ([1,2,3,4,5,6,8,10,11,13,14], "A - N") if which_set != "test" else ([16,17,18,19,20,21,22,23,24,25,26], "P - Z")
                    self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in self.targets]
                    
                    # shuffle the indices as we will need splits
                    random.shuffle(self.letter_indexes)
                    self.nr_letters = len(self.letter_indexes)
                        
                        # depending on setup we will need to half or third the used letters as we want even distribution of samples for comparison
                    if include_arpl: 
                            if include_counterfactuals:
                                self.letter_indexes = self.letter_indexes[math.ceil((self.nr_letters // 3))]
                                self.counterfactual_samples = self.counterfactual_samples[math.ceil((self.nr_letters // 3))]
                                self.arpl_samples = self.arpl_samples[math.ceil((self.nr_letters // 3))]
                                
                            else:
                                self.letter_indexes = self.letter_indexes[math.ceil((self.nr_letters // 2))]
                                self.arpl_samples = self.arpl_samples[math.ceil((self.nr_letters // 2))]
                                
                    elif include_counterfactuals:
                            self.letter_indexes = self.letter_indexes[math.ceil((self.nr_letters // 2))]
                            self.counterfactual_samples = self.counterfactual_samples[math.ceil((self.nr_letters // 2))]
                            print("after splitting")
                            print(type(self.counterfactual_samples))
                    
                else: 
                    self.targets, self.which_letters = (list(), "None") if which_set != "test" else ([16,17,18,19,20,21,22,23,24,25,26], "P - Z")
                    self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in self.targets]
                                        
            # In case no synthetic negative samples are used, use letters as unknowns
            # Letters A to N in train and val set, P to Z in test set
            else: 
                self.targets, self.which_letters = ([1,2,3,4,5,6,8,10,11,13,14], "A - N") if which_set != "test" else ([16,17,18,19,20,21,22,23,24,25,26], "P - Z")
                self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in self.targets]
                
            # Calculate the indices for splitting, # 80% for training   , 20% for Validation
            split_index_cf = int(0.8 * len(self.counterfactual_samples))  
            split_index_arpl = int(0.8 * len(self.arpl_samples))   
                    
            if which_set == "train":
                # Take the first 80% of the samples for training
                self.counterfactual_samples = self.counterfactual_samples[:split_index_cf]
                self.arpl_samples = self.arpl_samples[:split_index_arpl]
                
            elif which_set == "val":
                # Take the last 20% of the samples for validation
                self.counterfactual_samples = self.counterfactual_samples[split_index_cf:]
                self.arpl_samples = self.arpl_samples[split_index_arpl:]
                
            # FINALLY, ASSIGN THE SYNTHETIC SAMPLES
            print(type(self.counterfactual_samples))
            self.synthetic_samples = self.arpl_samples + self.counterfactual_samples
        
        # shuffle it too for good measures:
        random.shuffle(self.synthetic_samples)
        
        
  
        print(" ========= LENGTH OF DIGITS :" + str(len(self.mnist)))
        if include_unknown:
            print(" ========= LENGTH OF LETTER :" + str(len(self.letter_indexes)) + " , FROM LETTERS: " + self.which_letters)
        print(" ========= LENGTH OF SYNTHETIC SAMPLES :" + str(len(self.synthetic_samples)))
        print(" ========= CLASSES IN OVERALL DATASET :", self.classes )       
        print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ", end='\n')

    ''' Function to load counterfactual images. It reads a .dataset file from data_path which is
        hardcoded in this file. The dataset file contains the path to each image, which is read line by line.
        Each image is read in and necessary transformation is applied.
        
        Return: A list with image tensors and labels (-1)
    '''        
    def load_counterfactuals(self, data_path=GENERATED_COUNTERFACTUALS_DIR):
        
        samples = []
        counter = 0
        # Read the whole file at once
        with open(data_path, 'r') as file:
            file_content = file.readlines()
        
        # hardcode amount of negative samples to be equal to letters used for negative samples
        # first 52800 images for trainn split, last 8800 images for val split
        if self.which_set == "train":
            file_content = file_content[:52800]
        else:
            file_content = file_content[-8800:]
                        
        for item in file_content:
            try:
                item = json.loads(item.strip())
                image = Image.open(item["filename"])    
                image_tensor = transforms.Compose([
                    
                    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                    transforms.Resize((28, 28)),  # Resize to 28x28
                    
                    transforms.ToTensor(),
                    transpose # IS THIS STILL NECESSARY?
                ])(image)
                label = item["label"]
                samples.append((image_tensor, label))
                counter += 1

            except Exception as e:
                print(f"Error processing item {item}: {e}")
        
        return samples
    
    
    
    ''' Function to load arpl images. It reads a .dataset file from data_path which is
        hardcoded in this file. The dataset file contains the path to each image, which is read line by line.
        Each image is read in and necessary transformation is applied.
        
        Return: A list with image tensors and labels (-1)
    ''' 
    def load_arpl(self, data_path = GENERATED_ARPL_DIR):
        samples = []
        counter = 0
        # Read the whole file at once
        with open(data_path, 'r') as file:
            file_content = file.readlines()
        
        # hardcode amount of negative samples to be equal to letters used for negative samples
        # first 52800 images for trainn split, last 8800 images for val split
        if self.which_set == "train":
            file_content = file_content[:52800]
        else:
            file_content = file_content[-8800:]                       

        for item in file_content:
            try:
                item = json.loads(item.strip())
                image = Image.open(item["filename"])    
                image_tensor = transforms.Compose([
                    
                    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                    transforms.Resize((28, 28)),  # Resize to 28x28
                    
                    transforms.ToTensor(),
                    transpose # IS THIS STILL NECESSARY?
                ])(image)
                label = item["label"]
                samples.append((image_tensor, label))
                counter += 1

            except Exception as e:
                print(f"Error processing item {item}: {e}")
        
        return samples

    
    '''The getter handles all images as a big list of image tensors. 
        If letters or synthetic images are available, their list is
        conceptually appended to the list of all image tensors.
        (Not really appended but when iterating it behaves like it)
    '''
    def __getitem__(self, index):
        total_mnist = len(self.mnist)
        total_letters = len(self.letter_indexes)  # Use the length of letter_indexes, not self.letters
        data = None

        if index < total_mnist:
            data = self.mnist[index]
        elif index < total_mnist + total_letters:
            letter_index = index - total_mnist
            data = self.letters[self.letter_indexes[letter_index]][0], 10 if self.has_garbage_class else -1
        else:
            synthetic_index = index - (total_mnist + total_letters)
            data = self.synthetic_samples[synthetic_index][0], 10 if self.has_garbage_class else -1

        return data            
            

    '''
        len of all image tensor lists combined
    '''
    def __len__(self):
        return len(self.mnist) + len(self.letter_indexes) + len(self.synthetic_samples)
    
##################################################################################################################################################
##################################################################################################################################################
    
    '''
    TO BE DELETED!!
    '''


def create_labels_files(args):
    
    first_loss_func,second_loss_func,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]
    test_dataset = Dataset(args, args.dataset_root, which_set="test", include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns)


    test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.Batch_Size,
    pin_memory=True
    )
    
    train_data_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=args.Batch_Size,
    shuffle=True,
    num_workers=5,
    pin_memory=True
        )
    val_data_loader = torch.utils.data.DataLoader(
    validation_data,
    batch_size=args.Batch_Size,
    pin_memory=True
    )
    
    
    with open('emnist_classifier_labels.dataset', 'w') as file:
        from collections import Counter
        train_labels = []
        val_labels = []
        test_labels = []
        # Collect labels from the train data loader
        for _, l in train_data_loader:
            train_labels.extend(l.tolist())  # Assuming l is a tensor or similar structure that needs to be converted to a list

        # Collect labels from the validation data loader
        for _, l in val_data_loader:
            val_labels.extend(l.tolist())

        # Collect labels from the test data loader
        for _, l in test_loader:
            test_labels.extend(l.tolist())

        # Count the occurrences of each label and write to file
        train_label_counts = Counter(train_labels)
        val_label_counts = Counter(val_labels)
        test_label_counts = Counter(test_labels)

        # Writing the counts in a sorted manner by label
        file.write(f"train labels: {sorted(train_label_counts.items())}\n")
        file.write(f"val labels: {sorted(val_label_counts.items())}\n")
        file.write(f"test labels: {sorted(test_label_counts.items())}\n")  
     
     
'''
    Is this still used???
    ??
'''   
def create_fold():
    
    # We read through all letters and digits and create three different distributions.
    # The train / val / test splits are each time taken differently e.g. first n for train, then last n for train
    # Split 1: For GAN training, contains digits in train and val 
    # Split 2: For Classifier training: Test set contains letters as unknowns. Train and Val only digits as we later add counterfactuals as negatives
    # Split 3: For Classifier training: Train and val contain letters a-n as negatives (later expanded by generated counterfactuals), test contains p-z as unknowns 
    
    with open(DATA_DIR + 'digits.dataset', 'r') as digits,  open(DATA_DIR + 'letters.dataset', 'r') as letters: 
        test_size = 3600 
        train_size = 20400  # 85% of 24,000 for training in the first split
        train_len = test_len = 0  # Initialize counters
        
        digits = digits.readlines()
        letters = letters.readlines()
  
        digits = [json.loads(line) for line in digits]
        letters = [json.loads(line) for line in letters]
             
        letters_AtoM = [elem for elem in letters if elem["label"] in [1,2,3,4,5,6,8,10,11,13,14]]
        letters_PtoZ = [elem for elem in letters if elem["label"] in [16,17,18,19,20,21,22,23,24,25,26]]       
        
        print(" ========= WRITING DATASET FILES ==========")
        
        with open('emnist_split1.dataset', 'w') as file1:
            # Training data
            for element in digits[:train_size]:
                train_len += 1
                file1.write(json.dumps(element, sort_keys=True) + '\n')
                # We dont need the fold in the csv files and we need to label to be in the second row
                
            # Test data
            for element in digits[-test_size:]:
                test_len += 1
                element["fold"] = "test"
                file1.write(json.dumps(element, sort_keys=True) + '\n')

        print(" ==== CREATED  emnist_split1.dataset with only digits WITH FOLD SIZES:")
        print(" ==== TRAIN SPLIT: " + str(train_len) + " ========")
        print(" ==== VALIDATION SPLIT : " + str(val_len) + " ========")
        print(" ==== TEST SPLIT: " + str(test_len) + " ========")

        # Reset counters for the second file split
        test_size = 3600 
        val_size = 3600
        train_size = 16800  # 70% of 24,000 for training in the  split
        train_len = val_len = test_len = 0
            
        with open('emnist_split2.dataset', 'w') as file2:
            
            # Training data        
            for element in digits[-train_size:]:
                train_len += 1
                element["fold"] = "train"
                file2.write(json.dumps(element, sort_keys=True) + '\n')
                
            # Validation data
            for element in digits[train_size+val_size:train_size+val_size+test_size]:
                val_len += 1
                element["fold"] = "val"
                file2.write(json.dumps(element, sort_keys=True) + '\n')
                
            # Test data
            for element in letters_AtoM + digits[train_size:train_size+val_size]:
                test_len += 1
                element["fold"] = "test"
                file2.write(json.dumps(element, sort_keys=True) + '\n')
                

        print(" ==== CREATED emnist_split2.dataset with digits and letters A to M in test WITH FOLD SIZES:")
        print(" ==== TRAIN SPLIT: " + str(train_len) + " ========")
        print(" ==== VALIDATION SPLIT : " + str(val_len) + " ========")
        print(" ==== TEST SPLIT: " + str(test_len) + " ========")
        
        # Reset counters for the second file split
        train_len = val_len = test_len = 0
            
        with open('emnist_split3.dataset', 'w') as file3:
            # Training data
            for element in digits[-train_size:]:
                train_len += 1
                element["fold"] = "train"
                file3.write(json.dumps(element, sort_keys=True) + '\n')
                
            # Validation data          
            for element in letters_AtoM + digits [train_size+val_size:train_size+val_size+test_size]:
                val_len += 1
                element["fold"] = "val"
                file3.write(json.dumps(element, sort_keys=True) + '\n')

            # Test data
            for element in letters_PtoZ + digits[train_size:train_size+val_size]:
                test_len += 1
                element["fold"] = "test"

    print(" ==== CREATED emnist_split3.dataset with digits and letters P to Z in test WITH FOLD SIZES:")
    print(" ==== TRAIN SPLIT: " + str(train_len) + " ========")
    print(" ==== VALIDATION SPLIT : " + str(val_len) + " ========")
    print(" ==== TEST SPLIT: " + str(test_len) + " ========")          
    
    
""" Returns the loss function and the data for training and validation
    The code supports SoftMax, Garbage, EOS and Objectosphere, 
    In this implementation, only EOS was focused and tested
"""
def get_loss_functions(args):
    if args.approach == "SoftMax":
        print(" ========= Using SoftMax Loss ===========")
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args, args.dataset_root, include_unknown=False , include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns),
                    val_data = Dataset(args, args.dataset_root, which_set="val", include_unknown=False, include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns),
                )
    elif args.approach =="Garbage":
        print(" ========= Using Garbage Loss ===========")
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args, args.dataset_root, has_garbage_class=True, include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns),
                    val_data = Dataset(args, args.dataset_root, which_set="val", has_garbage_class=True, include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns)
                )
    elif args.approach == "EOS":
        print(" ========= Using Entropic Openset Loss ===========")
        return dict(
                    first_loss_func=EntropicOpensetLoss(num_of_classes=10),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data=Dataset(args, args.dataset_root, include_unknown=args.include_unknown, include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns),
                    val_data = Dataset(args, args.dataset_root, which_set="val", include_unknown=args.include_unknown, include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns)
                )
    elif args.approach == "Objectosphere":
        print(" ========= Using Objectosphere Loss ===========")
        return dict(
                    first_loss_func=EntropicOpensetLoss(num_of_classes=10),
                    second_loss_func=losses.objectoSphere_loss(args.Minimum_Knowns_Magnitude),
                    training_data=Dataset(args, args.dataset_root, include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns),
                    val_data = Dataset(args, args.dataset_root, which_set="val", include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns)
                )

'''Creates a string suffix that can be used when writing files'''
def get_experiment_suffix(args):
    suffix = ""
    letters = True
    if args.include_counterfactuals:
        suffix += "_counterfactuals"
        letters = False
    if args.include_arpl:
        suffix += "_arpl"
        letters = False
    if args.mixed_unknowns:
        suffix += "_mixed"
        letters = False
    if letters:
        suffix += "_letters"
    return suffix

def training(args): 
    # setup device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
        print(" ============== GPU Selected! =============")
    else:
        print("No GPU device selected, training will be extremely slow")
        set_device_cpu()
    
    torch.manual_seed(0)

    # get training and validation data and Entropic OpenSet Loss function    
    training_data=Dataset(args, args.dataset_root , include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns)
    first_loss_func=EntropicOpensetLoss(num_of_classes=len(training_data.classes))
    validation_data = Dataset(args, args.dataset_root, which_set="val", include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns)

    # Create .pth file to store the training result and the directory to store it if necessary
    suffix = get_experiment_suffix(args=args)
    results_dir = pathlib.Path(f"{args.arch}")
    model_file = f"{results_dir}/{suffix}.pth"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(training_data.classes)

    # instantiate network and data loader
    net = architectures.LeNet(num_classes=num_classes)
    net = tools.device(net)
    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.Batch_Size,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.Batch_Size,
        pin_memory=True
    )
    
    if args.solver == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.solver == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    logs_dir = results_dir/'Logs'
    writer = SummaryWriter(logs_dir)
    
    t_metrics = defaultdict(AverageMeter)
    v_metrics = defaultdict(AverageMeter)

    # train network
    prev_confidence = None
    
    # Training loop uses an early stopping mechanism that looks at the confidence score
    # If confidence score is not improving over n iterations, training is terminated
    BEST_SCORE = 0.0
    EARLY_STOPPING_COUNTER = 0
    LIMIT = args.early_stopping
    for epoch in range(1, args.no_of_epochs + 1, 1): 
        print ("======== TRAINING EPOCH: " + str(epoch) +" ===============")
        
        train(
            net=net,
            train_data_loader=train_data_loader,
            optimizer=optimizer, loss_func=first_loss_func,
            t_metrics=t_metrics,
            args=args
        )
        
        validate(
            net=net,
            val_data_loader=val_data_loader,
            optimizer=optimizer, loss_func=first_loss_func,
            v_metrics=v_metrics,
            num_classes=num_classes,
            args=args
        )
                                        
        # log statistics
        curr_score = v_metrics["conf_kn"].avg + v_metrics["conf_unk"].avg
        if BEST_SCORE < curr_score:
            BEST_SCORE = curr_score
            
            #reset counter:
            EARLY_STOPPING_COUNTER = 0
            
        # If there is no score improvement for n epochs, stop training
        else: 
            EARLY_STOPPING_COUNTER += 1
            if EARLY_STOPPING_COUNTER == LIMIT:
                logger.info(f"STOPPED TRAINING DUE TO TRRIGGERED EARLY STOPPING AT EPOCH: {epoch}")
                break


        # Logging metrics to tensorboard object
        writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
        writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
        # Validation metrics
        writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
        writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

        
        # t_metrics carries only a loss value, val metrics -> loss, conf_known, conf_unknown
                
        logger.info(f"Saving  model {model_file} at epoch: {epoch}")
        save_checkpoint(model_file, net, epoch, optimizer, curr_score, scheduler=None)

        def pretty_print(d):
                #return ",".join(f'{k}: {float(v):1.3f}' for k,v in dict(d).items())
                return dict(d)

        logger.info(
                f"ep:{epoch} "
                f"train:{pretty_print(t_metrics)} "
                f"val:{pretty_print(v_metrics)} ")
    
    print(" ====== TRAINING FINISHED WITH BEST SCORE: ", BEST_SCORE)


def train(net, train_data_loader, optimizer, loss_func, t_metrics, args):
    
    # Reset dictionary of training metrics
    for metric in t_metrics.values():
        metric.reset()

    loss = None
    net.train()
    
    for images, labels in train_data_loader:
        
        # load tensors and labels to device
        images = tools.device(images)
        labels = tools.device(labels)    
        batch_len = labels.shape[0]  
        optimizer.zero_grad()
        logits, features = net(images)
        
        # first loss is always computed, second loss only for some loss functions
        loss = loss_func(logits, labels)
        t_metrics["j"].update(loss.item(), batch_len)     

        loss.backward()
        optimizer.step()      


def validate(net, val_data_loader, optimizer, num_classes, loss_func, v_metrics, args):
    # Reset all validation metrics
    for metric in v_metrics.values():
        metric.reset()
    
    net.eval()
    with torch.no_grad():
        data_len = len(val_data_loader.dataset)  # size of dataset
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, num_classes), requires_grad=False))
        
        for i, (x,y) in enumerate(val_data_loader): 
            batch_len = y.shape[0]  # current batch size, last batch has different value 
            
            # load image tensor and label to device              
            x = tools.device(x)          
            y = tools.device(y)
            logits, features = net(x)             
            scores = torch.nn.functional.softmax(logits, dim=1)
            loss = loss_func(logits, y) 
            v_metrics["j"].update(loss.item(), batch_len)
                        
            start_ix = i * args.Batch_Size
            all_targets[start_ix: start_ix + batch_len] = y
            all_scores[start_ix: start_ix + batch_len] = scores
            
                
        kn_conf, kn_count, neg_conf, neg_count = confidence(
        scores=all_scores,
        target_labels=all_targets,
        offset= 1. /  num_classes,
        unknown_class = -1,
        last_valid_class = None)
            
        if kn_count:
                v_metrics["conf_kn"].update(kn_conf, kn_count)
        if neg_count:
                v_metrics["conf_unk"].update(neg_conf, neg_count)


def evaluate(args):
    
    print(" ========= Using Entropic Openset Loss ===========")
    val_dataset = Dataset(args,  args.dataset_root, which_set="val", include_arpl=args.include_arpl, include_counterfactuals=args.include_counterfactuals, mixed_unknowns=args.mixed_unknowns)
    test_dataset = Dataset(args, args.dataset_root, which_set="test")

    num_classes = len(val_dataset.classes)

    # create data loaders
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.Batch_Size,
        pin_memory=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.Batch_Size,
        pin_memory=True
        )

    # create device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
    else:
        print("No GPU device selected, evaluation will be slow")
        set_device_cpu()
    
    # create model
    suffix = get_experiment_suffix(args=args)
    
    # Result dir in "LeNet"
    results_dir = pathlib.Path(f"{args.arch}")
    model_path = f"{results_dir}/{suffix}.pth"

    net = architectures.LeNet(num_classes=num_classes)
    start_epoch, best_score = load_checkpoint(net, model_path)
    print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    net = tools.device(net)

    print("========== Evaluating ==========")
    print("Validation data:")
    # extracting arrays for validation
    val_targets, logits, val_features, scores = get_arrays(
        model=net,
        loader=val_loader,
        dataset= "VAL"
    )

    # Print summary statistics for validation data
    print(f"Number of validation samples: {len(val_targets)}")
    if scores is not None:
        print(f"Average score on validation data: {np.mean(scores):.4f}")
    if logits is not None:
        predicted_classes = np.argmax(logits, axis=1)
        accuracy = np.mean(predicted_classes == val_targets)
        print(f"Accuracy on validation data: {accuracy:.4f}")

    directory = pathlib.Path(f"{args.eval_directory}")

    file_path = directory / f"validation_{get_experiment_suffix(args=args)}.npz"
    np.savez(file_path, gt=val_targets, logits=logits, features=val_features, scores=scores)
    print(f"Target labels, logits, features, and scores saved in: {file_path}")

    # extracting arrays for test
    print("Test data:")
    test_targets, logits, test_features, scores = get_arrays(
        model=net,
        loader=test_loader,
        dataset= "TEST"
    )
    
    print("VALIDATION TARGETS:", val_targets[:10], test_targets[-10:], np.unique(val_targets))
    print("TEST TARGETS:", val_targets[:10], test_targets[-10:], np.unique(val_targets))
    
    # Print summary statistics for test data
    print(f"Number of test samples: {len(test_targets)}")
    if scores is not None:
        print(f"Average score on test data: {np.mean(scores):.4f}")
    if logits is not None:
        predicted_classes = np.argmax(logits, axis=1)
        accuracy = np.mean(predicted_classes == test_targets)
        print(f"Accuracy on test data: {accuracy:.4f}")

    file_path = directory / f"test_{get_experiment_suffix(args=args)}.npz"
    np.savez(file_path, gt=test_targets, logits=logits, features=test_features, scores=scores)
    print(f"Target labels, logits, features, and scores saved in: {file_path}")


   
def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(f_name), exist_ok=True)    
    
    
    # If model is DistributedDataParallel extracts the module.
    if isinstance(model, DistributedDataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    data = {"epoch": epoch + 1,
            "model_state_dict": state,
            "opt_state_dict": opt.state_dict(),
            "best_score": best_score_}
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, f_name)   
    
if __name__ == "__main__":
    args = command_line_options()
    print(args)
    
    suffix = get_experiment_suffix(args=args)
    print(args.include_counterfactuals)
    print(suffix)
    
    if args.task == "train":
        print(" TASK IS TO TRAIN")
        example = training(args = args)
    if args.task == "eval":
        print(" TASK IS TO EVALUATE")
        evaluate(args = args)
    if args.task == "dataset":
        print("creating labels file")
        create_fold(args=args)
