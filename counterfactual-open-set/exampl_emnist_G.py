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

#Server
DATA_DIR = '/home/user/heizmann/data/'

#created by grid labeler
GENERATED_NEGATIVES_DIR = "/home/user/heizmann/openset-comparison/counterfactual-open-set/generated_images_counterfactual.dataset"

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
    parser.add_argument("--arch", default='LeNet_plus_plus', choices=['LeNet', 'LeNet_plus_plus'])
    parser.add_argument('--second_loss_weight', "-w", help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--Minimum_Knowns_Magnitude', "-m", help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument("--solver", dest="solver", default='sgd',choices=['sgd','adam'])
    parser.add_argument("--lr", "-l", dest="lr", default=0.01, type=float)
    parser.add_argument('--batch_size', "-b", help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
    parser.add_argument("--no_of_epochs", "-e", dest="no_of_epochs", type=int, default=70)
    parser.add_argument("--eval_directory", "-ed", dest= "eval_directory", default ="evaluation", help="Select the directory where evaluation details are.")
    parser.add_argument("--dataset_root", "-d", dest= "dataset_root", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--gpu", "-g", type=int, nargs="?",dest="gpu", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")
    parser.add_argument("--include_counterfactuals", "-inc_c", type=bool, default=False, dest="include_counterfactuals", help="Include counterfactual images in the dataset")
    parser.add_argument("--include_arpl", "-inc_a", type=bool, default=False, dest="include_arpl", help="Include ARPL samples in the dataset")
    parser.add_argument("--mixed_unknowns", "-mu", type=bool, default=False, dest="mixed_unknowns", help="Mix unknown samples in the dataset")
    parser.add_argument("--download", "-dwn", type=bool, default=False, dest="download", help="donwload emnist dataset")

    return parser.parse_args()


def transpose(x):
    """Used for correcting rotation of EMNIST Letters"""
    return x.transpose(2,1)


"""A split dataset for our experiments. It uses MNIST as known samples and EMNIST letters as unknowns.
    Particularly, the 11 letters will be used as negatives (for training and validation), and the 11 other letters will serve as unknowns (for testing only) -- 
    we removed letters `g`, `l`, `i` and `o` due to large overlap to the digits.
    The MNIST test set is used both in the validation and test split of this dataset.

    For the test set, you should consider to leave the parameters `include_unknown` and `has_garbage_class` at their respective defaults -- this might make things easier.

    Parameters:

    dataset_root: Where to find/download the data to.

    which_set: Which split of the dataset to use; can be 'train' , 'test' or 'validation' (anything besides 'train' and 'test' will be the validation set)

    include_unknown: Include unknown samples at all (might not be required in some cases, such as training with plain softmax)

    has_garbage_class: Set this to True when training softmax with background class. This way, unknown samples will get class label 10. If False (the default), unknown samples will get label -1.
"""
class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, args, dataset_root, which_set="train", include_unknown=True, has_garbage_class=False):
        
        self.include_arpl = args.include_arpl
        self.include_counterfactuals = args.include_counterfactuals
        self.mixed_unknowns = args.mixed_unknowns
        
        self.includes_synthetic_samples = self.include_arpl or self.include_counterfactuals
        self.synthetic_samples = list()
        
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
        
        print(args.include_counterfactuals)
        print(" ========= INCLUDING COUNTERFACTUALS :" + str(self.include_counterfactuals))
        print(" ========= INCLUDING APRL:" + str(self.include_arpl))
        
        if self.includes_synthetic_samples:           
            # if we mix letters with synthetic samples in train and validation
            if self.mixed_unknowns:
                targets = [1,2,3,4,5,6,8,10,11,13,14] if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
            else: 
                targets = list() if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
            
            # Test set does not include synthetic samples
            if not self.which_set == "test":
                if self.include_arpl:
                    self.synthetic_samples.extend(self.load_arpl())
                    if self.include_counterfactuals:
                        self.synthetic_samples.extend(self.load_counterfactuals())
                elif self.include_counterfactuals:
                    self.synthetic_samples.extend(self.load_counterfactuals())
                                     
        # In case no synthetic negative samples are used, letters are unknowns
        else: 
            targets = list() if not include_unknown else [1,2,3,4,5,6,8,10,11,13,14] if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
            
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]

        
        print(" ++++++++++++++++++" + which_set + " DATASET LOADING +++++++++++++++++++ ")
        print(" ========= LENGTH OF DIGITS :" + str(len(self.mnist)))
        print(" ========= LENGTH OF LETTER :" + str(len(self.letters)))
        print(" ========= LENGTH OF SYNTHETIC SAMPLES :" + str(len(self.synthetic_samples)))       
    

        
        
    def load_counterfactuals(self, data_path=GENERATED_NEGATIVES_DIR):
        
        samples = []
        counter = 0
        # Read the whole file at once
        with open(data_path, 'r') as file:
            file_content = file.readlines()

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
                if counter == 20800 and self.which_set == "val":
                    break
                if counter == 124800 and self.which_set == "train":
                    break
            except Exception as e:
                print(f"Error processing item {item}: {e}")
        
        return samples
    
    def load_arpl(self):
        return

    def create_png(self):
        images_path = os.path.join(DATA_DIR, "emnist")
        if not os.path.exists(images_path):
            os.makedirs(images_path)
            
        for i in range(len(self.mnist)):
            image, label = self.mnist[i]
            image = transforms.ToPILImage()(image)  # This line converts the image tensor to a PIL Image
            filename = os.path.join(images_path, f'digits_{i:06d}.png')
            image.save(filename)  
            self.write_dataset(img=filename, label=label, category="digits")
            
        for i in range(len(self.letters)):
            image, label = self.letters[i]
            image = transforms.ToPILImage()(image)  # Convert tensor to PIL Image
            filename = os.path.join(images_path, f'letters_{i:06d}.png')
            image.save(filename)  
            self.write_dataset(img=filename, label=label, category="letters")
    
    #write two dataset file for each the digits and the letters, containing disctopnaries for each img with path, fold=train and label            
    def write_dataset(self, img, label, category):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # Path to the .dataset file
        dataset_path = os.path.join(DATA_DIR, f'{category}.dataset')
        
        # Dictionary to write
        data = {
            "filename": img,
            "fold": "train",
            "label": label
        }
        
        # Open the file in append mode and write the dictionary as a JSON line
        with open(dataset_path, 'a') as file:
            file.write(json.dumps(data) + "\n")

          
    def __getitem__(self, index):
        total_mnist = len(self.mnist)
        total_letters = len(self.letter_indexes)  # Use the length of letter_indexes, not self.letters
        data = None

        if index < total_mnist:
            data = self.mnist[index]
            data_kind = 'mnist'
        elif index < total_mnist + total_letters:
            letter_index = index - total_mnist
            data = self.letters[self.letter_indexes[letter_index]][0], 10 if self.has_garbage_class else -1
            data_kind = 'letter'
        else:
            synthetic_index = index - (total_mnist + total_letters)
            data = self.synthetic_samples[synthetic_index][0], 10 if self.has_garbage_class else -1
            data_kind = 'synthetic'

        # Check if the first element of the tuple (should be the image data) is a tensor
        if not isinstance(data[0], torch.Tensor):
            print(f"Non-tensor data found: {data_kind} data at index {index}")
            print(f"Data element: {data[0]}")  # Print the problematic data

        # Similarly, check the label if necessary, assuming it might not be a tensor and is an int or similar
        if not isinstance(data[1], (torch.Tensor, int)):
            print(f"Label data is not int or tensor: {data_kind} label at index {index}")
            print(f"Label element: {data[1]}")  # Print the problematic label data

        return data            
            
    '''def __getitem__(self, index):
        if index < len(self.mnist):
            return self.mnist[index] 
        elif index < len(self.mnist) + len(self.letters):
            # index provided as input is based on the combined length of both datasets, but each dataset needs to be accessed independently.
            # [0] extracts the image data
            print("------------- INDEX: "+ str(index)+"  ---------------")
            print("------------- LENGTH OF LETTER INDEXES: "+ str(len(self.letter_indexes))+"  ---------------")

            return self.letters[self.letter_indexes[index - len(self.mnist)]][0], 10 if self.has_garbage_class else -1 
        else: 
            return self.synthetic_samples[index - len(self.mnist) + len(self.letters)], 10 if self.has_garbage_class else -1 '''

    def __len__(self):
        return len(self.mnist) + len(self.letter_indexes) + len(self.synthetic_samples)
    
##################################################################################################################################################
##################################################################################################################################################
    
def create_labels_files(args):
    
    first_loss_func,second_loss_func,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]
    test_dataset = Dataset(args, args.dataset_root, which_set="test")


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
        
def create_fold():
    
    # We read through all letters and digits and create three different distributions.
    # The train / val / test splits are each time taken differently e.g. first n for train, then last n for train
    # Split 1: For GAN training, contains digits in train and val 
    # Split 2: For Classifier training: Test set contains letters as unknowns. Train and Val only digits as we later add counterfactuals as negatives
    # Split 3: For Classifier training: Train and val contain letters a-n as negatives (later expanded by generated counterfactuals), test contains p-z as unknowns 
    
    with open(DATA_DIR + 'digits.dataset', 'r') as digits,  open(DATA_DIR + 'letters.dataset', 'r') as letters: 
        val_size = 3600 
        test_size = 3600 
        train_size = 16800  # 70% of 24,000 for training in the first split
        train_len = val_len = test_len = 0  # Initialize counters
        
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
                
            # Validation data
            for element in digits[train_size:train_size+val_size]:
                val_len += 1
                element["fold"] = "val"
                file1.write(json.dumps(element, sort_keys=True) + '\n')
                
            # Test data
            for element in digits[train_size+val_size:train_size+val_size+test_size]:
                test_len += 1
                element["fold"] = "test"
                file1.write(json.dumps(element, sort_keys=True) + '\n')

        print(" ==== CREATED  emnist_split1.dataset with only digits WITH FOLD SIZES:")
        print(" ==== TRAIN SPLIT: " + str(train_len) + " ========")
        print(" ==== VALIDATION SPLIT : " + str(val_len) + " ========")
        print(" ==== TEST SPLIT: " + str(test_len) + " ========")

        # Reset counters for the second file split
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
    
def get_loss_functions(args):
    """Returns the loss function and the data for training and validation"""
    if args.approach == "SoftMax":
        print(" ========= Using SoftMax Loss ===========")
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args, args.dataset_root, include_unknown=False),
                    val_data = Dataset(args, args.dataset_root, which_set="val", include_unknown=False),
                )
    elif args.approach =="Garbage":
        print(" ========= Using Garbage Loss ===========")
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args, args.dataset_root, has_garbage_class=True),
                    val_data = Dataset(args, args.dataset_root, which_set="val", has_garbage_class=True)
                )
    elif args.approach == "EOS":
        print(" ========= Using Entropic Openset Loss ===========")
        return dict(
                    first_loss_func=EntropicOpensetLoss(num_of_classes=10),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data=Dataset(args, args.dataset_root),
                    val_data = Dataset(args, args.dataset_root, which_set="val")
                )
    elif args.approach == "Objectosphere":
        print(" ========= Using Objectosphere Loss ===========")
        return dict(
                    first_loss_func=EntropicOpensetLoss(num_of_classes=10),
                    second_loss_func=losses.objectoSphere_loss(args.Minimum_Knowns_Magnitude),
                    training_data=Dataset(args, args.dataset_root),
                    val_data = Dataset(args, args.dataset_root, which_set="val")
                )


def training(args): 
    # setup device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
        print(" ============== GPU Selected! =============")
    else:
        print("No GPU device selected, training will be extremely slow")
        set_device_cpu()
    
    torch.manual_seed(0)

    # get training data and loss function(s)
    #ONLY for objectosphere a real second loss functions is returned, else its a tensord filled a 0
    first_loss_func,second_loss_func,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]
    
        # Info on console
    print("\n========== Data ==========")
    print(f"Training dataset len:{len(training_data)}")
    print(f"Validation dataset len:{len(validation_data)}")

    results_dir = pathlib.Path(f"{args.arch}/{args.approach}")
    model_file = f"{results_dir}/{args.approach}.pth"
    results_dir.mkdir(parents=True, exist_ok=True)

    # instantiate network and data loader
    net = architectures.__dict__[args.arch](use_BG=args.approach == "Garbage")
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
    for epoch in range(1, args.no_of_epochs + 1, 1):  # loop over the dataset multiple times
        print ("======== TRAINING EPOCH: " + str(epoch) +" ===============")
        
        '''loss_history = []
        train_accuracy = torch.zeros(2, dtype=int)
        train_magnitude = torch.zeros(2, dtype=float)
        train_confidence = torch.zeros(2, dtype=float)'''
        
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
            args=args
        )
        
                                  
        # log statistics
        curr_score = v_metrics["conf_kn"].avg + v_metrics["conf_unk"].avg


        # Logging metrics to tensorboard object
        writer.add_scalar("train/loss", t_metrics["j"].avg, epoch)
        writer.add_scalar("val/loss", v_metrics["j"].avg, epoch)
        # Validation metrics
        writer.add_scalar("val/conf_kn", v_metrics["conf_kn"].avg, epoch)
        writer.add_scalar("val/conf_unk", v_metrics["conf_unk"].avg, epoch)

        ''' # save network based on confidence metric of validation set
            save_status = "NO"
            if prev_confidence is None or (val_confidence[0] > prev_confidence):
                torch.save(net.state_dict(), model_file)
                prev_confidence = val_confidence[0]
                save_status = "YES" '''

        logger.info(f"Saving  model {model_file} at epoch: {epoch}")
        save_checkpoint(model_file, net, epoch, optimizer, curr_score, scheduler=None)

        def pretty_print(d):
                #return ",".join(f'{k}: {float(v):1.3f}' for k,v in dict(d).items())
                return dict(d)

        logger.info(
                f"ep:{epoch} "
                f"train:{pretty_print(t_metrics)} "
                f"val:{pretty_print(v_metrics)} ")

'''def confidence(logits, target, negative_offset=0.1):
    """Measures the softmax confidence of the correct class for known samples,
    and 1 + negative_offset - max(confidence) for unknown samples.

    Parameters:

      logits: the output of the network, must be logits

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      confidence: the sum of the confidence values for the samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0

        pred = torch.nn.functional.softmax(logits, dim=1)
        #    import ipdb; ipdb.set_trace()
        kn_count = sum(known).item()    # Total known samples in data
        neg_count = sum(~known).item()  # Total negative samples in data
        kn_conf = 0.0
        neg_conf = 0.0
        
        if torch.sum(known):
            kn_conf = torch.sum(pred[known, target[known]]).item() / kn_count
        if torch.sum(~known):
            neg_conf += torch.sum(
                1.0 + negative_offset - torch.max(pred[~known], dim=1)[0]
            ).item() / neg_count

    return kn_conf, kn_count, neg_conf, neg_count'''


def train(net, train_data_loader, optimizer, loss_func, t_metrics, args):
    
    # Reset dictionary of training metrics
    for metric in t_metrics.values():
        metric.reset()

    loss = None
    net.train()
    
    for images, labels in train_data_loader:
        
        images = tools.device(images)
        labels = tools.device(labels)    
        batch_len = labels.shape[0]  # Samples in current batch
        optimizer.zero_grad()
        logits, features = net(images)
        
        # first loss is always computed, second loss only for some loss functions
        loss = loss_func(logits, labels)
        t_metrics["j"].update(loss.item(), batch_len)     

        loss.backward()
        optimizer.step()      


def validate(net, val_data_loader, optimizer, loss_func, v_metrics, args):
    # Reset all validation metrics
    for metric in v_metrics.values():
        metric.reset()
    
    num_classes = 10
    net.eval()
    with torch.no_grad():
        data_len = len(val_data_loader.dataset)  # size of dataset
        all_targets = device(torch.empty((data_len,), dtype=torch.int64, requires_grad=False))
        all_scores = device(torch.empty((data_len, num_classes), requires_grad=False))
        
        
        # might not be necessary to enumare as we have no batches
        for i, (x,y) in enumerate(val_data_loader): 
            batch_len = y.shape[0]  # current batch size, last batch has different value               
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
    
    # create datasets    
    if args.approach == "SoftMax":
        print(" ========= Using SoftMax Loss ===========")
        val_dataset = Dataset(args, args.dataset_root, which_set="val", include_unknown=False)
        test_dataset = Dataset(args, args.dataset_root, which_set="test", include_unknown=False)
            
    elif args.approach =="Garbage":
        print(" ========= Using Garbage Loss ===========")
        val_dataset = Dataset(args, args.dataset_root, which_set="val", has_garbage_class=True)
        test_dataset = Dataset(args, args.dataset_root, which_set="test", has_garbage_class=True)
                
    elif args.approach == "EOS":
        print(" ========= Using Entropic Openset Loss ===========")
        val_dataset = Dataset(args,  args.dataset_root, which_set="val",)
        test_dataset = Dataset(args, args.dataset_root, which_set="test")

    # Info on console
    print("\n========== Data ==========")
    print(f"Val dataset len:{len(val_dataset)}")
    print(f"Test dataset len:{len(test_dataset)}")

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
    loss_suffix = str(args.approach)
    model_path = 'LeNet_plus_plus/' + loss_suffix + '/' + loss_suffix + '.pth'

    net = architectures.__dict__[args.arch](use_BG=args.approach == "Garbage")
    start_epoch, best_score = load_checkpoint(net, model_path)
    print(f"Taking model from epoch {start_epoch} that achieved best score {best_score}")
    net = tools.device(net)
    

    '''
    net.load_state_dict(torch.load(model_path)) #used to have an attribute map_location=net.device
    '''

    print("========== Evaluating ==========")
    print("Validation data:")
    # extracting arrays for validation
    gt, logits, features, scores = get_arrays(
        model=net,
        loader=val_loader
    )
    
    directory = pathlib.Path(f"{args.eval_directory}")
    
    file_path = directory/ f"{args.approach}_val_arr{loss_suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")

    # extracting arrays for test
    print("Test data:")
    gt, logits, features, scores = get_arrays(
        model=net,
        loader=test_loader
    )
    file_path = directory/ f"{args.approach}_test_arr{loss_suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")
    
def save_checkpoint(f_name, model, epoch, opt, best_score_, scheduler=None):
    """ Saves a training checkpoint.

    Args:
        f_name(str): File name.
        model(torch module): Pytorch model.
        epoch(int): Current epoch.
        opt(torch optimizer): Current optimizer.
        best_score_(float): Current best score.
        scheduler(torch lr_scheduler): Pytorch scheduler.
    """
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
    if args.task == "train":
        print(" TASK IS TO TRAIN")
        example = training(args = args)
    if args.task == "eval":
        print(" TASK IS TO EVALUATE")
        evaluate(args = args)
    if args.task == "plot":
        print(" TASK IS TO PLOT")
        example = training(args = args)
    if args.task == "show":
        print("creating labels file")
        create_labels_files(args=args)