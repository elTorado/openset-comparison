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
from vast.tools import set_device_gpu, set_device_cpu
from openset_imagenet.train import get_arrays
import numpy as np

DATA_DIR = '/home/user/heizmann/data/'
_DATA_DIR = '/home/deanheizmann/data/'

GENERATED_NEGATIVES_DIR = "/home/user/heizmann/openset-comparison/counterfactual-open-set/generated_images_counterfactual.dataset"

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the main training script for all MNIST experiments. \
                    Where applicable roman letters are used as negatives. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approach", "-a", required=True, choices=['SoftMax', 'Garbage', 'EOS', 'Objectosphere'])
    parser.add_argument("--task", default='train', choices=['train', 'eval', "plot"])
    parser.add_argument("--arch", default='LeNet_plus_plus', choices=['LeNet', 'LeNet_plus_plus'])
    parser.add_argument('--second_loss_weight', "-w", help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--Minimum_Knowns_Magnitude', "-m", help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument("--solver", dest="solver", default='sgd',choices=['sgd','adam'])
    parser.add_argument("--lr", "-l", dest="lr", default=0.01, type=float)
    parser.add_argument('--batch_size', "-b", help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
    parser.add_argument("--no_of_epochs", "-e", dest="no_of_epochs", type=int, default=70)
    parser.add_argument("--dataset_root", "-d", dest= "dataset_root", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--gpu", "-g", type=int, nargs="?",dest="gpu", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")
    parser.add_argument("--include_counterfactuals", "-inc_c", type=bool, default=False, dest="include_counterfactuals", help="Include counterfactual images in the dataset")
    parser.add_argument("--include_arpl", "-inc_a", type=bool, default=False, dest="include_arpl", help="Include ARPL samples in the dataset")
    parser.add_argument("--mixed_unknowns", "-mu", type=bool, default=False, dest="mixed_unknowns", help="Mix unknown samples in the dataset")

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

    def __init__(self, args, dataset_root, which_set="train", include_unknown=True, has_garbage_class=False, include_counterfactuals = False, include_arpl = False, mixed_unknowns = False):
        
        include_arpl = args.include_arpl
        include_counterfactuals = args.include_counterfactuals
        mixed_unknowns = args.mixed_unknowns
        
        self.mnist = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=False,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=False,
            split='letters',
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.which_set = which_set
        
        #If no unknowns, then  targets is empty.
        #If we have unknowns, then train & val targets are A - M / Test O - Z or so
        targets = list() if not include_unknown else [1,2,3,4,5,6,8,10,11,13,14] if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]
        self.has_garbage_class = has_garbage_class
        self.synthetic_samples = list() if not include_counterfactuals else self.load_arpl() if include_arpl else self.load_counterfactuals()
        print(" ========= LENGTH OF DIGITS :" + str(len(self.mnist)))
        print(" ========= LENGTH OF LETTER :" + str(len(self.letters)))
        print(" ========= LENGTH OF SYNTHETIC SAMPLES :" + str(len(self.synthetic_samples)))       
        print(" ========= LENGTH OF CLASS ITSELF :" + str(len(self)))
        
        # Print tensor sizes for each dataset sample
        if len(self.mnist) > 0:
            print(f"MNIST sample tensor size: {self.mnist[0][0].size()}")
        if len(self.letter_indexes) > 0:
            print(f"Letters sample tensor size: {self.letters[self.letter_indexes[0]][0].size()}")
        if len(self.synthetic_samples) > 0:
            print(f"Synthetic sample tensor size: {self.synthetic_samples[0][0].size()}")

        
        
    def load_counterfactuals(self, data_path=GENERATED_NEGATIVES_DIR):
        
        samples = []
        counter = 0
        # Read the whole file at once
        with open(data_path, 'r') as file:
            file_content = file.readlines()
            print(len(file_content))

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
                if counter%500 == 0:
                    print("ADDED "+ str(counter) +" NEGATIVE COUNTERFACTUAL SAMPLES")
                if counter == 6400:
                    break
            except Exception as e:
                print(f"Error processing item {item}: {e}")
        
        print("======= ADDED "+ str(counter) +" NEGATIVE COUNTERFACTUAL SAMPLES TO EMNIST DATSET =========")
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
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data=Dataset(args, args.dataset_root),
                    val_data = Dataset(args, args.dataset_root, which_set="val")
                )
    elif args.approach == "Objectosphere":
        print(" ========= Using Objectosphere Loss ===========")
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=losses.objectoSphere_loss(args.Minimum_Knowns_Magnitude),
                    training_data=Dataset(args, args.dataset_root),
                    val_data = Dataset(args, args.dataset_root, which_set="val")
                )


def train(args):
    # setup device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
        print(" ============== GPU Selected! =============")
    else:
        print("No GPU device selected, training will be extremely slow")
        set_device_cpu()
    
    torch.manual_seed(0)

    # get training data and loss function(s)
    first_loss_func,second_loss_func,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]

    results_dir = pathlib.Path(f"{args.arch}/{args.approach}")
    model_file = f"{results_dir}/{args.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)

    # instantiate network and data loader
    # WHY IS GARBAGE HARDCODED?
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

    # train network
    prev_confidence = None
    for epoch in range(1, args.no_of_epochs + 1, 1):  # loop over the dataset multiple times
        print ("======== TRAINING EPOCH: " + str(epoch) +" ===============")
        
        loss_history = []
        train_accuracy = torch.zeros(2, dtype=int)
        train_magnitude = torch.zeros(2, dtype=float)
        train_confidence = torch.zeros(2, dtype=float)
        net.train()
        
        debug = True
        
        for x, y in train_data_loader:
          
            x = tools.device(x)
            y = tools.device(y)    
                    
            optimizer.zero_grad()
            logits, features = net(x)
            
            # first loss is always computed, second loss only for some loss functions
            loss = first_loss_func(logits, y) + args.second_loss_weight * second_loss_func(features, y)

            # metrics on training set
            train_accuracy += losses.accuracy(logits, y)
            train_confidence += losses.confidence(logits, y)
            if args.approach not in ("SoftMax", "Garbage"):
                train_magnitude += losses.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach in args.approach == "Objectosphere" else None)

            loss_history.extend(loss.tolist())
            loss.mean().backward()
            optimizer.step()

        # metrics on validation set
        with torch.no_grad():
            val_loss = torch.zeros(2, dtype=float)
            val_accuracy = torch.zeros(2, dtype=int)
            val_magnitude = torch.zeros(2, dtype=float)
            val_confidence = torch.zeros(2, dtype=float)
            net.eval()
            
            for x,y in val_data_loader:                
                x = tools.device(x)          
                y = tools.device(y)
                outputs = net(x)             
                loss = first_loss_func(outputs[0], y) + args.second_loss_weight * second_loss_func(outputs[1], y)               
                
                val_loss += torch.tensor((torch.sum(loss), len(loss)))
                val_accuracy += losses.accuracy(outputs[0], y)
                val_confidence += losses.confidence(outputs[0], y)
                if args.approach not in ("SoftMax", "Garbage"):
                    val_magnitude += losses.sphere(outputs[1], y, args.Minimum_Knowns_Magnitude if args.approach == "Objectosphere" else None)
                    
                debug = False
                
        # log statistics
        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        writer.add_scalar('Loss/train', epoch_running_loss, epoch)
        writer.add_scalar('Loss/val', val_loss[0] / val_loss[1], epoch)
        writer.add_scalar('Acc/train', float(train_accuracy[0]) / float(train_accuracy[1]), epoch)
        writer.add_scalar('Acc/val', float(val_accuracy[0]) / float(val_accuracy[1]), epoch)
        writer.add_scalar('Conf/train', float(train_confidence[0]) / float(train_confidence[1]), epoch)
        writer.add_scalar('Conf/val', float(val_confidence[0]) / float(val_confidence[1]), epoch)
        writer.add_scalar('Mag/train', train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else 0, epoch)
        writer.add_scalar('Mag/val', val_magnitude[0] / val_magnitude[1], epoch)

        # save network based on confidence metric of validation set
        save_status = "NO"
        if prev_confidence is None or (val_confidence[0] > prev_confidence):
            torch.save(net.state_dict(), model_file)
            prev_confidence = val_confidence[0]
            save_status = "YES"

        # print some statistics
        print(f"Epoch {epoch} "
              f"train loss {epoch_running_loss:.10f} "
              f"accuracy {float(train_accuracy[0]) / float(train_accuracy[1]):.5f} "
              f"confidence {train_confidence[0] / train_confidence[1]:.5f} "
              f"magnitude {train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else -1:.5f} -- "
              f"val loss {float(val_loss[0]) / float(val_loss[1]):.10f} "
              f"accuracy {float(val_accuracy[0]) / float(val_accuracy[1]):.5f} "
              f"confidence {val_confidence[0] / val_confidence[1]:.5f} "
              f"magnitude {val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else -1:.5f} -- "
              f"Saving Model {save_status}")

def evaluate(args):
    
    # create datasets    
    if args.approach == "SoftMax":
        print(" ========= Using SoftMax Loss ===========")
        val_dataset = Dataset(args, args.dataset_root, include_unknown=False)
        test_dataset = Dataset(args, args.dataset_root, which_set="test", include_unknown=False)
            
    elif args.approach =="Garbage":
        print(" ========= Using Garbage Loss ===========")
        val_dataset = Dataset(args, args.dataset_root, has_garbage_class=True),
        test_dataset = Dataset(args, args.dataset_root, which_set="test", has_garbage_class=True)
                
    elif args.approach == "EOS":
        print(" ========= Using Entropic Openset Loss ===========")
        val_dataset=Dataset(args, args.dataset_root),
        test_dataset = Dataset(args, args.dataset_root, which_set="test")

    # Info on console
    print("\n========== Data ==========")
    print(f"Val dataset len:{len(val_dataset)}")
    print(f"Test dataset len:{len(test_dataset)}")

    # create data loaders
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)

    # create device
    if args.gpu is not None:
        set_device_gpu(index=args.gpu)
    else:
        print("No GPU device selected, evaluation will be slow")
        set_device_cpu()

    '''
    if args.loss == "garbage":
        n_classes = val_dataset.label_count # we use one class for the negatives
    else:
        n_classes = val_dataset.label_count - 1  # number of classes - 1 when training with unknowns
    '''
    
    # create model
    loss_suffix = str(args.approach)
    
    net = architectures.__dict__[args.arch](use_BG=args.approach == "Garbage")
    net = tools.device(net)
    
    model_path = 'LeNet_plus_plus/' + loss_suffix + '/' + loss_suffix + '.model'
    print(f"Taking model from:  {model_path}")
    net.load_state_dict(torch.load(model_path, map_location=net.device))


    print("========== Evaluating ==========")
    print("Validation data:")
    # extracting arrays for validation
    gt, logits, features, scores = get_arrays(
        model=net,
        loader=val_loader
    )
    file_path = args.output_directory / f"{args.loss}_val_arr{suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")

    # extracting arrays for test
    print("Test data:")
    gt, logits, features, scores = get_arrays(
        model=net,
        loader=test_loader
    )
    file_path = args.output_directory / f"{args.loss}_test_arr{suffix}.npz"
    np.savez(file_path, gt=gt, logits=logits, features=features, scores=scores)
    print(f"Target labels, logits, features and scores saved in: {file_path}")
    
    
    
if __name__ == "__main__":
    args = command_line_options()
    if args.task == "train":
        print(" TASK IS TO TRAIN")
        example = train(args = args)
    if args.task == "eval":
        print(" TASK IS TO EVALUATE")
        evaluate(args = args)
    if args.task == "plot":
        print(" TASK IS TO PLOT")
        example = train(args = args)