import os
import numpy as np
import json
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--protocol', type=str, required=True, help='protocol, can be 1, 2, or 3')

options = vars(parser.parse_args())


_DATA_DIR = '/home/user/heizmann/data/'
DATA_DIR = 'local/scratch/datasets/ImageNet/ILSVRC2012/'

'''
    Downloads the ImageNet dataset and creates a .dataset file. The file contains datasplits for GAN training (80% train, 20% test). 
'''

""" Code based on: Bhoumik, A. (2021). Open-set Classification on ImageNet."""
def transform(img):
    return img.permute(0, 3, 1,2)

class ImagenetDataset(Dataset):
    """ Imagenet Dataset. """

    def __init__(self, csv_file, imagenet_path, transform=None):
        """ Constructs an Imagenet Dataset from a CSV file. The file should list the path to the
        images and the corresponding label. For example:
        val/n02100583/ILSVRC2012_val_00013430.JPEG,   0

        Args:
            csv_file(Path): Path to the csv file with image paths and labels.
            imagenet_path(Path): Home directory of the Imagenet dataset.
            transform(torchvision.transforms): Transforms to apply to the images.
        """
        self.dataset = pd.read_csv(csv_file, header=None)
        self.imagenet_path = Path(imagenet_path)
        self.transform = transform
        self.label_count = len(self.dataset[1].unique())
        self.unique_classes = np.sort(self.dataset[1].unique())

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Returns a tuple (image, label) of the dataset at the given index. If available, it
        applies the defined transform to the image. Images are converted to RGB format.

        Args:
            index(int): Image index

        Returns:
            image, label: (image tensor, label tensor)
        """
        if torch.is_tensor(index):
            index = index.tolist()

        jpeg_path, label = self.dataset.iloc[index]
        image = Image.open(self.imagenet_path / jpeg_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # convert int label to tensor
        label = torch.as_tensor(int(label), dtype=torch.int64)
        return image, label

    def has_negatives(self):
        """ Returns true if the dataset contains negative samples."""
        return -1 in self.unique_classes

    def replace_negative_label(self):
        """ Replaces negative label (-1) to biggest_label + 1. This is required if the loss function
        is BGsoftmax. Updates the array of unique labels.
        """
        # get the biggest label, which is the number of classes - 1 (since we have the -1 label inside)
        biggest_label = self.label_count - 1
        self.dataset[1].replace(-1, biggest_label, inplace=True)
        self.unique_classes[self.unique_classes == -1] = biggest_label
        self.unique_classes.sort()

    def remove_negative_label(self):
        """ Removes all negative labels (<0) from the dataset. This is required for training with plain softmax"""
        self.dataset.drop(self.dataset[self.dataset[1] < 0].index, inplace=True)
        self.unique_classes = np.sort(self.dataset[1].unique())
        self.label_count = len(self.dataset[1].unique())

    def calculate_class_weights(self):
        """ Calculates the class weights based on sample counts.

        Returns:
            class_weights: Tensor with weight for every class.
        """
        # TODO: Should it be part of dataset class?
        counts = self.dataset.groupby(1).count().to_numpy()
        class_weights = (len(self.dataset) / (counts * self.label_count))
        return torch.from_numpy(class_weights).float().squeeze()

    
    def create_gan_training_splits(self):
        """Create training and validation splits for GAN training and save them to a dataset file.

            Splits the dataset into 80% training and 20% validation sets. The entries are saved to a file named
            "imagenet_p<protocol>.dataset". Each entry contains the filename, fold (either 'train' or 'test'), and the label.

        """        
        start_val_index = int(len(self) * 0.8)  # Calculate index for 20% validation split
        
        with open("imagenet_p"+options["protocol"]+".dataset", 'w') as file: 
            for i in range(len(self)):
                jpeg_path, label = self.dataset.iloc[i]
                # Determine whether the entry should be 'train' or 'test'
                entry = {
                    "filename": jpeg_path,
                    "fold": "test" if i >= start_val_index else "train",
                    "label": int(label)
                }
                if label >= 0:
                    file.write(json.dumps(entry, sort_keys=True) + '\n')

if __name__ == '__main__':
    #imagenet = ImagenetDataset(csv_file="/home/deanheizmann/masterthesis/openset-imagenet/protocols/p2_train.csv", imagenet_path=DATA_DIR)
    imagenet = ImagenetDataset(csv_file="/home/user/heizmann/openset-comparison/protocols/p"+options["protocol"]+"_train.csv", imagenet_path=DATA_DIR)
    imagenet.create_gan_training_splits()