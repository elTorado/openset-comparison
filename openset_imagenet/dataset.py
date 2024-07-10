""" Code based on: Bhoumik, A. (2021). Open-set Classification on ImageNet."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import json
import os


class ImagenetDataset(Dataset):
    """ Imagenet Dataset. """

    def __init__(self, which_set, csv_file, include_unknown, imagenet_path, counterfactuals_path, arpl_path, mixed_unknowns, transform=None):
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
        
        #preparations for negative handling
        negative_count = (self.dataset[1] == -1).sum()
        negative_indices = self.dataset[self.dataset[1] == -1].index.tolist()
        np.random.shuffle(negative_indices)

        def replace_image_paths(indices, image_list, description):
            initial_count = len(indices)
            replace_count = 0
            for idx in indices:
                if image_list:
                    image_info = image_list.pop(0)  # Get the next dictionary from the list
                    image_path = json.loads(image_info)

                    self.dataset.at[idx, 0] = image_path['filename']  # Set the path  
                    replace_count += 1
                else:
                    break  # Break if no more images are available to avoid index errors
            print(f"Replaced {replace_count} of {initial_count} paths with {description}.")

        # For training with e.g. softmax we dont use negative labels. For our experiments we want them still in test set.
        if not include_unknown and which_set != "test":
            self.remove_negative_label()
        
        if counterfactuals_path:
            with open(counterfactuals_path, 'r') as cf_file:
                counterfactual_images = cf_file.read().splitlines()
                
                # for the validation set, we replace original images from the back instead of from the front
                if which_set != "train":
                    counterfactual_images.reverse()

            
            if arpl_path:
                with open(arpl_path, 'r') as arpl_file:
                    arpl_images = arpl_file.read().splitlines()
                    
                if which_set != "train":
                    arpl_images.reverse()

                if mixed_unknowns:
                    third = len(negative_indices) // 3
                    cf_indices = negative_indices[:third]
                    arpl_indices = negative_indices[third:2*third]
                    remaining_negatives = negative_indices[2*third:]
                    print("Mixing negatives: one-third counterfactuals, one-third ARPL, one-third existing negatives.")
                else:
                    half = len(negative_indices) // 2
                    cf_indices = negative_indices[:half]
                    arpl_indices = negative_indices[half:]
                    print("Splitting negatives 50/50 between counterfactuals and ARPL.")

                replace_image_paths(cf_indices, counterfactual_images, "counterfactual images")
                replace_image_paths(arpl_indices, arpl_images, "ARPL images")

            elif mixed_unknowns:
                half = len(negative_indices) // 2
                cf_indices = negative_indices[:half]
                print("Replacing half of negatives with counterfactual images only.")
                replace_image_paths(cf_indices, counterfactual_images, "counterfactual images")
            else:
                cf_indices = negative_indices
                print("Replacing all negatives with counterfactual images.")
                replace_image_paths(cf_indices, counterfactual_images, "counterfactual images")

        # Finally check the case where no counterfactuals but ARPL is present
        elif arpl_path:
            with open(arpl_path, 'r') as arpl_file:
                arpl_images = arpl_file.read().splitlines()
                arpl_indices = negative_indices
            
            if mixed_unknowns:
                half = len(negative_indices) // 2
                arpl_indices = negative_indices[half:]
                print("Mixing negatives with half ARPL images when no counterfactuals available.")
            
            print("Replacing negatives with ARPL images.")
            replace_image_paths(arpl_indices, arpl_images, "ARPL images")

                

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
        
        # try to get real imagenet iamge, if not possible, try get synthetic image
        try:
        # Try to open the image from the ImageNet directory
            image = Image.open(self.imagenet_path / jpeg_path).convert("RGB")
        except IOError:
            # If the image is not found, fall back trajectories
            try:
                
                user_path = "/home/user/heizmann/openset-comparison/counterfactual-open-set/"
                full_path = os.path.join(user_path, jpeg_path)

                image = Image.open(full_path).convert("RGB")
            except IOError:
                raise IOError(f"Unable to open image from both paths: {self.imagenet_path / jpeg_path} and {jpeg_path}")
        

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
        print(" REMOVE ALL NEGATIVE LABELS ")
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