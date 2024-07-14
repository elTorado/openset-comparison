#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np
import csv

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--output_filename', required=True, help='directory and name of created .dataset filename')
parser.add_argument('--dataset', required=True, help='dataset of which images are processed')
options = vars(parser.parse_args())

'''
    The goal of this file is to create a ".dataset" file which contains the paths to synthetic counterfactual images.
    This .dataset file can then be used to train the open-set classifier.
    This file can handle both, EMNIST and ImageNet dataset. 
    The code will list all images it finds in the respective dataset directory in tranjectories/, 
    iterate trough them and write them into a .dataset file and save in in the given output directory.

'''

def ls(dirname, ext=None):
    """
    Args:
        dirname (string): directory path

    Returns:
        list: list with all filenames as string
    """    
    files = os.listdir(dirname)
    if ext:
        files = [f for f in files if f.endswith(ext)]
    files = [os.path.join(dirname, f) for f in files]
    return files


# Generate a cool filename for it and save it
def save_image(pixels):
    """_summary_

    Args:
        pixels (tensor): image tensor to sace

    Returns:
        string: where the image was saved
    """    
    import uuid
    from PIL import Image
    pixels = (255 * pixels).astype(np.uint8).squeeze()
    img = Image.fromarray(pixels)
    filename = os.path.join('trajectories', uuid.uuid4().hex) + '.png'
    img.save(filename)
    return os.path.join(os.getcwd(), filename)

# This function does two things related to generated images. First it writes the paths and labels into a simples text file.
# Then it creates a csv file that contains the positive train and validation closed-set samples combined with the generated negative samples
def write_dataset(examples, filename):
    """_summary_

    Args:
        examples (list): list of images paths
        filename : filename to write the paths into
    """    
    with open(filename, 'w') as fp:
        for e in examples:
            fp.write(json.dumps(e))
            fp.write('\n')
                


def grid_from_filename(filename):
    """_summary_
    load image grid of batch size from path
    Args:
        filename (string): path

    Returns:
        numpy tensor: tensor with image tensors of batch size
    """    
    grid = np.load(filename)
    n, height, width, channels = grid.shape
    return grid

examples = []
errorcount = 0


if options["dataset"] == "emnist":
    print("CREATING EMNIST - COUNTERFACTUAL DATASET FILE")
    grids = 0
    images = 0
    for filename in ls('trajectories/emnist/counterfactual/', '.npy'):
        grids += 1
        grid = grid_from_filename(filename)
        
        for image in grid:
            
            try:
                saved_filename = save_image(image)  
                examples.append({
                    'filename': saved_filename,
                    'label': -1,
                })
                images += 1
            except Exception as e:  
                errorcount += 1  # Increment error count if an error occurs (if e.g. grid dimenssion are unexpected - might happen when e.g. last batch is not as large as others)
    print("GRIDS FOUND: ", grids)
    print("IMAGES ADDED: ", images)


if options["dataset"] == "imagenet":
    print("CREATING IMAGENET - COUNTERFACTUAL DATASET FILE")
    grids = 0
    images = 0
    for filename in ls('trajectories/imagenet/counterfactual/', '.npy'):
        grids += 1
        grid = grid_from_filename(filename)
        
        for image in grid:
            
            try:
                saved_filename = save_image(image)  
                examples.append({
                    'filename': saved_filename,
                    'label': -1,
                })
                images += 1
            except Exception as e:  
                errorcount += 1  # Increment error count if an error occurs (if e.g. grid dimenssion are unexpected - might happen when e.g. last batch is not as large as others)
    print("GRIDS FOUND: ", grids)
    print("IMAGES ADDED: ", images)


print(f"Total errors encountered: {errorcount}")

write_dataset(examples, options['output_filename'])
