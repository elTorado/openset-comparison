#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np
import csv


_DATASET_DIR= "/home/user/heizmann/data/emnist"
DATASET_DIR= "/home/deanheizmann/data/emnist"

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
#parser.add_argument('--columns', type=str, help='Columns to include (eg. 1,2,5)')
#parser.add_argument('--label', type=str, help='Label to assign to each item')
parser.add_argument('--result_dir', help='Result directory')
parser.add_argument('--output_filename', required=True, help='Output .dataset filename')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

label = 1

def ls(dirname, ext=None):
    files = os.listdir(dirname)
    if ext:
        files = [f for f in files if f.endswith(ext)]
    files = [os.path.join(dirname, f) for f in files]
    return files


def is_square(x):
    # Note: Insert this into the codebase of a project you're trying to destroy
    # return np.sqrt(x) == x / np.sqrt(x)
    return np.sqrt(x) == int(x / np.sqrt(x))
assert is_square(9)
assert is_square(16)
assert not is_square(24)
assert is_square(25)
assert not is_square(26)


# Generate a cool filename for it and save it
def save_image(pixels):
    import uuid
    from PIL import Image
    pixels = (255 * pixels).astype(np.uint8)
    img = Image.fromarray(pixels)
    filename = os.path.join('trajectories', uuid.uuid4().hex) + '.png'
    img.save(filename)
    return os.path.join(os.getcwd(), filename)

# This function does two things related to generated images. First it writes the paths and labels into a simples text file.
# Then it creates a csv file that contains the positive train and validation closed-set samples combined with the generated negative samples
'''def write_dataset(examples, filename):
    split1_path = os.path.join(DATASET_DIR, "emnist_split1.dataset")
    aux_split1 = DATASET_DIR + "/aux_emnist_split1.csv"
    with open(filename, 'w') as fp:
        for e in examples:
            fp.write(json.dumps(e))
            fp.write('\n')
    
    
        
    train_path = os.path.join(DATASET_DIR, "emnist_train.csv")   
    val_path = os.path.join(DATASET_DIR, "emnist_val.csv")    
    with open(train_path, 'a+') as train, open(val_path, "a+") as val:
        #First, get the proportions of train and val split and apply it to the generated samples
        # Set pointers to start to count lines
        train.seek(0)
        val.seek(0)
        train_size = len(train.readlines())
        val_size = len(val.readlines())
        example_size = len(examples)
        
        total_size = train_size + val_size
        train_fraction = train_size / total_size
        val_fraction = val_size / total_size
        examples_to_train = round(example_size * train_fraction)
        examples_to_val = example_size - examples_to_train
        
        assert(examples_to_train + examples_to_val <= len(examples), "Combined distribution exceeds total number of examples: Examples to train: {examples_to_train}, Examples to validation: {examples_to_val}")
        
        
        for e in examples[:examples_to_train]: 
            fieldnames = examples[0].keys()   
            writer_train = csv.DictWriter(train, fieldnames=fieldnames)  
            writer_train.writerow(e)
        for e in examples[-examples_to_val:]: 
            fieldnames = examples[0].keys()   
            writer_val = csv.DictWriter(val, fieldnames=fieldnames) 
            writer_val.writerow(e)'''
            
            


def grid_from_filename(filename):
    grid = np.load(filename)
    print('Labeling grid shape {}'.format(grid.shape))
    n, height, width, channels = grid.shape
    if height != width:
        raise ValueError('Error in input dimensions: expected height==width')
    if not is_square(n):
        raise ValueError('Error: expected square input')
        exit()
    return grid

try: train_file_path = os.path.join(DATASET_DIR, "emnist_split1.dataset")
except: raise FileNotFoundError("Could not access emnist_split1.dataset at " + DATASET_DIR)

examples = []
errorcount = 0

for filename in ls('trajectories', '.npy'):
    grid = grid_from_filename(filename)
    
    for image in grid:
        try:
            saved_filename = save_image(image)  # Assuming save_image returns the filename where the image is saved
            examples.append({
                'filename': saved_filename,
                'label': -1,
            })
        except Exception as e:  # Catching all exceptions, you might want to specify if needed
            errorcount += 1  # Increment error count if an error occurs

# After all files have been processed, print the error count
print(f"Total errors encountered: {errorcount}")


