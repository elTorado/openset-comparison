#!/usr/bin/env python
import argparse
import json
import os
import sys
import numpy as np

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
#parser.add_argument('--columns', type=str, help='Columns to include (eg. 1,2,5)')
#parser.add_argument('--label', type=str, help='Label to assign to each item')
parser.add_argument('--result_dir', help='Result directory')
parser.add_argument('--output_filename', required=True, help='Output .dataset filename')
parser.add_argument('--dataset_name', required=True, help='dataset')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def ls(dirname, ext=None):
    files = os.listdir(dirname)
    if ext:
        files = [f for f in files if f.endswith(ext)]
    files = [os.path.join(dirname, f) for f in files]
    return files


# Generate a cool filename for it and save it
def save_image(pixels):
    import uuid
    from PIL import Image
    pixels = (255 * pixels).astype(np.uint8)
    img = Image.fromarray(pixels)
    filename = os.path.join('trajectories', uuid.uuid4().hex) + '.png'
    img.save(filename)
    return os.path.join(os.getcwd(), filename)


def write_dataset(examples, filename):
    print("WRITING DATASET FILE: ", filename)
    with open(filename, 'w') as fp:
        for e in examples:
            fp.write(json.dumps(e))
            fp.write('\n')


examples = []

if options["dataset_name"]=="imagenet":
    directory = 'trajectories/imagenet/arpl'
if options["dataset_name"]=="emnist":
    directory = 'trajectories/emnist/arpl'

images = 0
for filename in ls(directory, '.jpg'):
    if not "grid" in filename:
        examples.append({
            'filename': filename,
            'label': -1,
        })
        images += 1
    print("IMAGES ADDED: ", images)
        


write_dataset(examples, options['output_filename'])
