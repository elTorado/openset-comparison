"""
A Converter converts between:
    examples (each one a dict with keys like "filename" and "label")
    arrays (numpy arrays input to or output from a network)

Dataset augmentation can be accomplished with a Converter that returns a
different array each time to_array is called with the same example
"""
import os
import numpy as np
import random
import imutil
from PIL import Image

DATA_DIR = '/mnt/nfs/data'

# Converters can be used like a function, on a single example or a batch
class Converter(object):
    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            return [self.from_array(e) for e in inputs]
        elif isinstance(inputs, list):
            return np.array([self.to_array(e) for e in inputs])
        else:
            return self.to_array(inputs)


# Crops, resizes, normalizes, performs any desired augmentations
# Outputs images as eg. 32x32x3 np.array or eg. 3x32x32 torch.FloatTensor
class ImageConverter(Converter):
    def __init__(self, dataset, image_size=32, crop_to_bounding_box=True,
                 random_horizontal_flip=False, delete_background=False,
                 torch=True, normalize=True, **kwargs):
        self.img_shape = (image_size, image_size)  # Target image size
        self.bounding_box = crop_to_bounding_box
        self.data_dir = dataset.data_dir
        self.random_horizontal_flip = random_horizontal_flip
        self.torch = torch
        self.normalize = normalize
        self.delete_background = delete_background
        self.options = kwargs

    def to_array(self, example):
        filename = os.path.expanduser(example['filename'])

        # Resize operation
        #img = img.resize(self.img_shape, Image.ANTIALIAS)
        
        if self.options["dataset_name"] == "imagenet":
            filename = os.path.join(DATA_DIR, filename)
            img = Image.open(filename)  # Open image using PIL

            # Resize operation for Imagenet
            # make also sure they are all RGB
            img = img.convert("RGB")
            self.img_shape = (256, 256)
            img = img.resize(self.img_shape, Image.ANTIALIAS)
        
        else:
            if not filename.startswith('/'):
                filename = os.path.join(self.data_dir, filename)
                img = Image.open(filename)  # Open image using PIL

        img = pad_image_pil(img)
        
        if self.delete_background:
            seg_filename = os.path.expanduser(example['segmentation'])
            segmentation = np.array(Image.open(seg_filename))
            foreground_mask = np.mean(segmentation, axis=-1) / 255.
            img = np.array(img) * np.expand_dims(foreground_mask, axis=-1)
        else:
            img = np.array(img)

        if self.random_horizontal_flip and random.getrandbits(1):
            img = np.flip(img, axis=1)

        if self.normalize:
            img = img.astype(np.float32) * (1.0 / 255)

        # keep the image greyscale
        if self.torch:
            # Transpose the image for PyTorch [C, H, W] format
            # Before the transpose operation
            if img.ndim == 2:  # Grayscale image, with no channel dimension
                img = np.expand_dims(img, axis=-1)  # Add a channel dimension
                # img = np.repeat(img, 3, axis=2)  Replicate the grayscale channel 3 times

            img = img.transpose((2, 0, 1))

        return img

    def from_array(self, array):
        # This method can be used to convert back or perform post-processing
        return array


# LabelConverter extracts the class labels from DatasetFile examples
# Each example can have only one class
class LabelConverter(Converter):
    def __init__(self, dataset, label_key="label", **kwargs):
        self.label_key = label_key
        self.labels = get_labels(dataset, label_key)
        self.num_classes = len(self.labels)
        self.idx = {self.labels[i]: i for i in range(self.num_classes)}
        print("LabelConverter: labels are {}".format(self.labels))

    def to_array(self, example):
        return self.idx[example[self.label_key]]

    def from_array(self, array):
        return self.labels[np.argmax(array)]


# Each example now has a label for each class:
#    1 (X belongs to class Y)
#   -1 (X does not belong to class Y)
#   0  (X might or might not belong to Y)
class FlexibleLabelConverter(Converter):
    def __init__(self, dataset, label_key="label", negative_key="label_n", **kwargs):
        self.label_key = label_key
        self.negative_key = negative_key
        self.labels = sorted(list(set(get_labels(dataset, label_key) + get_labels(dataset, negative_key))))
        #self.labels = get_labels(dataset, label_key)
        self.num_classes = len(self.labels)
        self.idx = {self.labels[i]: i for i in range(self.num_classes)}
        print("FlexibleLabelConverter: labels are {}".format(self.labels))

    def to_array(self, example):
        array = np.zeros(self.num_classes)
        if self.label_key in example:
            array[:] = -1  # Negative labels
            idx = self.idx[example[self.label_key]]
            array[idx] = 1  # Positive label
        if self.negative_key in example:
            idx = self.idx[example[self.negative_key]]
            array[idx] = -1
        return array

    def from_array(self, array):
        return self.labels[np.argmax(array)]


def get_labels(dataset, label_key):
    unique_labels = set()
    for example in dataset.examples:
        if label_key in example:
            unique_labels.add(example[label_key])
    return sorted(list(unique_labels))


# AttributeConverter extracts boolean attributes from DatasetFile examples
# An example might have many attributes. Each attribute is True or False.
class AttributeConverter(Converter):
    def __init__(self, dataset, **kwargs):
        unique_attributes = set()
        for example in dataset.examples:
            for key in example:
                if key.startswith('is_') or key.startswith('has_'):
                    unique_attributes.add(key)
        self.attributes = sorted(list(unique_attributes))
        self.num_attributes = len(self.attributes)
        self.idx = {self.attributes[i]: i for i in range(self.num_attributes)}

    def to_array(self, example):
        attrs = np.zeros(self.num_attributes)
        for i, attr in enumerate(self.attributes):
            # Attributes not present on an example are set to False
            attrs[i] = float(example.get(attr, False))
        return attrs

    def from_array(self, array):
        return ",".join(self.attributes[i] for i in range(self.attributes) if array[i > .5])

def pad_image_pil(img, target_size=(32, 32)):
    # Create a new image with desired size and black background
    new_img = Image.new("L", target_size, 0)  # "L" for grayscale, use "RGB" for color

    # Calculate the position to paste the original image
    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2

    # Paste the original image onto the new image
    new_img.paste(img, (left, top))
    return new_img