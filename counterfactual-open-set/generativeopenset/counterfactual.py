import os
import time
import torch
import numpy as np
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from vector import clamp_to_unit_sphere

from logutil import TimeSeries
import imutil

'''
    This code creates grids of counterfactual images and open-set images and stores them in 
    the dataset specific subdirectory of the trajectories/ directory. A trained GAN needs to be provided
    as well as the dataset name.

'''


def to_torch(z, requires_grad=False):
    '''Convert a numpy array to a PyTorch Variable '''
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()


def to_np(z):
    '''Convert a PyTorch Variable to a numpy array'''
    return z.data.cpu().numpy()


def generate_counterfactual(networks, dataloader, **options):
    """Generate 'counterfactual' images for each class by gradient descent of the class.

    This function generates counterfactual images for a specified number of classes (K). It starts with randomly-selected
    images from the dataloader and then generates a column of images corresponding to each target class. The generated images
    are saved in the trajectories directory as a grid of images.

    Args:
        networks (dict): A dictionary containing the neural networks used for generation, classification, and encoding.
        dataloader (object): DataLoader object containing dataset.
        options (dict): Additional options for the generation process.
            - 'result_dir' (str): Directory where the result files are to be stored.
            - 'dataset_name' (str): The name of the dataset (e.g., 'imagenet' or 'emnist').

    Returns:
        np.ndarray: A numpy array containing the generated counterfactual images.
    """
    
    print("**************** ENTERING COUNTERFACTUALS GENERATION PROCESS*****************")
    """
    # TODO: Fix Dropout/BatchNormalization outside of training -> CHAT GPT SAYS IT WORKS BY UNCOMMENTIG THAT PART
    for net in networks:
        networks[net].eval()
    """
    result_dir = options['result_dir']

    # NOTE: Too many classes in datasets like cub200
    K = min(dataloader.num_classes, 10) #LIMITS K TO 10 IF THERE ARE MORE CLASSES
    print("======================== NUMBER OF CLASSES CLASSES:===========================")
    print(K)
    # Make the batch size large enough to form a square grid
    cf_count = K + 2

    # Start with randomly-selected images from the dataloader
    start_images, _ = dataloader.get_batch()
    start_images = start_images[:cf_count]  # assume batch_size >= cf_count, we need at least as many start images, as cf_count

    batches = [start_images.cpu().numpy()] #convert start images to numpy arrays
    
    print("======================== BATCHES:===========================")
    print(batches)
    for target_class in range(K + 1):
        # Generate one column of the visualization, corresponding to a target class
        img_batch = generate_counterfactual_column(networks, start_images, target_class, **options)
        batches.append(img_batch)

    images = []
    for i in range(cf_count):
        for batch in batches:
            images.append(batch[i])

    images = np.array(images).transpose((0,2,3,1)) #re order dimensions for saving or displaying
    dummy_class = 0
    video_filename = make_video_filename(result_dir, dataloader, dummy_class, dummy_class, dataset=options["dataset_name"],  label_type='grid')

    # Save the images in npy/jpg format as input for the labeling system
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)
    imutil.show(images, display=False, filename=video_filename.replace('.mjpeg', '.jpg'))

    # Save the images in jpg format to display to the user
    name = 'counterfactual_{}.jpg'.format(int(time.time()))
    jpg_filename = os.path.join(result_dir, 'images', name)
    imutil.show(images, filename=jpg_filename)
    return images


# Generates 'unknown unknown' images unlike any known class
def generate_open_set(networks, dataloader, **options):
    """Generate 'unknown unknown' images from known classes.

    This function generates images that do not belong to any known class by using a counterfactual generation process. 
    The generated images are saved in the trajectories directory as a grid of images.

    Args:
        networks (dict): A dictionary containing the neural networks used for generation, classification, and encoding.
        dataloader (object): DataLoader object containing dataset.
        options (dict): Additional options for the generation process.
            - 'result_dir' (str): Directory where the result files are to be stored.
            - 'dataset_name' (str): The name of the dataset (e.g., 'imagenet' or 'emnist').

    Returns:
        np.ndarray: A numpy array containing the generated 'unknown unknown' images.
    """
    
    
    """
    # TODO: Fix Dropout/BatchNormalization outside of training
    for net in networks:
        networks[net].eval()
    """
    result_dir = options['result_dir']

    # Start with randomly-selected images from the dataloader
    start_images, _ = dataloader.get_batch()

    openset_class = dataloader.num_classes
    images = generate_counterfactual_column(networks, start_images, openset_class, **options)
    images = np.array(images).transpose((0,2,3,1))

    dummy_class = 0
    video_filename = make_video_filename(result_dir, dataloader, dummy_class, dummy_class, dataset=options["dataset_name"], label_type='grid')

    # Save the images in npy/jpg format as input for the labeling system
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)
    imutil.show(images, display=False, filename=video_filename.replace('.mjpeg', '.jpg'))

    # Save the images in jpg format to display to the user
    name = 'counterfactual_{}.jpg'.format(int(time.time()))
    jpg_filename = os.path.join(result_dir, 'images', name)
    
    # THIS WILL SHOW IMAGES WHILE GENERATING ON THE RUN
    #imutil.show(images, filename=jpg_filename)
    return images


log = TimeSeries('Counterfactual')
def generate_counterfactual_column(networks, start_images, target_class, **options):
    """Generates a grif of counterfactual images in batch size.

    Args:
        networks (dict): A dictionary containing the neural networks used for generation, classification, and encoding.
        start_images (torch.Tensor): A batch of starting images to generate counterfactuals from.
        target_class (int): The target class to which the counterfactuals should correspond.
        options (dict): Additional options for the generation process.
            - 'cf_speed' (float): The speed of adjustment in the latent space.
            - 'cf_max_iters' (int): The Number of grids to be generated. The amount of generated images is this times batch size.
            - 'cf_distance_weight' (float): The weight given to the distance loss in the loss function.
            - 'cf_gan_scale' (int): The scale used for the GAN.
            - 'dataset_name' (str): The name of the dataset.

    Returns:
        np.ndarray: A numpy array containing the generated counterfactual images grid.
    """  
    netG = networks['generator']
    netC = networks['classifier_k']
    netE = networks['encoder']
    speed = options['cf_speed']
    max_iters = options['cf_max_iters']
    distance_weight = options['cf_distance_weight']
    gan_scale = options['cf_gan_scale']
    cf_batch_size = len(start_images)
    
    # hardcode this for the moment, for greyscale gan_scale is 4
    if options["dataset_name"] == "imagenet":
        gan_scale = 1
    

    # Start with the latent encodings
    z_value = to_np(netE(start_images, gan_scale))
    z0_value = z_value

    # Move them so their labels match target_label
    target_label = Variable(torch.LongTensor(cf_batch_size)).cuda()
    target_label[:] = target_class

    print(" ============= ENTERING IMAGE GENERATION ====================")
    for i in range(max_iters):
        z = to_torch(z_value, requires_grad=True)
        z_0 = to_torch(z0_value)
        logits = netC(netG(z, gan_scale))
        augmented_logits = F.pad(logits, pad=(0,1))

        cf_loss = F.nll_loss(F.log_softmax(augmented_logits, dim=1), target_label)

        distance_loss = torch.sum(
                (
                    z.mean(dim=-1).mean(dim=-1)
                    -
                    z_0.mean(dim=-1).mean(dim=-1)
                ) ** 2
            ) * distance_weight

        total_loss = cf_loss + distance_loss

        scores = F.softmax(augmented_logits, dim=1)

        log.collect('Counterfactual loss', cf_loss)
        log.collect('Distance Loss', distance_loss)
        log.collect('Classification as {}'.format(target_class), scores[0][target_class])
        log.print_every(n_sec=1)

        dc_dz = autograd.grad(total_loss, z, total_loss)[0]
        z = z - dc_dz * speed
        z = clamp_to_unit_sphere(z, gan_scale)

        # TODO: Workaround for Pytorch memory leak
        # Convert back to numpy and destroy the computational graph
        # See https://github.com/pytorch/pytorch/issues/4661
        z_value = to_np(z)
        del z
    print(log)
    z = to_torch(z_value)

    images = netG(z, gan_scale)
    return images.data.cpu().numpy()


# Trajectories are written to result_dir/trajectories/
def make_video_filename(result_dir, dataloader, start_class, target_class, dataset, label_type='active'):
    """Generate a filename for storing a grid of generated images.

    Args:
        result_dir (str): Directory where the result files are to be stored.
        dataloader (object): DataLoader object containing dataset.
        start_class (int): The starting class.
        target_class (int): The target class.
        dataset (str): The name of the dataset being used ('emnist' or 'imagenet').
        label_type (str, optional): Type of label being used. Defaults to 'active'.

    Returns:
        str: The generated filename for the video.
    """    
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    start_class_name = dataloader.lab_conv.labels[start_class]
    target_class_name = dataloader.lab_conv.labels[target_class]
    video_filename = '{}-{}-{}-{}.mjpeg'.format(label_type, trajectory_id, start_class_name, target_class_name)
    
    subdirectory = ""
    if dataset == "emnist":
        subdirectory = "trajectories/emnist/counterfactual"
    elif dataset == "imagenet":
        subdirectory = "trajectories/imagenet/counterfactual"
    
    video_filename = os.path.join(subdirectory, video_filename)
    video_filename = os.path.join(result_dir, video_filename)
    path = os.path.join(result_dir, subdirectory)
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)
    return video_filename
