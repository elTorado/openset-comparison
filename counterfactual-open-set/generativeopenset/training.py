import time
import os
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable

from vector import make_noise
from dataloader import FlexibleCustomDataloader
import imutil
import logutil
from vast.tools import set_device_gpu, set_device_cpu, device
from gradient_penalty import calc_gradient_penalty


log = logutil.TimeSeries('Training GAN')

'''
    This file provides the training loop for the GAN trainnig. There is also additional code for trainig a classifier, which is not needed for our case.
    The code was extended to load the process to GPUs and is suitable for EMNIST and ImageNet. The training loop fiters negative samples - which is important
    for the ImageNet protocols to not train the generator on negative samples. 

'''

def train_gan(networks, optimizers, dataloader, epoch=None, **options): 
    """
    GAN training for generator, encode and discriminator. 
    Logs various metrics and saves intermediate results for monitoring the training progress.

    Args:
        networks (dict): Contains the networks to be trained, Encoder, Generator, Discriminator
        optimizers (dict): A dictionary containing the optimizers for each network - SGD or Adam.
        dataloader (DataLoader): Dataloader object for the dataset.
        epoch (int): The current epoch, used for loging purposes.

    Returns:
        bool: returns true when training is finished.
    """    
    
    args = options
    # setup device
    if args['gpu'] is not None:
        set_device_gpu(index=args['gpu'] )
        print(" ============== GPU Selected! =============")
    else:
        print("No GPU device selected, training will be extremely slow")
        set_device_cpu()
    
    for net in networks.values():
        net.train()
    netE = networks['encoder']
    netD = networks['discriminator']
    netG = networks['generator']
    netC = networks['classifier_k']
    optimizerE = optimizers['encoder']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerC = optimizers['classifier_k']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    latent_size = options['latent_size']
    
    #setup vast tools devices
    netE = device(netE)
    netD = device(netD)
    netG = device(netG)
    netC = device(netC)

    for i, (images, class_labels) in enumerate(dataloader):
        
        # ONLY USE POSITIVE CLASSES - only applied when training imagenet protocols!
        if (class_labels < 0).any():
            
            log.collect("Current Epoch: ", epoch)
            
            images = Variable(images)
            labels = Variable(class_labels)
            
            images = device(images)
            labels = device(class_labels)
            

            if options["dataset_name"] == "imagenet":
                ac_scale = 1
                sample_scale = 1
                
            else:
                ac_scale = 4
                sample_scale = 4
                
            ############################
            # Discriminator Updates
            ###########################
            netD.zero_grad()

            # Classify sampled images as fake
            noise = make_noise(batch_size, latent_size, sample_scale)
            noise = device(noise)
            
            fake_images = netG(noise, sample_scale)
            
            logits = netD(fake_images)[:,0]
            loss_fake_sampled = F.softplus(logits).mean()
            log.collect('Discriminator Sampled', loss_fake_sampled)
            loss_fake_sampled.backward()

            # Source Code commented out by Counterfactual Images authors
            """
            # Classify autoencoded images as fake
            more_images, more_labels = dataloader.get_batch()
            more_images = Variable(more_images)
            fake_images = netG(netE(more_images, ac_scale), ac_scale)
            logits_fake = netD(fake_images)[:,0]
            #loss_fake_ac = F.softplus(logits_fake).mean() * options['discriminator_weight']
            loss_fake_ac = logits_fake.mean() * options['discriminator_weight']
            log.collect('Discriminator Autoencoded', loss_fake_ac)
            loss_fake_ac.backward()
            """

            # Classify real examples as real
            logits = netD(images)[:,0]
            loss_real = F.softplus(-logits).mean() * options['discriminator_weight']
            #loss_real = -logits.mean() * options['discriminator_weight']
            loss_real.backward()
            log.collect('Discriminator Real', loss_real)
            gp = calc_gradient_penalty(netD, images.data, fake_images.data, options)
            gp.backward()
            log.collect('Gradient Penalty', gp)

            optimizerD.step()

            ############################

            ############################
            # Generator Update
            ###########################
            netG.zero_grad()

            # Source Code commented out by Counterfactual Images authors
            """
            # Minimize fakeness of sampled images
            noise = make_noise(batch_size, latent_size, sample_scale)
            fake_images_sampled = netG(noise, sample_scale)
            logits = netD(fake_images_sampled)[:,0]
            errSampled = F.softplus(-logits).mean() * options['generator_weight']
            errSampled.backward()
            log.collect('Generator Sampled', errSampled)
            """

            # Minimize fakeness of autoencoded images
            fake_images = netG(netE(images, ac_scale), ac_scale)
            logits = netD(fake_images)[:,0]
            #errG = F.softplus(-logits).mean() * options['generator_weight']
            errG = -logits.mean() * options['generator_weight']
            errG.backward()
            log.collect('Generator Autoencoded', errG)

            optimizerG.step()

            ############################
            # Autoencoder Update
            ###########################
            netG.zero_grad()
            netE.zero_grad()

            # Minimize reconstruction loss
            reconstructed = netG(netE(images, ac_scale), ac_scale)
            err_reconstruction = torch.mean(torch.abs(images - reconstructed)) * options['reconstruction_weight']
            err_reconstruction.backward()
            log.collect('Pixel Reconstruction Loss', err_reconstruction)

            optimizerE.step()
            optimizerG.step()
            ###########################

            ############################
            # Classifier Update
            ############################
            
            netC.zero_grad()
                    
            # Classify real examples into the correct K classes with hinge loss
                
            classifier_logits = netC(images)
            classifier_logits = device(classifier_logits) 
                    
            errC = F.softplus(classifier_logits * -labels).mean()
            errC.backward()
            log.collect('Classifier Loss', errC)

            optimizerC.step()
            
            ############################

            # Keep track of accuracy on positive-labeled examples for monitoring
            # log.collect_prediction('Classifier Accuracy', netC(images), labels)
            # log.collect_prediction('Discriminator Accuracy, Real Data', netD(images), labels)

            log.print_every()

            if i % 100 == 1:
                fixed_noise = make_noise(batch_size, latent_size, sample_scale, fixed_seed=42)
                # demo(networks, images, fixed_noise, ac_scale, sample_scale, result_dir, epoch, i)
    return True


def demo(networks, images, fixed_noise, ac_scale, sample_scale, result_dir, epoch=0, idx=0):
    netE = networks['encoder']
    netG = networks['generator']

    def image_filename(*args):
        image_path = os.path.join(result_dir, 'images')
        name = '_'.join(str(s) for s in args)
        name += '_{}'.format(int(time.time() * 1000))
        return os.path.join(image_path, name) + '.jpg'

    demo_fakes = netG(fixed_noise, sample_scale)
    img = demo_fakes.data[:16]

    filename = image_filename('samples', 'scale', sample_scale)
    caption = "S scale={} epoch={} iter={}".format(sample_scale, epoch, idx)
    imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)

    aac_before = images[:8]
    aac_after = netG(netE(aac_before, ac_scale), ac_scale)
    img = torch.cat((aac_before, aac_after))

    filename = image_filename('reconstruction', 'scale', ac_scale)
    caption = "R scale={} epoch={} iter={}".format(ac_scale, epoch, idx)
    imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)


def train_classifier(networks, optimizers, dataloader, epoch=None, **options):
    for net in networks.values():
        net.train()
    netC = networks['classifier_kplusone']
    optimizerC = optimizers['classifier_kplusone']
    batch_size = options['batch_size']
    image_size = options['image_size']

    dataset_filename = options.get('aux_dataset')
    if not dataset_filename or not os.path.exists(dataset_filename):
        raise ValueError("Aux Dataset not available")
    print("Using aux_dataset {}".format(dataset_filename))
    aux_dataloader = FlexibleCustomDataloader(dataset_filename, batch_size=batch_size, image_size=image_size)

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        ############################
        # Classifier Update
        ############################
        netC.zero_grad()

        # Classify real examples into the correct K classes
        classifier_logits = netC(images)
        augmented_logits = F.pad(classifier_logits, (0,1))
        _, labels_idx = labels.max(dim=1)
        errC = F.nll_loss(F.log_softmax(augmented_logits, dim=1), labels_idx)
        errC.backward()
        log.collect('Classifier Loss', errC)

        # Classify aux_dataset examples as open set
        aux_images, aux_labels = aux_dataloader.get_batch()
        classifier_logits = netC(Variable(aux_images))
        augmented_logits = F.pad(classifier_logits, (0,1))
        log_soft_open = F.log_softmax(augmented_logits, dim=1)[:, -1]
        errOpenSet = -log_soft_open.mean()
        errOpenSet.backward()
        log.collect('Open Set Loss', errOpenSet)

        optimizerC.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        log.collect_prediction('Classifier Accuracy', netC(images), labels)

        log.print_every()

    return True
