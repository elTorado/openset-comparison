#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
parser.add_argument('--aux_dataset', help='Path to aux_dataset file [default: None]')
parser.add_argument('--dataset', type=str, required=True, help='dataset')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from training import train_gan
from networks import build_networks, save_networks, get_optimizers
from options import load_options, get_current_epoch
from counterfactual import generate_counterfactual
from comparison import evaluate_with_comparison

# Load params and check for existing files
options = load_options(options)
assert os.path.exists(options['dataset']), f"Dataset file {options['dataset']} does not exist."
assert os.path.exists(options['comparison_dataset']), f"Comparison dataset file {options['comparison_dataset']} does not exist."

dataloader = FlexibleCustomDataloader(fold='train', **options)
eval_dataloader = CustomDataloader(fold='test', **options)

print(" ---------------------- Training Labels: --------------------")
print(dataloader.lab_conv.labels)
print(" ---------------------- Eval Labels: --------------------")
print(eval_dataloader.lab_conv.labels)


networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, **options)

start_epoch = get_current_epoch(options['result_dir']) + 1
for epoch in range(start_epoch, start_epoch + options['epochs']):
    print("############ GAN TRAINING ITERATION : CURRENT EPOCH: " + str(epoch)+ "#######################")
    train_gan(networks, optimizers, dataloader, epoch=epoch, **options)
    #generate_counterfactual(networks, dataloader, **options)
    eval_results = evaluate_with_comparison(networks, eval_dataloader, **options)
    pprint(eval_results)
    save_networks(networks, epoch, options['result_dir'])
