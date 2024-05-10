
from pathlib import Path
import argparse
import multiprocessing
import collections
import subprocess
import pathlib
import openset_imagenet
import os
import torch
import numpy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from matplotlib import pyplot, cm, colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, LogLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the main training script for all MNIST experiments. \
                    Where applicable roman letters are used as negatives. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approach", "-a", required=True, choices=['SoftMax', 'Garbage', 'EOS', 'Objectosphere'])
    parser.add_argument("--task", default='train', choices=['train', 'eval', "plot", "show"])
    parser.add_argument("--arch", default='LeNet_plus_plus', choices=['LeNet', 'LeNet_plus_plus'])
    parser.add_argument('--second_loss_weight', "-w", help='Loss weight for Objectosphere loss', type=float, default=0.0001)
    parser.add_argument('--Minimum_Knowns_Magnitude', "-m", help='Minimum Possible Magnitude for the Knowns', type=float,
                        default=50.)
    parser.add_argument("--solver", dest="solver", default='sgd',choices=['sgd','adam'])
    parser.add_argument("--lr", "-l", dest="lr", default=0.01, type=float)
    parser.add_argument('--batch_size', "-b", help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
    parser.add_argument("--no_of_epochs", "-e", dest="no_of_epochs", type=int, default=70)
    parser.add_argument("--eval_directory", "-ed", dest= "eval_directory", default ="evaluation", help="Select the directory where evaluation details are.")
    parser.add_argument("--dataset_root", "-d", dest= "dataset_root", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--gpu", "-g", type=int, nargs="?",dest="gpu", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")
    parser.add_argument("--include_counterfactuals", "-inc_c", type=bool, default=False, dest="include_counterfactuals", help="Include counterfactual images in the dataset")
    parser.add_argument("--include_arpl", "-inc_a", type=bool, default=False, dest="include_arpl", help="Include ARPL samples in the dataset")
    parser.add_argument("--mixed_unknowns", "-mu", type=bool, default=False, dest="mixed_unknowns", help="Mix unknown samples in the dataset")
    parser.add_argument("--download", "-dwn", type=bool, default=False, dest="download", help="donwload emnist dataset")

    return parser.parse_args()



#returns scores, which is a dictionary of arrays containing various data such as logits, scores, target labels, and feature norms,
#returns model 
def get_experiment_suffix(args):
    suffix = ""
    letters = True
    if args.include_arpl:
        suffix += "_counterfactuals"
        letters = False
    if args.include_arpl:
        suffix += "_arpl"
        letters = False
    if args.mixed_unknowns:
        suffix += "_mixed"
        letters = False
    if letters:
        suffix += "_letters"
    return suffix
def load_scores(args):
  
  

  
  args.protocols = ["1"]
  loss = "EOS"
  scores = {p:{} for p in args.protocols}
  epoch = {p:{} for p in args.protocols}
# collect all result files and evalaute, WE DONT NEEED THIS
  
  suffix = get_experiment_suffix(args=args)
  directory = "evaluation"
  results_dir = pathlib.Path("LeNet_plus_plus") # Can later be implemented dynamically
  model_file = f"{results_dir}/{suffix}.pth"
  
  eval_val_file = directory / f"validation_{get_experiment_suffix(args=args)}.npz"
  eval_test_file = directory / f"test_{get_experiment_suffix(args=args)}.npz"
  
  
  
  score_files = {"val":eval_val_file, "test": eval_test_file}

  print("Extracting scores of", model_file)           
          
  # remember files
  scores["1"][loss] = openset_imagenet.util.read_array_list(score_files) 
  checkpoint = torch.load(model_file, map_location="cpu")
  
  epoch["1"][loss] = (checkpoint["epoch"],checkpoint["best_score"])
  
  print(epoch)

  
  return scores, epoch

def plot_OSCR(args, scores):
  args.loss_functions = ["EOS"]
  args.protocols = ["1"]

    # plot OSCR
  # Only create one subplot directly
  fig, ax = pyplot.subplots(figsize=(5, 6))
  font = 15
  scale = 'linear' if args.linear else 'semilog'
  
  index = 0
  p = '1'

  val = [scores[p][l]["val"] if scores[p][l] is not None else None for l in args.loss_functions]
  test = [scores[p][l]["test"] if scores[p][l] is not None else None for l in args.loss_functions]
  
  '''openset_imagenet.util.plot_oscr(arrays=val, methods=args.loss_functions, scale=scale, title=f'EMNIST val Negative',
                ax_label_font=font, ax=ax, unk_label=-1,)'''
  
  openset_imagenet.util.plot_oscr(arrays=test, methods=args.loss_functions, scale=scale, title=f'EMNIST test Negative',
                ax_label_font=font, ax=ax, unk_label=-1,)

  ax.legend(args.labels, frameon=False, fontsize=font - 1, bbox_to_anchor=(0.8, -0.12), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
  ax.label_outer()
  ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
  ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')
  fig.text(0.5, 0.03, 'FPR', ha='center', fontsize=font)
  fig.text(0.08, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)
  
  # Figure labels
  fig.text(0.5, 0.03, 'FPR', ha='center', fontsize=font)
  fig.text(0.08, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)
  

def plot_confidences(args):
  args.loss_functions = ["EOS"]
  args.protocols = ["1"]
  
  # locate event paths
  event_files = {p:collections.defaultdict(list) for p in args.protocols}
  for protocol in args.protocols:
    protocol_dir = args.output_directory/f"Protocol_{protocol}"
    if os.path.exists(protocol_dir):
      files = sorted(os.listdir(protocol_dir))
      # get the event files
      for f in files:
        if f.startswith("event"):
          loss = f.split("-")[1].split(".")[0]
          # set (overwrite) event file
          event_files[protocol][loss].append(protocol_dir / f)

  P = len(args.protocols)
  linewidth = 1.5
  legend_pos = "lower right"
  font_size = 15
  color_palette = cm.get_cmap('tab10', 10).colors
  fig = pyplot.figure(figsize=(12,3*P+1))
  gs = fig.add_gridspec(P, 2, hspace=0.2, wspace=0.1)
  axs = gs.subplots(sharex=True, sharey=True)
  axs = axs.flat


  def load_accumulators(files):
    known_data, unknown_data = {}, {}
    for event_file in files:
      try:
        event_acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
        event_acc.Reload()
        for event in event_acc.Scalars('val/conf_kn'):
          known_data[event[1]+1] = event[2]
        for event in event_acc.Scalars('val/conf_unk'):
          unknown_data[event[1]+1] = event[2]
      except KeyError: pass

    # re-structure
    return zip(*sorted(known_data.items())), zip(*sorted(unknown_data.items()))

  max_len = 0
  min_len = 100
  for index, protocol in enumerate(args.protocols):
      ax_kn = axs[2 * index]
      ax_unk = axs[2 * index + 1]
      for c, loss in enumerate(args.loss_functions):
        step_kn, val_kn, step_unk, val_unk = [[]]*4
        if loss in event_files[protocol]:
          # Read data from the tensorboard object
          (step_kn, val_kn), (step_unk, val_unk) = load_accumulators(event_files[protocol][loss])
        else:
          step_kn, val_kn, step_unk, val_unk = [[]]*4

        # Plot known confidence
        ax_kn.plot(step_kn, val_kn, linewidth=linewidth, label = loss + ' kn', color=color_palette[c])
        # Plot unknown confidence
        ax_unk.plot(step_unk, val_unk, linewidth=linewidth, label = loss + ' unk', color=color_palette[c])
        if len(step_kn):
          max_len = max(max_len, max(step_kn))
          min_len = min(min_len, min(step_kn))
      # set titles
      ax_kn.set_title(f"$P_{protocol}$ Known", fontsize=font_size)
      ax_unk.set_title(f"$P_{protocol}$ Negative", fontsize=font_size)

  # Manual legend
  axs[-2].legend(args.labels, frameon=False,
                fontsize=font_size - 1, bbox_to_anchor=(0.8, -0.1), ncol=3, handletextpad=0.5, columnspacing=1)

  for ax in axs:
      # set the tick parameters for the current axis handler
      ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
      ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
      ax.set_xlim(min_len, max_len)
      ax.set_ylim(0, 1)
      # Thicklocator parameters
      ax.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
      ax.xaxis.set_major_locator(MaxNLocator(6))
      ax.label_outer()
  # X label
  fig.text(0.5, 0.05, 'Epoch', ha='center', fontsize=font_size)

def plot_softmax(args, scores):
  args.loss_functions = ["EOS"]
  args.protocols = ["1"]
  font_size = 15
  bins = 30
  unk_label = -1
  P = len(args.protocols)
  N = len(args.loss_functions)

  fig = pyplot.figure(figsize=(3*P+1, 2*N))
  gs = fig.add_gridspec(N, P, hspace=0.2, wspace=0.08)
  axs = gs.subplots(sharex=True, sharey=False)
  axs = axs.flat
  # Manual colors
  edge_unk = colors.to_rgba('indianred', 1)
  fill_unk = colors.to_rgba('firebrick', 0.04)
  edge_kn = colors.to_rgba('tab:blue', 1)
  fill_kn = colors.to_rgba('tab:blue', 0.04)

  index = 0
  for protocol in args.protocols:
    for l, loss in enumerate(args.loss_functions):
      # Calculate histogram
      drop_bg = loss == "garbage"  #  Drop the background class
      if scores[protocol][loss] is not None:
        kn_hist, kn_edges, unk_hist, unk_edges = openset_imagenet.util.get_histogram(
            scores[protocol][loss]["test"],
            unk_label=unk_label,
            metric='score',
            bins=bins,
            drop_bg=drop_bg
        )
      else:
        kn_hist, kn_edges, unk_hist, unk_edges = [], [0], [], [0]
      # Plot histograms
      axs[index].stairs(kn_hist, kn_edges, fill=True, color=fill_kn, edgecolor=edge_kn, linewidth=1)
      axs[index].stairs(unk_hist, unk_edges, fill=True, color=fill_unk, edgecolor=edge_unk, linewidth=1)

      # axs[ix].set_yscale('log')
      axs[index].set_title(f"$P_{{{protocol}}}$ {args.labels[l]}")
      index += 1

  # Share y axis of the histograms of the same protocol
  for p in range(P):
    for l in range(1,N):
      axs[N*p+l-1].sharey(axs[N*p+l])

  for ax in axs:
      # set the tick parameters for the current axis handler
      ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
      ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
      ax.yaxis.set_major_locator(MaxNLocator(6))
      ax.label_outer()

  # Manual legend
  axs[-2].legend(['Known', 'Unknown'],
                frameon=False,
                fontsize=font_size-1,
                bbox_to_anchor=(0.2, -0.08),
                ncol=2,
                handletextpad=0.3,
                columnspacing=1,
                markerscale=1)
  # X label
  fig.text(0.5, 0.02, 'Score', ha='center', fontsize=font_size)

def conf_and_ccr_table(args, scores, epochs):
  args.loss_functions = ["EOS"]
  args.protocols = ["1"]
  def find_nearest(array, value):
      """Get the closest element in array to value"""
      array = numpy.asarray(array)
      idx = (numpy.abs(array - value)).argmin()
      return idx, array[idx]

  query = [1e-3, 1e-2, 0.1,1.0]
  unk_label = -1

  with open(args.table, "w") as table:
    for p, protocol in enumerate(args.protocols):
      for l, loss in enumerate(args.loss_functions):
        for which in ["test"]:
          
          print(p, protocol)
          print(l, loss)
          print(which)
          
          array = scores[protocol][loss][which]
          gt = array['gt']
          values = array['scores']
          
          ccr_, fpr_ = openset_imagenet.util.calculate_oscr(gt, values, unk_label=unk_label)

          # get confidences on test set
          offset = 0 if loss == "garbage" else 1 / (numpy.max(gt)+1)
          last_valid_class = -1 if loss == "garbage" else None
          c = openset_imagenet.metrics.confidence(
              torch.tensor(values),
              torch.tensor(gt, dtype=torch.long),
              offset = offset, unknown_class=-1, last_valid_class=last_valid_class
          )


          # write loss and confidences
          table.write(f"$P_{protocol}$ - {args.labels[l]} & {epochs[protocol][loss][0]} & {c[0]:1.3f} & {c[2]:1.3f}")

          for q in query:
              idx, fpr = find_nearest(fpr_, q)
              error = round(100*abs(fpr - q) / q, 1)
              if error >= 10.0:  # If error greater than 10% then assume fpr value not in data
                  table.write(" & ---")
              else:
                  table.write(f" & {ccr_[idx]:1.3f}")
        table.write("\\\\\n")
      if p < len(args.protocols)-1:
        table.write("\\midrule\n")

if __name__ == "__main__":
  args = command_line_options()
  print("loaded args")


  print("Extracting and loading scores")
  scores, epoch = load_scores(args)  
    
  print("Writing file", args.plots)
  pdf = PdfPages(args.plots)
  
  try:
    # plot OSCR (actually not required for best case)
    print("Plotting OSCR curves")
    plot_OSCR(args, scores)
    pdf.savefig(bbox_inches='tight', pad_inches = 0)

    '''if not args.linear and not args.use_best and not args.sort_by_loss:
      # plot confidences
      print("Plotting confidence plots")
      plot_confidences(args)
      pdf.savefig(bbox_inches='tight', pad_inches = 0)

    if not args.linear and not args.sort_by_loss:
      # plot histograms
      print("Plotting softmax histograms")
      plot_softmax(args, scores)
      pdf.savefig(bbox_inches='tight', pad_inches = 0)'''

  finally:
    pdf.close()

  # create result table
  if not args.linear and not args.sort_by_loss:
    print("Creating Table")
    print("Writing file", args.table)
    conf_and_ccr_table(args, scores, epoch)
