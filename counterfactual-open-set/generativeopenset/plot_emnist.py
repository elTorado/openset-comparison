
from pathlib import Path
import argparse
import multiprocessing
import collections
import subprocess
import pathlib

import sys
import os

#Ugly way to deal with the fact that python stubornly refuses to import openset_imagenet. Adds it to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)
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

    parser.add_argument("--approach", "-a", required=False, choices=['SoftMax', 'Garbage', 'EOS', 'Objectosphere'], default=["EOS"])
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
    parser.add_argument("--include_unknown", "-iu", action='store_false', dest="include_unknown", help="Exclude unknowns")
    parser.add_argument("--all", action='store_true', dest="all", help="plott all")

    parser.add_argument("--download", "-dwn", type=bool, default=False, dest="download", help="donwload emnist dataset")
    parser.add_argument(
        "--labels",
        nargs="+",
        choices = ("S", "BG", "EOS"),
        default = ("S", "BG", "EOS"),
        help = "Select the labels for the plots"
    )
    parser.add_argument(
        "--use-best",
        action = "store_true",
        help = "If selected, the best model is selected from the validation set. Otherwise, the last model is used"
    )
    parser.add_argument(
        "--force", "-f",
        action = "store_true",
        help = "If set, score files will be recomputed even if they already exist"
    )
    parser.add_argument(
      "--linear",
      action="store_true",
      help = "If set, OSCR curves will be plot with linear FPR axis"
    )
    parser.add_argument(
      "--sort-by-loss", "-s",
      action = "store_true",
      help = "If selected, the plots will compare across protocols and not across algorithms"
    )
    parser.add_argument(
      "--plots",
      help = "Select where to write the plots into"
    )
    parser.add_argument(
      "--table",
      help = "Select the file where to write the Confidences (gamma) and CCR into"
    )

    args = parser.parse_args()

    suffixes = get_experiment_suffix( args = args)
    args.plots = [f"Results_{suffix}.pds" for suffix in suffixes]
    args.table = [f"Results_{suffix}.tex" for suffix in suffixes]


    return args



'''Creates a string suffix that can be used when writing files'''
def get_experiment_suffix(args):
  if args.all:
    return ["_no_negatives", "_letters", "_counterfactuals",
            "_counterfactuals_mixed", "_arpl", 
            "_arpl_mixed", "_counterfactuals_arpl", 
            "_counterfactuals_arpl_mixed"]
  
  else:
    suffix = ""
    letters = True
    if args.include_counterfactuals:
        suffix += "_counterfactuals"
        letters = False
    if args.include_arpl:
        suffix += "_arpl"
        letters = False
    if args.mixed_unknowns:
        suffix += "_mixed"
        letters = False
    if not args.include_unknown:
        suffix += "_no_negatives"
        letters = False
    if letters:
        suffix += "_letters"

    return [suffix]

  
def load_scores(args):
  
  suffixes = get_experiment_suffix(args=args)
  
  loss = "EOS"
  scores = {s:{} for s in suffixes}
  epoch = {s:{} for s in suffixes}
# collect all result files and evalaute, WE DONT NEEED THIS

  directory =  pathlib.Path("evaluation")
  results_dir = pathlib.Path("LeNet") 
  
  for suffix in suffixes:
    model_file = f"{results_dir}/{suffix}.pth"
    
    eval_val_file = directory / f"validation_{suffix}.npz"
    eval_test_file = directory / f"test_{suffix}.npz"
    
    score_files = {"val":eval_val_file, "test": eval_test_file}

    print("Extracting scores of", model_file)           
            
    # remember files
    scores[suffix][loss] = openset_imagenet.util.read_array_list(score_files) 
    checkpoint = torch.load(model_file, map_location="cpu")
    
    epoch[suffix][loss] = (checkpoint["epoch"],checkpoint["best_score"])
      
  return scores, epoch

def plot_OSCR(args, scores):
    
  # default entropic openset loss, can be implemented to different losses in the future
  loss = args.approach[0]
  suffix = scores[0]
  score = scores[1]
  # plot OSCR
  # Only create one subplot directly
  fig, ax = pyplot.subplots(figsize=(5, 6))
  font = 15
  scale = 'linear' if args.linear else 'semilog'

  ############ FINAL PREPARATIONS ##############
  val = [score[loss]["val"]]
  test = [score[loss]["test"]]
  
  print(val)
        
  red = pyplot.colormaps.get_cmap('tab10').colors[3]
  blue = pyplot.colormaps.get_cmap('tab10').colors[0]
  
  args.labels = ["Val", "Test"] 
  
  title = suffix.replace("_", " ")
  
  openset_imagenet.util.plot_oscr(arrays=val, methods=[args.approach], color=blue, scale=scale, title=title,
                ax_label_font=font, ax=ax, unk_label=-1,)
  
  openset_imagenet.util.plot_oscr(arrays=test, methods=[args.approach], color=red, scale=scale, title=title,
                ax_label_font=font, ax=ax, unk_label=-1,)

  ax.legend(args.labels, frameon=False, fontsize=font - 1, bbox_to_anchor=(0.8, -0.12), ncol=3, handletextpad=0.5, columnspacing=1, markerscale=3)
  ax.label_outer()
  ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
  ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')
  
  # Close grid at 10^0
  ax.set_xlim(left=10**-2, right=10**0)

  
  # Adding more white space around the figure
  fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
  
  fig.text(0.55, 0.06, 'FPR', ha='center', fontsize=font+1)
  fig.text(-0.01, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font+1)

def plot_many_OSCR(args, scores, pdf):
    # Default entropic openset loss, can be implemented to different losses in the future
    loss = args.approach[0]

    # create figure with 8 subplots
    fig, axs = plt.subplots(2, 4, figsize=(18, 6))
    font = 13
    scale = 'linear' if args.linear else 'semilog'
    axs = axs.flatten()
    
    for index, (suffix, score) in enumerate(scores.items()):
      
        ax = axs[index]  # Get the specific subplot for this score
                
        test = [score[loss]["test"]]
        val = [score[loss]["val"]]
                
        red = pyplot.colormaps.get_cmap('tab10').colors[3]
        blue = pyplot.colormaps.get_cmap('tab10').colors[0]

        title = suffix[1:].replace("_", " & ").replace("mixed", "letters")

        # Plot test scores
        
        '''
        openset_imagenet.util.plot_oscr(arrays=val, methods=[args.approach], color=blue, scale=scale, title=title,
                ax_label_font=font, ax=ax, unk_label=-1,)
        '''
        
        openset_imagenet.util.plot_oscr(arrays=test, methods=[args.approach], color=red, scale=scale, title=title,
                ax_label_font=font, ax=ax, unk_label=-1,)
        
        
        # Set grid and limits
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')
        ax.set_xlim(left=10**-2, right=10**0)
        ax.set_ylim(0, 1.05)
        
    # Set the legend with labels
   
    '''
    ax.legend(["Val", "Test"], frameon=False, fontsize=16, bbox_to_anchor=(-0.4, -0.25), ncol=3, loc='upper center', handletextpad=0.5, 
              columnspacing=1, markerscale=3)
    '''
    
    ax.legend(["Test"], frameon=False, fontsize=16, bbox_to_anchor=(-0.4, -0.25), ncol=3, loc='upper center', handletextpad=0.5, 
              columnspacing=1, markerscale=3)
    

    # Adjust layout to prevent overlap
    # fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.1, bottom=0.1, hspace=0.3)
    
    fig.text(0.45, 0.03, 'FPR', ha='center', fontsize=font+1)
    fig.text(0.05, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font+1)

    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

    plt.close(fig)  


def plot_OSCR_comparison(args, scores, pdf):
    # Default entropic openset loss, can be implemented to different losses in the future
    loss = args.approach[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    font = 15
    scale = 'linear' if args.linear else 'semilog'
    
    colormap = plt.get_cmap('tab10')
    colors = colormap.colors

    labels = []
    

    for index, (suffix, score) in enumerate(scores.items()):
        test = [score[loss]["test"]]
        
        color = colors[index % len(colors)]  # Cycle through colors if more than the colormap length

        # Add label for the current test score
        labels.append(suffix[1:].replace("_", " & ").replace("mixed", "letters"))

        # Plot test scores
        openset_imagenet.util.plot_oscr(arrays=test, methods=[args.approach], color=color, scale=scale, 
                                        title="Test score comparison", ax_label_font=font, ax=ax, unk_label=-1)

    # Set the legend with labels
    ax.legend(labels, frameon=False, fontsize=font - 1, bbox_to_anchor=(0.5, -0.1), ncol=3, loc='upper center', handletextpad=0.5, 
              columnspacing=1, markerscale=3)
    
    ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
    ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')
    
    ax.set_xlim(left=10**-2, right=10**0)
    
    fig.text(0.45, 0.08, 'FPR', ha='center', fontsize=font)
    fig.text(0.085, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)

    # Adjust layout to prevent overlap
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.15, bottom=0.15)

    # Save the figure to the PDF
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

    plt.close(fig)  # Close the figure after saving to the PDF

def plot_many_confidences(args, pdf):
    loss = "EOS"
    suffixes = get_experiment_suffix(args=args)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    event_dir = os.path.join(current_dir, '../LeNet/Logs/')
    
    # Locate event paths
    event_files = {}
    files = sorted(os.listdir(event_dir))
    
    for f in files:
        for suffix in suffixes:
            if f.startswith(suffix + "_event"):
                file_path = os.path.abspath(os.path.join(event_dir, f))
                event_files[suffix] = file_path

    linewidth = 1.5
    font_size = 15
    color_palette = cm.get_cmap('tab10', 10).colors
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    def load_accumulators(files, suffix):
        print(files)
        known_data, unknown_data = {}, {}
        event_file = files[suffix]
        try:
            event_acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
            event_acc.Reload()
            for event in event_acc.Scalars('val/conf_kn'):
                known_data[event.step + 1] = event.value
            for event in event_acc.Scalars('val/conf_unk'):
                unknown_data[event.step + 1] = event.value
        except KeyError:
            pass

        # Re-structure
        return zip(*sorted(known_data.items())), zip(*sorted(unknown_data.items()))

    for index, suffix in enumerate(suffixes):
        ax = axs[index]  # Get the specific subplot for this score
        print(suffix)
        (step_kn, val_kn), (step_unk, val_unk) = load_accumulators(event_files, suffix)

        # Plot known and unknown confidences
        ax.plot(step_kn, val_kn, linewidth=linewidth, label='Known', color=color_palette[1])
        ax.plot(step_unk, val_unk, linewidth=linewidth, label='Unknown', color=color_palette[0])

        ax.set_title(suffix[1:].replace("_", " & ").replace("mixed", "letters"), fontsize=font_size)

        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

        ax.set_xlim(min(step_kn + step_unk), max(step_kn + step_unk))
        ax.set_ylim(0.8, 1)

        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)

    

    # Adjust layout to prevent overlap
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.1, bottom=0.1)
    
    fig.text(0.5, 0.04, 'Epoch', ha='center', fontsize=font_size+1)
    fig.text(0.04, 0.5, 'Confidence', va='center', rotation='vertical', fontsize=font_size+1)
    
   
    ax.legend(["Confidence on Knowns", "Confidence on Negatives"], frameon=False, fontsize=15, bbox_to_anchor=(-0.2, -0.15), ncol=3, loc='upper center', handletextpad=0.5, 
              columnspacing=1, markerscale=3)

    # Save the final figure to the PDF
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

#these even t files are in the LeNet directory
# Can easily be adjustet to load all event file
def plot_confidences(args):
    loss = "EOS"
    suffixes = get_experiment_suffix(args=args)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    event_dir = os.path.join(current_dir, '../LeNet/Logs/')
    
    # Locate event paths
    event_files = {}
    files = sorted(os.listdir(event_dir))
    
    for suffix in suffixes:
    
        # Get the event files
        for f in files:
            if f.startswith(suffix + "_event"):
                file_path = os.path.abspath(os.path.join(event_dir, f))
                event_files[suffix] = file_path

        linewidth = 1.5
        font_size = 15
        color_palette = cm.get_cmap('tab10', 10).colors
        fig, ax = plt.subplots(figsize=(12, 6))

        def load_accumulators(files):
            known_data, unknown_data = {}, {}
            event_file = files[suffix]
            try:
                event_acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
                event_acc.Reload()
                for event in event_acc.Scalars('val/conf_kn'):
                    known_data[event.step + 1] = event.value
                for event in event_acc.Scalars('val/conf_unk'):
                    unknown_data[event.step + 1] = event.value
            except KeyError:
                pass

            # Re-structure
            return zip(*sorted(known_data.items())), zip(*sorted(unknown_data.items()))

        (step_kn, val_kn), (step_unk, val_unk) = load_accumulators(event_files)

        # Plot known and unknown confidences
        ax.plot(step_kn, val_kn, linewidth=linewidth, label='Known', color=color_palette[1])
        ax.plot(step_unk, val_unk, linewidth=linewidth, label='Unknown', color=color_palette[0])

        ax.set_title(suffix[1:].replace("_", " & ").replace("mixed", "letters"), fontsize=font_size)

        # Add grid
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

        # Set limits and labels
        ax.set_xlim(min(step_kn + step_unk), max(step_kn + step_unk))
        ax.set_ylim(0.8, 1)
        ax.set_xlabel('Epoch', fontsize=font_size)
        ax.set_ylabel('Confidence', fontsize=font_size)

        # Add legend
        ax.legend(loc='upper right', fontsize=font_size - 1)

        # Set tick parameters
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_softmax(args, suffix, score):
  
  loss = args.approach[0]


  fig, ax = pyplot.subplots(figsize=(5, 6))
  font = 15
  scale = 'linear' if args.linear else 'semilog'
  
  font_size = 15
  bins = 30
  unk_label = -1
  
  # Manual colors
  edge_unk = colors.to_rgba('indianred', 1)
  fill_unk = colors.to_rgba('firebrick', 0.04)
  edge_kn = colors.to_rgba('tab:blue', 1)
  fill_kn = colors.to_rgba('tab:blue', 0.04)

  drop_bg = loss == "garbage"
  
    
      # Calculate histogram
  kn_hist, kn_edges, unk_hist, unk_edges = openset_imagenet.util.get_histogram(
      score[loss]["test"],
      unk_label=unk_label,
      metric='score',
      bins=bins,
      drop_bg=drop_bg
        )

  # Plot histograms
  ax.stairs(kn_hist, kn_edges, fill=True, color=fill_kn, edgecolor=edge_kn, linewidth=1)
  ax.stairs(unk_hist, unk_edges, fill=True, color=fill_unk, edgecolor=edge_unk, linewidth=1)

  # axs[ix].set_yscale('log')
  ax.set_title(f"${{{suffix}}}$")

  # set the tick parameters for the current axis handler
  ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
  ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)
  ax.yaxis.set_major_locator(MaxNLocator(6))
  ax.label_outer()

  # Manual legend
  ax.legend(['Known', 'Unknown'],
                frameon=False,
                fontsize=font_size-1,
                bbox_to_anchor=(0.2, -0.08),
                ncol=2,
                handletextpad=0.3,
                columnspacing=1,
                markerscale=1)
  # X label
  fig.text(0.5, 0.02, 'Score', ha='center', fontsize=font_size)

def plot_many_softmax(args, scores, pdf):
  
  loss = args.approach[0]
  # create figure with 8 subplots
  fig, axs = plt.subplots(2, 4, figsize=(18, 6))
  font = 15
  axs = axs.flatten()

  bins = 30
  unk_label = -1

  
  # Manual colors
  edge_unk = colors.to_rgba('indianred', 1)
  fill_unk = colors.to_rgba('firebrick', 0.04)
  edge_kn = colors.to_rgba('tab:blue', 1)
  fill_kn = colors.to_rgba('tab:blue', 0.04)

  drop_bg = loss == "garbage"
  
  for index, (suffix, score) in enumerate(scores.items()):
    ax = axs[index]
    
      # Calculate histogram
    kn_hist, kn_edges, unk_hist, unk_edges = openset_imagenet.util.get_histogram(
        score[loss]["test"],
        unk_label=unk_label,
        metric='score',
        bins=bins,
        drop_bg=drop_bg
          )

    # Plot histograms
    ax.stairs(kn_hist, kn_edges, fill=True, color=fill_kn, edgecolor=edge_kn, linewidth=1)
    ax.stairs(unk_hist, unk_edges, fill=True, color=fill_unk, edgecolor=edge_unk, linewidth=1)

    title = suffix[1:].replace("_", " & ").replace("mixed", "letters")
    ax.set_title(title)

    # set the tick parameters for the current axis handler
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.label_outer()

  # Manual legend
  ax.legend(['Known', 'Unknown'],
                frameon=False,
                fontsize=font-1,
                 bbox_to_anchor=(-1, -0.25),
                ncol=2,
                handletextpad=0.3,
                columnspacing=1,
                markerscale=1)
  # X label
  fig.text(0.5, 0.02, 'Score', ha='center', fontsize=font)
  pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

def conf_and_ccr_table(args, scores, epochs):
  approach = args.approach
  args.protocols = ["1"]
  def find_nearest(array, value):
      """Get the closest element in array to value"""
      array = numpy.asarray(array)
      idx = (numpy.abs(array - value)).argmin()
      return idx, array[idx]

  query = [1e-3, 1e-2, 0.1,1.0]
  unk_label = -1

  for table in args.table:
    suffix = table[8:-4]
    with open(table, "w") as table:
      for p, protocol in enumerate(args.protocols):
        for l, loss in enumerate(args.approach):
          for which in ["test"]:
            array = scores[suffix][loss][which]
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
            table.write(f"${suffix}$ - {args.labels[l]} & {epochs[suffix][loss][0]} & {c[0]:1.3f} & {c[2]:1.3f}")

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

  suffix = get_experiment_suffix(args=args)

  print("Extracting and loading scores")
  scores, epoch = load_scores(args)  
  
  for suffix in scores.keys():
    
    print("Writing file", suffix)
    pdf = PdfPages(f"{suffix}.pdf")
    
    try:
      # plot OSCR (actually not required for best case)
      
       
      print("Plotting OSCR curves")
      plot_OSCR(args, scores= (suffix, scores[suffix]))
      pdf.savefig(bbox_inches='tight', pad_inches = 0)
      
      
      # plot confidences
      print("Plotting confidence plots")
      plot_confidences(args)
      pdf.savefig(bbox_inches='tight', pad_inches = 0)
      
      
      if not args.linear and not args.sort_by_loss:
        # plot histograms
        print("Plotting softmax histograms")
        plot_softmax(args, suffix, scores[suffix])
        pdf.savefig(bbox_inches='tight', pad_inches = 0)
      
    
    finally:
      pdf.close()
  
  if args.all:
    
    print("Writing Combined OSC Comparison")
    pdf = PdfPages("EMNIST_Combined_OSC_Comparison.pdf")
    plot_OSCR_comparison(args, scores, pdf)
    pdf.close()
  
 
    print("Writing combined OSC plots")
    pdf = PdfPages("EMNIST_All_OSC_plots.pdf")
    plot_many_OSCR(args, scores, pdf)
    pdf.close()
    
    print("Writing combined confidences")
    pdf = PdfPages("EMNIST_Combined_Confidences.pdf")
    plot_many_confidences(args, pdf)
    pdf.close()
    
     
    print("Writing combined SoftMax scores")
    pdf = PdfPages("EMNIST_Combined_Softmax_Scores.pdf")
    plot_many_softmax(args,scores, pdf)
    pdf.close()
   
    
    
  '''
  # create result table
  if not args.linear and not args.sort_by_loss:
    print("Writing file", "Conf & CCR scores")
    print("Creating Table")
    conf_and_ccr_table(args, scores, epoch)
  '''
  
 
  
