import argparse
import collections 
import pathlib
import openset_imagenet
import os
import torch
import numpy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

"""
  This code plots the open-set classifiers trained on the ImageNet dataset. The metrices plotted are
        - OSCR
        - Confidence
        - SoftMax Scores
  
  The code can either plot a single experiment, or all expeirments into a single pdf (a pdf for each metrics).

"""
def get_args():
    """ Arguments handler.

    Returns:
        parser: arguments structure
    """
    parser = argparse.ArgumentParser("Imagenet Plotting", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--protocols",
        type=int,
        choices = (1,2,3),
        nargs="+",
        default = (1,2,3),
        help="Select the protocols that should be evaluated"
    )
    parser.add_argument(
        "--loss-functions", "-l",
        nargs = "+",
        choices = ('softmax', 'garbage', 'entropic'),
        default = ('softmax', 'garbage', 'entropic'),
        help = "Select the loss functions that should be evaluated"
    )
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
        "--output-directory", "-o",
        type=pathlib.Path,
        default="experiments",
        help="Directory where the models are saved"
    )
    parser.add_argument(
        "--imagenet-directory",
        type=pathlib.Path,
        default=pathlib.Path("/local/scratch/datasets/ImageNet/ILSVRC2012/"),
        help="Imagenet root directory"
    )
    parser.add_argument(
        "--protocol-directory",
        type=pathlib.Path,
        default = "protocols",
        help = "Where are the protocol files stored"
    )
    parser.add_argument(
        "--gpu", "-g",
        type = int,
        nargs="?",
        default = None,
        const = 0,
        help = "Select the GPU index that you have. You can specify an index or not. If not, 0 is assumed. If not selected, we will train on CPU only (not recommended)"
    )
    
    parser.add_argument("--all", action='store_true', dest="all", help="plott all")  
    parser.add_argument(
      "--plots",
      help = "Select where to write the plots into"
    )
    parser.add_argument(
      "--table",
      help = "Select the file where to write the Confidences (gamma) and CCR into"
    )

    parser.add_argument(
        "--include_counterfactuals", "-inc_c",
        type=bool, default=False, 
        dest="include_counterfactuals",
        help="Include counterfactual images in the dataset")
    
    parser.add_argument("--include_arpl", "-inc_a",
                        type=bool, default=False,
                        dest="include_arpl", 
                        help="Include ARPL samples in the dataset")
    
    parser.add_argument("--mixed_unknowns", "-mu",
                        type=bool, default=False,
                        dest="mixed_unknowns",
                        help="Mix unknown samples in the dataset")
    
    parser.add_argument("--include_unknown", "-iu", action='store_false', dest="include_unknown", help="Exclude unknowns")

    args = parser.parse_args()

    suffix = 'linear' if args.linear else 'best' if args.use_best else 'last'
    if args.sort_by_loss:
      suffix += "_by_loss"
    args.plots = args.plots or f"Results_{suffix}.pdf"
    args.table = args.table or f"Results_{suffix}.tex"
    return args

def get_experiment_suffix(args):
  '''Provides a string of one or many suffix/-es that can be used when writing files or when loading files

    Return:
    list with file suffix in a list. Suffix relates to approach that was used for composing the negative samples.
  
  '''
  if args.all:
    return ["_no_negatives","_vanilla", "_counterfactuals",
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
        suffix += "_vanilla"

    return [suffix]

def suffix_to_label(suffix):
  """Does string operations to replace characters present in experiment specific filename
    (suffix) but unfit for plot labels. 
    

  Args:
      suffix (string): suffix of filename e.g. "_arpl_counterfactuals_"

  Returns:
      str: changed filename e.g. "arpl & counterfactuals"
  """  
  # little string operation on the subplot title
  if "no_negatives" in suffix:
    return "no negatives"
  elif "vanilla" in suffix:
    return "original negatives"
  else:
    title = suffix[1:].replace("_", " & ").replace("mixed", "originals")
  if title.endswith(" & best"):
      title = title[:-7]
  return title
  
def load_scores(args):
    """Load scores and epochs from model checkpoints, checks for files of given ImageNet protocol and loads its sccores.

    Args:
      args (dict): The parser arguments
    
    Returns:
      tuple: A tuple containing:
              - scores (dict): A dictionary with extracted scores from validation and test files.
              - epoch (dict): A dictionary with epoch and best score information from the model checkpoints.
    """ 
    
    suffixes = get_experiment_suffix(args=args)
    
    scores = {p:{} for p in args.protocols}
    epoch = {p:{} for p in args.protocols}
    
    for protocol in args.protocols:
      for suffix in suffixes:
        experiment_dir = (args.output_directory / f"Protocol_{protocol}" / suffix).resolve()
        suffix = suffix + "_best" if args.use_best else "_curr"
        score_files = {v : experiment_dir / f"{suffix}_{v}_arr.npz" for v in ("val", "test")}

        
        # remember files
        scores[protocol][suffix] = openset_imagenet.util.read_array_list(score_files)

        epoch = []
        """
        else:
          print ("Checkpoint file", checkpoint_file, "not found, skipping protocol", protocol, loss)
          scores[protocol][loss] = None
          epoch[protocol][loss] = (0, 0)
        """
    return scores, epoch

#####################################################################################

def plot_OSCR(args, scores, suffix):
    """Plots ths OSCR curve of a single model into a pdf document.
  
    Args:
    args (dict): The parser arguments
    scores (dict): The scores of the experiment.
        -scores has form {protocol:{"val" or "test": scores}}
    suffix(str): suffix of experiment
    """  

    protocol = args.protocols[0]
    suffix = suffix
  
    # default entropic openset loss, can be implemented to different losses in the future
    # plot OSCR
    # Only create one subplot directly
    fig, ax = plt.subplots(figsize=(5, 6))
    font = 15
    scale = 'linear' if args.linear else 'semilog'

    ############ FINAL PREPARATIONS ##############
          
    red = plt.colormaps.get_cmap('tab10').colors[3]
    blue = plt.colormaps.get_cmap('tab10').colors[0]
    
    args.labels = ["Val", "Test"] 
    
    val = [scores[protocol][suffix]["val"]]
    openset_imagenet.util.plot_oscr(arrays=val, methods=args.loss_functions, scale=scale, title=f'$P{protocol}{suffix}$ Val',
                  ax_label_font=font, ax=ax, unk_label=-1, color=blue)
    test = [scores[protocol][suffix]["test"]]
    openset_imagenet.util.plot_oscr(arrays=test, methods=args.loss_functions, scale=scale, title=f'$P{protocol}{suffix}$ Test',
                  ax_label_font=font, ax=ax, unk_label=-1,color=red)
    
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
  """Plots all OSCR curve of all given models into different subplots but into the same pdf document.
  
  Args:
    args (dict): The parser arguments
    scores (dict): The scores of the experiment.
        -scores has form {suffix:{loss:{"val" or "test": scores}}}
    pdf: the pdf document where the plots are saved to
  """  
  protocol = args.protocols[0]
  
  # create figure with 8 subplots
  fig, axs = plt.subplots(2, 4, figsize=(18, 6))
  font = 13
  scale = 'linear' if args.linear else 'semilog'
  axs = axs.flatten()
  
  #choose scores from given protocol
  scores = scores[protocol]
  for index, (suffix, score) in enumerate(scores.items()):
  
    print(suffix)
    print("===================================================")

    loss = "EOS"
    # if suffix == _vanilla:
    #   loss = "SoftMax"
    
    ax = axs[index]  # Get the specific subplot for this score
    val = [score["val"]]
    test = [score["test"]]
    
    red = plt.colormaps.get_cmap('tab10').colors[3]
    blue = plt.colormaps.get_cmap('tab10').colors[0]

    title = suffix_to_label(suffix)
    # Plot test scores
    '''
    openset_imagenet.util.plot_oscr(arrays=val, methods=[loss], color=blue, scale=scale, title=title,
            ax_label_font=font, ax=ax, unk_label=-1,)
    '''
    
    openset_imagenet.util.plot_oscr(arrays=test, methods=[loss], color=red, scale=scale, title=title,
            ax_label_font=font, ax=ax, unk_label=-1,)

    # Set grid and limits
    ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
    ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')
    ax.set_xlim(left=10**-2, right=10**0)
    ax.set_ylim(0, 1.05)
    

  # Set the legend with labels
  
  ax.legend(["Test"], frameon=False, fontsize=16, bbox_to_anchor=(-0.4, -0.25), ncol=3, loc='upper center', handletextpad=0.5, 
            columnspacing=1, markerscale=3)
  '''
  ax.legend(["Val", "Test"], frameon=False, fontsize=16, bbox_to_anchor=(-0.4, -0.25), ncol=3, loc='upper center', handletextpad=0.5, 
            columnspacing=1, markerscale=3)
  '''

  # Adjust layout to prevent overlap
  # fig.tight_layout(pad=2.0)
  plt.subplots_adjust(left=0.1, bottom=0.1, hspace=0.3)
  
  fig.text(0.45, 0.03, 'FPR', ha='center', fontsize=font+1)
  fig.text(0.05, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font+1)

  pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

  plt.close(fig)  

def plot_OSC_comparison(args, scores, pdf):
  """Plots the OSCR curve on the test set of all given models into the same plot and saves in a pdf document.
  
   Args:
    args (dict): The parser arguments
    scores (dict): The scores of the experiment.
        -scores has form {suffix:{loss:{"val" or "test": scores}}}
     pdf: the pdf document where the plots are saved to
  """  
  protocol = args.protocols[0]
  
  # create figure with 8 subplots
  fig, ax = plt.subplots(figsize=(8, 7))
  font = 13
  scale = 'linear' if args.linear else 'semilog'
  colormap = plt.get_cmap('tab10')
  colors = colormap.colors

  labels = []
  
  #choose scores from given protocol
  scores = scores[protocol]
  for index, (suffix, score) in enumerate(scores.items()):
    loss = "EOS"

    labels.append(suffix_to_label(suffix))
    test = [score["test"]]
    color = colors[index % len(colors)]  # Cycle through colors if more than the colormap length
    
    openset_imagenet.util.plot_oscr(arrays=test, methods=[loss], color=color, scale=scale, title="Test score comparison",
            ax_label_font=font, ax=ax, unk_label=-1,)
    
    # Set the legend with labels
  ax.legend(labels, frameon=False, fontsize=font - 1, bbox_to_anchor=(0.5, -0.1), ncol=3, loc='upper center', handletextpad=0.5, 
            columnspacing=1, markerscale=3)
  
  ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
  ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')
  
  ax.set_xlim(left=10**-2, right=10**0)
  ax.set_ylim(0, 0.8)
  
  fig.text(0.45, 0.08, 'FPR', ha='center', fontsize=font)
  fig.text(0.085, 0.5, 'CCR', va='center', rotation='vertical', fontsize=font)

  # Adjust layout to prevent overlap
  fig.tight_layout(pad=2.0)
  plt.subplots_adjust(left=0.15, bottom=0.15)

  # Save the figure to the PDF
  pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

  plt.close(fig)  # Close the figure after saving to the PDF
  
#####################################################################################

def plot_confidences(args):

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

def plot_many_confidences(args, pdf):
    """Plots the confidence of each model into a own subplot and saves them into the same pdf document
    
    Args:
      args (dict): The parser arguments
      pdf: the pdf document where the plots are saved to
  
    """
  
    suffixes = get_experiment_suffix(args=args)
  
    # we dont plot no negatives since no data on validation
    # suffixes = [s for s in suffixes if s != "_no_negatives"]
    protocol = args.protocols[0]
  
    event_files = {}
  
    for suffix in suffixes:
        event_dir = (args.output_directory / f"Protocol_{protocol}" / suffix).resolve()
        # Locate event paths
        files = sorted(os.listdir(event_dir))
        for f in files:
            if f.startswith("events") and not f.endswith("Identifier"):
                file_path = os.path.abspath(os.path.join(event_dir, f))
                event_files[suffix] = file_path
                              
    linewidth = 1.5
    font_size = 15
    color_palette = cm.get_cmap('tab10', 10).colors

    # Adjust the number of subplots to match the number of suffixes
    num_plots = len(suffixes)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    def load_accumulators(files, suffix):
        known_data, unknown_data = {}, {}
        event_file = files[suffix]
        try:
            event_acc = EventAccumulator(str(event_file), size_guidance={'scalars': 0})
            event_acc.Reload()
          
            tags = event_acc.Tags()
          
            for event in event_acc.Scalars('val/conf_kn'):
                known_data[event.step + 1] = event.value
            for event in event_acc.Scalars('val/conf_unk'):
                unknown_data[event.step + 1] = event.value
        except KeyError:
            print("KEY ERROR IN EVENT ACCUMULATOR")
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

        title = suffix_to_label(suffix)
        
        ax.set_title(title, fontsize=font_size)
        ax.grid(axis='x', linestyle=':', linewidth=1, color='gainsboro')
        ax.grid(axis='y', linestyle=':', linewidth=1, color='gainsboro')

        ax.set_xlim(min(step_kn + step_unk), max(step_kn + step_unk))
        ax.set_ylim(0.0, 1)

        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='in')
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, labelsize=font_size)

    # Remove the last subplot if there are fewer suffixes than subplots
    for index in range(num_plots, len(axs)):
        fig.delaxes(axs[index])

    # Adjust layout to prevent overlap
    fig.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.1, bottom=0.1)
    
    fig.text(0.5, 0.04, 'Epoch', ha='center', fontsize=font_size + 1)
    fig.text(0.04, 0.5, 'Confidence', va='center', rotation='vertical', fontsize=font_size + 1)

    ax.legend(["Confidence on Knowns", "Confidence on Negatives"], frameon=False, fontsize=15, bbox_to_anchor=(-0.2, -0.15), ncol=3, loc='upper center', handletextpad=0.5, 
              columnspacing=1, markerscale=3)

    # Save the final figure to the PDF
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

#####################################################################################

def plot_softmax(args, scores):
    """
    Plots the SoftMax score of a single model

    Args:
      args (dict): The parser arguments
      suffix (string): The suffix of the model name
      score (dict): The scores of the experiment.
        -scores has form {loss:{"val" or "test": scores}}
    """
    font_size = 15
    bins = 30
    unk_label = -2
    P = len(args.protocols)
    N = len(args.loss_functions)

    fig = plt.figure(figsize=(3*P+1, 2*N))
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

def plot_many_softmax(args, scores, pdf):
  """
    Plots the SoftMax score of all models in different subplots and saves them into the same pdf document


    Args:
      args (dict): The parser arguments
      pdf: pdf document where to save the subplots
      score (dict): The scores of the experiment.
        -scores has form {loss:{"val" or "test": scores}}
  """ 
  # create figure with 8 subplots
  fig, axs = plt.subplots(2, 4, figsize=(18, 6))
  font = 15
  axs = axs.flatten()

  bins = 30
  unk_label = -1
  protocol = args.protocols[0]
  
  # Manual colors
  edge_unk = colors.to_rgba('indianred', 1)
  fill_unk = colors.to_rgba('firebrick', 0.04)
  edge_kn = colors.to_rgba('tab:blue', 1)
  fill_kn = colors.to_rgba('tab:blue', 0.04)

  #choose scores from given protocol
  scores = scores[protocol]

  for index, (suffix, score) in enumerate(scores.items()):
    ax = axs[index]
    
    # Calculate histogram
    kn_hist, kn_edges, unk_hist, unk_edges = openset_imagenet.util.get_histogram(
        score["test"],
        unk_label=unk_label,
        metric='score',
        bins=bins,
        drop_bg=False
          )

    # Plot histograms
    ax.stairs(kn_hist, kn_edges, fill=True, color=fill_kn, edgecolor=edge_kn, linewidth=1)
    ax.stairs(unk_hist, unk_edges, fill=True, color=fill_unk, edgecolor=edge_unk, linewidth=1)
    ax.set_ylim(0, 750)
    
    title = suffix_to_label(suffix)
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
#####################################################################################

def conf_and_ccr_table(args, scores, epochs):
  """
    Calculates the confidence scores and CCR values for different protocols and approaches,
    writing the results into a LaTeX table format.

    Args:
        args (dict): The parser arguments
        scores (dict): Scores from validation and test files.
        epochs (dict): Eepoch and best score information from the loaded model checkpoints.
  """
  def find_nearest(array, value):
      """Get the closest element in array to value"""
      array = numpy.asarray(array)
      idx = (numpy.abs(array - value)).argmin()
      return idx, array[idx]

  query = [1e-3, 1e-2, 0.1,1.0]
  unk_label = -2

  with open(args.table, "w") as table:
    for p, protocol in enumerate(args.protocols):
      for l, loss in enumerate(args.loss_functions):
        for which in ["test"]:
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
              offset = offset, unknown_class=-2, last_valid_class=last_valid_class
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


def main():
  args = get_args()

  # HARDCODED AS WE USE ONE PROTOCOL ONLY. 
  protocol = 2
  
  print("Extracting and loading scores")
  scores, epoch = load_scores(args)

  pdf = PdfPages(args.plots)
  
  for suffix in scores[protocol].keys():
  
    try:
      '''
      # plot OSCR (actually not required for best case)
      print("Plotting OSCR curves")
      plot_OSCR(args, scores, suffix)
      pdf.savefig(bbox_inches='tight', pad_inches = 0)

     
      if not args.linear and not args.use_best and not args.sort_by_loss:
        # plot confidences
        print("Plotting confidence plots")
        plot_confidences(args)
        pdf.savefig(bbox_inches='tight', pad_inches = 0)

      if not args.linear and not args.sort_by_loss:
        # plot histograms
        print("Plotting softmax histograms")
        plot_softmax(args, scores)
        pdf.savefig(bbox_inches='tight', pad_inches = 0)
      '''
      
    finally:
      pdf.close()
    
  if args.all:
       
    '''
    print("Writing combined OSC plots")
    pdf = PdfPages("ImageNet_All_OSC_plots.pdf")
    plot_many_OSCR(args, scores, pdf)
    pdf.close()
    '''
    
    print("Writing OSC comparison")
    pdf = PdfPages("ImageNet_OSC_comparison.pdf")
    plot_OSC_comparison(args, scores, pdf)
    pdf.close()
       
    
    print("Writing combined confidences")
    pdf = PdfPages("ImageNet_Combined_Confidences.pdf")
    plot_many_confidences(args, pdf)
    pdf.close()
    
  
    print("Writing combined SoftMax scores")
    pdf = PdfPages("ImageNet_Combined_Softmax_Scores.pdf")
    plot_many_softmax(args,scores, pdf)
    pdf.close()
    '''
  
  # create result table
  if not args.linear and not args.sort_by_loss:
    print("Creating Table")
    print("Writing file", args.table)
    conf_and_ccr_table(args, scores, epoch)
  '''
if __name__ == "__main__":
  main()