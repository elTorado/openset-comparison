# EMNIST and ImageNet Experiments with synthetic negatives

This guide provides instructions on how to create synthetic negatives samples from EMNIST and ImageNet datasets and train an open-set classifier using them. The process involves generating counterfactual and ARPL images, and subsequently training a classifier. The necessary source code repositories are provided.

The commands are listed with required parameters and an example value.

## Source Code

- Counterfactuals Generation & Classifier Training: [Counterfact-Openset](https://github.com/elTorado/openset-comparison/tree/main)
- ARPL Generation: [ARPL](https://github.com/elTorado/ARPL/tree/main)

## EMNIST Experiments

### Generating Counterfactual EMNIST Images

#### Setup

1. Create a Conda environment `conda env create -f environment.yaml`. Activate it `conda activate synthetic-openset`.

2. Execute `download_emnist.py` to download EMNIST data into the hardcoded `DATA_DIR`. This script also creates several `.dataset` files with image paths.
    - **DATA_DIR**: Directory where EMNIST dataset will be stored.

3. Run `python example_emnist_G.py --task "dataset"` to create different folds. The file `emnist_split1.dataset` is required for GAN training.
    - **task**: Specifies the task to be performed. Use `"dataset"` to create the dataset files.

#### GAN Training
1. Verify the correct dataset path in the `params.json` file.
2. Run `python generativeopenset/train_gan.py --dataset_name emnist --epochs 50 -g 4`.
    - **dataset_name**: Name of the dataset (use `emnist`).
    - **epochs**: Number of training epochs (default 50).
    - **-g**: GPU to use for training.

#### Image Generation
1. Run `python generativeopenset/generate_open_set.py --result_dir /home/user/username/openset-comparison/counterfactual-open-set --count 2484 --dataset emnist -g 3`.
    - **result_dir**: Directory where the generated images will be saved.
    - **count**: Number of batches to process (Number of images will be count * batch size).
    - **dataset**: Name of the dataset (use `emnist`).
    - **-g**: GPU to use for generation.

2. Split generated image grids into individual images using `python generativeopenset/auto_label.py --dataset emnist --output_filename /home/user/username/openset-comparison/counterfactual-open-set/emnist_counterfactual.dataset`.
    - **dataset**: Name of the dataset (use `emnist`).
    - **output_filename**: Name of the output file containing paths to each image and its label - must end on `.dataset`.

### ARPL EMNIST Images

#### Setup
1. The same conda environment can be used, as for the other code.
2. Set the correct path to EMNIST in the `main_worker()` function in `osr.py`.

#### GAN Training
1. Run `python osr.py --dataset emnist --loss ARPLoss --gpu 1 --max-epoch 50 --cs --result_dir /home/user/username/arpl/ARPL/generated_emnist`.
    - **dataset**: Name of the dataset (use `emnist`).
    - **loss**: Loss function to use (use `ARPLoss`).
    - **gpu**: GPU to use for training.
    - **max-epoch**: Maximum number of epochs (default 50).
    - **cs**: Conditional sampling flag - MUST BE SET!.
    - **result_dir**: Directory where the networks will be saved.

#### Image Generation
1. Run `python generate.py --max_epoch 50 --loss ARPLoss --number_images 800 --result_dir /home/user/username/arpl/ARPL/generated_emnist --gpu 1`.
    - **max_epoch**: Defines the generator model chosen (default 50).
    - **loss**: Loss function to use (use `ARPLoss`).
    - **number_images**: Number of batches to generate (Number of images will be batch size * this number).
    - **result_dir**: Directory where the generated images will be saved - Arpl images will have to be transferred manually later.
    - **gpu**: GPU to use for generation (set to 1).

After creating ARPL images, they need to be moved manually to the source code directory of the counterfactual GAN / Classifier training.
Optimally save the images in `/trajectories/emnist/arpl`.

#### Create Dataset File
For classifier training, a file containing the paths to the generated negative samples needs to be created. This can be done using this command.

1. Run `python generativeopenset/auto_label.py --output_filename /home/user/username/openset-comparison/counterfactual-open-set/emnist_arpl.dataset --dataset_name emnist`.
    - **output_filename**: Name of the output file containing paths to each image and its label (must end on `.dataset`).
    - **dataset_name**: Name of the dataset (use `emnist`).

### Classifier Training
1. The main file for classifier training with emnist is `emnist_main.py`.
2. You need to hardcode the directories to the EMNIST dataset and the directories of the generated samples at the top of the file.

3. Run the command: `python emnist_main.py --task "train" --dataset_root "/home/user/username/data/EMNIST/" --approach "EOS" --no_of_epochs 50 -g 2`
    - **task**: Set to `train` for training.
    - **dataset_root**: Location of the EMNIST dataset.
    - **approach**: Approach to use (use `EOS`, if training with negatives).
    - **gpu**: GPU to use for training.
    - **max-no_of_epochs**: Maximum number of epochs (default 50).

For different compositions of negative data in the training and validation set, the following parameters shall be passed:
- **include_unknown**: If this flag is set, NO negatives will be used in training.
- **include_counterfactuals**: Set this flag to true, to include counterfactual negatives.
- **include_arpl**: Set this flag to true, to include ARPL negatives.
- **mixed_unknowns**: Set this to true, to mix generated samples with original negative samples.

### Classifier Evaluation

Run the command: `python exampl_emnist_G.py --task "eval" --dataset_root "/home/user/heizmann/data/EMNIST/" --approach "EOS" --include_counterfactuals True --include_arpl True -g 1`
- **task**: Set to `train` for training.
- **dataset_root**: Location of the EMNIST dataset.
- **approach**: Approach to use (use `EOS`, if training with negatives).
- **gpu**: GPU to use for training.

To instruct the script which classifier to evaluate, use the same flag as it was trained with:
- **include_unknown**: If this flag is set, to evaluate the model trained with no unknowns.
- **include_counterfactuals**: Bool.
- **include_arpl**: Bool.
- **mixed_unknowns**: Bool.

### Plotting

You can plot a single classifier using: `python generativeopenset/plot_emnist.py` and indicating which model to plot by the flags:
- **include_unknown**: If this flag is set, to plot the model trained with no unknowns.
- **include_counterfactuals**: Bool.
- **include_arpl**: Bool.
- **mixed_unknowns**: Bool.

OR you can plot all available classifiers. This will also combine each subplot into a pdf for comparison:
Run: `python generativeopenset/plot_emnist.py --all`

## ImageNet Experiments

### Counterfactual ImageNet Images

#### Setup
1. Unzip the protocols file.
2. Convert the protocol CSV file into a `.dataset` file by running: `python datasets/imagenet.py --protocol 2`.
    - **protocol**: Protocol to use for creating the dataset.

#### GAN Training
1. Verify the correct dataset path in the params file or uncomment it.
2. Run `python generativeopenset/train_gan.py --dataset_name imagenet --epochs 50 -g 4`.
    - **dataset_name**: Name of the dataset (use `imagenet`).
    - **epochs**: Number of training epochs (default to 50).
    - **-g**: GPU to use.

#### Image Generation
1. Run `python generativeopenset/generate_open_set.py --result_dir /home/user/heizmann/openset-comparison/counterfactual-open-set --count 2484 --dataset imagenet -g 3`.
    - **result_dir**: Directory where the generated images will be saved.
    - **count**: Number of batches to process.
    - **dataset**: Name of the dataset (use `imagenet`).
    - **-g**: GPU to use.

2. Split generated image grids into individual images using `python generativeopenset/auto_label.py --dataset imagenet --output_filename /home/user/heizmann/openset-comparison/counterfactual-open-set/imagenet_images_counterfactual.dataset`.
    - **dataset**: Name of the dataset (use `imagenet`).
    - **output_filename**: Name of the output file containing paths to each image and its label.

### ARPL ImageNet Images

#### Setup
1. Set the path to ImageNet and protocol files in the `main_worker()` function in `osr.py`.

#### GAN Training
1. Run `python osr.py --dataset imagenet --loss ARPLoss --protocol 2 --max-epoch 300 --cs --result_dir /home/user/username/arpl/ARPL/generated_imagenet --gpu 0`.
    - **dataset**: Name of the dataset (use `imagenet`).
    - **loss**: Loss function to use (set to `ARPLoss`).
    - **protocol**: Protocol to use for creating the dataset (set to 2).
    - **max-epoch**: Maximum number of epochs (set to 300).
    - **cs**: Conditional sampling flag.
    - **result_dir**: Directory where the generated images will be saved.
    - **gpu**: GPU to use for training (set to 0).

#### Image Generation
1. Run `python generate.py --dataset imagenet --protocol 2 --max_epoch 300 --loss ARPLoss --number_images 1 --result_dir /home/user/username/arpl/ARPL/generated_imagenet --gpu 0`.
    - **dataset**: Name of the dataset (use `imagenet`).
    - **protocol**: Protocol to use for creating the dataset.
    - **max_epoch**: Defines the generator model chosen.
    - **loss**: Loss function to use (set to `ARPLoss`).
    - **number_images**: Number of batches to generate (actual images created will be batch size times number_images).
    - **result_dir**: Directory where the generated images will be saved.
    - **gpu**: GPU to use for generation.

#### Create Dataset File
1. Run `python generativeopenset/datasetfile_arpl.py --output_filename /home/user/username/openset-comparison/counterfactual-open-set/generated_arpl_imagenet.dataset --dataset_name imagenet`.
    - **output_filename**: Name of the output file containing paths to each image and its label.
    - **dataset_name**: Name of the dataset (use ending `generated_arpl_imagenet.dataset` as the code is hardcoded to look for this name).

### Classifier Training

##### Notes
- Ensure all generated images are moved to the appropriate directories ("trajectories/" subdirectory in the Openset-comparison repository).
- Certain required parameters are hardcoded in `configs/train.yaml`
    - Filenames of the ARPL & Counterfactual .dataset files
    - Loss
    - Epochs
- For certain commands, the `--loss` must always be set to `entropic`. This does not have any logical impact. The idea was to also be able to train and plot for different Loss Functions, but was not realized.

1. Run `train_imagenet.py config/train.yaml 2 -g 5`
    - The `2` indicates the protocol to use, `-g` the GPU to use.

    Indicate which data splits to use in training by using these flags:
    - **include_unknown**: If this flag is set, NO negatives will be used in training.
    - **include_counterfactuals**: Set this flag to true to include counterfactual negatives.
    - **include_arpl**: Set this flag to true to include ARPL negatives.
    - **mixed_unknowns**: Set this to true to mix generated samples with original negative samples.

### Classifier Evaluation

Run the command: `evaluate_imagenet.py config/train.yaml --loss entropic --protocol 2 --use-best --gpu 5`
- **loss**: Loss function used in classifier (leave as `entropic`).
- **protocol**: ImageNet protocol.
- **use-best**: Tells the script to use the best model of this approach, in case there are several.
- **gpu**: GPU to use.

To instruct the script which classifier to evaluate, use the same flag as it was trained with:
- **include_unknown**: If this flag is set, to evaluate the model trained with no unknowns.
- **include_counterfactuals**: Bool.
- **include_arpl**: Bool.
- **mixed_unknowns**: Bool.

### Plotting

You can plot a single classifier using: `plot_imagenet.py --use-best --protocols 2 -l entropic`
- **l**: Loss function used in classifier (leave as `entropic`).
- **protocols**: ImageNet protocol.
- **use-best**: Tells the script to use the best model of this approach, in case there are several.

To instruct the script which classifier to evaluate, use the same flag as it was trained with:
- **include_unknown**: If this flag is set, to evaluate the model trained with no unknowns.
- **include_counterfactuals**: Bool.
- **include_arpl**: Bool.
- **mixed_unknowns**: Bool.

OR use: `plot_imagenet.py --use-best --protocols 2 -l entropic --all`
This will combine the subplots of the individual models into a pdf.
