
# EMNIST and ImageNet Experiments

This guide provides instructions on how to create synthetic negatives samples from EMNIST and ImageNet datasets and train an open-set classifier using them. The process involves generating counterfactual and ARPL images, and subsequently training a classifier. The necessary source code repositories are provided.

## Source Code

- Counterfactuals Generation & Classifier Training: [Counterfact-Openset](https://www.github.com/counterfact-openset)
- ARPL Generation: [ARPL](https://www.github.com/ARPL)

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
1. Run `python generativeopenset/generate_open_set.py --result_dir /home/user/heizmann/openset-comparison/counterfactual-open-set --count 2484 --dataset emnist -g 3`.
    - **result_dir**: Directory where the generated images will be saved.
    - **count**: Number of batches to process (Number of images will be count * batch size).
    - **dataset**: Name of the dataset (use `emnist`).
    - **-g**: GPU to use for generation.

2. Split generated image grids into individual images using `python generativeopenset/auto_label.py --dataset emnist --output_filename /home/user/heizmann/openset-comparison/counterfactual-open-set/emnist_counterfactual.dataset`.
    - **dataset**: Name of the dataset (use `emnist`).
    - **output_filename**: Name of the output file containing paths to each image and its label - must end on `.dataset`.

### ARPL EMNIST Images

#### Setup
1. The same conda environment can be used, as for the other code.  
2. Set the correct path to EMNIST in the `main_worker()` function in `osr.py`.

#### GAN Training
1. Run `python osr.py --dataset emnist --loss ARPLoss --gpu 1 --max-epoch 50 --cs --result_dir /home/user/heizmann/arpl/ARPL/generated_emnist`.
    - **dataset**: Name of the dataset (use `emnist`).
    - **loss**: Loss function to use (use `ARPLoss`).
    - **gpu**: GPU to use for training.
    - **max-epoch**: Maximum number of epochs (default 50).
    - **cs**: Conditional sampling flag - MUST BE SET!.
    - **result_dir**: Directory where the networks will be saved.

#### Image Generation
1. Run `python generate.py --max_epoch 50 --loss ARPLoss --number_images 800 --result_dir /home/user/heizmann/arpl/ARPL/generated_emnist --gpu 1`.
    - **max_epoch**: Defines the generator model chosen (default 50).
    - **loss**: Loss function to use (use `ARPLoss`).
    - **number_images**: Number of batches to generate (Number of images will be batch size * this number).
    - **result_dir**: Directory where the generated images will be saved - Arpl images will have to be transferred manually later.
    - **gpu**: GPU to use for generation (set to 1).

After creating Arpl images, the need to be moved manually to the source code directory of the counterfactual GAN / Classifier training.
Optimally save the images in /trajectories/emnist/arpl


#### Create Dataset File
1. Run `python generativeopenset/auto_label.py --output_filename /home/user/heizmann/openset-comparison/counterfactual-open-set/emnist_arpl.dataset --dataset_name emnist`.
        Creates a 
    - **output_filename**: Name of the output file containing paths to each image and its label (must end on `.dataset`).
    - **dataset_name**: Name of the dataset (use `emnist`).

### Classifier Training
1. The main file for classifier training with emnist is emnist_main.py.
2. You need to hardcode the directories to the eminst dataset and the directories of the 
    generated samples at the top of the file.

3. Run the command: `python emnist_main.py --task "train" --dataset_root "/home/user/       heizmann/data/EMNIST/" --approach "EOS" --no_of_epochs 50 -g 2`
    - **task**: Set to `train` for training.
    - **dataset_root**: Location of the EMNIST dataset.
    - **approach**: Approach to use (use `EOS`, if training with negatives).
    - **gpu**: GPU to use for training.
    - **max-no_of_epochs**: Maximum number of epochs (default 50).

For different compositions of negative data in the training and validation set, the following parameters shall be passed:
    - **include_unknown**: If this flag is set, NO negatives will be used in training.
    - **include_counterfactuals**: Set this flag to true, to include counterfactual negatives.
    - **include_arpl**: Set this flag to true, to include ARPL negatives.
    - **mixed_unknowns**: Set this to true, to mix generated samples with original negative     samples - letters in this case.

### Classifier Evaluation


### Plotting


## ImageNet Experiments

### Counterfactual ImageNet Images

#### Setup
1. Unzip the protocols file.
2. Convert the protocol CSV file into a `.dataset` file using `python datasets/imagenet.py --protocol 2`.
    - **protocol**: Protocol to use for creating the dataset (set to 2).

#### GAN Training
1. Verify the correct dataset path in the params file and unlock it.
2. Run `python generativeopenset/train_gan.py --dataset_name imagenet --epochs 50 -g 4`.
    - **dataset_name**: Name of the dataset (use `imagenet`).
    - **epochs**: Number of training epochs (set to 50).
    - **-g**: GPU to use for training (set to 4).

#### Image Generation
1. Run `python generativeopenset/generate_open_set.py --result_dir /home/user/heizmann/openset-comparison/counterfactual-open-set --count 2484 --dataset imagenet -g 3`.
    - **result_dir**: Directory where the generated images will be saved.
    - **count**: Number of batches to process (set to 2484).
    - **dataset**: Name of the dataset (use `imagenet`).
    - **-g**: GPU to use for generation (set to 3).

2. Split generated image grids into individual images using `python generativeopenset/auto_label.py --dataset imagenet --output_filename /home/user/heizmann/openset-comparison/counterfactual-open-set/imagenet_images_counterfactual.dataset`.
    - **dataset**: Name of the dataset (use `imagenet`).
    - **output_filename**: Name of the output file containing paths to each image and its label.

### ARPL ImageNet Images

#### Setup
1. Set the path to ImageNet and protocol files in the `main_worker()` function in `osr.py`.

#### GAN Training
1. Run `python osr.py --dataset imagenet --loss ARPLoss --protocol 2 --max-epoch 300 --cs --result_dir /home/user/heizmann/arpl/ARPL/generated_imagenet --gpu 0`.
    - **dataset**: Name of the dataset (use `imagenet`).
    - **loss**: Loss function to use (set to `ARPLoss`).
    - **protocol**: Protocol to use for creating the dataset (set to 2).
    - **max-epoch**: Maximum number of epochs (set to 300).
    - **cs**: Conditional sampling flag.
    - **result_dir**: Directory where the generated images will be saved.
    - **gpu**: GPU to use for training (set to 0).

#### Image Generation
1. Run `python generate.py --dataset imagenet --protocol 2 --max_epoch 300 --loss ARPLoss --number_images 1 --result_dir /home/user/heizmann/arpl/ARPL/generated_imagenet --gpu 0`.
    - **dataset**: Name of the dataset (use `imagenet`).
    - **protocol**: Protocol to use for creating the dataset (set to 2).
    - **max_epoch**: Defines the generator model chosen (set to 300).
    - **loss**: Loss function to use (set to `ARPLoss`).
    - **number_images**: Number of batches to generate (set to 1).
    - **result_dir**: Directory where the generated images will be saved.
    - **gpu**: GPU to use for generation (set to 0).

#### Create Dataset File
1. Run `python generativeopenset/datasetfile_arpl.py --output_filename /home/user/heizmann/openset-comparison/counterfactual-open-set/generated_arpl_imagenet.dataset --dataset_name imagenet`.
    - **output_filename**: Name of the output file containing paths to each image and its label.
    - **dataset_name**: Name of the dataset (use `imagenet`).

## Additional Notes
- Ensure all generated images are moved to the appropriate directories as specified.
- Verify all paths and parameters are correctly set in the respective scripts before execution.
- The dataset files will be used for classifier training and contain the paths to all generated images.
