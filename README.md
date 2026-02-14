# Project 2: Classification and Feature Selection

## Installation

We suggest setting up a virtual environment for this project. This may be done using Conda or Python's built-in `venv` module. For example, using `venv`, you can create and activate a virtual environment as follows:

```bashpython -m venv prml
source prml/bin/activate  # On Windows, use `prml\Scripts\activate`
```

To install PyTorch, you can follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). Make sure to select the appropriate options for your system (e.g., operating system, package manager, Python version, and CUDA version if applicable).

Once you have your virtual environment and PyTorch set up and activated, you can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Dataset Generation

To generate the pose classification dataset from the PSUTMM dataset, you can use the provided `generate_dataset.py` script. This script will process the raw data and create a structured dataset suitable for training and evaluating classification models. To run the script, use the following command:

```bash
python generate_dataset.py --n 20 --downsample_rate 5 --num_takes 2
```

This will create a csv file with the processsed dataset with the naming convention `N_{n}_Takes_{num_takes}.csv`.

The three important arguments are:
- `--n`: This specifies the number of frames in the window around each labeled keyframe. 
- `--downsample_rate`: This determines how many frames to skip when creating the dataset, effectively controlling the size of the dataset and the temporal resolution of the data. 5 is a good starting point, but you are free to experiment with different values to see how it affects model performance.
- `--num_takes`: This specifies how many takes (or sequences) to include in from each subject. Each subject in PSUTMM has multiple recorded takes, and this parameter allows you to control how much data from each subject is included in the dataset. Using more takes can provide more training data, but it may also increase the computational requirements for training your model. 2 is a reasonable starting point, but you can adjust this based on your needs and resources.


## Training and Evaluation

To run training and inference in a LOSO manner, use the `classification_starter.py` script. It takes in the arguments:
- `--dataset_path`: The path to the generated dataset csv file.
- `--features`: Which features you want to include during training/testing. The options are `positions` and `eulers`. You can specify one or both (e.g. `--features positions eulers`).

There are two additional arguments that you can use to control what model is used during your experiments. You may use or adjust these as you see fit.

We provide a sample function that performs classification on a single subject with no feature selection. We suggest that you use this as an example when creating your `classification` function.

## Quickstart

To quickly run the provided code, after creating an environment and installing PyTorch, you may run:

```bash
pip install -r requirements.txt
python generate_dataset.py --n 20 --downsample_rate 5 --num_takes 2
python classification_starter.py --features positions
```