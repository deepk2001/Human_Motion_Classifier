# Project 2: Classification and Feature Selection

## Installation

We suggest setting up a virtual environment for this project. This may be done using Conda or Python's built-in `venv` module. For example, using `venv`, you can create and activate a virtual environment as follows:

```bash
python3 -m venv .venv
source .venv/bin/activate  
# On Windows, use 
.venv\Scripts\activate
```

To install PyTorch, you can follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). Make sure to select the appropriate options for your system (e.g., operating system, package manager, Python version, and CUDA version if applicable).

Once you have your virtual environment and PyTorch set up and activated, you can install the required dependencies using pip:

```bash
pip3 install -r requirements.txt
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
python3 -m venv .venv
source .venv/bin/activate # my system is Mac, please use the windows counterpart if your system requires it
pip3 install -r requirements.txt
python generate_dataset.py --n 20 --downsample_rate 5 --num_takes 2
python classification_starter.py --features  eulers  --dataset_path Datasets/N_20_Takes_2.csv --traditional_model KNN
```


# Function Descriptions

## classification_starter.py

### `load_new_dataset(dataset_path, verbose=False, subject_index=9, features=["euler"])`
Loads the dataset from the given path, selects requested feature types (e.g., positions, eulers), removes zero-variance features, and performs Leave-One-Subject-Out (LOSO) splitting into training and testing sets.

---

### `lda_projection(train_feats, train_labels, test_feats, reg=1e-6)`
Implements Linear Discriminant Analysis (LDA) classification. Computes within-class and between-class scatter matrices, estimates class priors, and returns predicted labels for both training and test sets.

---

### `fisher_projection(train_feats, train_labels)`
Computes Fisher’s Linear Discriminant projection matrix for dimensionality reduction. Returns the top eigenvectors used to project features into a lower-dimensional LDA space.

---

### `filter_method(feats, labels, topK=15)`
Performs statistical feature ranking using variance ratio (between-class variance / within-class variance). Returns indices of the top-ranked features.

---

### `wrapper_method(feats, labels, filterIndices, maxFeatures=15)`
Performs wrapper-based feature selection using KNN with cross-validation. Iteratively selects features that maximize validation accuracy.

---

### `feature_selection(feats, labels, method="filter")`
Feature selection hub function. Calls either filter-based or wrapper-based feature selection and returns selected feature indices.

---

### `convert_features_to_loader(train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size)`
Converts NumPy feature arrays into PyTorch tensors and creates DataLoader objects for training and testing.

---

### `deep_learning(train_feats_proj, train_labels, test_feats_proj, test_labels, ...)`
Trains and evaluates deep learning models (MLP or CNN). Performs training, testing, computes accuracy and macro F1 score, and generates confusion matrices.

---

### `perform_traditional(train_feats_proj, train_labels, test_feats_proj, test_labels, key, projection)`
Handles training and evaluation for traditional classifiers and deep learning models. Applies optional feature selection and returns accuracy and F1 metrics.

---

### `example_classification(args, class_info)`
Demonstrates LDA classification for a single subject split. Computes both unmerged and merged class performance and generates confusion matrices.

---

### `classification(args)`
Main evaluation pipeline. Performs LOSO cross-validation across all subjects, evaluates all classifiers with and without LDA projection, aggregates metrics, and saves final results.

---

### `main()`
Entry point of the program. Calls the full LOSO classification pipeline.