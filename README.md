# PRML Project 2 Starter

Starter code for keypose classification using motion-capture data.

## Project layout
- `generate_dataset.py` : Builds a flattened CSV dataset from `.mat` mocap files and keypose annotations.
- `classification_starter.py` : Baseline pipeline for loading the dataset, projecting features, and running classifiers.
- `model_starter.py` : Model definitions (traditional classifier placeholder, MLP, and CNN).
- `assets/` : Class metadata and keypose indices.
- `Data/` : Raw input data (per-subject `.mat` files).
- `Datasets/` : Generated CSV datasets.
- `results/` : Output plots (confusion matrices, etc.).

## Dependencies
This project expects a Python environment with:
- `numpy`, `pandas`, `h5py`
- `scikit-learn`, `matplotlib`
- `torch`


## Generate the dataset
`generate_dataset.py` reads motion-capture `.mat` files and produces a CSV with flattened joint coordinates plus labels.

Defaults (can be edited in the script):
- `data_path="Data"`
- `keypose_csv="assets/keyposes.csv"`
- `output_dir="Datasets"`
- `num_takes=2`, `subjects=1..10`
- `downsample_rate=5`, `n=20`

Output example: `Datasets/N_20_Takes_2.csv`

Note: This function is called from `classification_starter.py`. To create a dataset, set `create_dataset = True`. You can choose the value of `N` (e.g., 20 or 50). The default is 20. You can also try both and compare results to pick the better setting.
## Run classification
`classification_starter.py` loads the dataset, does a leave-one-subject-out split, and runs a baseline classifier. You need to run it 10 times, leaving each subject out once, to complete cross-validation. I suggest adding a `for` loop to automate all runs instead of doing them manually.

Run:
```bash
python classification_starter.py
```

Key points in the pipeline:
- `load_new_dataset(...)` loads CSV and holds out one subject for testing.
- `fisher_projection(...)` is a TODO for LDA-based dimensionality reduction.
- `perform_traditional(...)` runs either the placeholder traditional classifier, the MLP, or CNN.
- `example_classification(...)` shows LDA classification and confusion matrices.

## TODOs in the starter code
- `feature_selection(...)` in `classification_starter.py`
- `convert_features_to_loader(...)` in `classification_starter.py`
- `cnn_learning(...)` in `classification_starter.py`
- `fisher_projection(...)` in `classification_starter.py`
- `TraditionalClassifier.evaluate(...)` in `model_starter.py`

## Notes
- The dataset CSV contains flattened joint features, then `Label`, `Subject`, and `Take` as the last three columns.
- `classification_starter.py` currently sets `create_dataset = False`; set to `True` to regenerate the CSV.
