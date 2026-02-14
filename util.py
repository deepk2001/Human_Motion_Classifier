import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Utility functions for classification tasks."
    )
    parser.add_argument(
        "--traditional_model",
        type=str,
        default=None,
        help="Type of traditional classifier to use, if any",
    )
    parser.add_argument(
        "--deep_model",
        type=str,
        default=None,
        help="Type of deep learning model to use, if any",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Datasets/N_20_Takes_2.csv",
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["eulers"],
        choices=["eulers", "positions"],
        help="Features to use for classification",
    )
    # From here, you may want to add additional arguments as needed (e.g. epochs, batch size, learning rate, etc.)
    return parser.parse_args()


def plot_conf_mat(
    targets,
    predictions,
    title,
    filename,
    results_dir="results",
    class_names=None,
    save_prefix="",
    normalize="true",
):
    """
    Plot and save a single normalized confusion matrix.

    Args:
        targets (array-like): Ground truth labels
        predictions (array-like): Predicted labels
        title (str): Plot title
        filename (str): Base filename (without .png)
        class_names (list, optional): List of class names indexed by label
        save_prefix (str, optional): Prefix added to filename
        normalize (str or None): Passed to sklearn confusion_matrix
    """

    cm = confusion_matrix(targets, predictions, normalize=normalize)

    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 14))

    unique_indices = np.unique(targets)
    if class_names is not None:
        display_labels = [class_names[i] for i in unique_indices]
    else:
        display_labels = unique_indices

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    disp.plot(ax=ax, xticks_rotation="vertical", include_values=False, cmap="viridis")

    disp.im_.set_clim(0, 1)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.set_title(title, fontsize=18)

    cbar = ax.images[-1].colorbar
    cbar.set_label("Recall (Proportion of Correct Predictions)", fontsize=12)

    plt.tight_layout()

    full_filename = f"{save_prefix}_{filename}" if save_prefix else filename
    plt.savefig(f"{results_dir}/{full_filename}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def merge_classes(vals, class_map):
    # Convert the class map to ints if it's not already
    fixed_class_map = {int(k): int(v) for k, v in class_map.items()}
    mapped_arr = np.vectorize(fixed_class_map.get)(vals)
    return mapped_arr
