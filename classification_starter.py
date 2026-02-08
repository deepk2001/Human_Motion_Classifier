"""
Your Details: (The below details should be included in every python
file that you add code to.)
{
}
"""

import json
import os
from numpy.ma.core import mean
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.metrics import accuracy_score
from model_starter import MLP
from torch.optim.lr_scheduler import StepLR

from model_starter import TraditionalClassifier
from model_starter import MLP, CNN
from util import *
from generate_dataset import *


def lda_projection(train_feats, train_labels, test_feats, reg=1e-6):
    X = train_feats
    y = train_labels

    n_samples, n_features = X.shape
    class_labels = np.unique(y)

    mean_overall = np.mean(X, axis=0)

    S_w = np.zeros((n_features, n_features))
    S_b = np.zeros((n_features, n_features))
    class_means = {}
    class_priors = {}

    # Compute class means, S_w, and S_b
    for cls in class_labels:
        X_c = X[y == cls]
        mean_c = np.mean(X_c, axis=0)
        n_c = X_c.shape[0]

        class_means[cls] = mean_c
        class_priors[cls] = n_c / n_samples

        # Within-class scatter
        X_centered = X_c - mean_c
        S_w += X_centered.T @ X_centered

        # Between-class scatter
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_b += n_c * (mean_diff @ mean_diff.T)

    S_w += reg * np.eye(n_features)
    S_w_inv = np.linalg.inv(S_w)

    def predict(X_input):
        scores = np.zeros((X_input.shape[0], len(class_labels)))

        for i, cls in enumerate(class_labels):
            mu = class_means[cls]

            scores[:, i] = (
                X_input @ S_w_inv @ mu
                - 0.5 * mu @ S_w_inv @ mu
                + np.log(class_priors[cls])
            )

        return class_labels[np.argmax(scores, axis=1)]

    # ---- Predictions ----
    pred_train_labels = predict(train_feats)
    pred_test_labels = predict(test_feats)

    return pred_train_labels, pred_test_labels


def feature_selection(feats):
    """
    TODO: Implement Feature Selection
    """
    raise NotImplementedError


def convert_features_to_loader(
    train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size
):
    """
    TODO: Convert NumPy arrays to PyTorch tensors and create DataLoader instances.

    1. Convert `train_feats_proj` to a PyTorch tensor with dtype `torch.float32`.
    2. Convert `train_labels` to a PyTorch tensor with dtype `torch.long` (required for classification tasks).
    3. Create a `TensorDataset` from `train_feats_proj` and `train_labels`.
    4. Initialize a `DataLoader` for training, specifying `batch_size` and enabling shuffling for better generalization.
    5. Convert `test_feats_proj` to a PyTorch tensor with dtype `torch.float32`.
    6. Convert `test_labels` to a PyTorch tensor with dtype `torch.long`.
    7. Create a `TensorDataset` from `test_feats_proj` and `test_labels`.
    8. Initialize a `DataLoader` for testing, specifying `batch_size` but disabling shuffling to maintain order.
    9. Return the `train_loader` and `test_loader` for use in model training and evaluation.

    """


def cnn_learning(
    train_feats_proj,
    train_labels,
    test_feats_proj,
    test_labels,
    hidden_dim=0,
    batch_size=0,
    learning_rate=0,
    epochs=0,
):
    """
        TODO: Train and evaluate a CNN classifier. Refer to deep_learning function for structure. It is almost similar
    1. Infer `input_dim` from the feature dimension of `train_feats_proj`.
       - Hint: `train_feats_proj.shape[1]`
    2. Infer `output_dim` from the labels.
       - Hint: if labels are 0..C-1 then C = max(train_labels) + 1
    3. Initialize the CNN model using `CNN(input_dim, output_dim, hidden_dim)`.
    4. Define the loss function as `nn.CrossEntropyLoss()`.
    5. Define the optimizer as `optim.Adam(model.parameters(), lr=learning_rate)`.
    6. Define a learning-rate scheduler using `StepLR(optimizer, step_size=5, gamma=0.1)`.
    7. Use `convert_features_to_loader(...)` to create `train_loader` and `test_loader`.
    8. Training loop (repeat for `epochs`):
       a) Set model to training mode using `model.train()`.
       b) For each batch from `train_loader`:
          i)   Zero gradients using `optimizer.zero_grad()`.
          ii)  Forward pass: `outputs = model(batch_data)`.
          iii) Compute loss: `loss = criterion(outputs, batch_labels)`.
          iv)  Backprop: `loss.backward()`.
          v)   Update parameters: `optimizer.step()`.
          vi)  Accumulate the batch loss into `total_loss`.
       c) Step the scheduler once per epoch using `scheduler.step()`.
       d) Print average loss for the epoch.
    9. Evaluate on the test set:
       a) Set model to eval mode using `model.eval()`.
       b) Disable gradients using `torch.no_grad()`.
       c) For each batch in `test_loader`:
          i)   Forward pass and compute predicted class using `torch.max(outputs, 1)`.
          ii)  Count correct predictions and total samples.
       d) Compute `test_acc` as percentage and print it.
    10. (Optional) Evaluate on the training set the same way to compute `train_acc`.
    11. Return `(model, train_acc, test_acc)`.
    12. Check the function parameters and fill in appropriate values where indicated.


    """


# TODO: Parameters value have been left blank. Fill in the parameters with appropriate values
def deep_learning(
    train_feats_proj,
    train_labels,
    test_feats_proj,
    test_labels,
    hidden_dim=64,
    num_layers=2,
    batch_size=128,
    learning_rate=0.001,
    epochs=50,
):

    # infer dims from data
    input_dim = int(train_feats_proj.shape[1])
    output_dim = int(
        np.max(train_labels) + 1
    )  # assumes labels are 0..C-1 (yours are 0..45)

    model = MLP(input_dim, output_dim, hidden_dim, nn.ReLU, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader, test_loader = convert_features_to_loader(
        train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)  # Model call
            loss = criterion(outputs, batch_labels)
            loss.backward()

            optimizer.step()  # Update model parameters
            total_loss += loss.item()
        scheduler.step()  # Update learning rate if using a scheduler (optional)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total * 100
    print(f"Deep Learning Test Accuracy: {accuracy:.2f}%")

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total * 100
    print(f"Deep Learning Train Accuracy: {accuracy:.2f}%")


def perform_traditional(train_feats_proj, train_labels, test_feats_proj, test_labels):

    classifiers = {
        "traditional_classifier": TraditionalClassifier(),
        "deep_learning": deep_learning,
        "CNN": cnn_learning,
    }

    for name, clf in classifiers.items():
        if name == "traditional_classifier":
            print(f"\n--- {name} Classifier ---")
            # Train the classifier
            clf.fit(train_feats_proj, train_labels)

            # Predict the labels of the training and testing data
            pred_train_labels = clf.predict(train_feats_proj)
            pred_test_labels = clf.predict(test_feats_proj)

            # Get statistics
            train_accuracy = accuracy_score(train_labels, pred_train_labels) * 100
            test_accuracy = accuracy_score(test_labels, pred_test_labels) * 100

            print(f"KNN Train Accuracy: {train_accuracy:.2f}%")
            print(f"KNN Test Accuracy: {test_accuracy:.2f}%")

            return train_accuracy, test_accuracy
            # See example_classification function for plotting confusion matrices and testing on the merged classes

        else:
            # TODO: Call deep learning and CNNfunction to train and evaluate the model
            """clf(train_feats_proj, train_labels, test_feats_proj, test_labels)"""
            continue


def load_new_dataset(dataset_path, verbose=True, subject_index=9):
    dataset = np.loadtxt(dataset_path, delimiter=",", dtype=float, skiprows=1)
    # Extract `person_idxs` (last column)
    person_idxs = dataset[:, -2]  # All rows, 2nd last column
    labels = dataset[:, -3]  # All rows, 3rd last column

    feats = dataset[:, :-3].T  # All rows, all columns except the last three, transposed
    # TODO: Feature selection (EXTRA CREDIT). You can comment out the feature selection part if you are not implementing it.
    # feats = feature_selection(feats)
    feature_mask = np.var(feats, axis=1) > 0
    # Leave one subject out (LOSO)
    train_mask = person_idxs != subject_index

    train_feats = feats[feature_mask, :][:, train_mask].T
    train_labels = labels[train_mask].astype(int)
    test_feats = feats[feature_mask, :][:, ~train_mask].T
    test_labels = labels[~train_mask].astype(int)

    if verbose:

        def print_set_info(name, f, l):
            classes, counts = np.unique(l, return_counts=True)
            print(f"--- {name} Set ---")
            print(
                f"\t# Classes: {len(classes)} | # Features: {f.shape[1]} | # Samples: {f.shape[0]}"
            )

            print("\tClass Counts:")
            grid_str = ""
            for i, (cls, count) in enumerate(zip(classes, counts)):
                grid_str += f"C{cls:02d}: {count:<4}  "
                if (i + 1) % 6 == 0:  # New line every 6 items
                    grid_str += "\n\t"
            print(f"\t{grid_str.strip()}")

        print(f"\n[INFO] {dataset_path} Loaded (Subject {subject_index} out)")
        print_set_info("Training", train_feats, train_labels)
        print_set_info("Testing", test_feats, test_labels)
        print("-" * 50)

    return train_feats, train_labels, test_feats, test_labels


def example_classification(args, class_info):
    """
    Example classification using Linear Discriminant Analysis (LDA). Subject 9 is left out for testing.
    """
    # Assume this returns train_feats, train_labels, test_feats, test_labels
    train_feats, train_labels, test_feats, test_labels = load_new_dataset(
        dataset_path=args.dataset_path, subject_index=9
    )

    pred_train_labels, pred_test_labels = lda_projection(
        train_feats, train_labels, test_feats
    )

    # 1. Standard (Unmerged) Results
    train_acc = accuracy_score(train_labels, pred_train_labels) * 100
    train_std = np.std(train_labels - pred_train_labels)
    test_acc = accuracy_score(test_labels, pred_test_labels) * 100
    test_std = np.std(test_labels - pred_test_labels)

    print(f"Unmerged LDA Train Accuracy: {train_acc:.2f}% ± {train_std:.2f}")
    print(f"Unmerged LDA Test Accuracy: {test_acc:.2f}% ± {test_std:.2f}")

    plot_conf_mat(
        targets=train_labels,
        predictions=pred_train_labels,
        title="LDA Training Confusion Matrix (Unmerged)",
        filename="unmerged_train_confusion_matrix.png",
        class_names=class_info["names"]["short_unmerged"],
        save_prefix="LDA",
    )
    plot_conf_mat(
        targets=test_labels,
        predictions=pred_test_labels,
        title="LDA Testing Confusion Matrix (Unmerged)",
        filename="unmerged_test_confusion_matrix.png",
        class_names=class_info["names"]["short_unmerged"],
        save_prefix="LDA",
    )

    # 2. Merged Results
    merged_train_labels = merge_classes(train_labels, class_info["class_map"])
    merged_test_labels = merge_classes(test_labels, class_info["class_map"])
    merged_pred_train_labels = merge_classes(pred_train_labels, class_info["class_map"])
    merged_pred_test_labels = merge_classes(pred_test_labels, class_info["class_map"])

    m_train_acc = accuracy_score(merged_train_labels, merged_pred_train_labels) * 100
    m_train_std = np.std(merged_train_labels - merged_pred_train_labels)
    m_test_acc = accuracy_score(merged_test_labels, merged_pred_test_labels) * 100
    m_test_std = np.std(merged_test_labels - merged_pred_test_labels)

    print(f"Merged LDA Train Accuracy: {m_train_acc:.2f}% ± {m_train_std:.2f}")
    print(f"Merged LDA Test Accuracy: {m_test_acc:.2f}% ± {m_test_std:.2f}")

    plot_conf_mat(
        targets=merged_train_labels,
        predictions=merged_pred_train_labels,
        title="LDA Training Confusion Matrix (Merged)",
        filename="merged_train_confusion_matrix.png",
        class_names=class_info["names"]["short_merged"],
        save_prefix="LDA",
    )
    plot_conf_mat(
        targets=merged_test_labels,
        predictions=merged_pred_test_labels,
        title="LDA Testing Confusion Matrix (Merged)",
        filename="merged_test_confusion_matrix.png",
        class_names=class_info["names"]["short_merged"],
        save_prefix="LDA",
    )


def fisher_projection(train_feats, train_labels):
    """
    TODO: Implement Fisher's Linear Discriminant Analysis (LDA) for dimensionality reduction.

    Steps:
    1. Compute the overall mean of the training features.
    2. Calculate the mean vector for each class.
    3. Compute the within-class scatter matrix (S_w):
       - For each class, compute the deviation of each sample from the class mean.
       - Compute the scatter contribution for each class and sum them to obtain S_w.
    4. Compute the between-class scatter matrix (S_b):
       - Compute the deviation of each class mean from the overall mean.
       - Compute the scatter contribution for each class weighted by the number of samples.
       - Sum them to obtain S_b.
    5. Compute the transformation matrix J(W) = S_w^-1 * S_b.
    6. Solve for the eigenvalues and eigenvectors of J(W).
    7. Sort eigenvectors in descending order based on their absolute eigenvalues.
    8. Select the top two eigenvectors for dimensionality reduction.
    9. Return the selected eigenvectors for projecting data into a lower-dimensional space.

    Note: Ensure numerical stability while computing the inverse of S_w.
    """
    reg = 1e-6
    X = train_feats
    y = train_labels
    samplesSize, featureSize = X.shape
    classLabels = np.unique(y)

    # computing the overall mean using numpy mean function
    overallMean = np.mean(X, axis=0)

    # calculating the mean for each class
    classMeans = {}

    withinClassScatter = np.zeros((featureSize, featureSize))
    betweenClassScatter = np.zeros((featureSize, featureSize))

    for label in classLabels:
        Xc = X[y == label]
        classMeans[label] = np.mean(Xc, axis=0)
        classSize = Xc.shape[0]

        Xcentered = Xc - classMeans[label]
        withinClassScatter += Xcentered.T @ Xcentered

        meanDiff = (classMeans[label] - overallMean).reshape(-1, 1)
        betweenClassScatter += classSize * (meanDiff @ meanDiff.T)

    # computing J(W) = S_w^{-1} S_b (with regularization)
    withinClassScatter += reg * np.eye(
        featureSize
    )  # regularizing the within class scatter matrix
    withinClassScatterInv = np.linalg.inv(withinClassScatter)
    J = withinClassScatterInv @ betweenClassScatter

    # computing the eigenvalues and eigenvectors of J
    eigenValues, eigenVectors = np.linalg.eig(J)

    # sorting the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    # selecting the top two eigenvectors
    eigenVectors = eigenVectors[:, :5].real

    return eigenVectors


def classification():
    """
      TODO: Implement Leave-One-Subject-Out (LOSO) cross-validation.
      You dont need to implement this fully here. You can modify this as you wnat

     Loop over all subjects:
    - Each iteration, select one subject index as the test set.
    - Use all other subjects as the training set.
    Store accuracy scores:
    - Save the accuracy score for each test subject.
    - Repeat this process for all subjects.
    - Then find the average of all 10 subjects to get the final accuracy score
    """
    numSubjects = 10
    accuracyWithProjectionTest = []
    accuracyWithProjectionTrain = []
    accuracyWithoutProjectionTest = []
    accuracyWithoutProjectionTrain = []
    for subject in range(numSubjects):
        train_feats, train_labels, test_feats, test_labels = load_new_dataset(
            dataset_path="Datasets/N_20_Takes_2.csv", subject_index=subject + 1
        )
        train_accuracy, test_accuracy = perform_traditional(
            train_feats, train_labels, test_feats, test_labels
        )
        accuracyWithoutProjectionTest.append(test_accuracy)
        accuracyWithoutProjectionTrain.append(train_accuracy)
        train_eigens = fisher_projection(train_feats, train_labels)
        train_feats_proj = train_feats @ train_eigens
        test_feats_proj = test_feats @ train_eigens
        train_accuracy, test_accuracy = perform_traditional(
            train_feats_proj, train_labels, test_feats_proj, test_labels
        )
        accuracyWithProjectionTest.append(test_accuracy)
        accuracyWithProjectionTrain.append(train_accuracy)

    print("LOSO Mean accuracies ------------------")
    print(f"Accuracy with projection test: {np.mean(accuracyWithProjectionTest)}")
    print(f"Accuracy with projection train: {np.mean(accuracyWithProjectionTrain)}")
    print(f"Accuracy without projection test: {np.mean(accuracyWithoutProjectionTest)}")
    print(
        f"Accuracy without projection train: {np.mean(accuracyWithoutProjectionTrain)}"
    )


def main():
    args = parse_args()
    with open("assets/class_info.json", "r") as f:
        class_info = json.load(f)

    if create_dataset:
        create_dataset_file(
            data_path="Data",
            keypose_csv="assets/keyposes.csv",
            output_dir="Datasets",
            num_takes=2,
            subjects=list(range(1, 11)),
            downsample_rate=5,
            n=20,
            seed=42,
        )
    example_classification(args=args, class_info=class_info)

    # TODO: Call the classification function to perform LOSO classification
    classification()


if __name__ == "__main__":
    create_dataset = True
    main()
