"""
Your Details: (The below details should be included in every python
file that you add code to.)
{
    Name : Deep Hiren Kotecha
    email: djk6507@psu.edu
}
"""

import json
import pandas as pd
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
    f1_score,
)
from sklearn.metrics import accuracy_score
from model_starter import MLP
from torch.optim.lr_scheduler import StepLR

from model_starter import TraditionalClassifier
from model_starter import MLP, CNN
from util import *
from generate_dataset import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


globalArgs = parse_args()
with open("assets/class_info.json", "r") as f:
    globalClassInfo = json.load(f)

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


def filter_method(feats, labels, topK=15):
    """
    Ranks features by Variance Ratio without changing the shape of feats.
    Returns the INDICES of the topK features.
    """
    numFeatures = feats.shape[1]
    classLabels = np.unique(labels)
    overallMean = np.mean(feats, axis=0)
    varianceRatios = np.zeros(numFeatures)

    for i in range(numFeatures):
        featureCol = feats[:, i]
        betweenClassVar = 0
        withinClassVar = 0

        for label in classLabels:
            classFeats = featureCol[labels == label]
            if len(classFeats) < 2:
                continue
            classMean = np.mean(classFeats)
            classSize = classFeats.shape[0]

            betweenClassVar += classSize * (classMean - overallMean[i]) ** 2
            withinClassVar += np.sum((classFeats - classMean) ** 2)

        varianceRatios[i] = betweenClassVar / (withinClassVar + 1e-6)

    rankedIndices = np.argsort(varianceRatios)[::-1]
    return rankedIndices[:topK]


def wrapper_method(feats, labels, filterIndices, maxFeatures=15):
    selectedIndices = []
    remainingIndices = list(filterIndices)

    while len(selectedIndices) < maxFeatures and remainingIndices:
        bestScore = -1
        bestFeature = None
        print(len(selectedIndices))

        for feature in remainingIndices:
            currentTrial = selectedIndices + [feature]
            clf = KNeighborsClassifier(n_neighbors=3)

            # Use 3-fold CV on the training data to evaluate the feature set
            scores = cross_val_score(clf, feats[:, currentTrial], labels, cv=3)
            meanScore = np.mean(scores)

            if meanScore > bestScore:
                bestScore = meanScore
                bestFeature = feature

        selectedIndices.append(bestFeature)
        remainingIndices.remove(bestFeature)

    return selectedIndices


def feature_selection(feats, labels, method="filter"):
    """
    Utilization Hub: Returns the indices to keep based on the method requested.
    The shape of the 'feats' passed in is preserved.
    """
    if method == "filter":
        return filter_method(feats, labels, topK=15)
    elif method == "wrapper":
        # Step 1: Statistical ranking (Filter) to reduce search space for speed
        topKIndices = filter_method(feats, labels, topK=30)
        # Step 2: Search within those indices using the Wrapper
        return wrapper_method(feats, labels, topKIndices, maxFeatures=15)
    else:
        # Return all indices if no selection is requested
        return np.arange(feats.shape[1])


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
    trainFeatsTensor = torch.tensor(train_feats_proj, dtype=torch.float32)
    trainLabelsTensor = torch.tensor(train_labels, dtype=torch.long)
    trainDataset = TensorDataset(trainFeatsTensor, trainLabelsTensor)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testFeatsTensor = torch.tensor(test_feats_proj, dtype=torch.float32)
    testLabelsTensor = torch.tensor(test_labels, dtype=torch.long)
    testDataset = TensorDataset(testFeatsTensor, testLabelsTensor)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    return trainLoader, testLoader


# TODO: Parameters value have been left blank. Fill in the parameters with appropriate values
def deep_learning(
    train_feats_proj,
    train_labels,
    test_feats_proj,
    test_labels,
    input_dim=0,
    output_dim=0,
    hidden_dim=64,
    num_layers=2,
    batch_size=128,
    learning_rate=0.001,
    epochs=100,
    modelkey="MLP",
):
    models = {
        "MLP": MLP(input_dim, output_dim, hidden_dim, nn.ReLU, num_layers),
        "CNN": CNN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim),
    }
    model = models[modelkey]
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
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    trainAccuracy = accuracy_score(all_labels, all_preds) * 100
    trainF1 = f1_score(all_labels, all_preds, average="macro")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    sub_dir = "_".join(globalArgs.features)
    results_dir = os.path.join("results", sub_dir)

    plot_conf_mat(
        targets=all_labels,
        predictions=all_preds,
        title="LDA Testing Confusion Matrix (Unmerged)",
        filename=f"unmerged_test_confusion_matrix_{modelkey}.png",
        class_names=globalClassInfo["names"]["short_unmerged"],
        save_prefix="LDA",
        results_dir=results_dir,
    )

    testAccuracy = accuracy_score(all_labels, all_preds) * 100
    testF1 = f1_score(all_labels, all_preds, average="macro")

    plot_conf_mat(
        targets=all_labels,
        predictions=all_preds,
        title="LDA Testing Confusion Matrix (Unmerged)",
        filename=f"unmerged_test_confusion_matrix_{modelkey}.png",
        class_names=globalClassInfo["names"]["short_unmerged"],
        save_prefix="LDA",
        results_dir=results_dir,
    )

    return trainAccuracy, testAccuracy, trainF1, testF1


def perform_traditional(
    train_feats_proj,
    train_labels,
    test_feats_proj,
    test_labels,
    key="traditional_classifier-w_filter",
):

    classifiers = {
        "traditional_classifier": TraditionalClassifier(),
        "MLP": deep_learning,
        "CNN": deep_learning,
    }
    trainAccuracy, testAccuracy = 0, 0
    trainF1Score, testF1Score = 0, 0
    config = key.split("-")
    clf = classifiers[config[0]]
    if "traditional_classifier" in key:

        featureSelectionMethod = config[1].split("_")[1]
        print(f"\n\n{key}\n\n")
        train_feats_proj_copy = train_feats_proj
        test_feats_proj_copy = test_feats_proj
        selected_idx = feature_selection(
            train_feats_proj_copy, train_labels, method=featureSelectionMethod
        )
        train_feats_proj_feats = train_feats_proj_copy[:, selected_idx]
        test_feats_proj_feats = test_feats_proj_copy[:, selected_idx]

        # Train the classifier
        clf.fit(train_feats_proj_feats, train_labels)

        # Predict the labels of the training and testing data
        pred_train_labels = clf.predict(train_feats_proj_feats)
        pred_test_labels = clf.predict(test_feats_proj_feats)

        # Get statistics
        trainAccuracy = accuracy_score(train_labels, pred_train_labels) * 100

        testAccuracy = accuracy_score(test_labels, pred_test_labels) * 100

        trainF1Score = f1_score(train_labels, pred_train_labels, average="macro")
        testF1Score = f1_score(test_labels, pred_test_labels, average="macro")

        sub_dir = "_".join(globalArgs.features)
        results_dir = os.path.join("results", sub_dir)

        plot_conf_mat(
            targets=train_labels,
            predictions=pred_train_labels,
            title="LDA Training Confusion Matrix (Unmerged)",
            filename=f"unmerged_test_confusion_matrix_{key}.png",
            class_names=globalClassInfo["names"]["short_unmerged"],
            save_prefix="LDA",
            results_dir=results_dir,
        )
        plot_conf_mat(
            targets=test_labels,
            predictions=pred_test_labels,
            title="LDA Testing Confusion Matrix (Unmerged)",
            filename=f"unmerged_test_confusion_matrix_{key}.png",
            class_names=globalClassInfo["names"]["short_unmerged"],
            save_prefix="LDA",
            results_dir=results_dir,
        )

        print(f"KNN Train Accuracy: {trainAccuracy:.2f}%")
        print(f"KNN Test Accuracy: {testAccuracy:.2f}%")

        # See example_classification function for plotting confusion matrices and testing on the merged classes

    else:
        # TODO: Call deep learning and CNN function to train and evaluate the model
        input_dim = int(train_feats_proj.shape[1])
        output_dim = int(np.max(train_labels) + 1)
        testAccuracy, trainAccuracy, trainF1Score, testF1Score = clf(
            train_feats_proj,
            train_labels,
            test_feats_proj,
            test_labels,
            input_dim=input_dim,
            output_dim=output_dim,
            modelkey=key,
        )
    return trainAccuracy, testAccuracy, trainF1Score, testF1Score


def load_new_dataset(dataset_path, verbose=False, subject_index=9, features=["euler"]):

    # Determine which features to load in
    df_headers = pd.read_csv(dataset_path, nrows=0)
    col_names = df_headers.columns.tolist()

    features_to_use = features
    selected_indices = []

    for j in range(17):
        # 1. Add Positions if requested
        if "positions" in features_to_use:
            selected_indices.extend(
                [
                    col_names.index(f"joint{j}_x"),
                    col_names.index(f"joint{j}_y"),
                    col_names.index(f"joint{j}_z"),
                ]
            )

        # 2. Add Euler if requested
        if "eulers" in features_to_use:
            selected_indices.extend(
                [
                    col_names.index(f"joint{j}_yaw"),
                    col_names.index(f"joint{j}_pitch"),
                    col_names.index(f"joint{j}_roll"),
                ]
            )
        # 3. Add Confidence always
        selected_indices.append(col_names.index(f"joint{j}_conf"))

    dataset = np.loadtxt(dataset_path, delimiter=",", skiprows=1)

    # ...,Label,Subject,Take
    person_idxs = dataset[:, -2]  # All rows, 2nd last column
    labels = dataset[:, -3]  # All rows, 3rd last column

    # Feature extraction based on selected features
    feats = dataset[:, selected_indices]

    # Here we just use 0 variance feature removal as an example
    feature_mask = np.var(feats, axis=0) > 0
    feats = feats[:, feature_mask]

    # Leave one subject out (LOSO)
    train_mask = person_idxs != subject_index
    train_feats = feats[train_mask, :]
    train_labels = labels[train_mask].astype(int)
    test_feats = feats[~train_mask, :]
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
        dataset_path=args.dataset_path, subject_index=9, features=args.features
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

    # Sub dir will be based on features used
    sub_dir = "_".join(args.features)
    results_dir = os.path.join("results", sub_dir)

    # The below function plots the confusion matrix with the actual class names
    plot_conf_mat(
        targets=train_labels,
        predictions=pred_train_labels,
        title="LDA Training Confusion Matrix (Unmerged)",
        filename="unmerged_train_confusion_matrix.png",
        class_names=class_info["names"]["short_unmerged"],
        save_prefix="LDA",
        results_dir=results_dir,
    )
    plot_conf_mat(
        targets=test_labels,
        predictions=pred_test_labels,
        title="LDA Testing Confusion Matrix (Unmerged)",
        filename="unmerged_test_confusion_matrix.png",
        class_names=class_info["names"]["short_unmerged"],
        save_prefix="LDA",
        results_dir=results_dir,
    )
    # Alternatively, you can use the class indices instead of names
    plot_conf_mat(
        targets=test_labels,
        predictions=pred_test_labels,
        title="LDA Testing Confusion Matrix (Unmerged)",
        filename="unmerged_test_confusion_matrix_indices.png",
        class_names=None,
        save_prefix="LDA",
        results_dir=results_dir,
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
        results_dir=results_dir,
    )
    plot_conf_mat(
        targets=merged_test_labels,
        predictions=merged_pred_test_labels,
        title="LDA Testing Confusion Matrix (Merged)",
        filename="merged_test_confusion_matrix.png",
        class_names=class_info["names"]["short_merged"],
        save_prefix="LDA",
        results_dir=results_dir,
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
    eigenVectors = eigenVectors[:, :45].real

    return eigenVectors


def classification(args):
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
    accuracyMetrics = {
        "traditional_classifier-w_filter": {
            "accuracyWithProjectionTest": [],
            "accuracyWithProjectionTrain": [],
            "accuracyWithoutProjectionTest": [],
            "accuracyWithoutProjectionTrain": [],
            "f1WithProjectionTest": [],
            "f1WithProjectionTrain": [],
            "f1WithoutProjectionTest": [],
            "f1WithoutProjectionTrain": [],
        },
        "traditional_classifier-w_wrapper": {
            "accuracyWithProjectionTest": [],
            "accuracyWithProjectionTrain": [],
            "accuracyWithoutProjectionTest": [],
            "accuracyWithoutProjectionTrain": [],
            "f1WithProjectionTest": [],
            "f1WithProjectionTrain": [],
            "f1WithoutProjectionTest": [],
            "f1WithoutProjectionTrain": [],
        },
        "MLP": {
            "accuracyWithProjectionTest": [],
            "accuracyWithProjectionTrain": [],
            "accuracyWithoutProjectionTest": [],
            "accuracyWithoutProjectionTrain": [],
            "f1WithProjectionTest": [],
            "f1WithProjectionTrain": [],
            "f1WithoutProjectionTest": [],
            "f1WithoutProjectionTrain": [],
        },
        "CNN": {
            "accuracyWithProjectionTest": [],
            "accuracyWithProjectionTrain": [],
            "accuracyWithoutProjectionTest": [],
            "accuracyWithoutProjectionTrain": [],
            "f1WithProjectionTest": [],
            "f1WithProjectionTrain": [],
            "f1WithoutProjectionTest": [],
            "f1WithoutProjectionTrain": [],
        },
    }

    for subject in range(numSubjects):
        trainFeats, trainLabels, testFeats, testLabels = load_new_dataset(
            dataset_path="Datasets/N_20_Takes_2.csv",
            subject_index=subject + 1,
            features=args.features,
        )
        for key in accuracyMetrics.keys():
            # 1. Evaluate without projection
            # Note: Your perform_traditional returns a tuple, ensure it matches this unpacking
            trainAcc, testAcc, trainF1, testF1 = perform_traditional(
                trainFeats, trainLabels, testFeats, testLabels, key=key
            )
            accuracyMetrics[key]["accuracyWithoutProjectionTest"].append(testAcc)
            accuracyMetrics[key]["accuracyWithoutProjectionTrain"].append(trainAcc)
            accuracyMetrics[key]["f1WithoutProjectionTest"].append(testF1)
            accuracyMetrics[key]["f1WithoutProjectionTrain"].append(trainF1)

            # 2. Apply Fisher Projection
            trainEigens = fisher_projection(trainFeats, trainLabels)
            trainFeatsProj = trainFeats @ trainEigens
            testFeatsProj = testFeats @ trainEigens

            # 3. Evaluate with projection
            trainAcc, testAcc, trainF1, testF1 = perform_traditional(
                trainFeatsProj, trainLabels, testFeatsProj, testLabels, key=key
            )
            accuracyMetrics[key]["accuracyWithProjectionTest"].append(testAcc)
            accuracyMetrics[key]["accuracyWithProjectionTrain"].append(trainAcc)
            accuracyMetrics[key]["f1WithProjectionTest"].append(testF1)
            accuracyMetrics[key]["f1WithProjectionTrain"].append(trainF1)

    file_id = "_".join(globalArgs.features)
    output_file = f"metrics_{file_id}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        header = "\nLOSO Mean Accuracy and F1 Score (Mean ± SD) ------------------"
        print(header)
        f.write(header + "\n")

        for name, metrics in accuracyMetrics.items():
            # Helper to format Mean ± SD
            def format_metric(data):
                return f"{np.mean(data):.2f} ± {np.std(data):.2f}"

            # Construct the block of text for this classifier
            output_block = [
                f"\nClassifier: {name}",
                f"  Accuracy (With Projection Test):  {format_metric(metrics['accuracyWithProjectionTest'])}",
                f"  Accuracy (No Projection Test):    {format_metric(metrics['accuracyWithoutProjectionTest'])}",
                f"  Accuracy (With Projection Train): {format_metric(metrics['accuracyWithProjectionTrain'])}",
                f"  Accuracy (No Projection Train):   {format_metric(metrics['accuracyWithoutProjectionTrain'])}",
                f"  F1 Score (With Projection Test):  {format_metric(metrics['f1WithProjectionTest'])}",
                f"  F1 Score (No Projection Test):    {format_metric(metrics['f1WithoutProjectionTest'])}",
                f"  F1 Score (With Projection Train): {format_metric(metrics['f1WithProjectionTrain'])}",
                f"  F1 Score (No Projection Train):   {format_metric(metrics['f1WithoutProjectionTrain'])}"
            ]

            # Join the block with newlines and print/write
            full_text = "\n".join(output_block)
            print(full_text)
            f.write(full_text + "\n")

    print(f"\n--- Results successfully saved to {output_file} ---")


def main():
    example_classification(args=globalArgs, class_info=globalClassInfo)

    # TODO: Call the classification function to perform LOSO classification
    classification(args=globalArgs)


if __name__ == "__main__":
    main()
