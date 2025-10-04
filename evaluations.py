import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from dataloader import DatasetWalking, DatasetArmActivities
from classifier import Classifier


def load_trained_model(parameters, arm_activities_detection):
    """Loads torch model with chosen classifier

    Args:
        parameters (str): Saved weights of the model 
        arm_activities_detection (boolean): If True, uses 3-class model, else, 2-classes

    Returns:
        torch.nn.Module: loaded torch model
    """
    model_name = "eldernet_ft"
    repo_name = 'yonbrand/ElderNet'
    model = torch.hub.load(repo_name, model_name)

    if arm_activities_detection:
        model.classifier = Classifier(50, 3)

    model.load_state_dict(torch.load(
        parameters, weights_only=True, map_location=device))
    model.eval()

    return model.to(device)


def run_model(input, model) -> torch.tensor:
    """Uses the loaded model to classify the occurrence of gait activity.

    Args:
        input (np.ndarray): Accelerometer data as 10-second windows.
        model (torch.nn.Module): Pre-trained model

    Returns:
        torch.Tensor: Model output for each window.
"""

    x = torch.FloatTensor(input).to(device)

    with torch.no_grad():
        ft_output = model(x)

    return ft_output


def auc(probabilities, labels, plot):
    """Computes AUC and gives optional plot.

    Args:
        probabilities (np.array): Predicted probabilities for each class.
        labels (np.array): Binary labels.
        plot (bool): Whether to plot the ROC curve.

    Returns:
        float: Computed AUC score.
    """

    y_scores = np.array(probabilities)[:, 1].squeeze()
    y_true = np.array(labels)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    auc = metrics.roc_auc_score(y_true, y_scores)
    if plot:
        plt.figure()
        plt.plot(fpr, tpr, color="red", label="ROC curve tuned eldernet")
        plt.plot([0, 1], [0, 1], "go--")
        plt.title(f"ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

    return auc


def precision_recall(probabilities, labels, plot) -> None:
    """Given models outputs, the function computes 
    Precision-Recall Curve with optional plot

    Args:
        probabilities (np.array): Predicted probabilities for each class.
        labels (np.array): Binary labels.
        plot (bool): Whether to plot the Precision-Recall curve.

    Returns:
        None
    """
    y_scores = np.array(probabilities)[:, 1].squeeze()
    y_true = np.array(labels)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    if plot:
        plt.figure()
        plt.plot(recall, precision, color="red",
                 label="Precision-Recall curve")
        plt.title(f"Precision-Recall curve (Tuned Eldernet)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

    return None


def get_probabilities(dataset, fold, idx, file_name, arm_activities):
    """Gets model predictions for each file

    Args:
        dataset (torch.utils.data.Dataset): Dataset containing windowed input data and labels.
        fold (str): Fold number used to select the model.
        idx (list[int]): List of sample indices to evaluate.
        file_name (str): File name for the model checkpoint. 
        arm_activities (bool): If True, uses 3-class model, else, 2-classes.

    Returns:
        tuple(list[np.ndarray],list[np.ndarray]): Model output probabilities 
        and corresponding true labels."""

    path_model = os.path.join(os.path.dirname(__file__), file_name)

    model = load_trained_model(f"{path_model}/{fold}.pt", arm_activities)

    test_sampler = torch.utils.data.SubsetRandomSampler(idx)
    dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler)

    all_probs = []
    all_labels = []

    for subject in dataloader:
        # each subject has 2 files - MAS and LAS
        for i in range(len(subject)):

            batch = subject[i]

            windows = (batch["windows"].squeeze(0)).float()
            y_true = (batch["labels"].squeeze(0))

            output = run_model(windows, model)

            y_pred = torch.softmax(output, dim=1)

            if arm_activities:
                y_pred = y_pred[:, 2].detach().cpu().numpy()
                y_true = np.where(y_true == 2, 1, 0)
            else:
                y_pred = y_pred.detach().cpu().numpy()
                y_true = y_true.detach().cpu().numpy()

            all_probs.append(y_pred)
            all_labels.append(y_true)

    return all_probs, all_labels


def get_metrics(threshold: float, probabilities: np.array, labels: np.array, multiclass: bool) -> dict:
    """Calculates metrics for the model.

    Args:
        threshold (float): The threshold for gait gait classification.
        probabilities (np.array): Probabilities of each class.
        labels (np.array): Labels for each window. 
        multiclass (bool): If True, uses 3-class model, else, 2-classes.

    Returns:
        dict: Dictionary of metrics.
    """

    results = {}

    if multiclass:
        predictions = np.where(probabilities > threshold, 1, 0)
    else:
        predictions = np.where(probabilities[:, 1] > threshold, 1, 0)

    results["accuracy"] = accuracy_score(labels, predictions)
    results["recall"] = recall_score(labels, predictions)
    results["specificity"] = recall_score(labels, predictions, pos_label=0)
    results["confusion_matrix"] = confusion_matrix(
        labels, predictions).tolist()
    results["f1"] = f1_score(labels, predictions)

    if multiclass:
        results["auc"] = metrics.roc_auc_score(labels, probabilities)
        results["accuracy_all_classes"] = accuracy_score(labels, predictions)
        results["recall_all_classes"] = recall_score(
            labels, predictions, average="macro")
        results["f1_all_classes"] = f1_score(
            labels, predictions, average="macro")

    else:
        results["auc"] = auc(probabilities, labels, False)

    return results


def run_evaluations(file_name: str, multiclass: bool):
    """ Runs cross-validation evaluations and saves results to JSON

    Args:
        file_name (str): Model weights file name.
        multiclass (bool): If True, uses 3-class model, else, 2-classes.

    """

    path_splits = os.path.join(os.path.dirname(__file__), "splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    # if not multiclass:
    with open("threshold.json", "r") as f:
        threshold = json.load(f)

    k = data["folds"]

    all_probabilities = []
    all_labels = []
    results = {}

    for fold in range(k):
        test_idx = data[str(fold)]
        for index in test_idx:

            probabilities, labels = get_probabilities(
                dataset, fold, [index], file_name, multiclass)
            threshold = threshold[str(fold)]

            results1 = get_metrics(
                threshold, probabilities[0], labels[0], multiclass)
            results1["id"] = dataset.data[index][0]["file"]

            results2 = get_metrics(
                threshold, probabilities[1], labels[1], multiclass)
            results2["id"] = dataset.data[index][1]["file"]

            results[fold] = [results1, results2]

            # flattening
            all_probabilities.extend(np.concatenate(probabilities).tolist())
            all_labels.extend(np.concatenate(labels).tolist())

    with open("results.json", "w") as file:
        json.dump(results, file)
    if not multiclass:
        auc(all_probabilities, all_labels, True)
        precision_recall(all_probabilities, all_labels, True)


def set_specificity_threshold(file_name: str, arm_activities: bool) -> None:
    """
    Args:
        file_name (string): Name of directory with tuned models.
        arm_activities (bool): If True, uses 3-class model, else, 2-classes.
    """

    path_splits = os.path.join(os.path.dirname(__file__), "splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    k = data["folds"]

    all_threshold = {}

    for fold in range(k):
        test_idx = data[str(fold)]
        train_idx = [i for i in range(len(dataset)) if i not in test_idx]

        probabilities, labels = get_probabilities(
            dataset, fold, train_idx, file_name, arm_activities)

        if arm_activities:
            probabilities = np.concatenate(probabilities)[:, :, 2]
            labels = np.concatenate(labels)[:, :, 2].tolist()
            y_scores = probabilities[:, 1].squeeze()
        else:
            y_scores = np.concatenate(probabilities)[:, 1]
            labels = np.concatenate(labels).tolist()

        y_true = np.array(labels)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        specificity = 1-fpr

        pass_tresh = np.where(specificity >= 0.95)
        max_tpr = np.argmax(tpr[pass_tresh])
        threshold = thresholds[max_tpr]

        all_threshold[int(fold)] = float(threshold)

    print("Threshold sucessfully saved")

    path_splits = os.path.join(os.path.dirname(__file__), "threshold.json")

    with open(path_splits, "w") as file:
        json.dump(all_threshold, file)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # test model
    device = torch.device(
        "cuda" if torch.backends.cuda.is_available() else
        "mps" if torch.backends.mps.is_available()
        else "cpu")

    # CONFIGS
    model = "model1"  # model1 - geit detection, model2 - gait without other arm activities
    file_name = 'cross_validation_models_3_e'  # directory with models

    if model == "model1":
        dataset = DatasetWalking()
        set_specificity_threshold(file_name, False)
        run_evaluations(file_name, False)

    elif model == "model2":
        dataset = DatasetArmActivities()
        set_specificity_threshold(file_name, True)
        run_evaluations(file_name, True)
