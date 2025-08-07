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
    """Loades torch model with choosen classifier in evaluation mode

    Args:
        parameters (str): saved weights of the model 
        arm_activities_detection (boolean): detecting gait (true) or detecting gait without arm activites (false)

    Returns:
        torch.nn.Module: loaded torch model set to train
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
    """The function uses loaded model to clasify occurance of gait

    Args:
        input (np.array): accelometer data as 10s windows 
        model : loaded model which should be used to make predictions

    Returns:
        1-D tensor : prediction of gait for each window 
    """

    repo_name = 'yonbrand/ElderNet'

    x = torch.FloatTensor(input).to(device)

    with torch.no_grad():
        ft_output = model(x)

    return ft_output


def auc(probabilities, labels, plot):
    """_summary_

    Args:
        probabilities (_type_): _description_
        labels (_type_): _description_
        plot (_type_): _description_

    Returns:
        _type_: _description_
    """

    y_scores = np.array(probabilities)[:, 1].squeeze()
    y_true = np.array(labels)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
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


def precision_recall(probabilities, labels, plot):
    """_summary_

    Args:
        probabilities (): 
        labels (_type_): _description_
        plot (_type_): _description_

    Returns:
        _type_: _description_
    """
    y_scores = np.array(probabilities)[:, 1].squeeze()
    y_true = np.array(labels)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

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
    """_summary_

    Args:
        dataset (_type_): _description_
        fold (_type_): _description_
        idx (_type_): _description_
        arm_activities (_type_): _description_

    Returns:
        _type_: _description_
    """

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
            y_true = (batch["labels"].squeeze(0)).numpy()

            output = run_model(windows, model)

            # binary classification
            y_pred = (torch.nn.functional.softmax(
                output, dim=1).to(device)).cpu().numpy().squeeze()
            if arm_activities:
                y_pred = y_pred[:, 2]
                y_true = np.where(y_true == 2, 1, 0)

            all_probs.append(y_pred)
            all_labels.append(y_true)

    return all_probs, all_labels


def get_metrics(treshold, probabilities, labels, multiclass):
    """_summary_

    Args:
        treshold (): _description_
        probabilities (np.array): probabilities of each class (n,m) where m={2,3} depending on configs 
        labels (np.array): labeles for each window (n,)
        multiclass (boolean): detecting gait (true) or detecting gait without arm activites (false)

    Returns:
        _type_: _description_
    """

    results = {}
    print("here", probabilities.shape, labels.shape)

    if multiclass:
        predictions = np.where(probabilities > treshold, 1, 0)
    else:
        predictions = np.where(probabilities[:, 1] > treshold, 1, 0)

    results["accuracy"] = accuracy_score(labels, predictions)
    results["recall"] = recall_score(labels, predictions)
    results["specificity"] = recall_score(labels, predictions, pos_label=0)
    results["confusion__matrix"] = confusion_matrix(
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


def run_evaluations(file_name, multiclass):
    """_summary_

    Args:
        file_name (_type_): _description_
        multiclass (_type_): _description_
    """

    path_splits = os.path.join(os.path.dirname(__file__), "splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    # if not multiclass:
    with open("tresholds.json", "r") as f:
        tresholds = json.load(f)

    k = data["folds"]

    all_probabilities = []
    all_labels = []
    results = {}

    for fold in range(k):
        test_idx = data[str(fold)]
        for index in test_idx:

            probabilities, labels = get_probabilities(
                dataset, fold, [index], file_name, multiclass)
            treshold = tresholds[str(fold)]

            results1 = get_metrics(
                treshold, probabilities[0], labels[0], multiclass)
            results1["id"] = dataset.data[index][0]["file"]

            results2 = get_metrics(
                treshold, probabilities[1], labels[1], multiclass)
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


def set_treshold_spe(file_name, arm_activities):
    """

    Args:
        file_name (string): name of directory with tuned models
        arm_activities (Boolean): detecting gait (true) or detecting gait without arm activites (false)
    """

    path_splits = os.path.join(os.path.dirname(__file__), "splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    k = data["folds"]

    all_tresholds = {}

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

        all_tresholds[int(fold)] = float(threshold)

    print("threshold sucessfully saved")

    path_splits = os.path.join(os.path.dirname(__file__), "tresholds.json")

    with open(path_splits, "w") as file:
        json.dump(all_tresholds, file)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # test model
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("GPU: ", torch.mps.is_available())

    # CONFIGS
    model = "model1"  # model1 - geit detection, model2 - gait without other arm activities
    file_name = 'cross_validation_models'  # directory with models

    if model == "model1":
        dataset = DatasetWalking()
        set_treshold_spe(file_name, False)
        run_evaluations(file_name, False)

    elif model == "model2":
        dataset = DatasetArmActivities()
        set_treshold_spe(file_name, True)
        run_evaluations(file_name, True)
