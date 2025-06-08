

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from dataloader import DatasetWalking, DatasetArmActivities
from torch.utils.data import DataLoader
from training import train
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import numpy as np
from classifier import Classifier
import torch
import json


def load_trained_model(parameters, arm_activities_detection):
    model_name = "eldernet_ft"
    repo_name = 'yonbrand/ElderNet'
    model = torch.hub.load(repo_name, model_name)
    
    if arm_activities_detection:
        model.classifier = Classifier(50,3)

    model.load_state_dict(torch.load(parameters, weights_only=True,map_location=device))

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

    print(f"Fine-tuned Model Output Shape: {ft_output.shape}")

    return ft_output

def auc(probabilities,labels, plot):

    y_scores = np.array(probabilities)[:,1].squeeze()
    y_true = np.array(labels)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    auc = metrics.roc_auc_score(y_true,y_scores)
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
    y_scores = np.array(probabilities)[:,1].squeeze()
    y_true = np.array(labels)

    precision, recall, thresholds = precision_recall_curve(y_true,y_scores)

    if plot:
        plt.figure()  
        plt.plot(recall, precision, color="red", label="Precision-Recall curve")
        plt.title(f"Precision-Recall curve (Tuned Eldernet)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

    return None

def get_probabilities(dataset, fold, idx, arm_activities):

    path_model = os.path.join(os.path.dirname(__file__),"cross_validation_models")

    model = load_trained_model(f"{path_model}/{fold}.pt",arm_activities) 

    test_sampler = torch.utils.data.SubsetRandomSampler(idx)
    dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler)

    all_probs = []
    all_labels = []

    for subject in dataloader:
        # each subject has 2 files - MAS and LAS
        for i in range(len(subject)):

            batch = subject[i]
            # print(f"Testing file: {batch["study_id"]}, {batch["side"]}")

            windows = (batch["windows"].squeeze(0)).float()
            y_true =  (batch["labels"].squeeze(0)).numpy()

            output = run_model(windows,model)

            #model output to binary classification

            y_pred = (torch.nn.functional.softmax(output, dim=1).to(device)).cpu().numpy().squeeze()
    
            all_probs.append(y_pred)
            all_labels.append(y_true)
        
    return all_probs, all_labels

def get_metrics(treshold, probabilities, labels, multiclass):

    results = {}
    if multiclass:

        predictions = np.argmax(probabilities,axis=1)
        # print(list(predictions).count(0),list(predictions).count(1),list(predictions).count(2))
        # print(list(labels).count(0),list(labels).count(1),list(labels).count(2))

        results["accuracy"] =  accuracy_score(labels, predictions)
        results["recall"] = recall_score(labels,predictions, average="macro")
        results["f1"] = f1_score(labels,predictions, average="macro")

        # checking scores of gait without other arm activities
        pred_temp = np.where(predictions==2,1,0)
        labels_temp = np.where(labels==2,1,0)

        results["accuracy_without"] =  accuracy_score(labels_temp, pred_temp)
        results["recall_without"] = recall_score(labels_temp, pred_temp)
        results["f1_without"] = f1_score(labels_temp, pred_temp)
        print(probabilities[:,2].shape)
        results["auc_without"] = metrics.roc_auc_score(labels_temp, probabilities[:,2])
        results["specificity_without"] = recall_score(labels_temp, pred_temp, pos_label=0)



    if not multiclass:
        predictions = np.where(probabilities[:,1]>treshold,1,0)
        results["accuracy"] =  accuracy_score(labels, predictions)
        results["recall"] = recall_score(labels,predictions)
        results["specificity"] = recall_score(labels, predictions, pos_label=0)
        results["confusion__matrix"] =  confusion_matrix(labels, predictions).tolist()
        results["f1"] = f1_score(labels,predictions)
        results["auc"] = auc(probabilities,labels, False)
        
    return results


def run_evaluations(multiclass):

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)
    
    if not multiclass:
        with open("tresholds_f1.json", "r") as f:
            tresholds = json.load(f)

    k = data["folds"] 

    all_probabilities = []
    all_labels = []
    results = {}

    for fold in range(k):
        test_idx = data[str(fold)] 
        for index in  test_idx:
            
            probabilities, labels = get_probabilities(dataset, fold, [index], multiclass)
            
            if multiclass:
                results1 = get_metrics(None, probabilities[0], labels[0], True)
                results1["id"] = dataset.data[index][0]["file"]

                results2 = get_metrics(None, probabilities[1], labels[1], True)
                results2["id"] = dataset.data[index][1]["file"]

                results[fold] = [results1,results2]

            if not multiclass:
                treshold = tresholds[str(fold)]
                
                results1 = get_metrics(treshold, probabilities[0], labels[0], False)
                results1["id"] = dataset.data[index][0]["file"]

                results2 = get_metrics(treshold, probabilities[1], labels[1], False)
                results2["id"] = dataset.data[index][1]["file"]

                results[fold] = [results1,results2]

            #flattening 
            all_probabilities.extend(np.concatenate(probabilities).tolist())
            all_labels.extend(np.concatenate(labels).tolist())

    with open("results.json", "w") as file:
        json.dump(results, file)
    if not multiclass:
        auc(all_probabilities,all_labels,True)
        precision_recall(all_probabilities,all_labels,True)


def set_treshold_spe(arm_activities):

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    k = data["folds"] 

    all_tresholds = {}

    for fold in range(k):
        test_idx = data[str(fold)]
        train_idx = [i for i in range(len(dataset)) if i not in test_idx]

        probabilities, labels = get_probabilities(dataset, fold, train_idx, arm_activities)


        probabilities = np.concatenate(probabilities).tolist()
        labels = np.concatenate(labels).tolist()
        
        y_scores = np.array(probabilities)[:,1].squeeze()
        y_true = np.array(labels)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        specificity = 1-fpr

        pass_tresh = np.where(specificity>=0.95)
        max_tpr = np.argmax(tpr[pass_tresh])
        threshold = thresholds[max_tpr]
   
        all_tresholds[int(fold)] = float(threshold)
    
    print("threshold sucessfully saved")

    path_splits = os.path.join(os.path.dirname(__file__),"tresholds.json")


    with open(path_splits, "w") as file:
        json.dump(all_tresholds, file)

def set_treshold_f1(arm_activities):
    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    k = data["folds"] 

    all_tresholds = {}

    for fold in range(k):
        test_idx = data[str(fold)]
        train_idx = [i for i in range(len(dataset)) if i not in test_idx]

        probabilities, labels = get_probabilities(dataset, fold, train_idx, arm_activities)
        probabilities = np.concatenate(probabilities).astype(float)
        labels = np.concatenate(labels).astype(float)
        
        # maximising f1
        best = 0.5
        for threshold in np.linspace(0,1,100):
            predictions = (probabilities[:,1]>threshold).astype(int)
            current_score = f1_score(labels,predictions)
            if current_score>best:
                best = current_score
                best_thr = threshold
        print("fold:",best_thr)
        all_tresholds[int(fold)] = float(best_thr)
    
    print("threshold sucessfully saved")

    path_splits = os.path.join(os.path.dirname(__file__),"tresholds_f1.json")


    with open(path_splits, "w") as file:
        json.dump(all_tresholds, file)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    #test model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = "model2"

    if model == "model1":
        dataset = DatasetWalking()
        # set_treshold_spe()
        # set_treshold_f1(False)  
        run_evaluations(False)

    elif model == "model2":
        dataset = DatasetArmActivities() 
        run_evaluations(True)


