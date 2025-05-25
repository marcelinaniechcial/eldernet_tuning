

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from dataloader import DatasetWalking
from torch.utils.data import DataLoader
from training import train
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import numpy as np
import torch
import json


def load_trained_model( parameters):

    model_name = "eldernet_ft"
    repo_name = 'yonbrand/ElderNet'
    model = torch.hub.load(repo_name, model_name)
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

def get_probabilities(dataset, fold, idx):

    path_model = os.path.join(os.path.dirname(__file__),"cross_validation_models")

    model = load_trained_model(f"{path_model}/{fold}.pt") 

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

def get_metrics(treshold, probabilities, labels):

    results = {}
    predictions = np.where(probabilities[:,1]>treshold,1,0)

    results["accuracy"] =  accuracy_score(labels, predictions)
    results["recall"] = recall_score(labels,predictions)
    results["specificity"] = recall_score(labels, predictions, pos_label=0)
    results["confusion__matrix"] =  confusion_matrix(labels, predictions).tolist()
    results["f1"] = f1_score(labels,predictions)
    results["auc"] = auc(probabilities,labels, False)
    
    return results



def run_evaluations():

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)
    
    with open("tresholds.json", "r") as f:
        tresholds = json.load(f)

    k = data["folds"] 

    all_probabilities = []
    all_labels = []
    results = {}

    for fold in range(k):
        test_idx = data[str(fold)] 
        for index in  test_idx:
            # print("fold",fold,",Indexes to test ",test_idx,"Current",index)
            treshold = tresholds[str(fold)]
            probabilities, labels = get_probabilities(dataset, fold, [index])
            results1 = get_metrics(treshold, probabilities[0], labels[0])
            results1["id"] = dataset.data[index][0]["file"]

            results2 = get_metrics(treshold, probabilities[1], labels[1])
            results2["id"] = dataset.data[index][1]["file"]

            results[fold] = [results1,results2]

            #flattening 
            all_probabilities.extend(np.concatenate(probabilities).tolist())
            all_labels.extend(np.concatenate(labels).tolist())

    with open("results.json", "w") as file:
        json.dump(results, file)

    auc(all_probabilities,all_labels,True)
    precision_recall(all_probabilities,all_labels,True)


def set_treshold():

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    k = data["folds"] 

    all_tresholds = {}

    for fold in range(k):
        test_idx = data[str(fold)]
        train_idx = [i for i in range(len(dataset)) if i not in test_idx]

        probabilities, labels = get_probabilities(dataset, fold, train_idx)


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


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    #test model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DatasetWalking()

    set_treshold()
    run_evaluations()


