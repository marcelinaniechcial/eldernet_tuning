

import os, sys
# temporary solution
sys.path.append(os.getcwd())
import pandas as pd
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from sklearn import metrics
from eldernet_tuning.dataloader import CustomDataset
from torch.utils.data import DataLoader
from eldernet_tuning.training_model1 import train
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
import json


def load_model(tuned, parameters):

    model_name = "eldernet_ft"
    repo_name = 'yonbrand/ElderNet'
    model = torch.hub.load(repo_name, model_name)

    if tuned:
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

def auc(probabilities,labels):

    y_scores = np.array(probabilities)[:,1].squeeze()
    print(y_scores.shape)
    y_true = np.array(labels)
    print(y_true.shape)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)

    plt.figure()  
    plt.plot(fpr, tpr, color="red", label="ROC curve tuned eldernet")
    plt.plot([0, 1], [0, 1], "go--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
    


def get_probabilities(dataset, fold, test_idx):

    path_model = os.path.join(os.path.dirname(__file__),"cross_validation_models")

    model = load_model(True, f"{path_model}/{fold}.pt") 

    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler)

    all_probs = []
    all_labels = []

    for batch in dataloader:
        print(f"Testing file: {batch["study_id"] }")

        windows = (batch["windows"].squeeze(0)).float()
        y_true =  (batch["labels"].squeeze(0)).numpy()

        output = run_model(windows,model)

        #model output to binary classification

        y_pred = (torch.nn.functional.softmax(output, dim=1).to(device)).numpy().squeeze()

        all_probs.extend(y_pred)
        all_labels.extend(y_true)
    
    return all_probs, all_labels

def run_evaluations():

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "r") as f:
        data = json.load(f)

    k = data["folds"] 

    all_probabilities = []
    all_labels = []

    for fold in range(k):
        test_idx = data[str(fold)] 
        probabilities, labels = get_probabilities(dataset, fold, test_idx)

        all_probabilities.extend(probabilities)
        all_labels.extend(labels)

    auc(all_probabilities,all_labels)





def run_cross_validation(dataset, n_splits) -> tuple:
    """This function is used for n-fold cross validation

    Args:
        dataset (Dataset): pytorch custom dataset 
        device : pytorch device setting

    """

    kf = GroupKFold(n_splits=n_splits,shuffle=True)
    groups = [d["study_id"] for d in dataset]

    #saves indicies of test data and number of folds
    splits = {}
    splits["folds"] = n_splits

    path_models = os.path.join(os.path.dirname(__file__),"cross_validation_models")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, groups=groups)):
        print(f"fold: {fold}, Training: {train_idx}, Testing: {test_idx}")

        splits[fold] = test_idx.tolist()

        parameters = train(5, 0.0001, dataset, train_idx)
        
        torch.save(parameters, f"{path_models}/{fold}.pt")

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "w") as f:
            json.dump(splits, f) 


        
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    #test model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset()
    run_cross_validation(dataset,5)
    run_evaluations()


