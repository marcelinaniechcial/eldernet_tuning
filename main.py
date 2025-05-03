
from sklearn.model_selection import KFold
from dataset_model1 import CustomDataset
from torch.utils.data import DataLoader
from training_model1 import train
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
import os
import pandas as pd


def load_model(tuned, parameters):

    model_name = "eldernet_ft"
    repo_name = 'yonbrand/ElderNet'
    model = torch.hub.load(repo_name, model_name)

    if tuned:
        model.load_state_dict(torch.load(parameters, weights_only=True,map_location=device))
    model.eval()
    return model.to(device)

def eval(input, model) -> torch.tensor:
    """The function uses loaded model to clasify occurance of gait

    Args:
        input (np.array): accelometer data as 10s windows 
        model : loaded model which should be used to make predictions

    Returns:
        1-D tensor : prediction of gait for each window 
    """

    repo_name = 'yonbrand/ElderNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.FloatTensor(input).to(device)

    with torch.no_grad():
        ft_output = model(x)

    print(f"Fine-tuned Model Output Shape: {ft_output.shape}")

    return ft_output

def evaluation(true,predicted) -> None:
    """ This function prints model's accuracy metrics

    Args:
        true (numpy.array): true labels of dataset
        predicted (numpy.array): predicted labels of dataset
    """
    print("Accuracy: ",accuracy_score(true,predicted))
    print("Precision: ",precision_score(true,predicted))
    print("Recall: ",recall_score(true,predicted))
    print("F1: ",f1_score(true,predicted))
    print("Non-gait vs gait true: ", np.bincount(true))
    print("Non-gait vs gait model: ", np.bincount(predicted))


def test(dataset, device, file, test_idx):

    model = load_model(True, f"cross_validation_models/fold_{file}.pt") 
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
    dataloader = DataLoader(dataset, batch_size=1, sampler=test_sampler)

    accuracy = []
    sensitivity = []
    specificity = []

    for batch in dataloader:
        print(f"Testing file: {batch["study_id"] }")

        windows = (batch["windows"].squeeze(0)).float()
        y_true =  (batch["labels"].squeeze(0)).numpy()

        output = eval(windows,model)

        #model output to binary classification
        y_pred = (torch.argmax(output, dim=1).to(device)).numpy()


        #printing 
        evaluation(y_true, y_pred)

        accuracy.append(accuracy_score(y_true,y_pred))
        sensitivity.append(recall_score(y_true,y_pred))
        specificity.append(recall_score(y_true, y_pred, pos_label=0))

    return accuracy, sensitivity, specificity
    


def run(dataset, device) -> tuple:
    """This function is used for n-fold cross validation

    Args:
        dataset (Dataset): pytorch custom dataset 
        device : pytorch device setting

    """

    kf = KFold(n_splits=10,shuffle=True)

    all_acc = []
    all_sens = []
    all_spec = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"fold: {fold}, Training: {train_idx}, Testing: {test_idx}")

        parameters = train(5, 0.001, dataset, train_idx)
        
        torch.save(parameters, f"cross_validation_models/fold_{fold}.pt")

        # make array for all metrics 
        acc, sens, spec = test(dataset, device, fold, test_idx)

        all_acc.extend(acc)
        all_sens.extend(sens)
        all_spec.extend(spec)
    
    return all_acc, all_sens, all_spec
        
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    #test model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset()
    run(dataset,device)
