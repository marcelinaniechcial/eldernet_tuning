from sklearn.utils.class_weight import compute_class_weight
from dataloader import DatasetWalking
from sklearn.model_selection import KFold
import json
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import os

def load_model():
    model_name = "eldernet_ft"
    repo_name = 'yonbrand/ElderNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(repo_name, model_name)
    model.train()

    return model.to(device)


def train(epochs, lr, dataset, train_idx) -> dict:
    """Main training loop

    Args:
        epochs (int): number of training epochs 
        lr (float): learning rate
        dataset (Dataset): whole dataset 
        train_idx: indicies of train set
    """
    np.random.seed(42)
    torch.manual_seed(42)
    model = load_model()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

  
    #adjusting weight to balance classes 
    #did NOT work well, overcompensated minority class

    # all_train_labels = []
    # for i in train_idx:
    #     all_train_labels.extend(dataset[i]["labels"].flatten().tolist())

    # weights = compute_class_weight(class_weight="balanced", classes=np.unique(all_train_labels), y = all_train_labels)
    # weights = weights/weights.sum()
    # weights = torch.tensor(weights,dtype=torch.float)
    # criterion = torch.nn.CrossEntropyLoss(weight=weights)

    criterion = torch.nn.CrossEntropyLoss()

    for i in range(epochs):
        print(f"Epoch {i}")

        train_idx = np.random.permutation(train_idx)
        training_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        dataloader = DataLoader(dataset, batch_size=1, sampler=training_sampler)

        for batch in dataloader:
            # each batch has 2 files for MAS and LAS
            for j in range(len(batch)):
                print(f"Processing file: {batch[j]["study_id"]}, {batch[j]["side"]}")

                windows = batch[j]["windows"].squeeze(0).float()
                labels = batch[j]["labels"].squeeze(0)

                optimizer.zero_grad()
                output = model(windows)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                print(f"Loss:{loss}")
    
    return model.state_dict()
    

def run_cross_validation(dataset, n) -> tuple:
    """This function is used for n-fold cross validation

    Args:
        dataset (Dataset): pytorch custom dataset 
        device : pytorch device setting

    """

    kf = KFold(n_splits=n,shuffle=True)
    #saves indicies of test data and number of folds
    splits = {}
    splits["folds"] = n

    path_models = os.path.join(os.path.dirname(__file__),"cross_validation_models")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):

        print(f"fold: {fold}, Training: {train_idx}, Testing: {test_idx}")

        splits[fold] = test_idx.tolist()

        parameters = train(5, 0.0001, dataset, train_idx)
        
        torch.save(parameters, f"{path_models}/{fold}.pt")

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    with open(path_splits, "w") as f:
            json.dump(splits, f) 

    print("Models sucessfully saved")

if __name__=="__main__":
    dataset = DatasetWalking()
    run_cross_validation(dataset,3)
