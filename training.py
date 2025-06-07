from sklearn.utils.class_weight import compute_class_weight
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import DatasetWalking
from dataloader import DatasetArmActivities
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch import nn
import torch
import os

class Windows(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels 

    def __len__(self):
        return self.windows.shape[0]
    
    def __getitem__(self, index):
        window = torch.tensor(self.windows[index], dtype=torch.float32).to(device)
        label = torch.tensor(self.labels[index], dtype=torch.long).to(device)
        return window,label


class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


def load_model_training(device, arm_activities_detection):
    model_name = "eldernet_ft"
    repo_name = 'yonbrand/ElderNet'
    model = torch.hub.load(repo_name, model_name)

    if arm_activities_detection:
        model.classifier = Classifier(50,3)

    model.train()

    return model.to(device)

def train(fold, epochs, lr, dataset, train_idx, batch_size, device) -> dict:
    """Main training loop

    Args:
        epochs (int): number of training epochs 
        lr (float): learning rate
        dataset (Dataset): whole dataset 
        train_idx: indicies of train set
    """
    np.random.seed(42)
    torch.manual_seed(42)
    model = load_model_training(device, True)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=5e-5)

    all_train_labels = []
    all_train_windows = []

    for i in train_idx:
        all_train_windows.append(dataset[i][0]["windows"])
        all_train_windows.append(dataset[i][1]["windows"])

        all_train_labels.extend(dataset[i][0]["labels"].flatten().tolist())
        all_train_labels.extend(dataset[i][1]["labels"].flatten().tolist())

    all_train_windows = np.concatenate(all_train_windows,axis=0)
    all_train_windows, all_train_labels = shuffle(all_train_windows,all_train_labels, random_state=42)

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(all_train_labels), y = all_train_labels)
    weights = torch.tensor(weights,dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)

    windows_dataset = Windows(all_train_windows,all_train_labels)
    windows_loader = DataLoader(windows_dataset, batch_size=batch_size, shuffle=True)

    for i in range(epochs):
        print(f"Epoch {i}")
        loss_avg = []

        for windows,labels in windows_loader:
            
            optimizer.zero_grad()
            output = model(windows)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_avg.append(float(loss))
            
        scheduler.step()

        print(f"Loss:{sum(loss_avg)/len(loss_avg)}")

        #saving after 15 epochs
        if i==14 or i==19 or i==24 or i == 29 or i==32 or i==34:
            path_models = os.path.join(os.path.dirname(__file__),"cross_validation_models_"+str(i+1)+"_e")
            os.makedirs(path_models,exist_ok=True)
            torch.save(model.state_dict(), f"{path_models}/{fold}.pt")

    return None
    

def run_cross_validation(dataset, n, epochs, batch_size, lr, device) -> tuple:
    """This function is used for n-fold cross validation

    Args:
        dataset (Dataset): pytorch custom dataset 
        device : pytorch device setting

    """

    kf = KFold(n_splits=n,shuffle=True, random_state=42)
    #saves indicies of test data and number of folds
    splits = {}
    splits["folds"] = n

    path_splits = os.path.join(os.path.dirname(__file__),"splits.json")

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        splits[fold] = test_idx.tolist()

    with open(path_splits, "w") as f:
        json.dump(splits, f)



    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
 
        print(f"fold: {fold}, Training: {train_idx}, Testing: {test_idx}")

        train(fold, epochs, lr, dataset, train_idx, batch_size, device)


  
        
    print("Models sucessfully saved")

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU",torch.cuda.is_available())
    # dataset = DatasetWalking()
    dataset = DatasetArmActivities()
    batch_size = 32
    epochs = 35
    lr = 0.0001
    run_cross_validation(dataset, len(dataset), epochs, batch_size, lr, device)
