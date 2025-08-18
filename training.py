import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import DatasetWalking, DatasetArmActivities
from models import Resnet, ElderNet
from classifier import Classifier


class Windows(Dataset):
    """A custom torch dataset that handles windows and labels for efficient batch loading

    Args:
        Windows (np.array): (nx3x300) 
        Labels (np.array): (n) 
    """

    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, index):
        window = torch.tensor(self.windows[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return window, label


def load_model_training(device, arm_activities_detection, use_ssl):
    """Loades torch model with choosen classifier in training mode

    Args:
        device (torch.device)
        arm_activities_detection (boolean)

    Returns:
        torch.nn.Module: loaded torch model set to train
    """

    if use_ssl:
        model_name = "eldernet_ft"
        repo_name = 'yonbrand/ElderNet'
        model = torch.hub.load(repo_name, model_name)
    else:

        feature_extractor = Resnet(
            is_eva=False, is_simclr=False, is_mtl=False)

        model = ElderNet(feature_extractor=feature_extractor.feature_extractor,
                         is_eva=True, is_simclr=False, is_mtl=False, head='fc')

    if arm_activities_detection:
        model.classifier = Classifier(50, 3)

    model.train()

    return model.to(device)


def train(fold, epochs, lr, dataset, train_idx, batch_size, device, arm_activities, use_ssl_model) -> dict:
    """ Executes the complete training loop for a single cross-validation fold,

    Args:
        fold (int): Fold number from LOO CV
        epochs (int): Number of training epochs
        lr (float): Learing rate
        dataset (torch.dataset): Custom torch dataset
        train_idx (int): Indicies contained in trained dataset  
        batch_size (int): Batch size
        device (torch.device)
        arm_activities (boolean): Detecting gait (true) or detecting gait without arm activites (false)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    model = load_model_training(device, arm_activities, use_ssl_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)

    all_train_labels = []
    all_train_windows = []

    # computing classes distribution
    for i in train_idx:
        all_train_windows.append(dataset[i][0]["windows"])
        all_train_windows.append(dataset[i][1]["windows"])

        all_train_labels.extend(dataset[i][0]["labels"].flatten().tolist())
        all_train_labels.extend(dataset[i][1]["labels"].flatten().tolist())

    all_train_windows = np.concatenate(all_train_windows, axis=0)
    all_train_windows, all_train_labels = shuffle(
        all_train_windows, all_train_labels)

    weights = compute_class_weight(class_weight="balanced", classes=np.unique(
        all_train_labels), y=all_train_labels)
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    windows_dataset = Windows(all_train_windows, all_train_labels)
    windows_loader = DataLoader(
        windows_dataset, batch_size=batch_size, shuffle=True)

    # training loop
    for i in range(epochs):

        print(f"Epoch {i}")
        loss_avg = []

        for windows, labels in windows_loader:

            windows = windows.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(windows)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_avg.append(float(loss))

        scheduler.step()

        print(f"Average loss:{sum(loss_avg)/len(loss_avg)}")

        path_models = os.path.join(os.path.dirname(
            __file__), "cross_validation_models_"+str(i+1)+"_e")
        os.makedirs(path_models, exist_ok=True)
        torch.save(model.state_dict(), f"{path_models}/{fold}.pt")

    return None


def run_cross_validation(dataset, n, epochs, batch_size, lr, device, arm_activities, use_ssl_model) -> tuple:
    """ Manages the k-fold cross-validation 

    Args:
        dataset (torch.dataset): Custom torch dataset
        n (int): Number of folds
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        lr (float): Learing rate
        device (torch.device)
        arm_activities (boolean): Detecting gait (true) or detecting gait without arm activites (false)

    """

    kf = KFold(n_splits=n, shuffle=True, random_state=42)
    splits = {}
    splits["folds"] = n

    path_splits = os.path.join(os.path.dirname(__file__), "splits.json")

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):

        print(f"Fold: {fold}")

        train(fold, epochs, lr, dataset, train_idx, batch_size,
              device, arm_activities, use_ssl_model)
        splits[fold] = test_idx.tolist()

    # saves indicies of test data and number of folds
    with open(path_splits, "w") as f:
        json.dump(splits, f)
    print("Models sucessfully saved")


if __name__ == "__main__":

    print("Is MPS avilable: ", torch.mps.is_available())
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    model = "model1"  # model1 - geit detection, model2 - gait without other arm activities
    use_ssl_model = True
    batch_size = 32
    epochs = 9
    lr = 0.0001

    if model == "model1":
        dataset = DatasetWalking()
        arm_activities = False
    else:
        dataset = DatasetArmActivities()
        arm_activities = True

    run_cross_validation(dataset, len(dataset), epochs,
                         batch_size, lr, device, arm_activities, use_ssl_model)
