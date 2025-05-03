from sklearn.utils.class_weight import compute_class_weight
from dataset_model1 import CustomDataset
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

            print(f"Processing file: {batch["study_id"]}")

            windows = batch["windows"].squeeze(0).float()
            labels = batch["labels"].squeeze(0)

            optimizer.zero_grad()
            output = model(windows)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            print(f"Loss:{loss}")
    
    return model.state_dict()
    

   