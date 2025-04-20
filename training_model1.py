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

def train(epochs, lr):
    """Main training loop

    Args:
        epochs (int): number of training epochs 
        lr (float): learning rate
    """
    model = load_model()
    dataset = CustomDataset()
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

    test = ["hbv014_MAS","hbv014_LAS","hbv072_LAS","hbv072_MAS"]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for i in range(epochs):
        for batch in dataloader:
            
            # # training only on PD 
            # if batch["group"]=="control":
            #     continue

            print(f"Processed file: {batch["study_id"]}")
            
            if batch["study_id"] in test:
                continue

            optimizer.zero_grad()
            output = model(torch.squeeze(batch["windows"]).float())
            loss = criterion(output, torch.squeeze(batch["labels"]))
            loss.backward()
            optimizer.step()
        
    torch.save(model.state_dict(), "eldernet_tuned_gait.pt")
    print("Model sucessfully saved")
    
if __name__=="__main__":

    np.random.seed(42)
    torch.manual_seed(42)

    epochs = 1
    learning_rate = 0.001
    train(epochs, learning_rate)

   