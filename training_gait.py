import torch.optim.adagrad
from dataset import CustomDataset
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    dataset = CustomDataset()
    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for i in range(epochs):
        for batch in dataloader:
            print(f"Processed file: {batch["study_id"]}")

            optimizer.zero_grad()
            output = model(torch.squeeze(batch["windows"]).float())
            loss = criterion(output, torch.squeeze(batch["labels"]))
            loss.backward()
            optimizer.step()
        
    torch.save(model.state_dict(), "eldernet_tuned_gait.pt")
    
if __name__=="__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    epochs = 1
    learning_rate = 0.001
    train(epochs, learning_rate)

   