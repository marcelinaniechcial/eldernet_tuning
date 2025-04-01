
from data_parkinson_home.data_processing import make_windows
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import torch
import os
import pandas as pd

def load_model(repo_name, model_name, device):
    model = torch.hub.load(repo_name, model_name)
    return model.to(device)

def main(input):
    repo_name = 'yonbrand/ElderNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate sample data (10 seconds window, 30Hz sampling rate)
    x = torch.FloatTensor(input).to(device)
    
    ft_model = load_model(repo_name, 'eldernet_ft', device)
    
    # Generate outputs
    with torch.no_grad():
        ft_output = ft_model(x)
    
    # print(f"SSL Model Output Shape: {ssl_output.shape}")
    print(f"Fine-tuned Model Output Shape: {ft_output.shape}")
    return ft_output

def evaluation(true,predicted):
    print("Accuracy: ",accuracy_score(true,predicted))
    print("Precision: ",precision_score(true,predicted))
    print("Recall: ",recall_score(true,predicted))

def run(with_parkinson):

    all_true = []
    all_pred = []

    if pd:
        directory = "data_parkinson_home/processed_data/pd"
    else:
        directory = "data_parkinson_home/processed_data/control"

    for f in os.listdir(directory):
        data = pd.read_parquet(directory + "/" + f)
        input, true_labels = make_windows(data)
        ft_output = main(input)
        predicted_labels = torch.argmax(ft_output, dim=1).numpy()

        all_true.extend(true_labels)
        all_pred.extend(predicted_labels)

        print(f)
        evaluation(true_labels,predicted_labels)

    print("Avarage:")
    evaluation(all_true,all_pred)

        
    return None

if __name__ == "__main__":
    run(True)
