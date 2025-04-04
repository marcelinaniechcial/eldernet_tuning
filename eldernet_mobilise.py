
from data_parkinson_home.data_processing import make_windows
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
import os
import pandas as pd

def load_model(repo_name, model_name, device):
    model = torch.hub.load(repo_name, model_name)
    return model.to(device)

def main(input) -> torch.tensor:
    """The function uses loaded model to clasify occurance of gait

    Args:
        input (np.array): accelometer data as 10s windows 

    Returns:
        1-D tensor : prediction of gait for each window 
    """

    repo_name = 'yonbrand/ElderNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate sample data (10 seconds window, 30Hz sampling rate)
    x = torch.FloatTensor(input).to(device)

    ft_model = load_model(repo_name, 'eldernet_ft', device)
    
    with torch.no_grad():
        ft_output = ft_model(x)
    
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


def run(with_parkinson: bool) -> None:
    """This function runs model on parkinson patients (true) or control group (False).
      The function prints accuracy metrics for each file and for whole dataset

    Args:
        with_parkinson (Boolean): True for pd, False for control 
    """

    all_true = []
    all_pred = []

    if with_parkinson:
        directory = "data_parkinson_home/processed_data/pd"
    else:
        directory = "data_parkinson_home/processed_data/control"

    for f in os.listdir(directory):
        try:
            data = pd.read_parquet(directory + "/" + f)
            input, true_labels = make_windows(data)

            ft_output = main(input)
            predicted_labels = torch.argmax(ft_output, dim=1).cpu().numpy()

            all_true.extend(true_labels)
            all_pred.extend(predicted_labels)

            # printing file name and accuracy metrics
            print("Processing file: ",f)
            evaluation(true_labels,predicted_labels)
        except:
            print("Could not proccess file: ",f)
            continue

    print("Avarage:")
    evaluation(all_true,all_pred)

    
if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(42)

    #test model 
    run(True)
