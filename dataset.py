import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_parkinson_home.data_processing import make_windows

class CustomDataset(Dataset):
    def __init__(self):
        self.data = []
        self.file_paths = ["data_parkinson_home/processed_data/pd", "data_parkinson_home/processed_data/control"]

        for file_path in self.file_paths:

            group = "pd"
            if file_path=="data_parkinson_home/processed_data/control":
                group = "control"

            for f in os.listdir(file_path):
                file = pd.read_parquet(file_path + "/" + f)
                input, labels = make_windows(file)
                sample = {"windows": input, "labels": labels, "group": group, "study_id": f}
                self.data.append(sample) 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def my_collate_fn(batch):
        windows_all = []
        labels_all = []
        groups = []
        study_id_all = []

        for sample in batch:
            windows_all.append(sample["windows"])
            labels_all.append(sample["labels"])
            groups.append(sample["group"]*len(labels_all))
            study_id_all.append(sample["study_id"]*len(labels_all))

        all_windows = torch.cat(all_windows, dim=0)  
        all_labels = torch.cat(all_labels, dim=0)

        new_data = {"windows": windows_all, "labels": labels_all, "group": groups, "study_id": study_id_all}

        return new_data
