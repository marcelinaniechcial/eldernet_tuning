import os 
import pandas as pd
from torch.utils.data import Dataset
from data_parkinson_home.data_processing import make_windows

class CustomDataset(Dataset):
    def __init__(self):
        self.data = []

        path = os.path.dirname(os.path.abspath(__file__))
        self.file_paths = [os.path.join(path,"data_parkinson_home/processed_data_model1/pd")]

        for file_path in self.file_paths:

            group = "pd"
            if file_path=="data_parkinson_home/processed_data_model1/control":
                group = "control"
            
            for f in os.listdir(file_path):
                side = "LAS"
                if "MAS" in f:
                    side = "MAS"

                file = pd.read_parquet(file_path + "/" + f)
                input, labels = make_windows(file)
                sample = {"windows": input, "labels": labels, "group": group, "study_id": f[:6], "side" : side }
                self.data.append(sample) 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


