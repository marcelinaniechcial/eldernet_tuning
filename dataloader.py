import os 
import pandas as pd
from torch.utils.data import Dataset
from data_parkinson_home.data_processing import make_windows
from data_parkinson_home.data_processing import make_windows_arm_labels

#gait recognition
class DatasetWalking(Dataset):
    def __init__(self):
        self.data = []

        path = os.path.dirname(os.path.abspath(__file__))
        self.file_paths = [os.path.join(path,"data_parkinson_home/processed_data_model1/pd")]

        positions = {}
        counter = 0 

        for file_path in self.file_paths:

            group = "pd"
            if file_path=="data_parkinson_home/processed_data_model1/control":
                group = "control"
            
            sorted_filenames = sorted(os.listdir(file_path))

            for f in sorted_filenames:
                # small datavbse for testing
                # files1 = [ "hbv058_LAS.parquet","hbv002_LAS.parquet","hbv058_MAS.parquet","hbv002_MAS.parquet", "hbv022_LAS.parquet", "hbv022_MAS.parquet"]
                # if str(f) in files1:
                #     pass
                # else:
                #     continue    
                study_id = f[:6]
                if positions.get(study_id, -1) == -1:
                    positions[study_id] = counter
                    self.data.append([])
                    counter +=1 

                index = positions[study_id]

                side = "LAS"
                if "MAS" in f:
                    side = "MAS"

                file = pd.read_parquet(file_path + "/" + f)
                input, labels = make_windows(file)
                sample = {"windows": input, "labels": labels, "group": group, "study_id": f[:6], "side" : side, "file" : f}
                self.data[index].append(sample) 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
#dataset for arm activities
class DatasetArmActivities(Dataset):
    def __init__(self):
        self.data = []

        path = os.path.dirname(os.path.abspath(__file__))
        self.file_paths = [os.path.join(path,"data_parkinson_home/processed_data_model2/pd")]

        positions = {}
        counter = 0 

        for file_path in self.file_paths:

            group = "pd"
            if file_path=="data_parkinson_home/processed_data_model1/control":
                group = "control"
            
            sorted_filenames = sorted(os.listdir(file_path))

            for f in sorted_filenames:
                # small datavbse for testing
                # files1 = [ "hbv058_LAS.parquet","hbv002_LAS.parquet","hbv058_MAS.parquet","hbv002_MAS.parquet", "hbv022_LAS.parquet", "hbv022_MAS.parquet"]
                # if str(f) in files1:
                #     pass
                # else:
                #     continue    
                study_id = f[:6]
                if positions.get(study_id, -1) == -1:
                    positions[study_id] = counter
                    self.data.append([])
                    counter +=1 

                index = positions[study_id]

                side = "LAS"
                if "MAS" in f:
                    side = "MAS"

                file = pd.read_parquet(file_path + "/" + f)
                input, labels = make_windows_arm_labels(file)
                sample = {"windows": input, "labels": labels, "group": group, "study_id": f[:6], "side" : side, "file" : f}
                self.data[index].append(sample) 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    







