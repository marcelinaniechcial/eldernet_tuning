import os
import pandas as pd
from torch.utils.data import Dataset
from data_processing.data_processing import make_windows, make_windows_arm_labels

########################################################################################################
# Details about data can be found here: https://data.ru.nl/collections/di/dcmn/DSC_pdhasq_t0000123a_971
########################################################################################################


class DatasetWalking(Dataset):
    """
    Torch dataset with labeled arm activities. 

        Where:
        0 - no gait
        1 - gait 

    """

    def __init__(self):
        self.data = []

        path = os.path.dirname(os.path.abspath(__file__))
        self.file_paths = [os.path.join(
            path, "data_processing/processed_data_model1/pd")]

        positions = {}
        counter = 0

        for file_path in self.file_paths:

            group = "pd"
            if file_path == "data_processing/processed_data_model1/control":
                group = "control"

            sorted_filenames = sorted(os.listdir(file_path))

            for f in sorted_filenames:
                study_id = f[:6]
                if positions.get(study_id, -1) == -1:
                    positions[study_id] = counter
                    self.data.append([])
                    counter += 1

                index = positions[study_id]

                side = "LAS"
                if "MAS" in f:
                    side = "MAS"

                file = pd.read_parquet(file_path + "/" + f)
                input, labels = make_windows(file)
                sample = {"windows": input, "labels": labels,
                          "group": group, "study_id": f[:6], "side": side, "file": f}
                self.data[index].append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Dataset with labeled arm activities


class DatasetArmActivities(Dataset):
    """Torch dataset with labeled arm activities. 

        Where:
        0 - no gait
        1 - gait with other arm acitivites (e.g: holding objects)
        2 - gait without other arm activities (i.e pure gait segments)

    """

    def __init__(self):
        self.data = []

        path = os.path.dirname(os.path.abspath(__file__))
        self.file_paths = [os.path.join(
            path, "data_processing/processed_data_model2/pd")]

        positions = {}
        counter = 0

        for file_path in self.file_paths:

            group = "pd"
            if file_path == "data_processing/processed_data_model2/control":
                group = "control"

            sorted_filenames = sorted(os.listdir(file_path))

            for f in sorted_filenames:
                study_id = f[:6]
                if positions.get(study_id, -1) == -1:
                    positions[study_id] = counter
                    self.data.append([])
                    counter += 1

                index = positions[study_id]

                side = "LAS"
                if "MAS" in f:
                    side = "MAS"

                file = pd.read_parquet(file_path + "/" + f)
                input, labels = make_windows_arm_labels(file)
                sample = {"windows": input, "labels": labels,
                          "group": group, "study_id": f[:6], "side": side, "file": f}
                self.data[index].append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
