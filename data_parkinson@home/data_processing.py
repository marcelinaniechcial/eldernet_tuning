from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
import os

def pd_recognition(data):

    if "pre_or_post" in data:
        return True
    return False


def drop_gyroscope(data):
    """Dropping unnecessery data

    Args:
        data (df): time-stamped data for one patient with labels

    Returns:
        data (df): data without columns to_drop
    """

    to_drop = ["gyroscope_x","gyroscope_y","gyroscope_z","arm_label","pre_or_post","tremor_label"]
    columns = [column for column in list(data) if column in to_drop]
    
    return data.drop(columns=columns)


def downsample(data):
    """Downsampling data to 30HZ by interpolation for float values and nearest label for string values (free_living_label)

    Args:
        data (df): time-stamped data for one patient with labels

    Returns:
        data: modified data with 30HZ
    """
    accelerometer = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
    cs = CubicSpline(data.iloc[:, 0], data[accelerometer])

    sampled_time = np.arange(data["time"].iloc[0], data["time"].iloc[-1], 1/30)

    sampled_data = pd.DataFrame(cs(sampled_time),columns=accelerometer)
    sampled_data.insert(0, "time", sampled_time)
    sampled_data = pd.merge_asof(sampled_data, data[["time", "free_living_label"]], on="time", direction="nearest")

    return sampled_data


def walking_to_binary(data):
    """changes free_living_labels into binary encoding where 1 is gait and 0 is non gait

    Args:
        data (df): time-stamped data for one patient with labels

    Returns:
        data: modified data with bianry-gait column and changed column label
    """

    data['free_living_label'] = np.where(
        (data['free_living_label'] == "Walking"), 1, 0)
    data.columns = ["time", "accelerometer_x", "accelerometer_y", "accelerometer_z", "gait"]

    return data

def make_windows(data):
    """The function splits accelometer data into 300 samples (10s) windows. 
    Each window is labeled 1 if tthreshold for gait is passed (acceptance_parameter) and 0 otherwise. 
    

    Args:
        data (df): _description_

    Returns:
        windows: array of size (number_of_windows, 3, 300) 
        labels: array of size (number_of_windows) . It includes labels indicating gait for each window (1 or 0)
    """
    windows = []
    labels = []
    acceptance_parameter = 0.9
    accelerometer = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]

    input = data[accelerometer].values
    output = data["gait"].values

    for i in range(0,data.shape[0]-300,300):

        windows.append(input[i:i+300,:].T)

        if sum(output[i:i+300])>=300*acceptance_parameter:
            labels.append(1)
        else:
            labels.append(0)
            
    windows = np.array(windows)
    labels = np.array(labels)


    return windows,labels

def process(data):
    data = drop_gyroscope(data)
    data = downsample(data)
    data = walking_to_binary(data)

    return data


directory_origin = "data_parkinson@home/baseline_data"

directory_processed_controls = "data_parkinson@home/processed_data/control"
directory_processed_pd = "data_parkinson@home/processed_data/pd"

#view example file
# temp = pd.read_parquet(directory_origin  + "/" + "hbv053_LAS.parquet")arquet")
# print(temp.head())


#loading and processing data
for f in os.listdir(directory_origin):

    file = pd.read_parquet(directory_origin + "/" + f)

    if pd_recognition(file):
        directory = directory_processed_pd
    else: 
        directory = directory_processed_controls

    processed = process(file)
    processed.to_parquet(directory + "/" + f)

