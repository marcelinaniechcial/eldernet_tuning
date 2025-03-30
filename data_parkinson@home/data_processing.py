import pandas as pd
import numpy as np
import os


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
    """Downsampling data to 30HZ

    Args:
        data (df): time-stamped data for one patient with labels

    Returns:
        data: modified data with 33HZ
    """
    return data.iloc[::3,:]


def walking_to_binary(data):
    """changes free_living_labels into binary encoding where 1 is gait and 0 is non gait

    Args:
        data (df): time-stamped data for one patient with labels

    Returns:
        data: modified data with bianry-gait column and changed column label
    """

    data['free_living_label'] = np.where(
        (data['free_living_label'] == "Walking") | 
        (data['free_living_label'] == "Running") |
        (data['free_living_label'] == "Walking upstairs") |
        (data['free_living_label'] == "Walking downstairs") |
        (data['free_living_label'] == "Turning"), 1, 0)
    data.columns = ["time", "accelerometer_x", "accelerometer_y", "accelerometer_z", "gait"]

    return data

def process(data):
    data = drop_gyroscope(data)
    data = downsample(data)
    data = walking_to_binary(data)

    return data


directory_origin = "data_parkinson@home/baseline_data"
directory_processed = "data_parkinson@home/processed_data"

#loading and processing data
for f in os.listdir(directory_origin):
    file = pd.read_parquet(directory_origin + "/" + f)
    processed = process(file)
    processed.to_parquet(directory_processed + "/" + f)

