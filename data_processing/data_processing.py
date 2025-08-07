from scipy.interpolate import CubicSpline
import pandas as pd
import numpy as np
import os

######################################################################################################
# Details about data can be found here: https://data.ru.nl/collections/di/dcmn/DSC_pdhasq_t0000123a_971
######################################################################################################


def pd_recognition(data: pd.DataFrame) -> pd.DataFrame:
    """The function distinguishes between PD and control group before pre-processing the data. 

    Args:
        data (df): time-stamped data with labels

    Returns:
        Boolean : True if PD, False if control 
    """
    if "pre_or_post" in data:
        return True
    return False


def drop_gyroscope(data: pd.DataFrame, to_drop: list) -> pd.DataFrame:
    """Dropping unnecessery data

    Args:
        data (df): time-stamped data for one patient with labels

    Returns:
        data (df): data without columns to_drop
    """

    columns = [column for column in list(data) if column in to_drop]

    return data.drop(columns=columns)


def downsample(data: pd.DataFrame, label_column) -> pd.DataFrame:
    """Downsampling data to 30HZ by interpolation for float values and nearest label for string values (free_living_label)

    Args:
        data (df): time-stamped data with labels

    Returns:
        data: modified data with 30HZ
    """
    accelerometer = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
    cs = CubicSpline(data["time"], data[accelerometer])

    sampled_time = np.arange(data["time"].iloc[0], data["time"].iloc[-1], 1/30)

    sampled_data = pd.DataFrame(cs(sampled_time), columns=accelerometer)
    sampled_data.insert(0, "time", sampled_time)
    sampled_data = pd.merge_asof(
        sampled_data, data[["time", label_column]], on="time", direction="nearest")

    return sampled_data


def walking_to_binary(data: pd.DataFrame) -> pd.DataFrame:
    """changes free_living_labels into binary encoding where 1 is gait and 0 is non gait

    Args:
        data (df): time-stamped data for one person with labels

    Returns:
        data: modified data with bianry-gait column and changed column label
    """

    data['free_living_label'] = np.where(
        (data['free_living_label'] == "Walking"), 1, 0)

    data.columns = ["time", "accelerometer_x",
                    "accelerometer_y", "accelerometer_z", "gait"]

    return data


def arm_activities_labeling(data: pd.DataFrame) -> pd.DataFrame:
    """changes free_living_labels into binary encoding where 1 is gait and 0 is non gait

    Args:
        data (df): time-stamped data for one person with labels

    Returns:
        data: modified data with bianry-gait column and changed column label
    """

    data['arm_label'] = np.where(
        (data['arm_label'] == "non_gait"), 0,
        np.where((data['arm_label'] == "Gait without other behaviours or other positions"), 1, 2))

    data.columns = ["time", "accelerometer_x", "accelerometer_y",
                    "accelerometer_z", "gait_arms_activities"]

    return data


def make_windows(data: pd.DataFrame) -> pd.DataFrame:
    """The function splits accelometer data into 300 samples (10s) windows and normalising data
    Each window is labeled 1 if threshold for gait is passed (acceptance_parameter) and 0 otherwise. 


    Args:
        data (): _description_

    Returns:
        windows: array of size (number_of_windows, 3, 300) 
        labels: array of size (number_of_windows) . It includes labels indicating gait for each window (1 or 0)
    """
    windows = []
    labels = []
    acceptance_parameter = 0.5
    accelerometer = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
    data[accelerometer] = (
        data[accelerometer]-data[accelerometer].mean())/data[accelerometer].std()
    input = data[accelerometer].values
    output = data["gait"].values

    for i in range(0, data.shape[0]-300, 300):

        window = input[i:i+300, :].T

        if sum(output[i:i+300]) >= 300*acceptance_parameter:
            labels.append(1)
        else:
            labels.append(0)

        windows.append(window)

    windows = np.array(windows)
    labels = np.array(labels)

    return windows, labels


def make_windows_arm_labels(data: pd.DataFrame) -> pd.DataFrame:
    """The function splits accelometer data into 300 samples (10s) windows and normalising data
    Each window is labeled 1 if threshold for gait is passed (acceptance_parameter) and 0 otherwise. 


    Args:
        data (): _description_

    Returns:
        windows: array of size (number_of_windows, 3, 300) 
        labels: array of size (number_of_windows)
        0 - no gait, 1 - gait with other arm acitivites, 2 - gait without other arm activities
    """
    windows = []
    labels = []
    acceptance_parameter = 0.5
    accelerometer = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
    data[accelerometer] = (
        data[accelerometer]-data[accelerometer].mean())/data[accelerometer].std()
    input = data[accelerometer].values
    output = data["gait_arms_activities"].values

    for i in range(0, data.shape[0]-300, 300):

        if np.count_nonzero(output[i:i+300] == 2) >= 300*acceptance_parameter:
            labels.append(2)
        elif np.count_nonzero(output[i:i+300] == 1) >= 300*acceptance_parameter:
            labels.append(1)
        else:
            labels.append(0)

        window = input[i:i+300, :].T
        windows.append(window)

    windows = np.array(windows)
    labels = np.array(labels)

    return windows, labels


def process_model1(data: pd.DataFrame) -> pd.DataFrame:
    """This funtion takes raw data and returns processed data. 
    Processing includes dropping uneccessery features, downsampling and one-hot encoding and normalisation 

    Args:
        data (df):  time-stamped data for one person with labels

    Returns:
        df: processed data
    """
    to_drop = ["gyroscope_x", "gyroscope_y", "gyroscope_z",
               "arm_label", "pre_or_post", "tremor_label"]
    label_column = "free_living_label"

    data = drop_gyroscope(data, to_drop)
    data = downsample(data, label_column)
    data = walking_to_binary(data)
    return data


def process_model2(data: pd.DataFrame) -> pd.DataFrame:
    """This funtion takes raw data and returns processed data. 
    Processing includes dropping uneccessery features, downsampling and one-hot encoding and normalisation 

    Args:
        data (df):  time-stamped data for one person with labels

    Returns:
        df: processed data
    """

    to_drop = ["gyroscope_x", "gyroscope_y", "gyroscope_z",
               "free_living_label", "pre_or_post", "tremor_label"]
    label_column = "arm_label"
    data = drop_gyroscope(data, to_drop)
    data = downsample(data, label_column)
    data = arm_activities_labeling(data)
    return data


directory_origin = "data_processing/baseline_data"

directory_processed_controls_model1 = "ddata_processing/processed_data_model1/control"
directory_processed_pd_model1 = "data_processinge/processed_data_model1/pd"

directory_processed_pd_model2 = "data_processing/processed_data_model2/pd"

# view example file

# temp = pd.read_parquet(directory_processed_pd_model2 + "/" + "hbv002_LAS.parquet")
# print(temp.head())


# loading and processing data
if __name__ == "__main__":

    for f in os.listdir(directory_origin):

        file = pd.read_parquet(directory_origin + "/" + f)

        if pd_recognition(file):
            directory = directory_processed_pd_model2
        else:
            # no controls for arm label detection
            continue

        processed_model2 = process_model2(file)
        processed_model2.to_parquet(directory + "/" + f)
