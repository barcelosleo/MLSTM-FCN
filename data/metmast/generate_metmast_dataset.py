import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

files_root = Path(r"/content/MLSTM-FCN/data/metmast")

file = files_root / "kegnes__1991_03_07_09_45__to__1991_06_11_09_35__with_anomaly.csv"

def load_data(file_path, multiple = False):
    df = pd.read_csv(file_path)
    date_start = file_path.stem.split("__")[1]
    date_end = file_path.stem.split("__")[3]

    date_start = datetime.strptime(date_start, "%Y_%m_%d_%H_%M")
    date_end = datetime.strptime(date_end, "%Y_%m_%d_%H_%M")

    # Split the timeseries in half
    date_mid = date_start + (date_end - date_start) / 2

    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    df = df.set_index("Date/Time")

    train_df = df.loc[date_start:date_mid]
    test_df = df.loc[date_mid:date_end].iloc[1:,:]

    # Train dataset
    X_train = train_df.iloc[:, 0:8]  # Sensor data except RH
    if multiple:
        y_train = train_df.iloc[:, 10:19]
    else:
        y_train = train_df.iloc[:, -1]

    # Test dataset
    X_test = test_df.iloc[:, 0:8]  # Sensor data except RH
    if multiple:
        y_test = test_df.iloc[:, 10:19]
    else:
        y_test = test_df.iloc[:, -1]

    np.save(file_path.parent / 'X_train.npy', X_train)
    np.save(file_path.parent / 'y_train.npy', y_train)
    np.save(file_path.parent / 'X_test.npy', X_test)
    np.save(file_path.parent / 'y_test.npy', y_test)

load_data(file_path=file, multiple=False)



