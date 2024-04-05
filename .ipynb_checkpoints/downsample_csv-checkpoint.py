import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def downsample_csv(name, current_freq, desired_freq):
    path = f'./normal/normal/{name}'
    # Load CSV file into a DataFrame
    df = pd.read_csv(path, header=None)
    column_names = [
        'tachometer_signal',  # Column 1
        'underhang_bearing_axial', 'underhang_bearing_radial', 'underhang_bearing_tangential',  # Columns 2 to 4
        'overhang_bearing_axial', 'overhang_bearing_radial', 'overhang_bearing_tangential',  # Columns 5 to 7
        'microphone'  # Column 8
    ]
    df.columns = column_names
    ratio = int(current_freq/desired_freq)
    downsampled_df = df.iloc[::ratio]
    print(downsampled_df.head())
    print(f'shape: {downsampled_df.shape}')
    downsampled_df.to_csv(f'./330Hz/{name}', index=False)



def combine_csv(folder_path):
    # List all files in the directory
    file_names = os.listdir(directory_path)
    df = pd.DataFrame()
    # Print the list of file names
    print("Files in directory:")
    for file_name in file_names:
        path = folder_path + "/" + file_name
        print(path)
        df_new = pd.read_csv(path, header=None)
        df = df.join(df_new, how='inner')
        print(df.shape)

    
combine_cvs('./330Hz')
# downsample_csv('12.288.csv', 51200, 330)

    