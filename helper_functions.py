import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def downsample_csv(name, current_freq, desired_freq):
    ''' Downsamples data 
    '''
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
    '''
    Combines all csvs in a folder into one
    '''
    file_names = os.listdir(folder_path)
    df = pd.DataFrame()
    # Print the list of file names
    print("Files in directory:")
    for file_name in file_names:
        path = folder_path + "/" + file_name
        print(path)
        df_new = pd.read_csv(path)
        df =  pd.concat([df, df_new], ignore_index=True)
        print(df.shape)
    df.to_csv(f'./combined/{folder_path}.csv', index=False)


def perform_fft(fs,window_size, path, band_size):
    '''
    Performs fft over the csv data
    '''
    df = pd.read_csv(path)
    
    
    samples_per_band = 0.5*(window_size/n_bands)
    T = 1/fs
    i = 0
    while i <= df.size-windowsize: 
        signal = df['overhang_bearing_axial'].iloc[i:i+window_size]       
        fourier = np.fft.fft(signal)
        n = signal.size
        freq = np.fft.fftfreq(n, d=T)
        i+=windowsize



def bucket_fourier(band_size, fs):
    n_bands = (fs*0.5)/band_size
    for i in range(n_bands):
        frequency = i * band_size
