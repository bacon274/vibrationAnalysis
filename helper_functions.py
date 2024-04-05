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
    Performs fft over the csv data, 
    '''
    df = pd.read_csv(path)
    
    n_bands = (fs*0.5)/band_size
    samples_per_band = 0.5*(window_size/n_bands)
    i = 0
    while i < df.size: 
        signal = df['overhang_bearing_axial'].iloc[i:i+window_size]
        T = 1/fs
        fourier = np.fft.fft(signal)
        n = signal.size
        
        freq = np.fft.fftfreq(n, d=T)
        i+=windowsize





    
# combine_csv('./330Hz')
# downsample_csv('12.288.csv', 51200, 330)

    void downSample(float *vData, uint16_t bufferSize, StaticJsonDocument<3000>& JSONdoc){
  uint16_t freq_bands = 10; // Hz range per band
  uint16_t n_bands = (samplingFrequency*0.5)/freq_bands;
  uint16_t samples_per_band = 0.5*(bufferSize/n_bands);
  double downsampledData[n_bands];

  

  for (uint16_t i = 0; i < n_bands+1; i++) {
    int frequency = i * freq_bands;
    double mag_max = 0.0;
    for (uint16_t j = i * samples_per_band; j < (i + 1) * samples_per_band; j++) {\
      if (vData[j] > mag_max) {
          mag_max = vData[j];
      }    
    }
    int ascii_int;
    char tag_string[15];
    char freq_string[10];
    ascii_int = 65 + i;
    tag_string[0] = ascii_int;
    tag_string[1] = '\0';
