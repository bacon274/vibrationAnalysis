import traceback
import os
import pandas as pd
import numpy as np


def load_data(csv_path):
    '''
    Load the data from the CSV file.
    '''
    # Check if the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    
    # Load the data into a DataFrame
    df = pd.read_csv(csv_path)
    
    return df

def window_data(df, window_size=128):
    windows = [df.iloc[i:i+window_size] for i in range(0, len(df), window_size) if len(df.iloc[i:i+window_size]) == window_size]
    return windows
def extract_features(window):
    '''
    Extract features from a single window of data (Series).
    '''
    # Time domain features
    features = {
        'mean': window.mean(),
        'std': window.std(),
        'max': window.max(),
        'min': window.min(),
        'median': window.median(),
        'var': window.var(),
        'range': window.max() - window.min(),
        'skew': window.skew(),
        'kurtosis': window.kurtosis(),
        'rms': (window**2).mean()**0.5,
        'energy': (window**2).sum(),
        'peak_to_peak': window.max() - window.min(),
        'zero_crossing_rate': ((window.diff().fillna(0) * window.shift(-1).fillna(0)) < 0).sum() / len(window),
    }

    # Frequency domain features
    fs = 500  # Sampling frequency in Hz
    X = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), d=1/fs)
    mag = np.abs(X)
    mag_sum = mag.sum()
    if mag_sum != 0:
        spectral_centroid = (mag * freqs).sum() / mag_sum
        spectral_bandwidth = np.sqrt(((freqs - spectral_centroid) ** 2 * mag).sum() / mag_sum)
        spectral_rolloff = freqs[np.where(np.cumsum(mag) >= 0.85 * mag_sum)[0][0]]
        dominant_frequency = freqs[np.argmax(mag)]
        power_band_0_50Hz = mag[(freqs >= 0) & (freqs < 50)].sum()
        power_band_50_100Hz = mag[(freqs >= 50) & (freqs < 100)].sum()
        power_band_100_250Hz = mag[(freqs >= 100) & (freqs < 250)].sum()
    else:
        spectral_centroid = 0
        spectral_bandwidth = 0
        spectral_rolloff = 0
        dominant_frequency = 0
        power_band_0_50Hz = 0
        power_band_50_100Hz = 0
        power_band_100_250Hz = 0

    features.update({
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'dominant_frequency': dominant_frequency,
        'power_band_0_50Hz': power_band_0_50Hz,
        'power_band_50_100Hz': power_band_50_100Hz,
        'power_band_100_250Hz': power_band_100_250Hz,
    })

    print(f"Extracted features: {features}")
    return pd.Series(features)

if __name__ == "__main__":
        try:
            # Load the data
            fault_data = load_data('./data/combined/fault_underhang_35g_bearing_500Hz.csv')
            fault_data.attrs['name'] = 'fault'
            normal_data = load_data('./data/combined/normal_500Hz.csv')
            normal_data.attrs['name'] = 'normal'
            column_names = [ 'tachometer_signal', 'underhang_bearing_radial', 'underhang_bearing_tangential','overhang_bearing_axial', 'overhang_bearing_radial', 'overhang_bearing_tangential', 'microphone']
            # Window the data
            fault_windows = window_data(fault_data['underhang_bearing_radial'], window_size=128)
            normal_windows = window_data(normal_data['underhang_bearing_radial'], window_size=128)
            # Extract features from each window
            fault_features = pd.DataFrame([extract_features(window) for window in fault_windows])
            normal_features = pd.DataFrame([extract_features(window) for window in normal_windows])
            fault_features['class'] = 1
            normal_features['class'] = 0
            features = pd.concat([fault_features, normal_features], ignore_index=True)
            features.to_csv('./data/features.csv', index=False)
            normal_features = pd.DataFrame([extract_features(window) for window in normal_windows])
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()  # Print the full traceback