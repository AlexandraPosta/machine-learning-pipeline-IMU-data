import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, butter


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def low_pass_filter(data):
    # Sampling time; the frequency of the data is the difference 
    # between consecutive time-stamps
    ts = data["Timestamp"].diff().median()  # Median sampling time
    fs = 1000/ts                            # Sampling frequency
    N = 3;                                  # Order of the Low Pass Filter (trial and error)
    cutoff = 25.0;                          # Cutoff frequency of the filter (trial and error)
    lpf_data = butter_lowpass_filter(data, cutoff, fs, N)
    return lpf_data


def rms(x):
    return np.sqrt(np.mean(x**2))


def max_gradient(x):
    return np.max(np.gradient(x))


def zero_crossings(x):
    return ((x[:-1] * x[1:]) < 0).sum()


# Define features
features = {
    'max': np.max,
    'min': np.min,
    'mean': np.mean,
    'std': np.std,
    'rms': rms,
    'maxgradient': max_gradient,
    'zero_crossings': zero_crossings
}


def generate_features(data, window_size, time_step):    
    results = {}

    # Apply the rolling windowing function to each column
    for column in data.columns:
        for feature_name, feature_func in features.items():
            results[f'{column}_{feature_name}'] = data[column].rolling(window=window_size).apply(feature_func, raw=True)
    
    # Combine results into a new DataFrame
    return pd.DataFrame(results)
