import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, butter


def low_pass_filter(ts, data):
    # Filter coefficients
    fs = 1000/ts            # Sampling frequency
    N = 3;                  # Order of the Low Pass Filter (trial and error)
    cutoff = 25.0;          # Cutoff frequency of the filter (trial and error)
    
    # Apply filter
    b, a = butter(N, cutoff, fs=fs, btype='low', analog=False)
    return lfilter(b, a, data)


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
    # Window parameters
    ts = data["Timestamp"].diff().median()      # Median sampling time
    stride = int(time_step/ts)                  # Stride length
    output = pd.DataFrame()

    for feature_name, feature_func in features.items():
        for col in data.columns:
            if feature_name == "maxgradient":
                output[f"{col}_{feature_name}"] = data[col].rolling(window_size).apply(feature_func)[::stride]
            else:
                output[f"{col}_{feature_name}"] = data[col].rolling(window_size, min_periods=1).apply(feature_func)[window_size-1::stride]

    return output
