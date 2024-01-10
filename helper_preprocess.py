import pandas as pd
import numpy as np

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
