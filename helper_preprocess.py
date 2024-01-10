def apply_windowing_funtion(data, window_size, time_step, ):
    pass

def generate_features(data, window_size, time_step):
    features = ["max", "min", "mean", "std", "rms", "maxgradient", "zero_crossings"]
    output_data = None
    
    for feature in features:
        output_data[f"{feature}"] = apply_windowing_funtion(data, window_size, time_step, feature)
    
    return output_data