import glob
import pandas as pd

def load_data(folder_path):
    file_list = glob.glob(folder_path + "/*.dat") 
    files =[]

    for i in range(1, len(file_list)): 
        df = pd.read_csv(file_list[i], delimiter='\t', header=None) 

        # Rename columns header to sensor location and sensor type
        new_headers = [f"{df.iloc[0][j]}_{df.iloc[1][j]}"
                       .replace("Wide Range Accelerometer", "Accelerometer")
                       .replace(" ", "_") 
                       for j in range(0, len(df.columns))]
        df.columns = new_headers
        df = df.drop([0, 1, 2, 3])
        df.reset_index(drop=True, inplace=True)

        # Convert all columns to numerical type
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        files.append(df)

    return files