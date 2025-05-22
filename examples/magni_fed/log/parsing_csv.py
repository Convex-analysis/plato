import os
import pandas as pd

def merge_accuracy_columns(folder_path):
    # List to hold dataframes
    dataframes = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            if 'accuracy' in df.columns:
                dataframes.append(df[['accuracy']])

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, axis=1)

    # Create the output file path
    output_file = os.path.join(folder_path, os.path.basename(folder_path) + '_merged.csv')

    # Save the merged dataframe to a CSV file
    merged_df.to_csv(output_file, index=False)
    return output_file


def add_average_column(file_path, output_file):
        # Read the CSV file
        file_name = os.path.join(file_path, output_file)
        df = pd.read_csv(file_name)
        
        # Calculate the row-wise average
        df['average'] = df.mean(axis=1)
        
        # Save the updated dataframe to a new CSV file
        df.to_csv(output_file, index=False)


file_path = "C:\\Users\\79944\\Desktop\\temp_config\\data\\MacroS1_Cifar10"
# Example usage
file = merge_accuracy_columns(file_path)
add_average_column(file_path, file)