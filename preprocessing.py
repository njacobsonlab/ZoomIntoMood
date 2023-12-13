import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset

def preprocess(csv_dir, save_dir):
    csv_files = os.listdir(csv_dir)
    
    int_dfs = []
    part_dfs = []

    for filename in csv_files:
        if "concat" in filename:
            file_path = os.path.join(csv_dir, filename)
            df = pd.read_csv(file_path)
            
            df_len = df.shape[0]
            df_drop_len = df.dropna().shape[0]
            missingness = (df_len - df_drop_len)/df_len
            if (missingness>0.5): 
                continue

            int_columns = [col for col in df.columns if 'int' in col]
            part_columns = [col for col in df.columns if 'part' in col]
            
            df_int = normalize_df(df[int_columns].copy())
            df_part = normalize_df(df[part_columns].copy())
            
            int_dfs.append(df_int)
            part_dfs.append(df_part)
            
            
    #  Implement more sophisticated padding/truncating method
    max_len = 4096
    
    int_tensor = [torch.tensor(df.to_numpy()) for df in int_dfs]
    part_tensor = [torch.tensor(df.to_numpy()) for df in part_dfs]
    int_tensor = torch.stack([torch.nn.functional.pad(df, (0, 0, 0, max_len - df.shape[0])) for df in int_tensor])
    part_tensor = torch.stack([torch.nn.functional.pad(df, (0, 0, 0, max_len - df.shape[0])) for df in part_tensor])
    
    indices = torch.randperm(int_tensor.size(0))
    train_indices, test_indices = train_test_split(indices, test_size=0.2)

    train_subset_int = Subset(int_tensor, train_indices)
    train_subset_part = Subset(part_tensor, train_indices)
    test_subset_int = Subset(int_tensor, test_indices)
    test_subset_part = Subset(part_tensor, test_indices)
    
    train_tensor_int = torch.stack([train_subset_int[i] for i in range(len(train_subset_int))])
    train_tensor_part = torch.stack([train_subset_part[i] for i in range(len(train_subset_part))])
    test_tensor_int = torch.stack([test_subset_int[i] for i in range(len(test_subset_int))])
    test_tensor_part = torch.stack([test_subset_part[i] for i in range(len(test_subset_part))])
    
    torch.save(train_tensor_int, f'{save_dir}train_tensor_int.pt')
    torch.save(train_tensor_part, f'{save_dir}train_tensor_part.pt')
    torch.save(test_tensor_int, f'{save_dir}test_tensor_int.pt')
    torch.save(test_tensor_part, f'{save_dir}test_tensor_part.pt')

def normalize_df(df):
    for col in df.columns:
        if col.endswith('_r'):
            df.loc[:, col] = df[col] / 5

    return df

def printt(print_text):
    # Get the current date and time
    currentDateTime = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    # Split the print_text into lines
    lines = print_text.split('\n')

    # Print the first line with the date and time
    print(f"{currentDateTime} {lines[0]}")

    # Print the remaining lines with a tab at the beginning
    for line in lines[1:]:
        print(f"\t{line}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Preprocess FAU CSVs')
    parser.add_argument('csv_dir', type=str, help='the csv directory')
    parser.add_argument('save_dir', type=str, help='the output directory to save processsed data')
    
    args = parser.parse_args()
    csv_dir = args.csv_dir
    save_dir = args.save_dir
        
    preprocess(csv_dir, save_dir)