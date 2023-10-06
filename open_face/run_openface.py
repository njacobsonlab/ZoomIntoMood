import argparse
import sys
import time
import os
import logging
import subprocess
import shutil
import datetime
import pandas as pd
import numpy as np

def execute():
    # Define paths based on input parameters
    pid_result_dir = out_dir+pid+"/"
    if int_bool: 
        final_result_dir = pid_result_dir + "interviewer/"
        df_csv_file = pid_result_dir + "int_" + pid + "_openface.csv"
        wip_df_csv_file = pid_result_dir + "WIP_int_" + pid + "_openface.csv"
    else: 
        final_result_dir = pid_result_dir + "participant/"
        df_csv_file = pid_result_dir + "part_" + pid + "_openface.csv"
        wip_df_csv_file = pid_result_dir + "WIP_part_" + pid + "_openface.csv"
        
    # Check if files have already been processed
    wip = os.path.exists(wip_df_csv_file)
    done = os.path.exists(df_csv_file)
    
    if done:
        return

    # Create necessary directories if they don't already exist
    if not wip:
        create_dir(pid_result_dir)
        create_dir(final_result_dir)
        
    # Process images and save results
    loop_through_imgs(pid_result_dir, final_result_dir, wip_df_csv_file, df_csv_file, wip)

def loop_through_imgs(result_dir, final_result_dir, df_csv_file, final_df_csv_file, wip):    
    # Construct path to current image directory based on whether it's an interviewer or participant image directory
    pid_in_dir = in_dir + pid + "/"
    if int_bool: 
        curr_img_dir = pid_in_dir + "interviewer_images/"
    else: 
        curr_img_dir = pid_in_dir + "participant_images/"

    # Initialize DataFrame to None and set starting index based on whether the work-in-progress csv exists
    df = None
    if wip:
        start_ind = pd.read_csv(df_csv_file).index[-1] + 1
    else:
        start_ind = 0
        
    # Get the images to process in the current image directory & sort files by integer value of x
    file_prefix = "int_" if int_bool else "part_"
    file_suffix = ".jpg"
    img_files = [
        f
        for f in os.listdir(curr_img_dir)
        if f.startswith(file_prefix + pid) and f.endswith(file_suffix)
        and int(f.split("_")[-1].split(".")[0]) >= start_ind
    ]
    sorted_files = sorted(img_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
    
    for img_name in sorted_files:
        img_file = curr_img_dir + img_name 
        i = int(img_file.split("_")[-1].split(".")[0])
        
        # If image exits
        if os.path.exists(img_file):
            # Run bash script which executes OpenFace command FaceLandmarkImg in a singularity image shell on the current image file and catch any exceptions that occur
            # Binds the home directory so that OpenFace has access to all files
            command = "singularity exec -B . ./openface.sif ./helper_files/run_openface.sh"
            command += " " + img_file + " " + final_result_dir
            
            try:
                subprocess.run(command, shell=True)
            except Exception as e:
                log_to_file(error_log, f'Error running openface: {e}')
                
                
            # Append the processed results to the DataFrame and check if tracking has failed
            df = append_lists(final_result_dir, img_file, df)
                
            # Append the current DataFrame to the CSV file every 500 images
            if (not i % 500):
                df = append_csvs(i, df, df_csv_file)
    
    # Append the final DataFrame to the CSV file and rename the file, marking it complete
    if df is not None:
        df = append_csvs(i, df, df_csv_file)
    os.rename(df_csv_file, final_df_csv_file)    
    
def append_csvs(i, df, df_csv_file):
    # Convert the dataframe to a new dataframe with the first row as column headers and 'idx' as index
    df_formatted = pd.DataFrame(df[1:], columns = df[0]).set_index('idx')
    
    # Check if the CSV file already exists, and append to it if it does, else create a new file with the data
    if (os.path.exists(df_csv_file)):
        df_formatted.to_csv(df_csv_file, mode='a', header=False)
    else:
        df_formatted.to_csv(df_csv_file)
    
    # Reset the dataframe
    df = None
    return df
    
def append_lists(final_result_dir, img_file, df):
    # Initialize image info
    processed_dir = final_result_dir + "processed/"
    img_file_name = img_file.split("/")[-1].split(".")[0]
    n_img = img_file_name.split("_")[-1]
    processed_csv = processed_dir+img_file_name+".csv"

    # Determine the current index based on int_bool value and image file information
    if int_bool: 
        curr_index = "int_" + pid + "_" + n_img
    else: 
        curr_index = "part_" + pid + "_" + n_img
    
    # If the data frame is empty, initialize with column names from the sample CSV file
    if df is None:
        df = [['idx']+sample_csv.columns.tolist()[1:]]
    
    try:
        # If the processed CSV file does not exist, create an empty row with current index and append to data frame
        if not os.path.exists(processed_csv):
            row_df = pd.DataFrame(None, index=[0], columns=df[0])
            row_df['idx'] = curr_index
            df.append(row_df.values.tolist()[0])
            
        # Otherwise, append the existing row(s) from the processed CSV to the data frame
        else:
            row_df = pd.read_csv(processed_csv).drop(columns=['face'])
            row_df.insert(0,"idx",[curr_index])
            df.append(row_df.values.tolist()[0])
    except Exception as e:
        log_to_file(error_log, f'Error appending row: {e}')

    # Remove the processed files
    empty_directory(processed_dir)

    # Return the updated data frame
    return df

def empty_directory(directory_path):
    try:
        # Ensure the directory exists
        if os.path.exists(directory_path):
            # Remove all files and subdirectories inside the directory
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    except Exception as e:
        print(f"Error: {str(e)}")
        log_to_file(error_log, f'Error emptying directory: {e}')

def create_dir(dir_name):
    # Split the path into individual directories
    dirs = os.path.abspath(dir_name).split(os.path.sep)
    created = False
    
    if os.path.exists(dir_name):
        created = False
    else:
        # Loop through each directory and ensure it exists
        for i in range(1, len(dirs)):
            subdir = os.path.sep.join(dirs[:i+1])
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        if os.path.exists(dir_name):
            created = True


def log_to_file(log_file_name, *args):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if the log file exists, and create it if it doesn't
    if not os.path.exists(log_file_name):
        with open(log_file_name, 'w') as new_log_file:
            pass  # Create an empty file
    
    with open(log_file_name, 'a') as log_file:
        log_file.write(current_time + ": " + " ".join(str(arg) for arg in args) + "\n")

if __name__ == "__main__":
    global in_dir, out_dir, log_dir, pid, int_bool, info_log, error_log, sample_csv

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process an image with OpenFace')

    # Define the command-line arguments
    parser.add_argument('in_dir', type=str, help='the input directory')
    parser.add_argument('out_dir', type=str, help='the output directory')
    parser.add_argument('log_dir', type=str, help='the log directory')
    parser.add_argument('pid', type=str, help='the participant ID')
    parser.add_argument('--int', action='store_true', help='the interviewer flag')
    
    # Parse the command-line arguments & set global variables
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    log_dir = args.log_dir
    int_bool = args.int
    pid = args.pid
    
    # Set log filenames
    info_log = log_dir + pid + "_info.txt"
    error_log = log_dir + pid + "_error.txt"
    
    sample_csv = pd.read_csv('./helper_files/sample.csv')
        
    try:
        execute()
    except Exception as e:
        log_to_file(error_log, f'Error executing: {e}')