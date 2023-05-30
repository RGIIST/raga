# TODO: Install boto3 to work with S3 in script

import os
import subprocess
from glob import glob
import math
import shutil
import boto3
from botocore.exceptions import NoCredentialsError
import numpy as np


# Initialize S3 client
s3 = boto3.client('s3')

# Function to upload a file to S3
def upload_to_s3(local_file, bucket_name, s3_file):
    try:
        s3.upload_file(local_file, bucket_name, s3_file)
        print(f"Upload successful: {s3_file}")
    except FileNotFoundError:
        print("File not found")
    except NoCredentialsError:
        print("Credentials not available")

# Function to remove a local file
def remove_local_file(file_path):
    try:
        os.remove(file_path)
        print(f"Local file removed: {file_path}")
    except OSError as e:
        print(f"Error removing file: {file_path}. Reason: {e.strerror}")

input_base_folder = "vid_folder"
output_base_folder = "weather_out"
s3_bucket_name = "gcp-vm-data"      
folders = np.load('filepath.npy')

for idx, folder in enumerate(folders):
    if not os.path.exists('local_video_frames_folder'):
        os.mkdir('local_video_frames_folder')
    folder_name = folder.split('/')[-2]
    output_file = os.path.join(output_base_folder, f"{folder_name}_embeddings.json")
    clas = os.path.join('local_video_frames_folder', folder.split('/')[-2])
    if not os.path.exists(clas):
        os.mkdir(clas)
    
    full_path = os.path.join(clas, folder.split('/')[-1])
    try:
        os.mkdir(full_path)
    except:
        continue
    subprocess.run(['aws', 's3', 'cp', folder, full_path, '--recursive'])
    

    if (idx%500==0 and idx!=0) or idx==(len(folders)-1):
        subprocess.run(['python3', 'weather_model_embeddings.py', '--input', 'local_video_frames_folder', '--output', output_file])
        s3_file_key = f"light_metrics/embeddings/{folder_name}_{idx}_embeddings.json"
        upload_to_s3(output_file, s3_bucket_name, s3_file_key)
        remove_local_file(output_file)
        shutil.rmtree('local_video_frames_folder')




print("All embeddings generated, uploaded to S3, and local files removed.")