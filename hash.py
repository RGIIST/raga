import hashlib
import numpy as np
import os
import json
import subprocess
import shutil
from tqdm import tqdm

folders = np.load('filepath.npy')
formats = (".png", ".jpg")

hash_code = {}
for idx, folder in tqdm(enumerate(folders)):
    if not os.path.exists('local_video_frames_folder'):
        os.mkdir('local_video_frames_folder')

    folder_name = folder.split('/')[-2]
    clas = os.path.join('local_video_frames_folder', folder.split('/')[-2])
    if not os.path.exists(clas):
        os.mkdir(clas)

    full_path = os.path.join(clas, folder.split('/')[-1])
    try:
        os.mkdir(full_path)
    except:
        continue
    subprocess.run(['aws', 's3', 'cp', folder, full_path, '--recursive'])

    for sub, _, files in os.walk('local_video_frames_folder'):
        for filename in files:
            if filename.endswith(formats):
                with open(os.path.join(sub, filename), "rb") as f:
                    file_data = f.read()
                hash_code[os.path.join(sub, filename)] = [4,7]#hashlib.md5(file_data).hexdigest()#.encode('utf-8', 'ignore')
                # hash_code[os.path.join(sub, filename)] = hashlib.md5(file_data).digest().decode('utf-8')
    
    shutil.rmtree('local_video_frames_folder')
    # if idx >1:
    #     break
    

with open('hash_code1.json', 'w') as f:
    json.dump(hash_code, f)