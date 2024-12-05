import os
import time
from datetime import datetime


def create_directory(base_name, root_dir='.'):
    timestamp = datetime.now().strftime("%m_%d_%Y")
    base_path = os.path.join(root_dir, f"{timestamp}_{base_name}")

    index = 1
    dir_path = f"{base_path}_{index}"
    while os.path.exists(dir_path):
        if not os.listdir(dir_path):
            return dir_path
        index += 1
        dir_path = f"{base_path}_{index}"

    os.makedirs(dir_path)
    return dir_path