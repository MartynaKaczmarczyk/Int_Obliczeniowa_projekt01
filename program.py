import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

archive_path = 'archive.zip'
train_path = 'Training/Training'
test_path = 'Testing/Testing'
valid_path = 'Validation/Validation'

current_directory = os.path.dirname(os.path.abspath(__file__))
archive_path = os.path.join(current_directory, archive_path)

# Rozpakowanie archiwum
with zipfile.ZipFile(archive_path, 'r') as zip_ref:
    zip_ref.extractall(current_directory)

def extract_from_path(path):
    full_path = []
    for i in sorted(os.listdir(path)):
        full_path.append(os.path.join(path, i))
    return full_path


train_images = extract_from_path(train_path)
test_images = extract_from_path(test_path)
valid_images = extract_from_path(valid_path)

print(f"The length of train_images is {len(train_images)}")
print(f"The length of test_images is {len(test_images)}")
print(f"The length of valid_images is {len(valid_images)}")