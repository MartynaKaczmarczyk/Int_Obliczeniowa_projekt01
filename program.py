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


def createDataFrame(image_paths):
    labels = [0 if 'notsmoking' in os.path.basename(path) else 1 for path in image_paths]
    names = ['notsmoking' if 'notsmoking' in os.path.basename(path) else 'smoking' for path in image_paths]
    df = pd.DataFrame({'path': image_paths, 'label': names,'label_id': labels})
    return df



train_images_df = createDataFrame(train_images)
test_images_df = createDataFrame(test_images)
valid_images_df = createDataFrame(valid_images)

def plotCount(df):
    name = os.path.split(df["path"][0])[0].split("/")[-1]  # Poprawiona linia
    smoking = df[df['label_id'] == 1]
    print(f"Total images in {name} is {len(df)}")
    print(f"Total smoking images is {len(smoking)}")
    print(f"Total non-smoking images is {len(df)-len(smoking)}")
    sns.set_style("whitegrid")
    sns.countplot(x='label_id', data=df)
    plt.title(f"Count of smoking vs non-smoking images in {name}")  # Dodanie tytułu wykresu
    plt.xlabel("Label ID")
    plt.ylabel("Count")
    plt.show()  # Wyświetlenie wykresu

plotCount(train_images_df)
plotCount(test_images_df)
plotCount(valid_images_df)