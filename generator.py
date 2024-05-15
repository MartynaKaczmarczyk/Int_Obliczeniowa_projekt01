from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
import pandas as pd

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
batch_size = 32

filenames = os.listdir("data/Training/Training")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if "smoking" in category:
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


df["category"] = df["category"].replace({0: 'non-smoking', 1: 'smoking'})
model2 = tf.keras.models.load_model('model_smokers.h5')
from sklearn.model_selection import train_test_split
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    df,
    "data/Training/Training",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "data/Training/Training",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    "data/Training/Training",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)




import matplotlib.pyplot as plt

x_batch, y_batch = next(train_generator)

random_index = np.random.randint(len(x_batch))
image = x_batch[random_index]
label = y_batch[random_index]

import os
import numpy as np
from PIL import Image

file_path = "image.jpg"
image2 = Image.fromarray((image * 255).astype(np.uint8))
image2.save(file_path)

plt.imshow(image)
plt.axis('off')
plt.show()



