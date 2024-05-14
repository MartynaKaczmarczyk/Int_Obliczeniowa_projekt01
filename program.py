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
    name = os.path.split(df["path"][0])[0].split("/")[-1]  
    smoking = df[df['label_id'] == 1]
    print(f"Total images in {name} is {len(df)}")
    print(f"Total smoking images is {len(smoking)}")
    print(f"Total non-smoking images is {len(df)-len(smoking)}")
    sns.set_style("whitegrid")
    sns.countplot(x='label_id', data=df)
    plt.title(f"Count of smoking vs non-smoking images in {name}") 
    plt.xlabel("Label ID")
    plt.ylabel("Count")
    plt.show()  

plotCount(train_images_df)
plotCount(test_images_df)
plotCount(valid_images_df)



import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def imgPreProcessing(image, label):
    img = tf.io.read_file(image)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=(IMAGE_SIZE))
    return img, label


def loadDataset(df: pd.DataFrame):
    dataset = tf.data.Dataset.from_tensor_slices((df['path'], df['label_id']))
    return (dataset
            .map(imgPreProcessing)
            .shuffle(BATCH_SIZE * 20)
            .batch(BATCH_SIZE))


train_data = loadDataset(train_images_df)
test_data = loadDataset(test_images_df)
valid_data = loadDataset(valid_images_df)

def plotRandom(data):
    for img,label in data.take(1):
        randomNum = random.randint(0, BATCH_SIZE - 1)
        text_label = "Smoking" if label[randomNum].numpy() == 1 else "Non-Smoking"
        plt.figure(figsize=(4, 4))
        plt.imshow(img[randomNum]/255.)
        plt.title(text_label)
        plt.axis('off')
        plt.show()

plotRandom(train_data)


def create_b0_base(lr: float = 0.001) -> tf.keras.Model:
    model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
    model.trainable = False

    input_ = tf.keras.layers.Input(shape=IMAGE_SIZE + (3,), name="input layer")
    x = model(input_)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output = tf.keras.layers.Dense(2, activation="sigmoid")(x)

    base_model = tf.keras.Model(input_, output)
    base_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])
    return base_model


def plot_history(model_history):
    plt.style.use("seaborn-v0_8-whitegrid")
    df = pd.DataFrame(model_history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    loss = df['loss']
    accuracy = df['accuracy']
    val_loss = df['val_loss']  
    val_accuracy = df['val_accuracy']  
    epochs = range(len(df['val_loss']))

    ax1.plot(epochs, loss, label='training_loss')
    ax1.plot(epochs, val_loss, label='val_loss')
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_title("Loss")

    ax2.plot(epochs, accuracy, label='training_accuracy')
    ax2.plot(epochs, val_accuracy, label='val_accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.set_title("Accuracy")

    plt.tight_layout()
    plt.show()


base_eff0 = create_b0_base()
base_eff0_history = base_eff0.fit(train_data, epochs=1, validation_data=valid_data)
plot_history(base_eff0_history)
