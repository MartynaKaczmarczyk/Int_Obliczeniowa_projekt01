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

def augmentation(image, label):
    img = tf.image.random_flip_left_right(image, seed=SEED)
    img = tf.image.random_brightness(img, 0.1, seed=SEED)
    img = tf.image.random_contrast(img, 0.2, 0.5, seed=SEED)
    img = tf.image.random_saturation(img, .5, 1, seed=SEED)
    img = tf.image.random_hue(img, 0.2, seed=SEED)
    return img, label

def loadDatasetWithAugmentation(df: pd.DataFrame):
    dataset = tf.data.Dataset.from_tensor_slices((df['path'], df['label_id']))
    return (dataset
            .map(imgPreProcessing)
            .map(augmentation)
            .shuffle(BATCH_SIZE * 20)
            .batch(BATCH_SIZE)
            )

data_size = len(train_images_df)
train_data_aug_20 = loadDatasetWithAugmentation(train_images_df.sample(frac=1)[:int(0.25 * data_size)])
plotRandom(train_data_aug_20)


# create a checkpoint callback to save the model
checkpoint_path = "25_percent_augmented/checkpoint.weights.h5"

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,  # set to False to save the entire model
                                                         save_best_only=True,  # save only the best model weights instead of a model every epoch
                                                         save_freq="epoch",  # save every epoch
                                                         verbose=1)
base_eff_modelOne = create_b0_base()
initial_epochs = 5
# compile and fit
base_eff_modelOne.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(),
                          metrics=["accuracy"])
base_eff_modelOne_history = base_eff_modelOne.fit(
    train_data_aug_20, epochs=initial_epochs,
    validation_data=valid_data,
    callbacks=[checkpoint_callback]
)



plot_history(base_eff_modelOne_history)
results_aug_25 = base_eff_modelOne.evaluate(test_data)
print(results_aug_25)
# Load the weights from the pre-trained model
base_eff_modelOne.load_weights(checkpoint_path)
base_loaded_weights = base_eff_modelOne.evaluate(test_data)
print(base_loaded_weights)

# create a new instance of the base model with lower learning rate
base_eff_modelTwo = create_b0_base(lr = 0.0001)
# Load the weights from the previous model chekpoint
base_eff_modelTwo.load_weights(checkpoint_path)

# Evaluate the test results to make sure they are same
base_eff_modelTwo.evaluate(test_data)

base_eff_modelTwo.summary()
print("Total trainable parameters in the model ", len(base_eff_modelTwo.trainable_variables))

for _,layer in enumerate(base_eff_modelTwo.layers):
    print("Layer no : ",_,"Trainable : ",layer.trainable, "Layer Name : ", layer.name,)

base_eff_modelTwo_base = base_eff_modelTwo.layers[1]


#==============================================================
from PIL import Image
from io import BytesIO
import requests


def plot_and_predict(url, img_shape=224):
    #     Download and preprocess
    response = requests.get(url)
    image_data = BytesIO(response.content)
    image = Image.open(image_data)
    image = image.convert("RGB")
    image = image.resize((img_shape, img_shape))
    image_array = np.array(image)

    #     Make predictions
    img = np.expand_dims(image_array, axis=0)
    prediction = base_eff_modelOne.predict(img, verbose=0)
    predicted_label = np.argmax(prediction)

    print(f"Smoking with probability {prediction[0][1] * 100}")
    print(f"Non-Smoking with probability {prediction[0][0] * 100}")

    plt.imshow(image_array)
    plt.axis('off')
    plt.show

plot_and_predict('https://img.freepik.com/free-photo/young-man-smoking_144627-29295.jpg')
plot_and_predict('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQZ2FnaBLHoNCw4OM00db5ahJdvs_LXEo45OQ&usqp=CAU')

