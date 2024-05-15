from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Załaduj model
model = tf.keras.models.load_model('model_smokers.h5')

# Funkcja do ładowania i przetwarzania obrazu
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filename = os.path.join(upload_folder, file.filename)
        file.save(filename)

        img_array = load_and_preprocess_image(filename)
        
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        print(f"Smoking with probability {prediction[0][1] * 100}")
        print(f"Non-Smoking with probability {prediction[0][0] * 100}")

        # Usunięcie pliku po przetworzeniu
        os.remove(filename)

        result = "Smoking" if predicted_label == 1 else "Non-Smoking"
        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
