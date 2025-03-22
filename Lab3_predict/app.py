from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  
    prediction = model.predict(image_array)
    prediction_class = np.argmax(prediction, axis=-1)

    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", labels_url)

    with open(labels_path, "r") as f:
        labels = f.read().splitlines()

    return jsonify({'prediction': prediction_class.tolist(),
                    'label': labels[prediction_class[0]]})

if __name__ == '__main__':
    app.run(debug=True)  
