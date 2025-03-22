import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import os

print("TensorFlow Version:", tf.__version__)

model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)
print("Model loaded successfully!")

image_url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
save_path = "E:/python/pug.jpg"

image_name = os.path.basename(save_path)

os.makedirs(os.path.dirname(save_path), exist_ok=True)

response = requests.get(image_url)
if response.status_code == 200:
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"Đã tải ảnh về: {image_name}")
else:
    print("Lỗi tải ảnh!")

image = Image.open(save_path).resize((224, 224))

plt.imshow(image)
plt.axis('off')
plt.title(f"Ảnh: {image_name}")
plt.show()

def preprocess_image(image):
    """
    Chuyển đổi ảnh về numpy array, chuẩn hóa về khoảng [0,1]
    và ép kiểu float32 để tương thích với TensorFlow.
    """
    image = np.array(image, dtype=np.float32) / 255.0  
    return image[np.newaxis, ...]    

processed_image = preprocess_image(image)
print("Image preprocessed successfully!")

predictions = model(processed_image).numpy()[0]
predicted_class = np.argmax(predictions)

print("🔎 Predicted class index:", predicted_class)

labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", labels_url)

with open(labels_path, "r") as f:
    labels = f.read().splitlines()

corrected_index = predicted_class + 1
predicted_label = labels[corrected_index] if corrected_index < len(labels) else "Unknown"

print(f"Ảnh: {image_name} | Dự đoán: **{predicted_label}**")

plt.imshow(image)
plt.title(f"Ảnh: {image_name}\nDự đoán: {predicted_label}")
plt.axis('off')
plt.show()
