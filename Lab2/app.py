from flask import Flask, request, jsonify, render_template
import os
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HUGGINGFACE_TOKEN = "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"

def detect_text(image_path):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-base-stage1"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    with open(image_path, "rb") as image:
        response = requests.post(API_URL, headers=headers, data=image)
    
    return response.json() if response.status_code == 200 else {"error": response.text}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không tìm thấy tệp tin'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Tên tệp tin không hợp lệ'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        text_result = detect_text(filepath)
        os.remove(filepath)

        return jsonify({'prediction': text_result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
