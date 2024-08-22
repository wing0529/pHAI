from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import torch
from arch import deep_wb_single_task
from arch import deep_wb_model
from sklearn.preprocessing import RobustScaler
import utilities.utils as utls
from utilities.deepWB import deep_wb
import os
from PIL import Image
import pandas as pd
import joblib
import numpy as np
import json
import logging

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Configuration
UPLOAD_FOLDER = './tmp/uploads'
MODEL_DIR = './models'
OUTPUT_DIR = './output'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_DIR'] = MODEL_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_scaler():
    try:
        rbs = RobustScaler()
        X_train = pd.read_csv('output_data.csv')[['R', 'G', 'B']]
        X_train_robust = rbs.fit_transform(X_train)
        svm_model = joblib.load('best_svm.joblib')
        return svm_model, rbs
    except FileNotFoundError as e:
        logging.error(f"Error loading model or scaler: {e}")
        return None, None

def convert_to_image(array):
    if isinstance(array, np.ndarray):
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)
    raise TypeError("Provided object is not a numpy array")

def extract_central_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Provided object is not a PIL.Image")

    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    return image.getpixel((center_x, center_y))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        task = request.form.get('task', 'AWB').lower()
        device_option = request.form.get('device_option', 'cpu')
        save_output = request.form.get('save_output', 'true') == 'true'

        device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')

        if 'input_image' not in request.files or not allowed_file(request.files['input_image'].filename):
            return "No file part or invalid file type", 400

        file = request.files['input_image']
        img = Image.open(file)
        name = os.path.splitext(file.filename)[0]

        # Model loading
        if task == 'awb':
            net_awb = deep_wb_single_task.deepWBnet() if os.path.exists(os.path.join(app.config['MODEL_DIR'], 'net_awb.pth')) else deep_wb_model.deepWBNet()
            net_awb.to(device)
            net_awb.load_state_dict(torch.load(os.path.join(app.config['MODEL_DIR'], 'net_awb.pth'), map_location=device), strict=False)
            net_awb.eval()

            # Image processing
            out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device)
            out_awb_image = convert_to_image(out_awb) if isinstance(out_awb, np.ndarray) else out_awb

            if save_output:
                os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
                result_image_path = os.path.join(app.config['STATIC_FOLDER'], name + '_AWB.png')
                out_awb_image.save(result_image_path)

            rgb = extract_central_rgb(out_awb_image)
            R, G, B = rgb

            svm_model, rbs = load_model_and_scaler()
            if svm_model and rbs:
                new_data = pd.DataFrame({'R': [R], 'G': [G], 'B': [B]})
                new_data_robust = rbs.transform(new_data)
                prediction = svm_model.predict(new_data_robust)

                result_json = {"prediction": prediction[0]}
                json_path = os.path.join(app.config['STATIC_FOLDER'], 'result.json')
                with open(json_path, 'w') as json_file:
                    json.dump(result_json, json_file)

            return render_template('result.html', image=url_for('static', filename=name + '_AWB.png'), prediction=prediction[0])

        return "Invalid task! Task should be: 'AWB'", 400

    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    try:
        with open(os.path.join(app.config['STATIC_FOLDER'], 'result.json'), 'r') as json_file:
            result_json = json.load(json_file)
        return jsonify(result_json)
    except FileNotFoundError:
        return jsonify({"error": "Result file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
