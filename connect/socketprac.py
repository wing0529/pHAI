from flask import Flask, send_from_directory, render_template
from flask_socketio import SocketIO, emit
import torch
from arch import deep_wb_single_task, deep_wb_model
from sklearn.preprocessing import RobustScaler
import os
from PIL import Image
import pandas as pd
import joblib
import numpy as np
import json
import logging
import time

app = Flask(__name__)

# WebSocket 설정
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = './tmp/uploads'
MODEL_DIR = './models'
OUTPUT_DIR = './output'
STATIC_FOLDER = './static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_DIR'] = MODEL_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure the necessary folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

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

def wait_for_files(folder, num_files, timeout=30):
    start_time = time.time()
    while True:
        files = [f for f in os.listdir(folder) if allowed_file(f)]
        if len(files) >= num_files:
            return sorted(files)
        if time.time() - start_time > timeout:
            logging.error("Timeout waiting for files")
            return []
        time.sleep(1)

def calculate_average_rgb(images):
    total_r, total_g, total_b = 0, 0, 0
    num_images = len(images)
    for img in images:
        r, g, b = extract_central_rgb(img)
        total_r += r
        total_g += g
        total_b += b
    return total_r / num_images, total_g / num_images, total_b / num_images

@app.route('/')
def index():
    return "<h1>Welcome to the WebSocket Server</h1>"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')
    emit('response', {'data': 'Connected to WebSocket'})

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')

@socketio.on('upload_files')
def process_images(data):
    files = wait_for_files(app.config['UPLOAD_FOLDER'], 10)

    if not files:
        emit('response', {'error': 'No uploaded files'}, 400)
        return

    task = 'awb'
    device_option = 'cpu'
    save_output = True

    device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    processed_images = []

    for file in files:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        img = Image.open(img_path)
        name = os.path.splitext(file)[0]

        if task == 'awb':
            model_path = os.path.join(app.config['MODEL_DIR'], 'net_awb.pth')
            if os.path.exists(model_path):
                net_awb = deep_wb_single_task.deepWBnet()
                net_awb.to(device)
                net_awb.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            else:
                net_awb = deep_wb_model.deepWBNet()
            
            net_awb.eval()

            # deep_wb 함수가 정의되지 않았으므로, 아래 코드를 실행할 수 없음
            # out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device)
            # out_awb_image = convert_to_image(out_awb) if isinstance(out_awb, np.ndarray) else out_awb

            # 여기서 테스트를 위해 out_awb_image를 그냥 원본 이미지로 설정
            out_awb_image = img  # Placeholder

            if save_output:
                result_image_path = os.path.join(app.config['OUTPUT_DIR'], name + '_AWB.png')
                out_awb_image.save(result_image_path)

            processed_images.append(out_awb_image)

    avg_r, avg_g, avg_b = calculate_average_rgb(processed_images)

    svm_model, rbs = load_model_and_scaler()
    if svm_model and rbs:
        new_data = pd.DataFrame({'R': [avg_r], 'G': [avg_g], 'B': [avg_b]})
        new_data_robust = rbs.transform(new_data)
        prediction = svm_model.predict(new_data_robust)

        result_json = {"prediction": prediction[0]}
        emit('response', result_json)

        # Save result to a JSON file
        json_path = os.path.join(app.config['STATIC_FOLDER'], 'result.json')
        with open(json_path, 'w') as json_file:
            json.dump(result_json, json_file)
    else:
        emit('response', {'error': 'Model or scaler loading error'}, 500)

@socketio.on('get_result')
def result():
    try:
        with open(os.path.join(app.config['STATIC_FOLDER'], 'result.json'), 'r') as json_file:
            result_json = json.load(json_file)
        emit('response', result_json)
    except FileNotFoundError:
        emit('response', {"error": "Result file not found"}, 404)

@socketio.on('get_output_file')
def output_file(data):
    filename = data.get('filename')
    if not filename:
        emit('response', {"error": "Filename is required"}, 400)
        return
    try:
        return send_from_directory(app.config['OUTPUT_DIR'], filename)
    except FileNotFoundError:
        emit('response', {"error": "File not found"}, 404)

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)
