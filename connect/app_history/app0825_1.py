from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
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
import time

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './tmp/uploads'
SELECTED_FOLDER = './tmp/selected'
MODEL_DIR = './models'
OUTPUT_DIR = './output'
STATIC_FOLDER = './static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_DIR'] = MODEL_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['SELECTED_FOLDER'] = SELECTED_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SELECTED_FOLDER'], exist_ok=True)


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
    print(f"width={width}height={height}\n")
    return image.getpixel((center_x, center_y))

def wait_for_files(folder, num_files):
    while True:
        files = [f for f in os.listdir(folder) if allowed_file(f)]
        if len(files) >= num_files:
            return sorted(files)  # 파일을 정렬하여 순서를 유지합니다.
        time.sleep(1)  # 파일이 업로드될 시간을 대기합니다.      

@app.route('/', methods=['GET'])
def index():
    # 10개의 사진이 업로드될 때까지 대기
    files = wait_for_files(app.config['UPLOAD_FOLDER'], 10)

    #딥러닝
    task = 'editing'  # 고정된 작업 값
    device_option = 'cpu'  # 고정된 디바이스 옵션
    save_output = True  # 고정된 출력 저장 옵션
    target_color_temp = 6000  # 목표 색온도 (6000K)

    device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'사용할 디바이스: {device}')

    processed_images = []
    
    for file in files:
        img_path = os.path.join(app.config['SELECTED_FOLDER'], file)
        img = Image.open(img_path)
        name = os.path.splitext(file)[0]

        if task == 'editing':
            # Tungsten 및 Shade 모델 로드
            net_t = deep_wb_single_task.deepWBnet()
            net_s = deep_wb_single_task.deepWBnet()
            
            net_t.to(device)
            net_s.to(device)
            
            net_t.load_state_dict(torch.load(os.path.join(app.config['MODEL_DIR'], 'net_t.pth'), map_location=device), strict=False)
            net_s.load_state_dict(torch.load(os.path.join(app.config['MODEL_DIR'], 'net_s.pth'), map_location=device), strict=False)
            
            net_t.eval()
            net_s.eval()

            # 이미지 처리
            out_t, out_s = deep_wb(img, task=task, net_t=net_t, net_s=net_s, device=device)
            
            # 목표 색온도로 보정
            out = utls.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
            out_image = convert_to_image(out) if isinstance(out, np.ndarray) else out

            if save_output:
                os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
                result_image_path = os.path.join(app.config['OUTPUT_DIR'], name + '_editing.png')
                out_image.save(result_image_path)
            
            processed_images.append(out_image)
    
    # 중앙 픽셀의 RGB 값 계산
    r, g, b = extract_central_rgb(img)
    print(f'Average R: {r}, Average G: {g}, Average B: {b}')
    
    # 머신러닝 모델을 사용한 예측
    svm_model, rbs = load_model_and_scaler()
    if svm_model and rbs:
        new_data = pd.DataFrame({'R': [r], 'G': [g], 'B': [b]})
        new_data_robust = rbs.transform(new_data)
        prediction = svm_model.predict(new_data_robust)

        result_json = {"prediction": prediction[0]}
        json_path = os.path.join(app.config['STATIC_FOLDER'], 'result.json')
        with open(json_path, 'w') as json_file:
            json.dump(result_json, json_file)

        return render_template('result.html', prediction=prediction[0])
    
    return render_template('waiting.html')
    


@app.route('/upload/multiple', methods=['POST'])
def upload_multiple():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('files')
    if not files or any(file.filename == '' for file in files):
        return jsonify({'error': 'No selected file(s)'}), 400
    
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure the upload folder exists
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
            
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    return jsonify({'message': 'File uploaded successfully'}), 200    


@app.route('/result', methods=['GET'])
def result():
    try:
        with open(os.path.join(app.config['STATIC_FOLDER'], 'result.json'), 'r') as json_file:
            result_json = json.load(json_file)
        return jsonify(result_json)
    except FileNotFoundError:
        return jsonify({"error": "Result file not found"}), 404
    
@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_DIR'], filename)    

if __name__ == '__main__':
    app.run(debug=True)