from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import torch
from arch import deep_wb_single_task
from arch import deep_wb_model
from sklearn.preprocessing import RobustScaler
import utilities.utils as utls
from utilities.deepWB import deep_wb
import os
from PIL import Image
import pandas as pd
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import json
import logging

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 설정
UPLOAD_FOLDER = './tmp/uploads'
MODEL_DIR = './models'
OUTPUT_DIR = './output'
STATIC_FOLDER = './static'  # JSON 파일을 저장할 폴더
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_DIR'] = MODEL_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# 로깅 설정
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
        logging.error(f"모델 또는 스케일러 로딩 오류: {e}")
        return None, None

def convert_to_image(array):
    if isinstance(array, np.ndarray):
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)
    raise TypeError("제공된 객체는 numpy 배열이 아닙니다.")

def extract_central_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("제공된 객체는 PIL.Image가 아닙니다.")

    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    return image.getpixel((center_x, center_y))

'''
@app.route('/', methods=['GET'])
def index():
    # 미리 업로드된 이미지 경로 정의
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'photo.jpg')

    if not os.path.isfile(input_image_path):
        return "업로드된 이미지가 없습니다", 400

    img = Image.open(input_image_path)
    name = os.path.splitext('photo.jpg')[0]

    task = 'awb'  # 고정된 작업 값
    device_option = 'cpu'  # 고정된 디바이스 옵션
    save_output = True  # 고정된 출력 저장 옵션

    device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'사용할 디바이스: {device}')

    if task == 'awb':
        net_awb = deep_wb_single_task.deepWBnet() if os.path.exists(os.path.join(app.config['MODEL_DIR'], 'net_awb.pth')) else deep_wb_model.deepWBNet()
        net_awb.to(device)
        net_awb.load_state_dict(torch.load(os.path.join(app.config['MODEL_DIR'], 'net_awb.pth'), map_location=device), strict=False)
        net_awb.eval()

        # 이미지 처리
        out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device)
        out_awb_image = convert_to_image(out_awb) if isinstance(out_awb, np.ndarray) else out_awb

        if save_output:
            os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
            result_image_path = os.path.join(app.config['OUTPUT_DIR'], name + '_AWB.png')
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

        return render_template('result.html', image=url_for('output_file', filename=name + '_AWB.png'), prediction=prediction[0])

    return "유효하지 않은 작업입니다! 작업은 'AWB' 여야 합니다.", 400
'''

@app.route('/', methods=['GET'])
def index():
    # 미리 업로드된 이미지 경로 정의
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'photo.jpg')

    img = Image.open(input_image_path)
    name = os.path.splitext('photo.jpg')[0]

    task = 'awb'  # 고정된 작업 값
    device_option = 'cpu'  # 고정된 디바이스 옵션
    save_output = True  # 고정된 출력 저장 옵션

    device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'사용할 디바이스: {device}')

    if task == 'awb':
        net_awb = deep_wb_single_task.deepWBnet() if os.path.exists(os.path.join(app.config['MODEL_DIR'], 'net_awb.pth')) else deep_wb_model.deepWBNet()
        net_awb.to(device)
        net_awb.load_state_dict(torch.load(os.path.join(app.config['MODEL_DIR'], 'net_awb.pth'), map_location=device), strict=False)
        net_awb.eval()

        # 이미지 처리
        out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device)
        out_awb_image = convert_to_image(out_awb) if isinstance(out_awb, np.ndarray) else out_awb

        if save_output:
            os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
            result_image_path = os.path.join(app.config['OUTPUT_DIR'], name + '_AWB.png')
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

        return render_template('result.html', image=url_for('output_file', filename=name + '_AWB.png'), prediction=prediction[0])

    return "유효하지 않은 작업입니다! 작업은 'AWB' 여야 합니다.", 400



@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다'}), 400
    if file and allowed_file(file.filename):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # 업로드 폴더가 존재하는지 확인
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # JSON 응답 반환
        return jsonify({'message': '파일이 성공적으로 업로드되었습니다'}), 200
    else:
        return jsonify({'error': '유효하지 않은 파일 형식입니다'}), 400

@app.route('/result', methods=['GET'])
def result():
    try:
        with open(os.path.join(app.config['STATIC_FOLDER'], 'result.json'), 'r') as json_file:
            result_json = json.load(json_file)
        return jsonify(result_json)
    except FileNotFoundError:
        return jsonify({"error": "결과 파일이 없습니다"}), 404

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_DIR'], filename)

if __name__ == '__main__':
    app.run(debug=True)
