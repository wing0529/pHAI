from flask import Flask, render_template, request, jsonify, send_from_directory, url_for,redirect
from werkzeug.utils import secure_filename
import torch
from arch import deep_wb_single_task
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
import threading
import shutil
from flask_cors import CORS
 
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './tmp/uploads'
MODEL_DIR = './models'
SELECTED_FOLDER = './tmp/selected'
OUTPUT_DIR = './output'
STATIC_FOLDER = './static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_DIR'] = MODEL_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['SELECTED_FOLDER'] = SELECTED_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SELECTED_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'bmp'}



def convert_to_image(array):
    if isinstance(array, np.ndarray):
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)
    raise TypeError("Provided object is not a numpy array")

def extract_central_rgb(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {e}")
    
    if not isinstance(image, Image.Image):
        raise TypeError("Provided object is not a PIL.Image")

    # Rotate the image 90 degrees counter-clockwise
    rotated_image = image.rotate(90, expand=True)
    width, height = rotated_image.size
    #print(f"width : {width}, height : {height}")

    center_x = width // 2
    center_y = height // 2
    
    return rotated_image.getpixel((center_x, center_y))

def get_single_file(folder):
    # 'selected' 폴더에서 하나의 파일만 선택
    files = [f for f in os.listdir(folder) if allowed_file(f)]
    if files:
        return files[0]  # 첫 번째 파일 선택
    return None



    
def find_closest_image_to_mean(upload_folder, selected_folder):
    """Find and save the image with RGB values closest to the mean of all images."""
    images = []
    rgb_values = []
    
    # 업로드 폴더의 이미지 처리
    for file_name in os.listdir(upload_folder):
        if file_name.lower().endswith(('jpg', 'jpeg', 'bmp')):
            image_path = os.path.join(upload_folder, file_name)
            try:
                central_rgb = extract_central_rgb(image_path)
            except Exception as e:
                logging.error(f"Error processing image {file_name}: {e}")
                continue
            images.append((file_name, image_path))
            rgb_values.append(central_rgb)
    
    if not rgb_values:
        raise ValueError("No valid images found in the upload folder.")

    # RGB 값을 DataFrame으로 변환
    df = pd.DataFrame(rgb_values, columns=['R', 'G', 'B'])
    
    # 평균 RGB 값 계산
    mean_values = df.mean()
    
    # 평균 값으로부터의 유클리드 거리 계산
    distances = df.apply(lambda row: np.linalg.norm(row - mean_values), axis=1)
    
    # 평균에 가장 가까운 이미지의 인덱스 찾기
    closest_index = distances.idxmin()
    
    # 가장 가까운 이미지 선택
    closest_image = images[closest_index][1]
    closest_image_name = images[closest_index][0]
    output_image_path = os.path.join(selected_folder, closest_image_name)
    
    # selected에 뭐가 있으면 미리 지우기
    if os.path.exists(selected_folder):
        for filename in os.listdir(selected_folder):
            file_path = os.path.join(selected_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logging.info(f"Deleted directory: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")
                
    #selected에 저장
    try:
        Image.open(closest_image).save(output_image_path)
    except Exception as e:
        logging.error(f"Failed to save image {closest_image_name}: {e}")
        raise RuntimeError(f"Failed to save the image: {e}")
    
    logging.info(f"The image closest to the mean is {closest_image_name}. Saved at: {output_image_path}")
    
    # 업로드 폴더의 모든 이미지 삭제
    for file_name in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file_name}")
        except Exception as e:
            logging.error(f"Failed to delete file {file_name}: {e}")

def process_files():

    find_closest_image_to_mean(app.config['UPLOAD_FOLDER'], app.config['SELECTED_FOLDER'])

    # Assume 'selected' folder contains a single file for processing
    file = get_single_file(app.config['SELECTED_FOLDER'])
    if not file:
        logging.error("No valid image file found in selected folder")
        return

    img_path = os.path.join(app.config['SELECTED_FOLDER'], file)
    try:
        img = Image.open(img_path)
    except FileNotFoundError as e:
        logging.error(f"Error opening image file: {e}")
        return

    task = 'editing'
    device_option = 'cpu'
    target_color_temp = 6000

    device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f'Device in use: {device}')

    name = os.path.splitext(file)[0]

    net_t = deep_wb_single_task.deepWBnet()
    net_s = deep_wb_single_task.deepWBnet()
    
    net_t.to(device)
    net_s.to(device)
    
    net_t.load_state_dict(torch.load(os.path.join(app.config['MODEL_DIR'], 'net_t.pth'), map_location=device), strict=False)
    net_s.load_state_dict(torch.load(os.path.join(app.config['MODEL_DIR'], 'net_s.pth'), map_location=device), strict=False)
    
    net_t.eval()
    net_s.eval()

    # Process the image
    out_t, out_s = deep_wb(img, task=task, net_t=net_t, net_s=net_s, device=device)
    
    # Adjust to target color temperature
    out = utls.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
    out_image = convert_to_image(out) if isinstance(out, np.ndarray) else out
    output_dir = app.config['OUTPUT_DIR']
    
    # 예전에 만들어진 editing 파일 삭제
    if os.path.exists(output_dir):
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logging.info(f"기존 파일 삭제: {file_name}")
            except Exception as e:
                logging.error(f"파일 삭제 실패: {file_name}. 오류: {e}")

    # output_dir을 생성 (이미 존재하면 무시)
    os.makedirs(output_dir, exist_ok=True)
    
    # awb 이미지를 output_dir에 저장
    result_image_path = os.path.join(output_dir, name + '_editing.png')
    out_image.save(result_image_path)
    logging.info('AWB 생성 완료')

    # awb 가운데 좌표의 R,G,B 구하기
    try:
        r, g, b = extract_central_rgb(result_image_path)
    except Exception as e:
        logging.error(f"Error extracting central RGB: {e}")
        return

    logging.info(f'R: {r}, G: {g}, B: {b}')
    
    # 역류성 식도염 진단
    def load_model_and_scaler():
        try:
            svm_model = joblib.load('0806_svm.joblib')
            return svm_model
        except FileNotFoundError as e:
            logging.error(f"Error loading model or scaler: {e}")
            return None, None
    
    svm_model = load_model_and_scaler()
    if svm_model:
        new_data = pd.DataFrame({'R': [r], 'G': [g], 'B': [b]})
        prediction = svm_model.predict(new_data)
        logging.info(f'역류성 식도염 진단 = {prediction[0]}')
        result_json = {"prediction": prediction[0]}
        json_path = os.path.join(app.config['STATIC_FOLDER'], 'result.json')
        with open(json_path, 'w') as json_file:
            json.dump(result_json, json_file)
        logging.info('JSON 생성 완료')            

def wait_for_files(folder):
    while True:
        files = [f for f in os.listdir(folder) if allowed_file(f)]
        current_count = len(files)
        print(f"현재 업로드된 파일 수: {current_count}")
        if(current_count > 1):
            break
        time.sleep(10)  # 5초 대기 후 다시 확인


@app.route('/file/ready',methods=['GET'])
def is_file_ready():
    if os.path.exists(app.config['SELECTED_FOLDER']):
        return jsonify({'status':'ready'}),200
    else:
        return jsonify({'status': 'not_ready'}), 404
    
@app.route('/', methods=['GET'])
def index():
    result_json_path = os.path.join('static', 'result.json')
    if os.path.exists(result_json_path):
        os.remove(result_json_path)
    wait_for_files(app.config['UPLOAD_FOLDER'])
    # 파일 처리 스레드 시작
    thread = threading.Thread(target=process_files)
    thread.start()
    return jsonify({"message": "파일 처리가 시작되었습니다. 결과를 확인하려면 /results를 요청하세요."})
   

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
    return jsonify({'success':'파일이 성공적으로\n업로드되었습니다.'}), 200    


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