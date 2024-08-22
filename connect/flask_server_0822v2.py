import logging
from flask import Flask, request, jsonify, url_for
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

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_DIR = './models'
OUTPUT_DIR = './output'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_DIR'] = MODEL_DIR
app.config['OUTPUT_DIR'] = OUTPUT_DIR
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Set up logging
log_file = 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Log to console as well
    ]
)

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error('No file part in the request.')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logging.error('No selected file.')
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        file.save(file_path)
        img = Image.open(file_path)
        name = os.path.splitext(filename)[0]

        task = 'awb'
        device_option = 'cpu'
        save_output = True

        device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')

        # Model loading
        if task == 'awb':
            model_path = os.path.join(app.config['MODEL_DIR'], 'net_awb.pth')
            if os.path.exists(model_path):
                net_awb = deep_wb_single_task.deepWBnet()
            else:
                net_awb = deep_wb_model.deepWBNet()
                
            net_awb.to(device)
            net_awb.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            net_awb.eval()

            # Image processing
            out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device)
            out_awb_image = convert_to_image(out_awb) if isinstance(out_awb, np.ndarray) else out_awb

            if save_output:
                if not os.path.exists(app.config['STATIC_FOLDER']):
                    os.makedirs(app.config['STATIC_FOLDER'])
                
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

            return jsonify({
                'image_url': url_for('static', filename=name + '_AWB.png'),
                'prediction': prediction[0]
            })

        return jsonify({"error": "Invalid task! Task should be: 'AWB'"}), 400
    
    logging.error('Invalid file type.')
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/result', methods=['GET'])
def result():
    try:
        json_path = os.path.join(app.config['STATIC_FOLDER'], 'result.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError("Result file does not exist.")
        
        with open(json_path, 'r') as json_file:
            result_json = json.load(json_file)
        return jsonify(result_json)
    except FileNotFoundError:
        logging.error('Result file not found.')
        return jsonify({"error": "Result file not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
    #app.run(debug=True)
