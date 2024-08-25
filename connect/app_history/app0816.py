from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
import os
from PIL import Image
from arch import deep_wb_model
import utilities.utils as utls
from utilities.deepWB import deep_wb
import arch.splitNetworks as splitter
from arch import deep_wb_single_task
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import numpy as np
import json

app = Flask(__name__)

# Function to load the SVM model and scaler
def load_model_and_scaler():
    try:
        rbs = RobustScaler()
        X_train = pd.read_csv('output_data.csv')[['R', 'G', 'B']]
        X_train_robust = rbs.fit_transform(X_train)
        svm_model = joblib.load('best_svm.joblib')
        return svm_model, rbs
    except FileNotFoundError as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

# Function to convert a numpy array to a PIL image
def convert_to_image(array):
    if isinstance(array, np.ndarray):
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)  # Scale to 0-255 if needed
        return Image.fromarray(array)
    else:
        raise TypeError("Provided object is not a numpy array")

# Function to extract the central RGB value from an image
def extract_central_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Provided object is not a PIL.Image")

    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    central_pixel = image.getpixel((center_x, center_y))

    return central_pixel

# Flask route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_dir = request.form.get('model_dir', './models')
        output_dir = request.form.get('output_dir', './output')
        task = request.form.get('task', 'AWB').lower()
        max_size = int(request.form.get('max_size', 656))
        device_option = request.form.get('device_option', 'cpu')
        show_output = request.form.get('show_output', 'true') == 'true'
        save_output = request.form.get('save_output', 'true') == 'true'

        device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        if 'input_image' not in request.files:
            return "No file part", 400

        file = request.files['input_image']
        if file.filename == '':
            return "No selected file", 400

        if file and allowed_file(file.filename):
            # Load image
            img = Image.open(file)
            name = os.path.splitext(file.filename)[0]

            # Load model based on the selected task
            if task == 'awb':
                if os.path.exists(os.path.join(model_dir, 'net_awb.pth')):
                    # Load AWB net
                    net_awb = deep_wb_single_task.deepWBnet()
                    net_awb.to(device=device)
                    try:
                        net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'), map_location=device),
                                                strict=False)
                    except RuntimeError as e:
                        return f"Error loading model: {e}", 500
                    net_awb.eval()
                elif os.path.exists(os.path.join(model_dir, 'net.pth')):
                    net = deep_wb_model.deepWBNet()
                    net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
                    net_awb, _, _ = splitter.splitNetworks(net)
                    net_awb.to(device=device)
                    net_awb.eval()
                else:
                    return "Model not found!", 500

                # Process image with the model
                out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device, s=max_size)

                # Convert to PIL.Image if necessary
                if isinstance(out_awb, np.ndarray):
                    out_awb_image = convert_to_image(out_awb)
                elif isinstance(out_awb, Image.Image):
                    out_awb_image = out_awb
                else:
                    return "Unexpected output type from deep_wb function.", 500

                # Save and show output
                if save_output:
                    if not os.path.exists('static'):
                        os.makedirs('static')
                    result_image_path = os.path.join('static', name + '_AWB.png')
                    out_awb_image.save(result_image_path)

                # Extract RGB and make a prediction
                rgb = extract_central_rgb(out_awb_image)
                R, G, B = rgb

                # Load the SVM model and scaler
                svm_model, rbs = load_model_and_scaler()
                if svm_model and rbs:
                    # Scale the RGB values and predict
                    new_data = pd.DataFrame({'R': [R], 'G': [G], 'B': [B]})
                    new_data_robust = rbs.transform(new_data)
                    prediction = svm_model.predict(new_data_robust)

                    # Save prediction to a JSON file
                    result_json = {"prediction": prediction[0]}
                    json_path = os.path.join('static', 'result.json')
                    with open(json_path, 'w') as json_file:
                        json.dump(result_json, json_file)

                return render_template('result.html', image=url_for('static', filename=name + '_AWB.png'), prediction=prediction[0])

            else:
                return "Wrong task! Task should be: 'AWB'", 400
    return render_template('index.html')

# New route to return the prediction result in JSON format
@app.route('/result', methods=['GET'])
def result():
    try:
        with open('static/result.json', 'r') as json_file:
            result_json = json.load(json_file)
        return jsonify(result_json)
    except FileNotFoundError:
        return jsonify({"error": "Result file not found"}), 404

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True)
