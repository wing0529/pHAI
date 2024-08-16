import streamlit as st
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

# Streamlit app title and description
st.title('Deep White-Balance Editing')
st.write('This application changes the white balance of an input image using a deep learning model and predicts health status based on RGB values.')

# Streamlit sidebar for user input
st.sidebar.title('Options')

model_dir = st.sidebar.text_input('Model Directory', './models')
input_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
output_dir = st.sidebar.text_input('Output Directory', './output')
task = st.sidebar.selectbox('Task', ['AWB'])
max_size = st.sidebar.slider('Max Image Size', 256, 1024, 656)
device_option = st.sidebar.selectbox('Device', ['cuda', 'cpu','mobile'])
show_output = st.sidebar.checkbox('Show Output', True)
save_output = st.sidebar.checkbox('Save Output', True)


def load_model_and_scaler():
    try:
        rbs = RobustScaler()
        X_train = pd.read_csv('output_data.csv')[['R', 'G', 'B']]
        X_train_robust = rbs.fit_transform(X_train)
        svm_model = joblib.load('best_svm.joblib')
        return svm_model, rbs
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

def convert_to_image(array):
    if isinstance(array, np.ndarray):
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)  # Scale to 0-255 if needed
        return Image.fromarray(array)
    else:
        raise TypeError("Provided object is not a numpy array")

def extract_central_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Provided object is not a PIL.Image")
            
    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    central_pixel = image.getpixel((center_x, center_y))
            
    return central_pixel
        
# Device setup
device = torch.device('cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu')
st.write(f'Using device: {device}')

if input_image is not None:
    # Load image
    img = Image.open(input_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    name = os.path.splitext(input_image.name)[0]

    # Load model based on the selected task
    if task.lower() == 'awb':
        if os.path.exists(os.path.join(model_dir, 'net_awb.pth')):
            # Load AWB net
            net_awb = deep_wb_single_task.deepWBnet()
            net_awb.to(device=device)
            try:
                net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'), map_location=device), strict=False)
            except RuntimeError as e:
                st.error(f"Error loading model: {e}")
                st.stop()
            net_awb.eval()
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            net = deep_wb_model.deepWBNet()
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, _, _ = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
        else:
            st.error('Model not found!')
            st.stop()

        # Process image with the model
        out_awb = deep_wb(img, task=task.lower(), net_awb=net_awb, device=device, s=max_size)

        # Convert to PIL.Image if necessary
        if isinstance(out_awb, np.ndarray):
            out_awb_image = convert_to_image(out_awb)
        elif isinstance(out_awb, Image.Image):
            out_awb_image = out_awb
        else:
            st.error("Unexpected output type from deep_wb function.")
            st.stop()

        # Save and show output
        if save_output:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            result_path = os.path.join(output_dir, name + '_AWB.png')
            out_awb_image.save(result_path)
            st.success(f'Image saved to {result_path}')

        if show_output:
            st.image(out_awb_image, caption='Processed Image', use_column_width=True)

        rgb = extract_central_rgb(out_awb_image)
        R, G, B = rgb

        # Load the SVM model and scaler
        svm_model, rbs = load_model_and_scaler()
        if svm_model and rbs:
            # Scale the RGB values and predict
            new_data = pd.DataFrame({'R': [R], 'G': [G], 'B': [B]})
            new_data_robust = rbs.transform(new_data)
            prediction = svm_model.predict(new_data_robust)
            
            # Display the prediction
            st.write("Prediction result (judge):", prediction[0])
    else:
        st.error("Wrong task! Task should be: 'AWB'")

else:
    st.info("Please upload an image.")
