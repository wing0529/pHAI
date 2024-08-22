import streamlit as st
from PIL import Image
import os

def display_image():
    # Flask에서 이미지를 저장하는 폴더와 일치해야 합니다
    image_folder = 'C:\\Narae\\kw\\2024\\MyCapstone\\app\\pHAI\\connect\\tmp\\uploads'

    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    if image_files:
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path)
            st.image(img, caption='Uploaded Image', use_column_width=True)
            name = os.path.splitext(image_file)[0]
            st.write(f"Image name: {name}")
    else:
        st.write("No images found.")

if __name__ == "__main__":
    display_image()