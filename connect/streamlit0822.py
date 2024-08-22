import os
import streamlit as st

def display_image():
    image_folder = 'C:\\Narae\\kw\\2024\\MyCapstone\\app\\pHAI\\connect\\tmp\\uploads'  # 실제 이미지 폴더 경로로 변경
    if not os.path.exists(image_folder):
        st.error(f"The folder '{image_folder}' does not exist.")
        return
    
    try:
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not image_files:
            st.warning("No image files found in the folder.")
            return

        st.write(f"Found {len(image_files)} images:")
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            st.image(image_path, caption=image_file)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    display_image()
