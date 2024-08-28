import os
import pandas as pd
from PIL import Image

def extract_central_rgb(image_path):
    image = Image.open(image_path)
    width, height = image.size
    central_pixel = image.getpixel((width // 2, height // 2))
    return central_pixel

def load_images_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(folder_path, filename))
    return image_paths

# data 경로
folder_path = "./dataset"
image_paths = load_images_from_folder(folder_path)

data = []

for image_path in image_paths:
    judge_value = 'error'
    
    # 이미지 파일 제목에서 pH 추출
    basename = os.path.splitext(os.path.basename(image_path))[0]
    pH_value = basename.split('(')[0]
    
    try:
        pH_float = float(pH_value)
        if 4.8 <= pH_float <= 5.6:
            judge_value = 'danger'
        elif 5.7 <= pH_float <= 6.0:
            judge_value = 'warning'
        elif 6.1 <= pH_float <= 6.4:
            judge_value = 'safe'
    except ValueError:
        pH_value = 'error'
    
    # 중앙 부분의 RGB 값을 추출
    rgb = extract_central_rgb(image_path)
    
    # 튜플로 데이터 생성
    image_data = (judge_value, pH_value, rgb[0], rgb[1], rgb[2])
    
    # 데이터 추가
    data.append(image_data)

# Extracting data for plotting
judge_values = [item[0] for item in data]
pH_values = [item[1] for item in data]
R_values = [item[2] for item in data]
G_values = [item[3] for item in data]
B_values = [item[4] for item in data]
    
# data dataframe로 저장    
df = pd.DataFrame(data, columns=['judge', 'pH', 'R', 'G', 'B'])

# DataFrame을 CSV 파일로 저장
csv_filename = 'output_dataset.csv'
df.to_csv(csv_filename, index=False)

# Optional: 결측값 확인
print(df.isnull().sum())

# Optional: Dummy variables 생성 (추후 분석에 활용할 수 있음)
print(pd.get_dummies(df['judge']))
